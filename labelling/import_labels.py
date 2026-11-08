"""
Turn the Dynamic World exports into training masks on the radar grid.

Run labelling/gee/export_dynamic_world.js in the Earth Engine Code Editor first, download
the files from the "echoscape_dw" Drive folder into data/dw_raw/, then:

    .venv\\Scripts\\python.exe labelling\\import_labels.py               # every region found
    .venv\\Scripts\\python.exe labelling\\import_labels.py --aoi iowa
    .venv\\Scripts\\python.exe labelling\\import_labels.py --min-confidence 0.25

Writes data/labels/<region>/<region>_labels.tif - single band, values
0-9, 0 = unlabelled - on EXACTLY the grid of that region's radar files.

HOW 10 m BECOMES 50 m

One 50 m pixel is 5 x 5 = 25 Dynamic World pixels. Rather than hand that to a
generic resampler, each export is first laid onto an exact 5x-finer copy of the
radar grid, then every 5 x 5 block votes:

    1. a 10 m pixel is usable if Dynamic World was at least DW_MIN_CONFIDENCE
       sure of it over the year, and its class maps to one of ours
    2. the block is labelled only if at least DW_MIN_VALID of its 25 pixels
       are usable, and the winner holds at least DW_MIN_PURITY of those
    3. otherwise it is left unlabelled
    4. a pixel called cropland whose year never greens up past
       DW_CROP_MIN_PEAK is relabelled grass (or bare) - see below

Doing it this way makes the output grid identical to the radar's by
construction rather than by hope, and it refuses mixed pixels - field edges,
shorelines, the edge of a town - which is where external labels are wrong most
often. An unlabelled pixel costs nothing: it is simply left out of the loss.

CROPLAND MUST GREEN UP

Dynamic World judges each image on its own, and dry seasonal grassland fools
it: it labelled the Mongolian steppe 46% cropland at a yearly peak NDVI of
0.23, where real cropland peaks at 0.71-0.81. Cropland is sown, grown and
harvested, so somewhere in the year it carries a full canopy; a "cropland"
pixel that never does is relabelled from the Sentinel-2 greenness already on
the radar grid. Same thresholds check_labels.py checks every mask against.
--no-crop-rule turns it off, to measure what it changed.

"""

import argparse
import json
import warnings

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

import sys
from pathlib import Path

# Every script runs from its own folder, where Python cannot see config.py
# or the other packages. Putting the project root first fixes that the same
# way for every script, however it is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

FACTOR = 5          # 50 m / 10 m
CLASSES = range(1, config.NUM_CLASSES)


def raw_file(aoi):
    """The export(s) for one region. Earth Engine names them dw2021_<aoi>.tif,
    and splits very large exports into numbered shards; both are handled."""
    found = sorted(config.DW_RAW.glob(f"dw2021_{aoi}*.tif"))
    # the shard pattern must not let "dw2021_nile" pick up another region
    return [p for p in found
            if p.stem == f"dw2021_{aoi}" or p.stem.startswith(f"dw2021_{aoi}-")]


def radar_grid(aoi):
    files = sorted((config.S1_MONTHLY / aoi).glob("*.tif"))
    if not files:
        raise SystemExit(f"{aoi}: no radar - the radar defines the grid.")
    with rasterio.open(files[0]) as src:
        return src.profile.copy(), files[0].name


def read_raw(paths):
    """The three exported bands as one array, merging shards if there are any."""
    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            return src.read(), src.transform, src.crs
    sources = [rasterio.open(p) for p in paths]
    try:
        data, transform = merge(sources, nodata=0)
        return data, transform, sources[0].crs
    finally:
        for s in sources:
            s.close()


def to_fine_grid(band, src_transform, src_crs, profile):
    """Nearest-neighbour onto a grid exactly FACTOR times finer than the radar.

    Nearest, not averaging: these are class codes and confidences. Averaging a
    water code (1) with a built-up code (7) would invent class 4.
    """
    fine = np.zeros((profile["height"] * FACTOR, profile["width"] * FACTOR),
                    dtype="uint8")
    reproject(source=band, destination=fine,
              src_transform=src_transform, src_crs=src_crs, src_nodata=0,
              dst_transform=profile["transform"] * Affine.scale(1 / FACTOR),
              dst_crs=profile["crs"], dst_nodata=0,
              resampling=Resampling.nearest)
    return fine


def vote(classes_fine, usable_fine, height, width):
    """Majority vote in every FACTOR x FACTOR block. Returns the 50 m mask."""
    blocks = (height, FACTOR, width, FACTOR)
    usable = usable_fine.reshape(blocks)
    cls = classes_fine.reshape(blocks)

    counts = np.stack([((cls == c) & usable).sum(axis=(1, 3)) for c in CLASSES])
    valid = counts.sum(axis=0)
    winner = counts.argmax(axis=0) + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(valid > 0, counts.max(axis=0) / valid, 0.0)

    keep = ((valid >= config.DW_MIN_VALID * FACTOR * FACTOR)
            & (purity >= config.DW_MIN_PURITY))
    return np.where(keep, winner, 0).astype("uint8")


def write_safely(mask, profile, path):
    """Write beside, reopen, prove the grid
    survived, then swap in. A mask on the wrong grid describes the wrong
    ground and nothing downstream would ever notice."""
    profile = profile.copy()
    profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mask, 1)
    with rasterio.open(tmp) as src:
        same = (src.crs == profile["crs"]
                and (src.width, src.height) == (profile["width"], profile["height"])
                and np.allclose(np.array(src.transform), np.array(profile["transform"]),
                                atol=1e-12)
                and (src.read(1) == mask).all())
    if not same:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{path.name}: grid or content changed on write - aborted")
    tmp.replace(path)


def yearly_peak_ndvi(aoi, profile):
    """How green each 50 m pixel gets at its best, over 2021.

    Read from the Sentinel-2 monthly composites, which fetch_s2.py already wrote
    onto this exact radar grid - so the comparison is pixel for pixel. The 85th
    percentile of the months rather than the maximum, so that a single month
    brightened by thin cloud cannot decide it. Returns None if there is no
    optical data; the rule is then skipped rather than guessed.
    """
    files = sorted((config.S2_MONTHLY / aoi).glob("*.tif"))
    if not files:
        return None
    months = []
    for path in files:
        with rasterio.open(path) as src:
            same_grid = ((src.width, src.height) == (profile["width"], profile["height"])
                         and src.crs == profile["crs"]
                         and np.allclose(np.array(src.transform),
                                         np.array(profile["transform"]), atol=1e-9))
            if not same_grid:
                raise SystemExit(f"{path.name} is not on the radar grid - run "
                                 f"acquisition/check_alignment.py --aoi {aoi}")
            red = src.read(3).astype("float32")
            nir = src.read(4).astype("float32")
            seen = src.read(7)
        ndvi = (nir - red) / (nir + red + 1e-9)
        ndvi[seen == 0] = np.nan
        months.append(ndvi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(np.stack(months), 85, axis=0)


def cropland_must_green(mask, peak):
    """Relabel Dynamic World "cropland" that never greens up.

    Cropland is sown, grown and harvested, so at some point in the year it
    carries a full canopy. A pixel called cropland whose year never reaches
    DW_CROP_MIN_PEAK is something Dynamic World misread - most of all dry
    steppe, which it judges from single images and mistakes for a field. It
    becomes grass if anything grows there, bare if not.

    Pixels with no optical reading are left exactly as they were: no evidence,
    no change.
    """
    if peak is None:
        return mask, 0, 0
    crop = mask == 4
    with np.errstate(invalid="ignore"):                # NaN compares False
        never_green = crop & (peak < config.DW_CROP_MIN_PEAK)
        to_grass = never_green & (peak >= config.DW_VEG_MIN_PEAK)
        to_bare = never_green & (peak < config.DW_VEG_MIN_PEAK)
    mask = mask.copy()
    mask[to_grass] = 5
    mask[to_bare] = 6
    return mask, int(to_grass.sum()), int(to_bare.sum())


def import_one(aoi, min_conf, crop_rule=True):
    paths = raw_file(aoi)
    if not paths:
        return None
    profile, radar_name = radar_grid(aoi)
    data, src_transform, src_crs = read_raw(paths)
    if data.shape[0] < 3:
        raise SystemExit(f"{aoi}: expected 3 bands (label, confidence, count), "
                         f"found {data.shape[0]}. Re-export with the provided script.")

    label = to_fine_grid(data[0], src_transform, src_crs, profile)
    conf = to_fine_grid(data[1], src_transform, src_crs, profile)
    seen = to_fine_grid(data[2], src_transform, src_crs, profile)

    lut = np.zeros(256, dtype="uint8")
    for exported, ours in config.DW_CLASS_MAP.items():
        lut[exported] = ours
    ours_fine = lut[label]
    usable = (ours_fine > 0) & (conf >= round(min_conf * 100))

    mask = vote(ours_fine, usable, profile["height"], profile["width"])

    # Applied BEFORE writing, so the mask on disk and every number reported
    # are the corrected ones.
    to_grass = to_bare = 0
    if crop_rule:
        mask, to_grass, to_bare = cropland_must_green(
            mask, yearly_peak_ndvi(aoi, profile))

    out = config.LABELS / aoi / f"{aoi}_labels.tif"
    write_safely(mask, profile, out)

    result = {"aoi": aoi, "radar": radar_name,
              "crop_to_grass": to_grass, "crop_to_bare": to_bare,
              "coverage": float((mask > 0).mean()),
              "median_views": float(np.median(seen[seen > 0])) if (seen > 0).any() else 0.0,
              "classes": {config.CLASSES[c]: float((mask == c).mean())
                          for c in CLASSES if (mask == c).any()}}

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default=None, choices=list(config.AOIS))
    parser.add_argument("--min-confidence", type=float,
                        default=config.DW_MIN_CONFIDENCE,
                        help="drop 10 m pixels Dynamic World was less sure of")
    parser.add_argument("--no-crop-rule", action="store_true",
                        help="keep Dynamic World's cropland as-is (for comparison)")
    args = parser.parse_args()

    if not config.DW_RAW.exists():
        raise SystemExit(f"No {config.DW_RAW}. Run labelling/gee/export_dynamic_world.js in "
                         f"Earth Engine and download the exports there first.")

    names = [args.aoi] if args.aoi else list(config.AOIS)
    results, missing = [], []

    print(f"confidence >= {args.min_confidence:.2f}, block needs "
          f">= {config.DW_MIN_VALID:.0%} usable and >= {config.DW_MIN_PURITY:.0%} agreeing")
    print("cropland rule: " + ("OFF" if args.no_crop_rule else
          f"cropland whose yearly peak NDVI stays under {config.DW_CROP_MIN_PEAK:.2f} "
          f"is relabelled") + "\n")
    print(f"{'region':<14}{'labelled':>9}{'views':>7}{'crop->grass':>12}")
    print("-" * 74)
    for name in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = import_one(name, args.min_confidence,
                           crop_rule=not args.no_crop_rule)
        if r is None:
            missing.append(name)
            continue
        results.append(r)
        flag = "   <-- little survived" if r["coverage"] < 0.3 else ""
        moved = r["crop_to_grass"] + r["crop_to_bare"]
        moved_txt = f"{moved:,}" if moved else "-"
        print(f"{name:<14}{100 * r['coverage']:>8.1f}%{r['median_views']:>7.0f}"
              f"{moved_txt:>12}{flag}")

    if missing:
        print(f"\nno export found for: {', '.join(missing)}")
        print(f"   expected data/dw_raw/dw2021_<region>.tif")

    # ---- pooled class balance, the thing that broke training before ----------
    if results:
        print("\nclass balance across everything imported:")
        for c in CLASSES:
            share = np.mean([r["classes"].get(config.CLASSES[c], 0) for r in results])
            print(f"   {config.CLASSES[c]:<24}{100 * share:>6.1f}% of pixels")


    if results:
        config.OUTPUTS.mkdir(parents=True, exist_ok=True)
        out = config.OUTPUTS / "label_import.json"
        out.write_text(json.dumps({"min_confidence": args.min_confidence,
                                   "crop_rule": not args.no_crop_rule,
                                   "crop_min_peak": config.DW_CROP_MIN_PEAK,
                                   "regions": results}, indent=2))
        print(f"\nwrote {out}")

    print("\nnext:  .venv\\Scripts\\python.exe training\\build_dataset.py")


if __name__ == "__main__":
    main()
