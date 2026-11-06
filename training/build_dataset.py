"""
Turn labelled regions into training tiles.

For every region that has a label mask, and for every month of radar that
exists for it, cut matching 128x128 tiles of radar and label, and write them
out with a manifest.

    .venv\\Scripts\\python.exe training\\build_dataset.py              # all regions
    .venv\\Scripts\\python.exe training\\build_dataset.py --dry-run    # just count
    .venv\\Scripts\\python.exe training\\build_dataset.py --aoi iowa

WHAT THE MODEL SEES

Three channels, all from radar, all in decibels and scaled by the fixed ranges
in config.py:

    VV          how brightly the ground reflects, same polarisation
    VH          cross-polarised - sensitive to volume scattering, so trees
    VV - VH     the ratio, in dB. Separates rough-but-bright from rough-and-
                volumetric, and gives three channels so a pretrained
                ImageNet encoder can be dropped in unchanged for experiment 2.

WHY ONE TILE PER REGION-MONTH

The mask is annual - one answer per place for the whole year - but the radar
is monthly. So each month of radar becomes its own training example sharing
the same labels. That is deliberate, and it is the heart of the brief's own
tension: the model is trained month-invariant and then asked for month-varying
output. It also multiplies the data twelvefold for free.

SPLITS ARE BY WHOLE REGION, NEVER BY TILE

Neighbouring tiles overlap in content even when they do not overlap in pixels:
the same river runs through both. Splitting at random would put near-identical
tiles on both sides and inflate the score. config.TEST_AOIS and VAL_AOIS name
whole regions, and nothing from them enters training.

NODATA

Radar pixels no satellite saw (band 3 == 0) are forced to label 0, unlabelled,
whatever the mask says. A pixel with no measurement cannot teach anything, and
leaving it in would train the model on whatever the fill value happened to be.
"""

import argparse
import csv
import warnings

import numpy as np
import rasterio

import sys
from pathlib import Path

# Every script runs from its own folder, where Python cannot see config.py
# or the other packages. Putting the project root first fixes that the same
# way for every script, however it is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def scale(values, low_high):
    """Squash a decibel band into roughly 0-1 using a FIXED range."""
    low, high = low_high
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def radar_channels(path):
    """Read one radar file and return (3, H, W) float32, plus the seen-mask."""
    with rasterio.open(path) as src:
        bands = src.read().astype("float32")
    vv, vh, looks = bands[0], bands[1], bands[2]

    # Linear power to decibels. Clip first: a zero or negative sneaks through
    # as -inf and then quietly becomes nan in every later sum.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vv_db = 10.0 * np.log10(np.clip(vv, 1e-6, None))
        vh_db = 10.0 * np.log10(np.clip(vh, 1e-6, None))

    stack = np.stack([scale(vv_db, config.VV_DB_RANGE),
                      scale(vh_db, config.VH_DB_RANGE),
                      scale(vv_db - vh_db, config.RATIO_DB_RANGE)])
    stack = np.nan_to_num(stack, nan=0.0)
    return stack.astype("float32"), looks > 0


def offsets(total, tile, stride):
    """Where to start each tile along one axis, covering the whole edge.

    A plain range leaves a strip at the far edge unused - 600 with 128 stops
    at 512 and throws away 88 pixels. Adding one final tile flush with the
    edge keeps that ground, at the cost of it overlapping its neighbour.
    """
    starts = list(range(0, total - tile + 1, stride))
    if not starts:
        return []
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def split_of(aoi):
    if aoi in config.TEST_AOIS:
        return "test"
    if aoi in config.VAL_AOIS:
        return "val"
    return "train"


def grid_check(label_path, radar_path):
    with rasterio.open(label_path) as a, rasterio.open(radar_path) as b:
        same = (a.crs == b.crs and (a.width, a.height) == (b.width, b.height)
                and np.allclose(np.array(a.transform), np.array(b.transform),
                                atol=1e-9))
    return same


def build(aoi, dry_run, rows):
    label_path = config.LABELS / aoi / f"{aoi}_labels.tif"
    if not label_path.exists():
        return 0, 0, np.zeros(config.NUM_CLASSES, dtype=np.int64)

    with rasterio.open(label_path) as src:
        mask = src.read(1)

    months = sorted((config.S1_MONTHLY / aoi).glob("*.tif"))
    if not months:
        print(f"  {aoi}: labelled but no radar - skipped")
        return 0, 0, np.zeros(config.NUM_CLASSES, dtype=np.int64)

    # The check that matters most. A mask on a different grid describes the
    # wrong ground and nothing downstream would ever notice.
    if not grid_check(label_path, months[0]):
        raise SystemExit(f"{aoi}: label grid does not match {months[0].name}. "
                         f"Run labelling/check_labels.py --aoi {aoi}.")

    split = split_of(aoi)
    out_dir = config.TILES / split
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    height, width = mask.shape
    row_starts = offsets(height, config.TILE_PX, config.TILE_STRIDE)
    col_starts = offsets(width, config.TILE_PX, config.TILE_STRIDE)
    need = config.MIN_LABELLED_FRACTION * config.TILE_PX * config.TILE_PX

    kept = dropped = 0
    counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)

    for month_path in months:
        channels, seen = radar_channels(month_path)
        month = config.month_of(month_path)

        # A pixel no satellite saw teaches nothing, so it is unlabelled here
        # regardless of what the human drew on it.
        labels = np.where(seen, mask, 0).astype("uint8")

        for row in row_starts:
            for col in col_starts:
                y = labels[row:row + config.TILE_PX, col:col + config.TILE_PX]
                if (y > 0).sum() < need:
                    dropped += 1
                    continue
                x = channels[:, row:row + config.TILE_PX,
                             col:col + config.TILE_PX]
                name = f"{aoi}_{month}_r{row:03d}_c{col:03d}.npz"
                if not dry_run:
                    np.savez_compressed(out_dir / name,
                                        x=x.astype("float16"), y=y)
                rows.append({"file": f"{split}/{name}", "split": split,
                             "aoi": aoi, "month": month, "row": row, "col": col,
                             "labelled_px": int((y > 0).sum())})
                counts += np.bincount(y.ravel(), minlength=config.NUM_CLASSES)
                kept += 1

    return kept, dropped, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default=None, choices=list(config.AOIS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = [args.aoi] if args.aoi else list(config.AOIS)

    # A full rebuild starts from empty. Tiles are named by region, month and
    # position, so a rebuild with different settings overwrites most of them -
    # but any tile that passed the labelled-fraction cut last time and does not
    # now would be left behind, and training reads every .npz it finds. That
    # would mix two builds into one training set without a word. Only the
    # script's own .npz output is removed, and only on a full build: with --aoi
    # the other regions' tiles must survive.
    if not args.dry_run and not args.aoi:
        stale = [f for split in ("train", "val", "test")
                 for f in (config.TILES / split).glob("*.npz")]
        for f in stale:
            f.unlink()
        if stale:
            print(f"cleared {len(stale):,} tiles from the previous build\n")
    rows = []
    totals = {"train": 0, "val": 0, "test": 0}
    counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)

    print(f"{'region':<14}{'split':<7}{'tiles':>7}{'dropped':>9}")
    print("-" * 74)
    for name in names:
        kept, dropped, got = build(name, args.dry_run, rows)
        if kept or dropped:
            print(f"{name:<14}{split_of(name):<7}{kept:>7,}{dropped:>9,}")
            totals[split_of(name)] += kept
            counts += got
    print("-" * 74)
    print(f"{'TOTAL':<14}{'':<7}{sum(totals.values()):>7,}")
    for split in ("train", "val", "test"):
        print(f"   {split:<6} {totals[split]:>6,} tiles")

    labelled = counts[1:].sum()
    if labelled:
        print("\nclass balance over the kept tiles:")
        for class_id in range(1, config.NUM_CLASSES):
            share = 100 * counts[class_id] / labelled
            bar = "#" * max(int(share / 2), 0)
            print(f"   {config.CLASSES[class_id]:<24}{counts[class_id]:>10,}"
                  f"{share:>7.1f}%  {bar}")
        unlabelled = 100 * counts[0] / counts.sum()
        print(f"   {'(unlabelled, ignored)':<24}{counts[0]:>10,}{unlabelled:>7.1f}%")

    if not args.dry_run and rows:
        config.TILES.mkdir(parents=True, exist_ok=True)
        manifest = config.TILES / "manifest.csv"
        with open(manifest, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {manifest}")
    elif args.dry_run:
        print("\ndry run - nothing written.")


if __name__ == "__main__":
    main()
