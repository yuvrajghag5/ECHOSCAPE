"""
Check label masks before they become training data.

Two things are checked, and they fail very differently.

1. THE GRID. A label mask must sit on exactly the same grid as the radar it
   describes. If it does not, every label points at the wrong ground, nothing
   crashes, and the only symptom is a mediocre model. This is a hard failure.

2. WHETHER THE LABELS ARE PLAUSIBLE. Each labelled zone is compared against
   what the optical imagery says is growing there, using the highest NDVI the
   pixel reached in the whole year. A zone called "trees" that never greened up
   is wrong; a zone called "water" that turns green in July is wrong. These are
   warnings, not failures - the tool cannot know which of the two is right.

Why the ANNUAL MAXIMUM rather than one month: the masks are annual, but NDVI is
seasonal. Cropland in winter is bare soil and deciduous forest in winter is bare
branches, so a single-month check would condemn correct labels.

    .venv\\Scripts\\python.exe labelling\\check_labels.py              # every mask found
    .venv\\Scripts\\python.exe labelling\\check_labels.py --aoi arabia # just one
"""

import argparse
import warnings

import numpy as np
import rasterio
from scipy import ndimage

import sys
from pathlib import Path

# Every script runs from its own folder, where Python cannot see config.py
# or the other packages. Putting the project root first fixes that the same
# way for every script, however it is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# Ignore anything smaller than this. Single stray pixels are the erase key's
# leftovers, not an opinion about the landscape.
MIN_ZONE_PX = 50

# What the yearly PEAK greenness should look like for each class. Deliberately
# wide - the point is to catch labels that are plainly contradicted, not to
# second-guess honest judgement calls.
PLAUSIBLE_NDVI_MAX = {
    1: (None, 0.30),   # water - should never green up
    2: (None, 0.60),   # built-up - cities have parks, so the ceiling is high
    3: (0.55, None),   # trees - must green up strongly at some point
    4: (0.40, None),   # cropland - must have a growing season
    5: (0.15, 0.80),   # grass
    6: (None, 0.30),   # bare - should stay brown all year
    7: (None, 0.80),   # shrub & scrub - arid scrub can stay low, so no floor
    8: (None, None),   # flooded vegetation - anything from open marsh to swamp forest
    9: (None, 0.30),   # snow & ice - should never green up
}

# The one pair the optical indices genuinely cannot separate: desert and city
# overlap almost completely in every index we have. Say so rather than
# pretending otherwise.
UNDECIDABLE = {2, 6}


def radar_for(aoi):
    """The radar file whose grid this region's labels must match."""
    folder = config.S1_MONTHLY / aoi
    found = sorted(folder.glob("*.tif"))
    if not found:
        raise SystemExit(f"No radar for {aoi} in {folder}. Run acquisition/run_all.py first.")
    return found[0]


def grid_matches(label_path, radar_path):
    """Compare the two grids field by field. Returns a list of differences."""
    with rasterio.open(label_path) as a, rasterio.open(radar_path) as b:
        problems = []
        if a.crs != b.crs:
            problems.append(f"CRS {a.crs} vs {b.crs}")
        if (a.width, a.height) != (b.width, b.height):
            problems.append(f"size {a.width}x{a.height} vs {b.width}x{b.height}")
        # Transforms are floats, so compare with a tolerance rather than ==.
        if not np.allclose(np.array(a.transform), np.array(b.transform), atol=1e-9):
            problems.append("transform differs")
        return problems


def annual_ndvi_max(aoi):
    """Highest NDVI each pixel reached across every optical month we have.

    Months where a pixel was never seen cloud-free are excluded, otherwise an
    unobserved pixel would read as a confident zero.
    """
    folder = config.S2_MONTHLY / aoi
    months = sorted(folder.glob("*.tif"))
    if not months:
        return None, 0
    stack = []
    for path in months:
        with rasterio.open(path) as src:
            bands = src.read().astype("float32")
        red, nir, seen = bands[2], bands[3], bands[6]
        ndvi = (nir - red) / (nir + red + 1e-9)
        ndvi[seen == 0] = np.nan
        stack.append(ndvi)
    # A pixel that was never seen cloud-free in any month is all-NaN, which
    # numpy warns about. That is expected here, not a problem worth printing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.dstack(stack), axis=2), len(months)


def check_one(aoi):
    """Report on one region. Returns True if the grid is sound."""
    label_path = config.LABELS / aoi / f"{aoi}_labels.tif"
    if not label_path.exists():
        return None

    print(f"\n{'=' * 74}\n{aoi.upper()}\n{'=' * 74}")

    radar_path = radar_for(aoi)
    problems = grid_matches(label_path, radar_path)
    if problems:
        print("  GRID MISMATCH - these labels describe the wrong ground:")
        for p in problems:
            print(f"     {p}")
        print(f"     compared against {radar_path.name}")
        return False
    print(f"  grid OK (matches {radar_path.name})")

    with rasterio.open(label_path) as src:
        mask = src.read(1)

    labelled = mask > 0
    print(f"  labelled {100 * labelled.mean():.1f}% of the image "
          f"({int(labelled.sum()):,} px)")
    if not labelled.any():
        print("  nothing labelled yet.")
        return True

    print("\n  by class:")
    for class_id, name in config.CLASSES.items():
        if class_id == 0:
            continue
        count = int((mask == class_id).sum())
        if count:
            print(f"     {name:<24} {count:>8,}  {100 * count / mask.size:5.1f}%")

    ndvi_max, n_months = annual_ndvi_max(aoi)
    if ndvi_max is None:
        print("\n  no optical imagery, so plausibility cannot be checked.")
        return True

    print(f"\n  plausibility, from the yearly peak greenness over {n_months} months:")
    warnings = 0
    for class_id in sorted(set(np.unique(mask)) - {0}):
        low, high = PLAUSIBLE_NDVI_MAX.get(int(class_id), (None, None))
        zones, count = ndimage.label(mask == class_id)
        for zone_id in range(1, count + 1):
            selected = zones == zone_id
            size = int(selected.sum())
            if size < MIN_ZONE_PX:
                continue
            with np.errstate(all="ignore"):
                median = float(np.nanmedian(ndvi_max[selected]))
            if np.isnan(median):
                continue
            too_green = high is not None and median > high
            too_bare = low is not None and median < low
            if too_green or too_bare:
                warnings += 1
                name = config.CLASSES[int(class_id)]
                said = "greener" if too_green else "barer"
                print(f"     {name:<22} {size:>7,} px  peak NDVI {median:.2f}"
                      f"   <-- much {said} than {name} should be")

    if not warnings:
        print("     nothing contradicted by the optical imagery.")

    if UNDECIDABLE <= set(np.unique(mask)):
        print("\n  NOTE: this mask uses both bare and built-up. No optical index")
        print("        separates those two - desert and dry city overlap almost")
        print("        exactly. Those zones need a human looking for structure:")
        print("        straight edges, a street grid, roads leading somewhere.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default=None, choices=list(config.AOIS),
                        help="just one region (default: every mask found)")
    args = parser.parse_args()

    names = [args.aoi] if args.aoi else list(config.AOIS)
    checked, bad_grid = [], []
    for name in names:
        result = check_one(name)
        if result is None:
            continue
        checked.append(name)
        if result is False:
            bad_grid.append(name)

    print(f"\n{'=' * 74}")
    if not checked:
        print("No label masks found yet. Label a region first.")
    else:
        print(f"checked {len(checked)}: {', '.join(checked)}")
        if bad_grid:
            print(f"GRID FAILURES - do not train on these: {', '.join(bad_grid)}")
        else:
            print("all grids sound.")
    print("=" * 74)


if __name__ == "__main__":
    main()
