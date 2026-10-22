"""
Check that the radar and the colour images describe the same ground.

This is the most important check in the whole data pipeline, and the easiest one
to skip. If the two are shifted relative to each other, every label will describe
the wrong piece of ground. Nothing will crash. The model will train happily and
be completely wrong, and you will not find out until the end.

So: compare the grids numerically, then look at them.

    .venv\\Scripts\\python.exe check_alignment.py
    .venv\\Scripts\\python.exe check_alignment.py --month 2021-09
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import config


def pick(matches, folder, pattern, hint):
    """Choose one file, and say so out loud if there was a choice to make.

    More than one resolution can sit in the same folder if the settings were
    changed between downloads. Silently taking the first one alphabetically
    would be a quiet way to compare the wrong files.
    """
    if not matches:
        raise SystemExit(f"Nothing matching {pattern} in {folder}.\n{hint}")
    if len(matches) == 1:
        return matches[0]
    preferred = [p for p in matches
                 if p.stem.endswith(f"_{config.TARGET_RES_M}m")]
    chosen = preferred[0] if preferred else matches[0]
    others = ", ".join(p.name for p in matches if p != chosen)
    print(f"  NOTE: several resolutions present. Using {chosen.name}")
    print(f"        (ignoring {others})")
    print(f"        Keep one resolution per dataset - mixing them will cause")
    print(f"        confusion later. See TARGET_RES_M in config.py.\n")
    return chosen


def stretch(values, low=2, high=98):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default=config.DEFAULT_AOI, choices=list(config.AOIS))
    parser.add_argument("--month", default=None, help="e.g. 2021-09")
    args = parser.parse_args()

    # Find files by looking on disk rather than building names from config.
    # config can be edited after the data was downloaded, and a name built from
    # the current setting would simply miss files that are already there.
    s2_dir = config.S2_MONTHLY / args.aoi
    pattern = f"*_{args.month}_*.tif" if args.month else "*.tif"
    s2_path = pick(sorted(s2_dir.glob(pattern)), s2_dir, pattern,
                   f"Run fetch_s2.py --aoi {args.aoi} first.")

    month = s2_path.stem.split("_")[1]
    s1_dir = config.S1_MONTHLY / args.aoi
    s1_path = pick(sorted(s1_dir.glob(f"*_{month}_*.tif")), s1_dir,
                   f"*_{month}_*.tif",
                   f"Run fetch_s1.py --months {int(month[-2:])} first.")

    with rasterio.open(s1_path) as radar, rasterio.open(s2_path) as colour:
        checks = [
            ("width", radar.width, colour.width),
            ("height", radar.height, colour.height),
            ("coordinate system", str(radar.crs), str(colour.crs)),
            ("bounds", tuple(round(v, 9) for v in radar.bounds),
                       tuple(round(v, 9) for v in colour.bounds)),
            ("pixel transform", tuple(round(v, 12) for v in radar.transform[:6]),
                                tuple(round(v, 12) for v in colour.transform[:6])),
        ]
        vv = radar.read(1).astype("float64")
        red = colour.read(3)
        green = colour.read(2)
        blue = colour.read(1)
        nir = colour.read(4)

    print(f"\nComparing   radar  {s1_path.name}")
    print(f"            colour {s2_path.name}\n")

    all_ok = True
    for name, left, right in checks:
        ok = left == right
        all_ok &= ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"          radar : {left}")
            print(f"          colour: {right}")

    if not all_ok:
        raise SystemExit("\nGrids do not match. Do not label anything until they do.")

    print("\n  Both images cover identical ground, pixel for pixel.")

    # A second, independent check: water should be dark in radar AND dark in the
    # near infrared. If the two agree about where the rivers are, they are aligned.
    with np.errstate(all="ignore"):
        vv_db = 10 * np.log10(np.where(vv > 0, vv, np.nan))
    radar_water = vv_db < np.nanpercentile(vv_db, 3)
    optical_water = nir < np.nanpercentile(nir[np.isfinite(nir)], 3)
    both = radar_water & optical_water
    agreement = both.sum() / max(radar_water.sum(), 1) * 100
    print(f"  Darkest 3% of radar vs darkest 3% of infrared: {agreement:.0f}% overlap")
    print("  (that is the rivers; a high number means they sit on top of each other)")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), dpi=150)
    axes[0].imshow(np.dstack([stretch(red), stretch(green), stretch(blue)]))
    axes[0].set_title(f"Colour (Sentinel-2)  {month}", fontsize=11)
    axes[1].imshow(stretch(vv_db), cmap="gray")
    axes[1].set_title(f"Radar (Sentinel-1)  {month}", fontsize=11)

    overlay = np.dstack([stretch(red), stretch(green), stretch(blue)])
    outline = radar_water & ~np.roll(radar_water, 1, axis=0)
    overlay[outline] = [1.0, 0.0, 0.0]
    axes[2].imshow(overlay)
    axes[2].set_title("Water found by RADAR, drawn on the COLOUR image\n"
                      "red lines should sit on the rivers", fontsize=11)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"Alignment check - {args.aoi} - {month}", fontsize=13, y=1.0)
    fig.tight_layout()

    out = config.DATA / f"alignment_{args.aoi}_{month}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved {out}")
    print("  Look at the right-hand panel. If the red lines follow the rivers,")
    print("  the two datasets are aligned and Phase 3 can start.")


if __name__ == "__main__":
    main()
