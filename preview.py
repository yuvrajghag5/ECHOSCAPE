"""
Look at what you downloaded.

Makes a PNG showing each monthly radar image, so you can check the data is
sensible before doing anything else. Water should be black, towns bright.

    .venv\\Scripts\\python.exe preview.py
    .venv\\Scripts\\python.exe preview.py --aoi mannheim
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import config


def stretch(values, low=2, high=98):
    """Scale an image so it looks right on screen, ignoring extreme outliers."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default=config.DEFAULT_AOI, choices=list(config.AOIS))
    parser.add_argument("--band", default="rgb", choices=["vv", "vh", "rgb"],
                        help="vv, vh, or a false-colour mix of both")
    args = parser.parse_args()

    folder = config.S1_MONTHLY / args.aoi
    files = sorted(folder.glob("*.tif"))
    if not files:
        raise SystemExit(f"Nothing in {folder}. Run fetch_s1.py first.")

    print(f"Found {len(files)} monthly images in {folder}")

    columns = min(4, len(files))
    rows = (len(files) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows), dpi=130)
    axes = np.atleast_1d(axes).ravel()

    for axis, path in zip(axes, files):
        with rasterio.open(path) as src:
            vv = src.read(1).astype("float64")
            vh = src.read(2).astype("float64")
            passes = src.read(3)

        with np.errstate(all="ignore"):
            vv_db = 10 * np.log10(np.where(vv > 0, vv, np.nan))
            vh_db = 10 * np.log10(np.where(vh > 0, vh, np.nan))

        if args.band == "vv":
            picture = stretch(vv_db)
            axis.imshow(picture, cmap="gray")
        elif args.band == "vh":
            picture = stretch(vh_db)
            axis.imshow(picture, cmap="gray")
        else:
            axis.imshow(np.dstack([stretch(vv_db), stretch(vh_db),
                                   stretch(vv_db - vh_db)]))

        month = path.stem.split("_")[1]
        covered = float(np.mean(passes > 0)) * 100
        axis.set_title(f"{month}   {int(np.nanmax(passes))} passes   "
                       f"{covered:.0f}% covered", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes[len(files):]:
        axis.axis("off")

    fig.suptitle(f"Sentinel-1 monthly composites - {args.aoi} - "
                 f"{config.TARGET_RES_M} m  ({args.band})", fontsize=12, y=1.0)
    fig.tight_layout()

    out = config.DATA / f"preview_{args.aoi}_{args.band}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
