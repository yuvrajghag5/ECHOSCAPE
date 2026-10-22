"""
Turn the downloaded .tif files into ordinary .png files you can double-click.

Why you need this: the .tif files are GeoTIFFs. They are not normal pictures.
Each pixel holds a real measurement stored as a decimal number, not a brightness
from 0 to 255, and the file also carries map coordinates. Windows Photos has no
idea what to do with that, so it refuses to open them. The file is fine - it is
just not a picture yet.

This script converts them into real pictures.

    .venv\\Scripts\\python.exe to_png.py
    .venv\\Scripts\\python.exe to_png.py --aoi mannheim

Output lands in data/png/ and opens in any image viewer.
"""

import argparse

import numpy as np
import rasterio
from PIL import Image

import config


def stretch(values, low=2, high=98):
    """Turn measurements into 0-255 brightness, ignoring extreme outliers."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype="uint8")
    lo, hi = np.percentile(finite, [low, high])
    scaled = np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)
    scaled[~np.isfinite(values)] = 0
    return (scaled * 255).astype("uint8")


def radar_to_png(path, out_path, band="rgb"):
    """Radar: VV and VH are measurements, so convert to decibels first.

    There is no single "radar picture". The file holds two measurements per
    pixel, and these are three ways of showing them:

      vv   just the VV measurement, as grey
      vh   just the VH measurement, as grey
      rgb  VV as red, VH as green, their ratio as blue - lets the eye see all
           of it at once, at the cost of looking nothing like a photograph

    All three show the same underlying numbers. None of them is the data.
    """
    with rasterio.open(path) as src:
        vv = src.read(1).astype("float64")
        vh = src.read(2).astype("float64")
    with np.errstate(all="ignore"):
        vv_db = 10 * np.log10(np.where(vv > 0, vv, np.nan))
        vh_db = 10 * np.log10(np.where(vh > 0, vh, np.nan))

    if band == "vv":
        picture = stretch(vv_db)
    elif band == "vh":
        picture = stretch(vh_db)
    else:
        picture = np.dstack([stretch(vv_db), stretch(vh_db),
                             stretch(vv_db - vh_db)])
    Image.fromarray(picture).save(out_path)


def optical_to_png(path, out_path):
    """Colour: bands 3, 2, 1 are red, green, blue."""
    with rasterio.open(path) as src:
        red = src.read(3).astype("float64")
        green = src.read(2).astype("float64")
        blue = src.read(1).astype("float64")
    picture = np.dstack([stretch(red), stretch(green), stretch(blue)])
    Image.fromarray(picture).save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default="all",
                        choices=["all"] + list(config.AOIS))
    parser.add_argument("--band", default="all",
                        choices=["all", "vv", "vh", "rgb"],
                        help="which view of the radar to write. "
                             "'all' writes vv, vh and rgb into separate folders")
    args = parser.parse_args()

    bands = ["vv", "vh", "rgb"] if args.band == "all" else [args.band]
    areas = list(config.AOIS) if args.aoi == "all" else [args.aoi]
    out_root = config.DATA / "png"
    made = 0

    for aoi_name in areas:
        # Radar: one folder per view, because there is no single correct picture
        radar_files = sorted((config.S1_MONTHLY / aoi_name).glob("*.tif"))
        for band in bands:
            if not radar_files:
                break
            out_dir = out_root / aoi_name / f"radar_{band}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for path in radar_files:
                try:
                    radar_to_png(path, out_dir / (path.stem + ".png"), band)
                except Exception as err:
                    print(f"   failed {path.name}: {err}")
                    continue
                made += 1
            label = {"vv": "VV only, grey", "vh": "VH only, grey",
                     "rgb": "VV+VH+ratio, false colour"}[band]
            print(f"   {aoi_name:10} radar {band:4} {len(radar_files):2d} images "
                  f"({label}) -> {out_dir}")

        # Optical: a real photograph, only one sensible view
        optical_files = sorted((config.S2_MONTHLY / aoi_name).glob("*.tif"))
        if optical_files:
            out_dir = out_root / aoi_name / "colour"
            out_dir.mkdir(parents=True, exist_ok=True)
            for path in optical_files:
                try:
                    optical_to_png(path, out_dir / (path.stem + ".png"))
                except Exception as err:
                    print(f"   failed {path.name}: {err}")
                    continue
                made += 1
            print(f"   {aoi_name:10} colour      {len(optical_files):2d} images "
                  f"(true colour) -> {out_dir}")

    if made == 0:
        raise SystemExit("Nothing found. Run fetch_s1.py first.")
    print(f"\nMade {made} pictures in {out_root}")
    print("Double-click any of them. For the real thing with coordinates,")
    print("open the .tif files in QGIS instead (free, qgis.org).")


if __name__ == "__main__":
    main()
