"""
Download Sentinel-2 colour imagery, on exactly the same grid as the radar.

Why this exists: radar cannot be labelled by eye. To build the answer key we need
a picture a human can actually read. Sentinel-2 is a camera, so it gives us that.
It is used ONLY to make labels. It never goes into the model.

Why "exactly the same grid" matters: if the colour image is shifted by even one
pixel relative to the radar, every label describes the wrong piece of ground. So
instead of recalculating the grid, this script reads it straight out of the radar
file that fetch_s1.py already produced. Alignment is then guaranteed rather than
hoped for.

Clouds block cameras. Over Mannheim in June 2021 there were 11 Sentinel-2 images
and only one was clear. This script therefore combines every image in a month,
keeping only the cloud-free pixels from each, and reports how much of the area
ended up visible.

    .venv\\Scripts\\python.exe fetch_s2.py --dry-run
    .venv\\Scripts\\python.exe fetch_s2.py
"""

import argparse
import math
import time

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile

import config

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox"
SESSION = requests.Session()

# The colours and infrared bands we need. Blue/green/red make a normal picture;
# near-infrared and the two shortwave bands are what actually separate vegetation
# from soil from water, and the helper model in Phase 3 will use them.
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
BAND_MEANING = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B08": "near infrared", "B11": "shortwave 1", "B12": "shortwave 2",
}

# Sentinel-2 ships a per-pixel scene classification. These values mean the pixel
# is unusable: no data, saturated, cloud shadow, cloud, thin cirrus, or snow.
UNUSABLE = {0, 1, 3, 8, 9, 10, 11}

# Reflectance values arrive multiplied by 10000 and stored as integers.
REFLECTANCE_SCALE = 10000.0


def radar_grid(aoi, year):
    """Read the exact grid the radar was written on, so the two line up."""
    folder = config.S1_MONTHLY / aoi
    files = sorted(folder.glob(f"{aoi}_{year}-*.tif"))
    if not files:
        raise SystemExit(
            f"No radar files in {folder}.\n"
            f"Run:  .venv\\Scripts\\python.exe fetch_s1.py --aoi {aoi}\n"
            f"The colour images must match the radar grid, so the radar comes first."
        )
    # Take the resolution tag from the radar filename we are actually matching,
    # NOT from config. If config has been changed since the radar was downloaded,
    # trusting it would label this file 100m while putting it on the 50m grid -
    # a mismatch nothing would catch later.
    resolution_tag = files[0].stem.rsplit("_", 1)[-1]
    with rasterio.open(files[0]) as src:
        return src.width, src.height, src.profile, tuple(src.bounds), resolution_tag


def search_optical(bbox, year, month, max_cloud):
    """Find camera images for one month, and say how many were clear enough."""
    last_day = [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30,
                31, 31, 30, 31, 30, 31][month - 1]
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day}",
        "limit": 200,
    }
    reply = SESSION.post(STAC, json=body, timeout=120)
    reply.raise_for_status()
    everything = reply.json().get("features", [])
    usable = [s for s in everything
              if s["properties"].get("eo:cloud_cover", 100) <= max_cloud]
    usable.sort(key=lambda s: s["properties"]["eo:cloud_cover"])
    return everything, usable


def download_bands(item_id, bounds, width, height, attempts=3):
    """Fetch the bands we need, cropped and resized by the server to our grid.

    Retries: single requests fail occasionally for no lasting reason, and over a
    full run a silently skipped image means a thinner composite or a missing
    month.
    """
    west, south, east, north = bounds
    url = f"{DATA_API}/{west},{south},{east},{north}/{width}x{height}.tif"
    for attempt in range(1, attempts + 1):
        try:
            reply = SESSION.get(
                url,
                params={"collection": "sentinel-2-l2a", "item": item_id,
                        "assets": BANDS + ["SCL"]},
                timeout=900,
            )
            reply.raise_for_status()
            break
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    with MemoryFile(reply.content) as memfile, memfile.open() as src:
        stack = src.read().astype("float32")

    spectral = stack[:len(BANDS)] / REFLECTANCE_SCALE
    scene_class = stack[len(BANDS)]

    clear = ~np.isin(np.rint(scene_class).astype("int16"), list(UNUSABLE))
    clear &= np.all(np.isfinite(spectral), axis=0) & (spectral[0] > 0)
    return spectral, clear


def month_composite(scenes, bounds, width, height):
    """Combine a month's images, keeping only the cloud-free pixels from each."""
    layers, masks = [], []
    for scene in scenes:
        try:
            spectral, clear = download_bands(scene["id"], bounds, width, height)
        except Exception as err:
            print(f"      skipped {scene['id'][:30]} ({err})")
            continue
        spectral[:, ~clear] = np.nan
        layers.append(spectral)
        masks.append(clear)
    if not layers:
        return None
    # Pixels that were cloudy in every single image of the month stay empty.
    # numpy warns about that; it is expected, so quieten it.
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        combined = np.nanmedian(np.stack(layers), axis=0)
    clear_count = np.sum(np.stack(masks), axis=0).astype("float32")
    return combined, clear_count


def write_geotiff(path, spectral, clear_count, profile):
    profile = dict(profile)
    profile.update(count=len(BANDS) + 1, dtype="float32", nodata=np.nan,
                   compress="deflate", driver="GTiff")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for index, band in enumerate(BANDS):
            dst.write(spectral[index], index + 1)
            dst.set_band_description(index + 1, f"{band} ({BAND_MEANING[band]})")
        dst.write(clear_count, len(BANDS) + 1)
        dst.set_band_description(len(BANDS) + 1, "cloud-free observations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default="all",
                        choices=["all"] + list(config.AOIS),
                        help="which area, or 'all' for every area (default)")
    parser.add_argument("--year", type=int, default=config.YEAR)
    parser.add_argument("--max-cloud", type=float, default=40.0,
                        help="ignore images cloudier than this percent (default 40)")
    parser.add_argument("--months", default="1-12")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is available, download nothing")
    args = parser.parse_args()

    if "-" in args.months:
        first, last = (int(x) for x in args.months.split("-"))
    else:
        first = last = int(args.months)

    areas = list(config.AOIS) if args.aoi == "all" else [args.aoi]
    grand_total = 0.0
    for position, aoi_name in enumerate(areas, start=1):
        if len(areas) > 1:
            print("\n" + "=" * 70)
            print(f"AREA {position} OF {len(areas)}:  {aoi_name}")
            print("=" * 70)
        grand_total += fetch_one_area(aoi_name, args, first, last)

    if len(areas) > 1:
        print("\n" + "=" * 70)
        print(f"ALL AREAS DONE.  {grand_total:.1f} MB written in total.")
        print("=" * 70)


def fetch_one_area(aoi_name, args, first, last):
    """Download every requested month for one area. Returns megabytes written."""
    width, height, profile, bounds, resolution_tag = radar_grid(aoi_name, args.year)
    print(f"\nArea      {aoi_name}")
    print(f"Grid      {width} x {height} pixels at {resolution_tag}, "
          f"taken from the radar files")
    print(f"Bounds    {bounds}")
    print(f"Year      {args.year}   cloud limit {args.max_cloud:.0f}%\n")

    print("Checking what the camera caught ...")
    available = {}
    for month in range(first, last + 1):
        everything, usable = search_optical(bounds, args.year, month, args.max_cloud)
        available[month] = usable
        bar = "#" * len(usable)
        note = "   <-- nothing usable" if not usable else ""
        print(f"   {args.year}-{month:02d}  {len(everything):2d} images, "
              f"{len(usable):2d} under {args.max_cloud:.0f}% cloud  {bar}{note}")

    if args.dry_run:
        print("\nDry run. Drop --dry-run to download.")
        return 0.0

    print("\nDownloading ...")
    out_dir = config.S2_MONTHLY / aoi_name
    total_mb = 0.0
    for month in range(first, last + 1):
        label = f"{args.year}-{month:02d}"
        out_path = out_dir / f"{aoi_name}_{label}_{resolution_tag}.tif"

        if out_path.exists():
            print(f"   {label}  already done, skipping")
            continue
        if not available[month]:
            print(f"   {label}  too cloudy all month, skipped")
            continue

        print(f"   {label}  combining {len(available[month])} images ...",
              end="", flush=True)
        result = month_composite(available[month], bounds, width, height)
        if result is None:
            print("  failed")
            continue
        spectral, clear_count = result
        write_geotiff(out_path, spectral, clear_count, profile)

        size_mb = out_path.stat().st_size / 1e6
        total_mb += size_mb
        visible = float(np.mean(clear_count > 0)) * 100
        flag = "  GOOD for labelling" if visible > 90 else "  partly cloudy"
        print(f"  saved {size_mb:4.1f} MB   {visible:5.1f}% of the area visible{flag}")

    print(f"\nDone with {aoi_name}. {total_mb:.1f} MB written to {out_dir}")
    print("Use the months with high visibility for labelling in Phase 3.")
    print("Contains modified Copernicus Sentinel data.")
    return total_mb


if __name__ == "__main__":
    main()
