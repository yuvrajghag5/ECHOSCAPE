"""
Download Sentinel-1 radar data and build one image per month.

What this does, in plain terms:

  1. Asks Microsoft's free satellite catalogue which radar images exist over
     our area during the year.
  2. Picks one orbit track and sticks to it, so every image is taken from the
     same angle.
  3. For each month, downloads every image from that track, already shrunk to
     50 m by the server (so we never download the huge originals).
  4. Takes the middle value of each pixel across those images. Radar is very
     noisy; averaging several passes cleans it up a lot.
  5. Saves one GeoTIFF per month.

No account, no API key, no login. Run:

    .venv\\Scripts\\python.exe fetch_s1.py --dry-run     # just look
    .venv\\Scripts\\python.exe fetch_s1.py               # actually download
"""

import argparse
import math
import sys
import time

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile

import config

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox"
SESSION = requests.Session()


def pixel_grid(bbox, res_m):
    """How many pixels wide and tall, so each pixel is about res_m on the ground.

    A degree of latitude is always ~111 km. A degree of longitude shrinks the
    further north you go, so we scale it by cos(latitude).
    """
    west, south, east, north = bbox
    mid_lat = (south + north) / 2
    height = round((north - south) * 111_320 / res_m)
    width = round((east - west) * 111_320 * math.cos(math.radians(mid_lat)) / res_m)
    return max(width, 16), max(height, 16)


def search_scenes(bbox, year):
    """Ask the catalogue what radar images exist over this box during this year."""
    body = {
        "collections": [config.COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{year}-01-01/{year}-12-31",
        "limit": 500,
    }
    reply = SESSION.post(STAC, json=body, timeout=120)
    reply.raise_for_status()
    return reply.json().get("features", [])


def choose_track(scenes, wanted_orbit):
    """Pick one orbit track - the satellite path that passes over most often.

    Radar brightness depends on the angle the satellite looked from. Mixing
    tracks means the same field reads differently on different dates, which the
    model would wrongly learn as a change on the ground.
    """
    tracks = {}
    for scene in scenes:
        props = scene["properties"]
        direction = props.get("sat:orbit_state")
        if wanted_orbit != "auto" and direction != wanted_orbit:
            continue
        key = (direction, props.get("sat:relative_orbit"))
        tracks.setdefault(key, []).append(scene)

    if not tracks:
        sys.exit("No scenes found. Try a different year, area, or ORBIT setting.")

    best = max(tracks, key=lambda k: len(tracks[k]))
    return best, tracks[best], tracks


def download_window(item_id, bbox, width, height, attempts=3):
    """Fetch one scene, cropped to our box and shrunk to our resolution.

    The server does the cropping and shrinking, so a whole scene that would be
    hundreds of megabytes arrives as a few megabytes.

    Retries, because single requests fail occasionally for no lasting reason.
    Over a full run that is 228 downloads, so "occasionally" happens several
    times - and a month that fails is a month silently missing from the dataset.
    """
    west, south, east, north = bbox
    url = f"{DATA_API}/{west},{south},{east},{north}/{width}x{height}.tif"
    for attempt in range(1, attempts + 1):
        try:
            reply = SESSION.get(
                url,
                params={"collection": config.COLLECTION, "item": item_id,
                        "assets": ["vv", "vh"]},
                timeout=600,
            )
            reply.raise_for_status()
            break
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    with MemoryFile(reply.content) as memfile, memfile.open() as src:
        stack = src.read().astype("float32")
        profile = src.profile
    vv, vh = stack[0], stack[1]
    # Band 3 marks which pixels the satellite actually covered.
    valid = stack[2] > 0 if stack.shape[0] > 2 else np.isfinite(vv)
    valid &= np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
    vv[~valid] = np.nan
    vh[~valid] = np.nan
    return vv, vh, valid, profile


def monthly_composite(scenes, bbox, width, height):
    """Combine several passes into one clean image using the middle value."""
    vv_stack, vh_stack, valid_stack, profile = [], [], [], None
    for scene in scenes:
        try:
            vv, vh, valid, prof = download_window(scene["id"], bbox, width, height)
        except Exception as err:
            print(f"      skipped {scene['id'][:28]} ({err})")
            continue
        vv_stack.append(vv)
        vh_stack.append(vh)
        valid_stack.append(valid)
        profile = profile or prof
    if not vv_stack:
        return None

    with np.errstate(all="ignore"):
        vv_med = np.nanmedian(np.stack(vv_stack), axis=0)
        vh_med = np.nanmedian(np.stack(vh_stack), axis=0)
    observations = np.sum(np.stack(valid_stack), axis=0).astype("float32")
    return vv_med, vh_med, observations, profile


def write_geotiff(path, vv, vh, observations, profile):
    """Save three layers: radar VV, radar VH, and how many passes saw each pixel."""
    profile = dict(profile)
    profile.update(count=3, dtype="float32", nodata=np.nan,
                   compress="deflate", driver="GTiff")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(vv.astype("float32"), 1)
        dst.write(vh.astype("float32"), 2)
        dst.write(observations, 3)
        dst.set_band_description(1, "VV gamma0 (monthly median)")
        dst.set_band_description(2, "VH gamma0 (monthly median)")
        dst.set_band_description(3, "number of valid passes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default="all",
                        choices=["all"] + list(config.AOIS),
                        help="which area, or 'all' for every area (default)")
    parser.add_argument("--year", type=int, default=config.YEAR)
    parser.add_argument("--res", type=int, default=config.TARGET_RES_M)
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what exists, download nothing")
    parser.add_argument("--months", default="1-12",
                        help="which months to fetch, e.g. 3 or 1-6 (default all)")
    args = parser.parse_args()

    if "-" in args.months:
        first, last = (int(x) for x in args.months.split("-"))
    else:
        first = last = int(args.months)
    wanted_months = range(first, last + 1)

    areas = list(config.AOIS) if args.aoi == "all" else [args.aoi]
    grand_total = 0.0
    for position, aoi_name in enumerate(areas, start=1):
        if len(areas) > 1:
            print("\n" + "=" * 70)
            print(f"AREA {position} OF {len(areas)}:  {aoi_name}")
            print("=" * 70)
        grand_total += fetch_one_area(aoi_name, args, wanted_months)

    if len(areas) > 1:
        print("\n" + "=" * 70)
        print(f"ALL AREAS DONE.  {grand_total:.1f} MB written in total.")
        print("=" * 70)


def fetch_one_area(aoi_name, args, wanted_months):
    """Download every requested month for one area. Returns megabytes written."""
    area = config.AOIS[aoi_name]
    bbox = config.bbox_for(aoi_name)
    width, height = pixel_grid(bbox, args.res)
    km_w = (bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    km_h = (bbox[3] - bbox[1]) * 111.32

    print(f"\nArea      {aoi_name}  [{area['class']}]  {area['note']}")
    print(f"Box       {bbox}")
    print(f"Ground    {km_w:.0f} x {km_h:.0f} km")
    print(f"Output    {width} x {height} pixels at {args.res} m")
    print(f"Year      {args.year}\n")

    print("Asking the catalogue what exists ...")
    scenes = search_scenes(bbox, args.year)
    (direction, track), track_scenes, all_tracks = choose_track(scenes, config.ORBIT)

    print(f"Found {len(scenes)} images in total, on these tracks:")
    for (d, t), group in sorted(all_tracks.items(), key=lambda kv: -len(kv[1])):
        mark = "  <- using this one" if (d, t) == (direction, track) else ""
        print(f"   {d:>10} track {t:<4} {len(group):3d} images{mark}")

    by_month = {m: [] for m in range(1, 13)}
    for scene in track_scenes:
        by_month[int(scene["properties"]["datetime"][5:7])].append(scene)

    print(f"\nImages per month on track {track}:")
    for month in range(1, 13):
        count = len(by_month[month])
        bar = "#" * count
        flag = "   <-- EMPTY" if count == 0 else ""
        print(f"   {args.year}-{month:02d}  {count:2d}  {bar}{flag}")

    if args.dry_run:
        print("\nDry run, nothing downloaded. Drop --dry-run to fetch it.")
        return 0.0

    print("\nDownloading ...")
    out_dir = config.S1_MONTHLY / aoi_name
    total_mb = 0.0
    for month in wanted_months:
        scenes_this_month = sorted(by_month[month],
                                   key=lambda s: s["properties"]["datetime"])
        label = f"{args.year}-{month:02d}"
        out_path = out_dir / f"{aoi_name}_{label}_{args.res}m.tif"

        if out_path.exists():
            print(f"   {label}  already done, skipping")
            continue
        if not scenes_this_month:
            print(f"   {label}  no images this month, skipped")
            continue

        print(f"   {label}  combining {len(scenes_this_month)} passes ...", end="", flush=True)
        result = monthly_composite(scenes_this_month, bbox, width, height)
        if result is None:
            print("  failed, no usable data")
            continue
        vv, vh, observations, profile = result
        write_geotiff(out_path, vv, vh, observations, profile)

        size_mb = out_path.stat().st_size / 1e6
        total_mb += size_mb
        covered = float(np.mean(observations > 0)) * 100
        median_db = 10 * math.log10(float(np.nanmedian(vv)))
        # Scenes listed in the catalogue can clip only the edge of our box, so
        # report how many passes actually reached a typical pixel.
        typical_passes = int(np.median(observations[observations > 0]))
        print(f"  saved {size_mb:4.1f} MB   {covered:5.1f}% covered"
              f"   {typical_passes} passes per pixel   typical VV {median_db:5.1f} dB")

    print(f"\nDone with {aoi_name}. {total_mb:.1f} MB written to {out_dir}")
    print("Sentinel-1 data from Microsoft Planetary Computer, CC-BY-4.0, "
          "contains modified Copernicus Sentinel data.")
    return total_mb


if __name__ == "__main__":
    main()
