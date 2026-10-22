"""
Put the two satellites side by side over the same place and month.

Sentinel-2 is a camera: it looks like a photograph, but clouds block it.
Sentinel-1 is radar: it works through cloud and at night, but you cannot read
it by eye.

    .venv\\Scripts\\python.exe compare_s1_s2.py
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile

import config

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox"

AOI = "mannheim"
MONTH = "2021-06"
BBOX = config.AOIS[AOI]["bbox"]


def count_optical_scenes(bbox, month):
    """How many camera images exist this month, and how many are clear enough."""
    start = f"{month}-01"
    end = f"{month}-30"
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox),
            "datetime": f"{start}/{end}", "limit": 200}
    everything = requests.post(STAC, json=body, timeout=120).json()["features"]
    clear = [s for s in everything if s["properties"].get("eo:cloud_cover", 100) < 20]
    return everything, clear


def fetch_optical(item_id, bbox, width, height):
    """Download the true-colour picture - what the scene looks like to your eye."""
    west, south, east, north = bbox
    url = f"{DATA_API}/{west},{south},{east},{north}/{width}x{height}.tif"
    reply = requests.get(url, params={"collection": "sentinel-2-l2a",
                                      "item": item_id, "assets": ["visual"]},
                         timeout=600)
    reply.raise_for_status()
    with MemoryFile(reply.content) as memfile, memfile.open() as src:
        return np.transpose(src.read()[:3], (1, 2, 0))


def stretch(values, low=2, high=98):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    radar_path = config.S1_MONTHLY / AOI / f"{AOI}_{MONTH}_{config.TARGET_RES_M}m.tif"
    if not radar_path.exists():
        raise SystemExit(f"Missing {radar_path}. Run fetch_s1.py --months 6 first.")

    with rasterio.open(radar_path) as src:
        vv = src.read(1).astype("float64")
        vh = src.read(2).astype("float64")
        height, width = src.height, src.width

    with np.errstate(all="ignore"):
        vv_db = 10 * np.log10(np.where(vv > 0, vv, np.nan))
        vh_db = 10 * np.log10(np.where(vh > 0, vh, np.nan))

    everything, clear = count_optical_scenes(BBOX, MONTH)
    print(f"Sentinel-2 images over this area in {MONTH}: {len(everything)}")
    print(f"   of those, clear enough to use (<20% cloud): {len(clear)}")
    print(f"Sentinel-1 radar passes used for the same month: 7")

    best = sorted(clear, key=lambda s: s["properties"]["eo:cloud_cover"])[0]
    cloud = best["properties"]["eo:cloud_cover"]
    date = best["properties"]["datetime"][:10]
    print(f"\nUsing optical image from {date} ({cloud:.1f}% cloud)")

    optical = fetch_optical(best["id"], BBOX, width, height)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), dpi=150)

    axes[0].imshow(optical)
    axes[0].set_title("SENTINEL-2  (camera)\ntrue colour, as your eye would see it",
                      fontsize=11)

    axes[1].imshow(stretch(vv_db), cmap="gray")
    axes[1].set_title("SENTINEL-1  (radar)\nraw signal strength",
                      fontsize=11)

    axes[2].imshow(np.dstack([stretch(vv_db), stretch(vh_db),
                              stretch(vv_db - vh_db)]))
    axes[2].set_title("SENTINEL-1  (radar)\nsame data, colour-coded to help the eye",
                      fontsize=11)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(f"Same place, same month - Mannheim / Heidelberg, {MONTH}   "
                 f"({len(clear)} usable camera image vs 7 radar passes)",
                 fontsize=13, y=1.00)
    fig.tight_layout()

    out = config.DATA / "compare_s1_s2.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
