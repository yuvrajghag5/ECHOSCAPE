# Architecture

How the pieces fit together.
(`design.md` covers what and why. `tech.md` covers which tools.)

**Legend:** ✅ built and tested · 🔜 planned

## Shape of the system

Two paths that must stay identical in their preprocessing, or the whole thing
fails silently.

```
TRAINING PATH (run once, offline)

  Planetary Computer
         |
         +-- Sentinel-1 radar ---> fetch_s1.py ✅ ---> data/s1_monthly/
         |                                                   |
         |                          (grid taken from here)   |
         |                                   v               |
         +-- Sentinel-2 colour ---> fetch_s2.py ✅ ---> data/s2_monthly/
                                                             |
                                                             v
                                              annotator (AI Vision) 🔜
                                                             |
                                                             v
                                                       data/labels/ 🔜
                                                             |
                            data/s1_monthly/ + data/labels/  |
                                                 v           v
                                            build_dataset.py 🔜
                                                     |
                                                     v
                                               data/tiles/ 🔜
                                                     |
                                                     v
                                                 train.py 🔜 ---> weights/
                                                     |
                                                     v
                                              evaluate.py 🔜


PREDICTION PATH (runs live, radar only)

  user: lat / lon / start month
         |
         v
  app/fetch.py 🔜  (SAME download + preprocessing as fetch_s1.py)
         |
         v
  app/infer.py 🔜  (load weights, tile, predict, stitch)
         |
         v
  app/stats.py 🔜  (percentages, month-to-month deltas, total delta)
         |
         v
  app/streamlit_app.py 🔜  (map picker, 12 maps, table, charts)
```

## The constraint that holds it together

**The same preprocessing recipe must run on both paths.**

If training data is built at 50 m with a monthly median over one orbit track, and
the live app fetches at 30 m mixing tracks, the model receives something it has
never seen. It will not error. It will produce confident, wrong maps.

This is enforced two ways:

1. Every setting lives in `config.py`. No script hard-codes a resolution, year or
   area.
2. `app/fetch.py` will reuse the functions in `fetch_s1.py` rather than
   reimplementing them.

## Alignment: structural, not hoped for

The labels are drawn on **optical** imagery and used to train on **radar**. If the
two grids differ by one pixel, every label describes the wrong ground, nothing
crashes, and the only symptom is a mediocre score that looks like a model problem.

So the grid is defined **once**, by the radar:

```
fetch_s1.py   computes the grid from the bounding box and resolution,
              writes GeoTIFFs

fetch_s2.py   OPENS a radar file, reads its width, height, bounds and
              transform, and requests optical imagery on exactly that grid

check_alignment.py   verifies it two independent ways
```

The check is deliberately not just "do the numbers match":

1. **Numerically** — width, height, CRS, bounds and transform must be identical.
   It refuses to continue otherwise.
2. **Physically** — water is dark in radar *and* dark in near infrared, for
   unrelated reasons. The darkest 3% of each are compared. Measured: **69%
   overlap**, with the radar's water outline landing on the rivers in the optical
   image.

Two satellites with different physics agreeing where the Rhine is, is real
evidence. Matching metadata alone is not.

## Components

### Built ✅

| File | Responsibility |
|---|---|
| `config.py` | Every setting. Areas, year, resolution, classes, paths. |
| `fetch_s1.py` | Search the catalogue, pick one orbit track, download cropped and resized windows, median-composite per month, write GeoTIFF. |
| `fetch_s2.py` | Same areas on the radar's grid. Per-pixel cloud masking, combine a month's images, write GeoTIFF. |
| `check_alignment.py` | Prove radar and optical describe the same ground. |
| `preview.py` | Visual sanity check of the radar composites. |
| `compare_s1_s2.py` | Side-by-side radar vs optical, for the report. |

### Planned 🔜

| File | Responsibility |
|---|---|
| `build_dataset.py` | Cut radar and masks into matching 128×128 tiles, drop mostly-empty tiles, split by whole area, write a manifest. |
| `train.py` | U-Net training loop, class-weighted loss, nodata ignored, radar-specific augmentation. |
| `evaluate.py` | Per-class IoU on the held-out area, confusion matrix, noise-floor measurement. |
| `app/` | The prototype: fetch, infer, stats, visualisation, Streamlit shell. |

## Data layout

```
data/
  s1_monthly/<area>/<area>_<year>-<mm>_50m.tif   ✅  radar, 3 bands
  s2_monthly/<area>/<area>_<year>-<mm>_50m.tif   ✅  optical, 7 bands
  labels/<area>/...                              🔜  masks, 1 band, values 0-6
  tiles/                                         🔜  training tiles + manifest
weights/                                         🔜
```

### Radar file — 3 bands

| Band | Contents |
|---|---|
| 1 | VV backscatter, monthly median (gamma0, linear) |
| 2 | VH backscatter, monthly median |
| 3 | **number of passes that saw this pixel** |

### Optical file — 7 bands

| Band | Contents |
|---|---|
| 1–6 | blue, green, red, near infrared, shortwave 1, shortwave 2 (reflectance) |
| 7 | **number of cloud-free views of this pixel** |

### Why the count bands exist

They are not decoration. Both are the guard against the most likely wrong number
in the submission.

A month with partial coverage has pixels no satellite saw. If those are counted in
the denominator when working out "what percentage of this area is water", every
class percentage is deflated, and the deflation differs between months — which
appears as month-to-month change that is not real.

So: **pixels with zero observations are excluded from every percentage.** Band 3
of the radar file is what makes that possible, and the optical count band lets
Phase 3 skip pixels that were only seen once and may still hold thin cloud.

## Why the server does the cropping

One Sentinel-1 scene is ~1 GB covering 250 × 170 km. The study area is 29 × 28 km
— about 2% of it.

| Approach | One area, one year |
|---|---|
| Download whole scenes, crop locally | ~84 GB |
| Ask the API for the crop at 50 m | **31 MB** |

The data API takes a bounding box and an output size and returns a correctly
georeferenced GeoTIFF. This single decision is what makes the project feasible on
home internet, and it also means no GDAL-level handling of enormous files.

## Invariants

Things that must stay true. Worth asserting in code.

1. Radar and optical for the same area and month share CRS, bounds, transform and
   shape. ✅ enforced by `check_alignment.py`
2. Labels share that grid too. 🔜
3. Class percentages sum to 100 each month, excluding nodata. 🔜
4. Month-to-month deltas across classes sum to approximately 0. 🔜
5. Total change equals the sum of the eleven monthly changes. 🔜
6. Train, validation and test splits never share an area. 🔜

Items 3–5 are three lines of assertion that catch the nodata bug instantly, and
belong in a unit test.
