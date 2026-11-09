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
         +-- Sentinel-1 radar ------> fetch_s1.py ✅ ------> data/s1_monthly/
         |                                          (defines the grid)
         |
         +-- Sentinel-2 colour -----> fetch_s2.py ✅ ------> data/s2_monthly/
         |                                          (yearly greenness, for one correction)
         |
  [Earth Engine] Dynamic World -> labelling/gee/export_dynamic_world.js ✅
         (10 m, from Sentinel-2)  -> Google Drive -> data/dw_raw/
                                  -> import_labels.py ✅  vote 10 m -> 50 m,
                                                         cropland-must-green rule
                                  -> data/labels/  ✅  the ONLY labels

   s1_monthly + labels  ->  build_dataset.py ✅  ->  data/tiles/{train,val,test}
                                                            |
                                                            v
                                     notebooks: three architectures 🔜  ->  weights/
                                                            |
                                                            v
                                     evaluate on the test regions 🔜


PREDICTION PATH (runs live, radar only)

  user: lat / lon / start month
         |
         v
  app/pipeline.py ✅  fetch: fetch_s1's own functions (track choice,
         |            server-side crop at 50 m, monthly median)
         |            preprocess: build_dataset.radar_channels - the training scaling
         |            infer: best model by validation, tiles overlapping by half
         |            stats: shares over ground seen in all 12 months,
         |            month-to-month and total change, PNG maps
         v
  app/server.py ✅  (FastAPI: run / progress / result / images)
         |
         v
  app/static/ ✅  (lat / lon / months in; radar | land cover side by side per month, year dial, change)
```

## The constraint that holds it together

**The same preprocessing recipe must run on both paths.**

If training data is built at 50 m with a monthly median over one orbit track, and
the live app fetches at 30 m mixing tracks, the model receives something it has
never seen. It will not error. It will produce confident, wrong maps.

This is enforced two ways:

1. Every setting lives in `config.py`. No script hard-codes a resolution, year or
   area.
2. `app/pipeline.py` imports `fetch_s1` and `build_dataset.radar_channels`
   instead of reimplementing them. The one adaptation is the date range
   (any 12 months, not one calendar year). Each month is written as a GeoTIFF
   with `fetch_s1.write_geotiff` and read back with `radar_channels`, exactly
   as the training files were.

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
| `acquisition/fetch_s1.py` | Search the catalogue, pick one orbit track that covers at least 90% of the box every month (measured from scene footprints, then the busiest), download cropped and resized windows, median-composite per month, write GeoTIFF tagged with its track. |
| `acquisition/fetch_s2.py` | Same areas on the radar's grid. Per-pixel cloud masking, combine a month's images, write GeoTIFF. |
| `acquisition/check_alignment.py` | Prove radar and optical describe the same ground. |
| `labelling/check_labels.py` | Prove a label mask sits on the radar grid, and flag zones the optical imagery contradicts. Run it before training on any new mask. |
| `labelling/gee/export_dynamic_world.js` | Paste into the Earth Engine Code Editor. Builds a 2021 Dynamic World composite per region and exports it to Drive. |
| `labelling/gee/export_dw_monthly.js` | Earth Engine again: Dynamic World's class shares per region per month of 2021, one small CSV (`outputs/dw_monthly_2021.csv`). The independent check for notebook 04. |
| `labelling/import_labels.py` | Lays each Dynamic World export on the radar grid, votes 10 m to 50 m, keeps all nine Dynamic World classes (renumbered, ids 1-9), and relabels "cropland" that never greens up. |
| `training/build_dataset.py` | Cut radar and masks into matching 128x128 tiles, one per region-month, drop mostly-empty tiles, split by whole region, write a manifest. |
| `training/model.py` | Three networks written from scratch, sized alike: U-Net (1.94 M), DeepLabV3+ (2.35 M), SegFormer (2.28 M). `build_model(name, **kwargs)` builds any of them; checkpoints store both, so the prototype rebuilds exactly what was trained. |
| `notebooks/01_train_unet.ipynb`, `02_train_deeplab`, `03_train_seg` | One per model, identical but for `MODEL`: class-weighted loss, unlabelled pixels ignored, radar-specific augmentation, overfit test, 40 full epochs with weight averaging (best kept), validation and test scores, confusion matrix, comparison table across all runs. |
| `notebooks/05_segformer_60_epochs.ipynb` | Notebook 03 with `EPOCHS = 60`, saved as run `segformer60`: tests whether longer training helps without touching the 40-epoch run. |
| `notebooks/06_segformer_big.ipynb` | Notebook 05 with a SegFormer shaped like MiT-B2 (27.4 M parameters, `SIZE`), saved as run `segformer_big`: tests whether a bigger model helps. Meant for the datalab GPU. Both 05 and 06 load tiles with `WORKERS` parallel processes. |
| `notebooks/07_segformer_rare_classes.ipynb` | Notebook 03's SegFormer and recipe on tiles rebuilt with six rare-class training regions (snow, flooded vegetation, shrub), saved as run `segformer_rare`. |
| `notebooks/04_monthly_change.ipynb` | The best run (by validation) maps all 12 months of every test region; class share per month and per season, a noise floor from built-up, the biggest changes, maps, and the comparison with monthly Dynamic World. Writes `outputs/monthly_change_<run>.json` for the prototype. |
| `visualisation/preview.py` | Visual sanity check of the radar composites. |
| `visualisation/compare_s1_s2.py` | Side-by-side radar vs optical, for the report. |
| `visualisation/to_png.py` | Convert GeoTIFFs to ordinary PNGs. The `.tif` files hold decimal measurements and map coordinates, so normal image viewers cannot open them. |

### Planned 🔜

| File | Responsibility |
|---|---|
| `evaluate.py` | Per-class IoU on the held-out regions, confusion matrix, noise-floor measurement. |
| `app/pipeline.py` | The prototype's engine: point + start month -> 12 radar composites -> 12 maps, shares, change. Cached in `data/app_cache/<lat>_<lon>_<start>_<size>km/`. |
| `app/server.py` | FastAPI on port 8010: `/api/info`, `/api/run` (starts a job), `/api/job/<key>` (progress), `/api/result/<key>`, `/api/image/<key>/<png>`. One run at a time. |
| `app/static/` | The page: `index.html`, `style.css`, `app.js`. |

## Code layout

Reorganised on 5 November 2026 from one flat folder into packages, by the stage
of the pipeline each file serves:

| Folder | Holds | Stage |
|---|---|---|
| (root) | `config.py` | every setting; read by everything |
| `acquisition/` | `fetch_s1`, `fetch_s2`, `run_all`, `check_alignment` | Phases 1-2 |
| `labelling/` | `check_labels`, `import_labels`, `gee/` | Phase 3 |
| `training/` | `build_dataset`, `model` | Phase 4, and the U-Net's shape |
| `notebooks/` | all model training | Phase 5 |
| `visualisation/` | `preview`, `to_png`, `compare_s1_s2` | looking at data |
| `app/` | the prototype | Phase 7 |

`config.py` stays at the root on purpose: it is the one file that locates
everything else from its own position (`ROOT = Path(__file__).parent`), so every
data path is unaffected by the move. Each script puts the root on Python's path
before `import config`, so it runs the same from anywhere, and the packages
import one another by full name - `from training.model import UNet`.

**Training lives in the notebooks** (since 5 November). `train.py` and
`baseline_rf.py` were folded into them and removed, so the submitted notebooks
contain the full procedure rather than calling out to scripts. `model.py` alone
stayed a file, deliberately: it is the network's *shape*, and the prototype must
build the identical network to load the trained weights. Defining it once and
importing it in both places means the two can never disagree - if they did, the
weights would simply refuse to load.

## Data layout

```
data/
  s1_monthly/<area>/<area>_<year>-<mm>_50m.tif   ✅  radar, 3 bands
  s2_monthly/<area>/<area>_<year>-<mm>_50m.tif   ✅  optical, 7 bands
  labels/<area>/<area>_labels.tif                ✅  Dynamic World masks, 0-9
  dw_raw/dw2021_<area>.tif                       ✅  Dynamic World exports, 10 m, 3 bands
  tiles/<split>/*.npz + manifest.csv             🔜  rebuild for the nine classes
weights/                                         🔜  trained models
outputs/                                         ✅  scores, curves, figures
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

## Rule: discover, do not predict

Filenames carry the resolution: `mannheim_2021-09_50m.tif`. Nothing builds that
name from `config.py`.

Configuration describes **intent for the next run**. It is not a description of
data already on disk, because it can be edited in between — and was. Every tool
therefore searches for `*_<month>_*.tif` and reads what it finds. `fetch_s2.py`
takes its resolution tag from the radar file it actually matched, so an output
file can never claim a resolution it was not written at.

If several resolutions are present, the tools say which file they chose and warn,
rather than silently taking whichever sorts first.

## Invariants

Things that must stay true. Worth asserting in code.

1. Radar and optical for the same area and month share CRS, bounds, transform and
   shape. ✅ enforced by `check_alignment.py`
2. Labels share that grid too. ✅ enforced by `check_labels.py`
3. Class percentages sum to 100 each month, excluding nodata. 🔜
4. Month-to-month deltas across classes sum to approximately 0. 🔜
5. Total change equals the sum of the eleven monthly changes. 🔜
6. Train, validation and test splits never share an area. ✅ enforced by
   whole-region splits in config.py, each region carrying its own `split`.
7. Every class appears in both validation and test. ✅ one region dominated
   by each class in each, checked against ESA WorldCover before download.

Items 3–5 are three lines of assertion that catch the nodata bug instantly, and
belong in a unit test.
