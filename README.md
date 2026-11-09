# ECHOSCAPE — Sentinel-1 Land Use / Land Cover Segmentation

SRH Hochschule Heidelberg — Deep Learning

A deep learning model that maps land cover from **Sentinel-1 radar only** — no
optical imagery at prediction time — plus a prototype that takes a latitude,
longitude and start month and returns 12 monthly land-cover maps with per-class
area percentages and month-to-month change.

Optical imagery never reaches the model. It is used only to make the labels —
Google Dynamic World, built from Sentinel-2 — and is then thrown away.

## Status

| Stage | State |
|---|---|
| 1. Download radar (Sentinel-1) | **done** — 37 regions, 12 months each |
| 2. Download colour (Sentinel-2), alignment checked | **done** — used only to build and check labels |
| 2c. Choose regions worldwide | **done** — 37 regions: 24 train, 6 validation, 7 test |
| 3. Labels | **done** — Dynamic World, all nine classes, on the radar grid |
| 4. Assemble the dataset | **done** — 7,164 / 1,800 / 2,100 tiles |
| 5. Train | **done** — U-Net, DeepLabV3+ and SegFormer from scratch, plus six variants. SegFormer serves the prototype. Longer training and a 12x bigger model both gained nothing; giving the model the whole year gained 24 points on test |
| 6. Evaluate | **done** — per-class IoU and F1 on test, monthly change on the 7 test regions, and a Mannheim case study checked against ESA WorldCover |
| 7. Prototype | **done** — `app/`, a web page: any point on Earth, any 12 months since 2017, with a model selector |
| 8. Submission | **done** — [docs/report.pdf](docs/report.pdf), eight executed notebooks, screen recording |

The labels are machine-made, so every score measures **agreement with Dynamic
World** — itself 73–84% accurate in published comparisons — not agreement with
ground truth. Read the numbers in that light.

**Results.** All numbers below are on the **test** split: seven regions that took
part in no decision at all. Validation scores higher throughout, and the report
explains why that gap is the point rather than a problem.

| | overall accuracy | mIoU, 6 main classes | mIoU, 9 classes |
|---|---|---|---|
| SegFormer (serves the prototype) | 66.7% | 46.6% | 31.1% |
| **Best model trained** (`temporal_mstat`) | — | **65.0%** | **43.5%** |

The nine-class mean is much lower than the six-class one because shrub & scrub,
flooded vegetation and snow & ice score near zero, and each still counts for one
ninth of the mean. Validation holds 324 snow pixels, so that class cannot be
measured there at all. Both figures are reported everywhere, never one alone.

U-Net, DeepLabV3+ and SegFormer are written from scratch in `training/model.py`,
one notebook each, all trained identically: 40 full epochs, weight averaging,
best epoch kept. `notebooks/04_monthly_change.ipynb` maps every month of 2021 and
reports how much of each region each class covered, month by month.

| notebook | what it answers |
|---|---|
| `01_train_unet`, `02_train_deeplab`, `03_train_seg` | three architectures, one recipe: curves, per-class IoU, confusion |
| `04_monthly_change` | the twelve monthly maps, class shares and month-to-month change |
| `05_segformer_60_epochs` | does training longer help? No. |
| `06_segformer_big` | does a 12x bigger model help? No. |
| `08_experiments` | nine experiments in one sweep - and the one that worked |
| `09_sixclass` | what dropping the three unmeasurable classes is worth |

Every notebook is committed with its outputs and figures. `docs/phase.md` lists
the remaining steps.

## What is submitted, and where it is

| Deliverable | Where |
|---|---|
| **Report** | [docs/report.pdf](docs/report.pdf) — 20 pages, with the figures taken from the notebooks |
| **Notebooks, with their outputs** | [notebooks/](notebooks/) — eight, every one committed with its results and figures |
| **Prototype source** | [app/](app/), plus `acquisition/`, `labelling/`, `training/` |
| **Setup, dependencies, API keys** | this file, below. No account or API key is needed to run the prototype |
| **Requirements file** | [requirements.txt](requirements.txt) — 36 pinned packages; torch installs separately, see Setup |
| **Dataset** | too large for GitHub (several GB): `TODO: paste the download link here` |
| **Trained weights** | not in the repository: `TODO: paste the download link here` |
| **Screen recording** | a full run over Mannheim: `TODO: paste the link here` |

The dataset and the weights are delivered separately because GitHub caps a file
at 100 MB. Everything needed to regenerate the dataset from scratch is in
`acquisition/` and `labelling/`, and the exact settings are in `config.py`.

To run the prototype you need the weights: put `segformer_best.pt` into
`weights/` and start the server. The page lists every model whose weights it
finds, so any subset works.

## Project layout

```
exam/
  config.py            every setting - regions, splits, year, resolution, classes, paths
  acquisition/         download radar and colour onto one grid, check they line up
    fetch_s1.py  fetch_s2.py  run_all.py  check_alignment.py
  labelling/           export Dynamic World, put it on the radar grid, check it
    gee/export_dynamic_world.js  import_labels.py  check_labels.py
  training/            cut tiles, and the networks' shapes (training is in notebooks/)
    build_dataset.py  model.py
  visualisation/       pictures of the data, for looking and for the report
    preview.py  to_png.py  compare_s1_s2.py
  notebooks/           ALL model training, run top to bottom
  app/                 the prototype: web server, pipeline, page
    server.py  pipeline.py  static/
  docs/                design, architecture, phases, tools, build log
  data/  outputs/  weights/
```

Run every script **from this root folder**, e.g.
`.venv\Scripts\python.exe training\build_dataset.py`. Each one puts the root on
Python's path itself, so `config.py` is found however it is launched.

## Documentation

| File | Contents |
|---|---|
| [docs/design.md](docs/design.md) | What this solves and for whom, the key decisions, known limitations |
| [docs/architecture.md](docs/architecture.md) | How the pieces fit, data layout, the invariants that must hold |
| [docs/phase.md](docs/phase.md) | All 8 phases, what each produces, current status |
| [docs/tech.md](docs/tech.md) | Every tool choice with its reason, and what was deliberately avoided |
| [docs/changes.md](docs/changes.md) | Build log — per phase: what was built, what broke, how it was fixed |
| [notebooks/](notebooks/) | Model training, with results |

`docs/changes.md` is the most useful of the five — it records what broke.

## Setup

No account and no API key are needed for the satellite data: Microsoft
Planetary Computer serves it anonymously. The labels come from Google Earth
Engine, which needs a (free, noncommercial) Google account — but only to
regenerate them; the exported labels ship with the dataset.

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Python 3.11 specifically — 3.14 has no CUDA build of PyTorch. PyTorch comes
from its own index because plain `pip install torch` gives a CPU-only build.

## The pipeline, in order

### 1. Radar and colour

```bash
.venv\Scripts\python.exe acquisition\run_all.py --dry-run      # see what exists first
.venv\Scripts\python.exe acquisition\run_all.py --skip-png     # download what is missing
```

`run_all.py` does radar, then colour, region by region, and skips anything
already on disk. **Order matters:** the colour imagery is downloaded onto the
grid the radar defines, which is what guarantees the two line up. It ends with
a completeness table naming any month that failed.

The colour imagery is still needed: `import_labels.py` uses its yearly
greenness to correct one known Dynamic World error (below).

### 2. Labels — Google Dynamic World

1. Paste `labelling/gee/export_dynamic_world.js` into the
   [Earth Engine Code Editor](https://code.earthengine.google.com) and Run it.
   It creates one export task per region listed in `EXPORT_ONLY` — click **RUN**
   on each in the Tasks tab. They land in a Google Drive folder, `echoscape_dw`.
2. Download every file into `data/dw_raw/`, then:

```bash
.venv\Scripts\python.exe labelling\import_labels.py
.venv\Scripts\python.exe labelling\check_labels.py
```

`import_labels.py` lays each 10 m export onto an exact 5×-finer copy of the
radar grid and lets every 50 m pixel take a majority vote of its 25 — leaving
mixed pixels such as field edges unlabelled. It also applies one correction:
**cropland must green up.** Dynamic World labelled the Mongolian steppe 46%
cropland at a yearly peak greenness of 0.23, where real cropland peaks at 0.7–0.8,
so "cropland" that never greens past 0.40 is relabelled grass, or bare.

`check_labels.py` confirms every mask sits on the radar grid and flags zones
the optical imagery contradicts.

### 3. Build the tiles

```bash
.venv\Scripts\python.exe training\build_dataset.py
```

Cuts every region into 128 × 128 tiles, one per region-month, and splits them
by **whole region** into train / val / test as `config.py` assigns them.

### 4. Train — in the notebooks

All model training happens in `notebooks/`. Each notebook has a settings cell
at the top, and runs an **overfit test** on ten tiles before any real run: if
the loss will not fall to near zero, the bug is in the code, and that costs two
minutes to find instead of a day.

### 5. How the land changed over the year

`notebooks/04_monthly_change.ipynb` maps every month of the test regions and
reports each class's share per month and per season. For its check against
optical data, run `labelling/gee/export_dw_monthly.js` in Earth Engine and
save the CSV as `outputs/dw_monthly_2021.csv`.

### 6. The prototype

```bash
.venv\Scripts\python.exe app\server.py
```

Then open http://127.0.0.1:8010. Type a latitude and longitude (or pick one of
the places listed), choose which 12 months, and press **Map 12 months**. The
page fetches Sentinel-1 for a 20 km box around the point, live, with the same
code that built the training data. It runs the best model (by validation) on
every month and shows:

- side by side, the month's radar picture and the land-cover map the model
  made from it; the month buttons or **Play** step through all 12;
- a year dial with each class's share per month;
- the month-to-month and whole-year change;
- whether the place was ever in training, validation or test.

A new point takes a few minutes, mostly downloading. Results are cached in
`data/app_cache/`, so a place opens instantly the second time.

## Key decisions

**50 m, not Sentinel-1's native 10 m.** The brief says not to train on
high-resolution data. Radar also carries heavy speckle noise, and averaging
pixels together removes most of it — so the coarser data is radiometrically
*cleaner*, not worse. Set by `TARGET_RES_M` in `config.py`.

**Year 2021.** Sentinel-1A and 1B were both operating, giving 6–8 passes every
month. Sentinel-1B failed in December 2021, so 2022–2024 have roughly half that.
Even coverage matters because the assignment asks for month-to-month change.

**One orbit track per region.** Radar brightness depends on the angle the
satellite looked from. Mixing passes from different tracks makes identical
ground read differently, which the model would learn as a false change.

**31 regions worldwide, split by region.** 18 for training, three per class at
class transitions. Validation and test are **one region dominated by each
class**, so every class is scored in both — chosen by measurement: each has
radar in all 12 months, and ESA WorldCover confirms the class dominates. That
check rejected two obvious-sounding picks (a "steppe" that was 81% lake, and a
lake box with no shoreline). Splitting by whole region, never by tile, keeps
near-identical neighbouring tiles from sitting on both sides and inflating the
score.

**Mannheim is a test region,** so the screen recording shows a city the model
never trained on.

## Data

Each radar file is a GeoTIFF with three layers:

| Band | Contents |
|---|---|
| 1 | VV backscatter, monthly median (gamma0, linear) |
| 2 | VH backscatter, monthly median (gamma0, linear) |
| 3 | how many satellite passes saw each pixel |

Band 3 matters: any pixel with zero passes must be excluded from the area
percentages, or months with partial coverage silently deflate every class and
invent month-boundary changes.

Labels are single-band GeoTIFFs on the identical grid, values 0–6, where 0
means unlabelled: water, built-up, trees, cropland, grass / low vegetation,
bare / sparse.

## Viewing the data

**The `.tif` files will not open in Windows Photos, and that is normal.** They
are GeoTIFFs, not pictures: each pixel holds a measurement, there are more
layers than a picture has, and the file carries map coordinates.

| Want | Use |
|---|---|
| Something you can double-click | `visualisation\to_png.py` → `data/png/` |
| A quick overview of a whole region | `visualisation\preview.py` |
| The real thing, with coordinates and pixel values | [QGIS](https://qgis.org) — free |

## Attribution

Sentinel-1 RTC and Sentinel-2 L2A from Microsoft Planetary Computer, CC-BY-4.0.
Contains modified Copernicus Sentinel data.

Labels: "This dataset is produced for the Dynamic World Project by Google in
partnership with National Geographic Society and the World Resources
Institute." CC-BY-4.0.

Region checks used ESA WorldCover 2021 (v200), CC-BY-4.0. Benchmark figures
quoted for context come from Copernicus-Bench (Wang et al., 2025,
[arXiv:2503.11849](https://arxiv.org/abs/2503.11849)) and the 2020 IEEE GRSS
Data Fusion Contest.
