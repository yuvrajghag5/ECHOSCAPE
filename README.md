# Sentinel-1 Land Use / Land Cover Segmentation

SRH Hochschule Heidelberg — Deep Learning

A deep learning model that maps land cover from **Sentinel-1 radar only** — no
optical imagery at prediction time — plus a prototype that takes a latitude,
longitude and start month and returns 12 monthly land-cover maps with per-class
area percentages and month-to-month change.

## Status

## Documentation

| File | Contents |
|---|---|
| `design.md` | What this solves and for whom, the key decisions, known limitations |
| `architecture.md` | How the pieces fit, data layout, the invariants that must hold |
| `phase.md` | All 8 phases, what each produces, current status |
| `tech.md` | Every tool choice with its reason, and what was deliberately avoided |
| `changes.md` | Build log — per phase: what was built, what broke, how it was fixed |

| Stage | State |
|---|---|
| 1. Download radar (Sentinel-1) | **working** |
| 2. Download colour (Sentinel-2) + alignment check | **working** |
| 3. Label the data | not started |
| 4. Assemble the dataset | not started |
| 5. Train the model | not started |
| 6. Evaluate | not started |
| 7. Prototype | not started |

## Setup

No account and no API key are needed. The satellite data comes from Microsoft
Planetary Computer, which serves it anonymously.

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.11 specifically — 3.14 has no CUDA build of PyTorch, which stage 3 needs.

## Usage

Look at what data exists before downloading anything:

```bash
.venv\Scripts\python.exe fetch_s1.py --dry-run
```

Download all 12 months for **all three areas** (~130 MB):

```bash
.venv\Scripts\python.exe fetch_s1.py
```

Then the colour imagery used to build labels, and confirm it lines up with the
radar:

```bash
.venv\Scripts\python.exe fetch_s2.py
.venv\Scripts\python.exe check_alignment.py
```

Both scripts do every area in one run by default. Add `--aoi mannheim` for just
one. Anything already downloaded is skipped, so re-running is safe.

Look at what you got:

```bash
.venv\Scripts\python.exe preview.py
.venv\Scripts\python.exe to_png.py
```

`preview.py` makes one overview image per area. `to_png.py` converts every
downloaded file into an ordinary PNG in `data/png/` that opens in any image
viewer — see **Viewing the data** below for why that is needed.

Other options:

```bash
.venv\Scripts\python.exe fetch_s1.py --aoi odenwald      # a different area
.venv\Scripts\python.exe fetch_s1.py --months 1-3        # just some months
.venv\Scripts\python.exe fetch_s1.py --year 2025         # a different year
.venv\Scripts\python.exe fetch_s1.py --res 100           # coarser pixels
.venv\Scripts\python.exe fetch_s2.py --max-cloud 60      # accept cloudier images
.venv\Scripts\python.exe compare_s1_s2.py                # radar vs camera
```

Everything else is configured in `config.py`.

**Order matters.** `fetch_s1.py` must run before `fetch_s2.py`, because the
colour imagery is downloaded onto the grid defined by the radar files. That is
what guarantees the two line up.

## Key decisions

**50 m, not Sentinel-1's native 10 m.** The brief says not to train on
high-resolution data. Radar also carries heavy speckle noise, and averaging
pixels together removes most of it — so the coarser data is radiometrically
*cleaner*, not worse. Set by `TARGET_RES_M` in `config.py`.

**Year 2021.** Sentinel-1A and 1B were both operating, giving 6–8 passes every
month. Sentinel-1B failed in December 2021, so 2022–2024 have roughly half that.
2025 recovers but its first four months are still thin. Even coverage matters
here because the assignment asks for month-to-month change — uneven pass counts
would show up as change that is not really on the ground.

**One orbit track per area.** Radar brightness depends on the angle the
satellite looked from. Mixing ascending and descending passes makes identical
ground read differently, which the model would learn as a false change.

**Monthly median, not mean.** The median ignores one-off bright returns and
suppresses speckle better.

## Data

Each file is a GeoTIFF with three layers:

| Band | Contents |
|---|---|
| 1 | VV backscatter, monthly median (gamma0, linear) |
| 2 | VH backscatter, monthly median (gamma0, linear) |
| 3 | how many satellite passes saw each pixel |

Band 3 matters: any pixel with zero passes must be excluded from the area
percentages, or months with partial coverage silently deflate every class and
invent month-boundary changes.

Convert to decibels for viewing or training with `10 * log10(value)`.

## Viewing the data

**The `.tif` files will not open in Windows Photos, and that is normal.** They are
GeoTIFFs, not pictures. Each pixel holds a decimal measurement rather than a
brightness from 0 to 255, there are more layers than a picture has, and the file
carries map coordinates. Ordinary image viewers cannot interpret any of that.

Three ways to look at them:

| Want | Use |
|---|---|
| Something you can double-click | `to_png.py` → `data/png/` |
| A quick overview of a whole area | `preview.py` |
| The real thing, with coordinates and pixel values | [QGIS](https://qgis.org) — free |

QGIS is the proper tool: it will show you the actual backscatter value under your
cursor and let you overlay the radar, the colour imagery and later your labels on
top of each other. Worth installing before Phase 3.

## Attribution

Sentinel-1 RTC from Microsoft Planetary Computer, CC-BY-4.0.
Contains modified Copernicus Sentinel data.
