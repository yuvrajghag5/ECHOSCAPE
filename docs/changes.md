# Build Log

What was built in each phase, what went wrong, and how it was fixed.

This file is written as the work happens, not afterwards. It feeds the written
report directly — the "issues" sections are the interesting part of a Data
Engineering project, not an embarrassment to hide.

---

# Phase 0 — Setup

*22 October 2026*

## 1. Initial build

A virtual environment plus a `config.py` holding every setting, so no script
hard-codes a number. The first attempt simply used the machine's default Python.

## 2. Issues faced

**The default Python cannot train the model.** The machine runs Python 3.14 as
default. PyTorch publishes no CUDA build for 3.14, so the graphics card would sit
unused and training would fall back to the processor — far too slow to be
practical.

This would not have surfaced until Phase 5, by which point the data pipeline would
have been built on the wrong interpreter and everything would need redoing.

## 3. How it was solved

Checked which Python versions were installed (`py -0p`) and found 3.11 available.
Created the environment with `py -3.11 -m venv .venv` instead.

**Lesson recorded:** check the constraint imposed by the *last* phase before
building the *first* one.

---

# Phase 1 — Radar acquisition

*22 October 2026 — complete*

## 1. Initial build

`fetch_s1.py`, which:

1. Searches Planetary Computer's catalogue for Sentinel-1 RTC over a bounding box
   and year.
2. Picks a single orbit track.
3. Downloads each pass, cropped to the box and resized to 50 m **by the server**.
4. Combines each month's passes by taking the median per pixel.
5. Writes a GeoTIFF per month with VV, VH and a count of valid passes.

Supporting scripts: `preview.py` to look at the result, `compare_s1_s2.py` to show
radar against optical.

**Result:** 12 monthly composites per area, roughly 2.6 MB each, about 31 MB for a
full year over a 29 × 28 km area.

## 2. Issues faced

**(a) The crop endpoint was not where it was assumed to be.**
The first attempt built URLs of the form `/item/crop/{bbox}.png`. Every request
returned `{"detail":"Not Found"}` — with HTTP 200, so it looked like a successful
download until the file was opened.

**(b) Pixels came out rectangular on the ground.**
The data is delivered in degrees of latitude and longitude, not metres. A degree
of latitude is always about 111 km, but a degree of longitude shrinks towards the
poles. Asking for equal numbers of pixels in both directions produced pixels that
were 50 m tall and 77 m wide — not the square 50 m pixels intended.

**(c) The number of satellite passes looked far better than it was.**
A first count returned 199 Sentinel-1 images over the area in 2025, suggesting
about 16 per month. That number is meaningless. A bounding box overlaps several
different orbit tracks, and images from different tracks view the ground from
different angles, so they cannot be averaged together.

**(d) The obvious year choice was wrong.**
2025 was chosen first, being the most recent with good coverage. Annual totals
supported it. But the month-by-month breakdown on a single track showed January
to April with only 2–3 passes, against 5 for the rest of the year, because
Sentinel-1C was still coming into service.

This matters more than it sounds: the assignment asks for **month-to-month
change**. Months built from fewer passes are noisier, and that extra noise looks
exactly like change on the ground.

**(e) The reported pass count was higher than the real one.**
The script reported "combining 7 passes" for June, but checking the output showed
99.9% of pixels had been seen by exactly **5**. Two of the seven scenes only
clipped the edge of the box and contributed nothing to most of it.

**(f) The first preview images were unreadable.**
Fixed brightness limits were used for display, and almost everything came out
white. The radar image looked broken when the data was fine.

**(g) Noise reduction was weaker than expected.**
Averaging 5×5 pixels together was expected to reduce the random variation by a
factor of 5. Measured on a real uniform field it was **2.2**.

## 3. How it was solved

**(a)** Fetched the service's own API description (`openapi.json`) and read the
route list rather than guessing. The correct form is
`/item/bbox/{minx},{miny},{maxx},{maxy}/{width}x{height}.tif`. Lesson: when an
endpoint 404s, ask the service what routes it has.

**(b)** Compute the pixel grid explicitly, scaling longitude by the cosine of the
latitude at the middle of the box:

```python
height = (north - south) * 111_320 / resolution
width  = (east - west) * 111_320 * cos(mid_latitude) / resolution
```

**(c)** Group scenes by `(orbit direction, relative orbit)` and use only the most
frequent track. Honest figures for a single area, per track: **2021 ≈ 5.0 passes
per month, 2023 ≈ 2.5, 2025 ≈ 4.2.**

**(d)** Added a `--dry-run` mode that prints a month-by-month bar chart of pass
counts without downloading anything, then compared years with it. 2021 showed
**6–8 passes in every single month**, because Sentinel-1A and 1B were both flying
with a 6-day repeat. Default year changed to 2021.

This is why the dry run exists: it answered a design question in seconds and
committed to nothing.

**(e)** Changed the report to use the median of the valid-pass layer, so it states
how many passes actually reached a typical pixel rather than how many scenes the
catalogue listed. The valid-pass layer is written into every output file as band 3
and is used later to exclude unseen pixels from the area percentages.

**(f)** Switched to a percentile stretch — map the 2nd to 98th percentile of each
image to black-to-white, rather than fixed limits.

**(g)** Investigated rather than accepting the expected figure. The reason is that
Sentinel-1 GRD data has **already** been averaged about 4.5 times during
processing, so a further 5×5 average starts from a much cleaner image than the
theory assumes. Measured chain on real data:

| Processing | Effective looks (higher is cleaner) |
|---|---|
| Single pass, 10 m | 4.2 |
| 5-pass monthly median, 10 m | 10.0 |
| Monthly median + 5×5 average, 50 m | 18.8 |

The monthly median turns out to do more of the cleaning than the resolution
reduction does. Worth stating in the report, because it justifies compositing as
the more important step.

**(h) The resolution was picked by feel, then checked.**
50 m was chosen at the start from reasoning alone. It was later changed to 100 m
on the grounds that coarser is cleaner and the professor asked for low
resolution — which sounded right and was wrong.

Downloading the same area and month at 25 m, 50 m and 100 m settled it:

| | 25 m | **50 m** | 100 m |
|---|---|---|---|
| Image size | 1159 × 1113 | **579 × 557** | 290 × 278 |
| Training tiles (128 px, 3 areas × 12 months) | 9,792 | **2,016** | **324** |
| Noise, effective looks | 26.7 | **60.1** | 113.6 |

100 m leaves only **324 training tiles in total**, which cannot train a U-Net to
separate six classes. And the noise argument for going coarser turns out to be
already spent: a single raw pass has about 4 effective looks, but by 25 m the
monthly median over 7–8 passes has already reached 26.7. **The compositing does
nearly all the noise removal before resolution is involved at all.**

So going 50 → 100 m bought cleanliness that was not needed and paid for it with
three quarters of the data. Reverted to 50 m, now with evidence rather than
assertion. Figure saved as `data/resolution_comparison.png`.

## Measurements worth keeping

| Quantity | Value |
|---|---|
| Download per area-month | 2.6 MB |
| Download per area-year | ~31 MB |
| Equivalent whole-scene download | ~84 GB |
| Coverage achieved | 100% of the box |
| Typical VV, mixed urban/rural | −7.3 dB |
| Optical images in June 2021 | 11, of which **1** usable |
| Radar passes, same month | 7, all usable |

---

# Phase 2 — Optical acquisition

*22 October 2026 — fetching and alignment working*

## 1. Initial build

`fetch_s2.py`, which collects Sentinel-2 colour and infrared imagery for the same
areas. Six bands are kept — blue, green, red, near infrared and two shortwave —
because those are what actually separate vegetation from soil from water, and the
helper model in Phase 3 will need them.

`check_alignment.py`, which verifies the colour images and the radar describe the
same ground.

**Result:** September 2021 over Mannheim, 6.0 MB, 99.5% of the area visible
through the clouds. Alignment confirmed.

## 2. Issues faced

**(a) The silent failure mode: misalignment.**
The labels drawn on the colour image are used to train a model on the radar. If
the two are offset by even one pixel, every label describes the wrong ground.
Nothing crashes. Training proceeds normally. The model is simply wrong, and there
is no error message anywhere to reveal it.

This is the single most dangerous bug available in this project, because the only
symptom is a mediocre accuracy number that looks like a modelling problem.

**(b) Scene-level cloud percentage is misleading.**
The catalogue reports one cloud figure per image, but an image covers a far larger
area than our 29 × 28 km box. An image listed as 35% cloudy might be perfectly
clear over Mannheim, or completely covered. Choosing images by that one number
either throws away usable data or keeps unusable data.

**(c) Clouds are worse than expected — much worse.**
Checking the whole of 2021 at a 40% cloud limit:

| Month | Images | Usable |
|---|---|---|
| Jan | 12 | 1 |
| Feb | 10 | 3 |
| Mar | 11 | 4 |
| Apr | 12 | 5 |
| May | 12 | 2 |
| Jun | 11 | 4 |
| Jul | 12 | 5 |
| Aug | 12 | 1 |
| Sep | 12 | 8 |
| **Oct** | 6 | **0** |
| **Nov** | 8 | **0** |
| Dec | 16 | 5 |

**October and November have no usable optical imagery at all.** The radar had 7
and 8 passes in those same months.

**(d) A warning that looked like a bug.**
numpy printed `RuntimeWarning: All-NaN slice encountered` during compositing.

**(e) Some pixels rest on a single observation.**
After compositing September, 13.3% of pixels had only **one** cloud-free view
across all eight images. For those, the "median" is just that single observation,
so any thin cloud the mask missed passes straight through.

**(f) Filenames were built from a setting that can change after the fact.**
Every script constructed its filenames from `TARGET_RES_M` in `config.py`:
`mannheim_2021-09_50m.tif`. That looks sensible until the setting is edited.

Changing `TARGET_RES_M` from 50 to 100 immediately broke three tools, which went
looking for `..._100m.tif` files that do not exist while a full year of
`..._50m.tif` sat right next to them.

Worse, and silently: `fetch_s2.py` reads its grid out of the radar file it finds
on disk — correctly, 50 m — but then named its output `..._100m.tif`. A file
labelled 100 m, actually written on the 50 m grid, with nothing downstream to
catch the lie. Exactly the class of silent mismatch this project keeps trying to
design out.

## 3. How it was solved

**(a)** The grid is no longer calculated twice. `fetch_s2.py` opens the radar file
that Phase 1 already wrote and takes its width, height, bounds and transform
directly, then requests the colour imagery on exactly that grid. Alignment becomes
structurally guaranteed rather than something to hope for.

`check_alignment.py` then verifies it two independent ways:

1. **Numerically** — width, height, coordinate system, bounds and pixel transform
   must match exactly. It refuses to continue if any differ.
2. **Visually and physically** — water is dark in radar *and* dark in near
   infrared, for completely unrelated physical reasons. The script finds the
   darkest 3% of the radar, draws its outline in red on the colour image, and
   those lines should land on the rivers. They do: **69% overlap** between the two
   independent water detections, and the red lines follow the Rhine and Neckar.

Two different satellites, two different physics, agreeing about where the rivers
are, is strong evidence the grids are correct.

**(b)** Use the scene-level figure only as a coarse filter (drop anything above
40%), then mask **per pixel** using Sentinel-2's own scene classification layer,
which marks cloud, cloud shadow, cirrus and snow for every pixel. Every image in
the month then contributes whichever of its pixels are clear, so a half-clouded
image is still useful over the half it can see.

This is why September reached 99.5% visibility from images that were individually
far from perfect.

**(c)** Accepted and recorded, because it *is* the project's argument. Optical
land-cover mapping is impossible over this area in October and November 2021.
Radar works every month. That is precisely why the brief asks for a radar-only
model, and the table above belongs in the report.

Practically: labelling happens in the clear months, and the model is then applied
to all twelve.

**(d)** Investigated before suppressing. It refers to the 0.5% of pixels that were
cloudy in every single image of the month, so there is nothing to take a median
of. That is correct behaviour, not a fault, so the warning is now suppressed with
a comment explaining why. Those pixels stay empty and are excluded later.

**(e)** The count of cloud-free observations is saved as band 7 of every output
file, so Phase 3 can require a minimum number of clear views before trusting a
pixel. Checked the residual contamination directly: only **0.8%** of pixels have
suspiciously bright blue reflectance, so the composite is clean enough to label.

**(f)** Stopped deriving filenames from configuration. The rule is now: **ask the
disk what exists, do not predict what it should be called.**

- `check_alignment.py` and `compare_s1_s2.py` search for `*_<month>_*.tif`
  instead of constructing an exact name.
- `fetch_s2.py` takes its resolution tag from the radar file it actually matched,
  so the output name always describes the grid the data is really on, whatever
  `config.py` currently says.

Verified by leaving `TARGET_RES_M` at 100 while the data on disk was 50 m: every
tool found the right files and reported `50m` correctly.

**The general lesson**, worth stating in the report: configuration describes
*intent for the next run*. It is not a reliable description of data already
written. Anything already on disk should be discovered, not predicted. A filename
built from a mutable setting is a guess dressed up as a fact.

**(g) A source file silently left version control.**
After a routine `git add -A` and commit, `fetch_s2.py` — 243 lines, the entire
Sentinel-2 downloader — was no longer tracked by Git. The commit recorded it as
deleted. The file was still sitting in the folder, so nothing looked wrong
locally, and it had already been pushed to GitHub in an earlier commit, so the
repository looked complete too. It was found only by listing tracked files
against the folder contents.

Nothing was lost: comparing the working copy against the last committed version
showed the two were identical apart from line endings.

**Cause.** No `.gitattributes`. Git on Windows was converting between LF and CRLF
on every operation, which made whole files appear completely rewritten — the diff
showed all 243 lines changed when not one character had. That churn is what the
file was lost inside.

**Fix.** Added `.gitattributes` with `* text=auto eol=lf`, so the repository
always stores LF regardless of what the working copy uses, and re-added the file.

**The lesson, which generalises past Git:** `add -A`, `--force`, `rm -r` and
similar sweeping commands do exactly what they are told across everything at
once, and report success either way. After any bulk operation on a repository,
compare what is tracked against what is actually there. It costs one command:

```
git ls-files | sort > /tmp/tracked
git status --short     # anything marked ?? that should not be
```

---

# Phase 3 — Labelling

*24-29 October 2026 - 14 of 19 regions*

## 1. Initial build

Labels are drawn on **Sentinel-2 optical**, which the model never sees, then
discarded. One mask per region, not per month, because land cover is annual:
19 masks instead of 228. Sparse by design - unlabelled pixels are excluded from
the loss, which is what makes 19 regions affordable.

The tool is the browser annotator built for the AI Vision course
(`Ai Vision/AI annotator/server.py`), running on this project's environment.
A matplotlib version came first and was abandoned: it could not cope with being
dragged between two monitors at different DPI scaling, a Tk-on-Windows limit
that neither window sizing nor redraw tuning fixed.

`check_labels.py` was written alongside it, and every mask goes through it. Two
checks, which fail very differently:

1. **The grid** - the mask must share CRS, size and transform with the radar.
   Hard failure. A mask on the wrong grid describes the wrong ground, nothing
   crashes, and the only symptom is a mediocre model.
2. **Plausibility** - each labelled zone against the **highest NDVI it reaches
   in any month**. Warnings only; the tool cannot know who is right.

The annual maximum matters: NDVI is seasonal but the mask is not. Checking
against a single month would condemn correct labels, because winter cropland
and winter deciduous forest both read as bare.

## 2. Issues faced

**Arabia - 26,612 px of open desert labelled built-up.** The labelled zones
were spectrally identical to the untouched desert around them: NDBI 0.092
against 0.105, same brightness. There is no city there.

**Finland - 41,167 px of built-up that was snow on forest.** The largest zone
read January NDVI 0.02 at brightness 0.268, July NDVI 0.71 at brightness 0.046
- winter **5.8x brighter** than summer. The trap was partly the tool's own
legend, which described built-up as "cyan, pale, white" in false-colour
infrared. So does snow.

**Delhi - 5,071 px of "water" that was farmed floodplain.** NDWI max median
-0.170 and NDVI max median 0.631, with only 18.2% of it overlapping real water.
The Yamuna's banks through Delhi are market gardens, so region-grow along the
river corridor takes the green banks instead of the channel.

**Amazon - doubted, and the doubt was wrong.** 27% water and 25% built-up in a
"rainforest" region looked implausible, and two separate reviews flagged it.
The region centre is -60.00, -3.10, which is **Manaus**, a city of two million
where the Rio Negro meets the Amazon. Measured per zone: water NDVI max -0.011,
the 88,836 px built-up zone 0.330, trees 0.870. All three correct.

**The grass class never got labelled.** 2,037 px against bare's 779,241.
Mongolia is 97% grass and was left unlabelled; NSW and Kenya were labelled
without using the class at all.

## 3. How it was solved

Arabia, Finland's built-up and Delhi's water were cleared class by class and
redone - each time writing to a temp file, reopening it and asserting the grid
was unchanged before replacing the original.

Four rules came out of it, and they are the useful part:

- **Bare versus built-up cannot be decided by any optical index.** Desert NDBI
  runs 0.08-0.16 and dry built-up 0.06-0.15 - they overlap. That pair needs a
  human looking for street-grid structure, straight edges and roads.
- **Never label a snowy region from a winter month.** Label boreal and
  continental regions from July or August.
- **Pick the month for the feature being labelled, not once per region.** Delhi
  shows 132 water pixels in its clearest month and 8,487 in September. The mask
  is annual, so the river may be labelled on one month and the fields on
  another.
- **Check the place before doubting the labels.** The Amazon episode cost two
  reviews and nearly a redo of a correct mask.

Requested from the annotator session as a result, and built there: multi-zone
select, zoom and pan with a verified click mapping, a measured class suggestion
(annual NDVI/NDWI/NDBI, proposing rather than deciding), a month picker with a
snow warning, a hover coordinate readout with a go-to box, a fine mode for thin
river channels, and real undo.

**Not used:** `prelabel.py` was written to propose water labels from NDWI
automatically - it agreed with the hand labels on 429,069 px against 283
conflicts - but it was never applied. Every label in the dataset is hand-made.

**Removed, 3 November 2026**, together with the unrun Phase 2b notebook. The
professor confirmed that an existing labelled dataset is acceptable and that
self-labelling is not graded, which made both redundant: a dedicated external
land-cover product does the job `prelabel.py` approximated, for every class
rather than water alone. The hand labels are kept and become the independent
test set.

## 4. Change of plan — 3 November 2026

**The professor confirmed that an existing labelled dataset is acceptable, and
that self-labelling is not graded.** That reversed the assumption Phase 3 was
built on.

**New plan: train on external labels, test on the hand labels.** The hand
masks are not wasted. They move to the test side and become the reference the
whole project is scored against — independent human judgement the model never
trains on. Testing on the external product instead would only measure how well
radar imitates another model.

**Why change, with five days left.** The hand-labelled results already showed
three limits that more clicking could not fix in time: grass at 2,037 pixels,
41% of every tile unlabelled, and a U-Net that overfitted after about six
epochs on ~233 distinct pieces of ground. Training takes three minutes at base
16, so the retraining itself is cheap; the cost is fetching labels.

**The source is still being chosen** between Impact Observatory io-lulc, ESA
WorldCover 2021 and Google Dynamic World. Checking them turned up one thing
worth recording: **ESA WorldCover, the most accurate of the three (83.8% in a
2024 comparison), was produced from Sentinel-1 as well as Sentinel-2.** A radar
model trained on it is not learning from a purely optical teacher. The
hand-labelled test set keeps the score honest regardless, but it has to be
declared. Dynamic World is the other notable trade-off: confidence bands per
pixel, but only on Google Earth Engine, which `tech.md` already lists under
deliberately-not-used because it forces an evaluator to register.

**Removed the same day:** `prelabel.py` and the unrun Phase 2b notebook, both
superseded (see above).

**Source chosen: Dynamic World.** Built the same day: an Earth Engine export
script, `import_labels.py`, and `--labels external` across `build_dataset.py`,
`train.py` and `baseline_rf.py`.

Three traps found while building it, each of which would have produced a wrong
result without an error:

- **Dynamic World's class 0 is water, and Earth Engine exports masked pixels as
  0.** Every pixel with no cloud-free view would have arrived as water. The
  export adds 1 to every label so that 0 can only mean "no data".
- **The class-weight cache was a single file.** `train.py` reused it whenever it
  existed, so training on Dynamic World would have weighted the loss with the
  hand labels' class counts. It is now one cache per label set.
- **A rebuild could leave stale tiles behind.** Tiles are named by region,
  month and position; a rebuild with different settings overwrites most, but a
  tile that no longer qualified would linger, and `train.py` reads every `.npz`
  it finds. Full builds now start from empty.

**And one error of reporting, corrected.** The U-Net was summarised as "51.4%
against the forest's 45.3%, a gap of +6.1 points". That compared the U-Net's
*validation* score with the forest's *test* score — two different pairs of
regions. On the same validation regions the forest scores 30.6%, so the real gap
is about **22 points**, and the U-Net has never been scored on test at all,
because `train.py` only evaluated validation. It now reports both, and the
rule is that every comparison quotes the same split.

---
## 5. Code reorganised — 5 November 2026

The thirteen scripts that had accumulated in the project root were moved into
four packages by pipeline stage: `acquisition/`, `labelling/`, `training/`,
`visualisation/`. `config.py` stays at the root.

Nothing about the data moved. `config.py` is the only file that finds paths
from its own location, and it did not move, so every data path is unchanged.
What did change: each script now puts the project root on Python's path before
`import config`; `train.py` imports `from training.model`; `run_all.py` launches
the moved scripts by their new paths; and the notebooks import from the
packages. Commands now name the folder - `training\train.py` rather than
`train.py`. **Commands quoted earlier in this log are as they were run at the
time**, before the move.

Verified by importing every module through its package, running
`labelling/check_labels.py` from the root against real data, and launching a
script from inside its own folder.


## 6. Dynamic World calls dry steppe "cropland" — 5 November 2026

**Built.** With all 19 regions imported, the Dynamic World tiles came to 5,436,
and grass rose from 0.1% to 3.3% of labelled pixels. Still low, which prompted
the question of whether to add grassland regions.

**Issue faced.** Grass was not missing — it was mislabelled. Per region,
Dynamic World called **Mongolia 45.8% cropland and 0.2% grass**. Against the
Sentinel-2 year, that "cropland" peaks at NDVI **0.23** with a swing of 0.09,
where real cropland peaks at 0.71 (Iowa) to 0.81 (NSW) and swings by over 0.5.
It is dry steppe. Dynamic World judges each image alone, and a seasonal
green-up in dry grassland looks like a field in a single snapshot. Left in, it
was roughly two million training examples of "steppe is cropland".

A second, smaller cause: the "Kenya savanna" region is centred on **Nairobi**
(36.80 E, 1.30 S). Dynamic World finds it 50.5% built-up, matching the hand
labels there. So one of the three grassland regions was never grassland — a
region-selection slip from Phase 2c, recorded here rather than fixed, because
re-siting a region this late costs a full re-download.

Adding regions was rejected: it costs hours, and new grassland would most
likely be mislabelled the same way.

**How it was solved.** A rule in `import_labels.py`: **cropland must green up.**
A "cropland" pixel whose yearly peak NDVI (85th percentile of the months) stays
under 0.40 becomes grass / low vegetation, or bare if it never passes 0.15.
Measured across every region before choosing 0.40:

| region | caught at < 0.40 |
|---|---|
| Mongolia | **99.1%** |
| Iowa, Pampas, Punjab, Nile Delta | 0.4–1.1% |
| Delhi | 6.9% |
| Kenya (Nairobi fringe) | 15.6% |

It uses the same thresholds `check_labels.py` applies to the hand masks, so the
machine labels meet the standard the human ones were held to. Pixels with no
optical reading are left untouched. `--no-crop-rule` reproduces the uncorrected
masks for comparison.


## 7. Training moved into the notebooks — 5 November 2026

`train.py` and `baseline_rf.py` were folded into notebooks 01 and 02 and
removed, at his request: the notebooks are what he runs and what is submitted,
so they now carry the full procedure — data loading, augmentation, capped class
weights, crash-safe checkpoints, resume, early stopping, validation-then-test
reporting — rather than calling out to scripts.

`model.py` was kept as a file on purpose. It defines the U-Net's shape, not its
training, and the prototype has to construct the identical network to load the
trained weights. One definition imported in both places cannot drift; two copies
eventually would, and a mismatch makes the weights fail to load.

Two leftovers dropped in the move: a second, dead docstring inside
`train_model`, and the `on_epoch` hook that had only served the live progress
table, replaced earlier by `tqdm` bars.

Checked by executing each notebook's setup and definition cells: imports
resolve, the GPU is found, and the base-16 U-Net builds at **1,942,679
parameters**. Nothing was trained.


## 8. Hand labels and the baseline dropped; new validation and test regions — 5 November 2026

**Decision (his).** Dynamic World becomes the only label source, the Random
Forest baseline is dropped, and three architectures are to be trained from
scratch and compared. The data lab's GPU removes the hardware ceiling, and "more
data is better" was the professor's advice.

**What was done.**

- The 14 hand masks moved to `data/archive/hand_labels/` — out of the pipeline,
  not deleted. Dynamic World masks moved into `data/labels/` under the standard
  name, so `check_labels.py` and `build_dataset.py` read them with no special
  mode, and the whole `--labels hand | external` machinery was removed rather
  than left as dead options. Results and weights from the hand-label era were
  archived the same way.
- The old tile sets were deleted (regenerable): `data/tiles/` held tiles cut from
  the hand labels, and anything trained before a rebuild would have used them
  without a word.
- **12 new regions** for validation and test, one dominated by each class in
  each. Measured before download: all 22 candidates had Sentinel-1 in all 12
  months, and ESA WorldCover 2021 gave each box's class mix. That rejected the
  Kazakh steppe (81% lake) and Lake Victoria (a box with no shoreline). Mannheim
  joined test, so the screen recording shows unseen ground.
- `config.bbox_for()` was checked identical, to the last decimal, to all 19
  existing radar files' bounds — so the Earth Engine script could be generated
  from config for the new regions before their radar existed, letting both
  downloads run in parallel. It exports only the 12 new regions (`EXPORT_ONLY`).

**The cost, recorded honestly.** With no hand labels, every score measures
agreement with Dynamic World — 73–84% accurate itself — rather than with ground
truth. And without the baseline, the report loses its measure of what spatial
context is worth; comparing three architectures replaces it as the experiment.


## 9. Six regions partly empty: the track was chosen by count, not coverage — 5 November 2026

**Built.** The 12 new regions downloaded in 43 minutes, and `run_all.py`'s
completeness table reported every region 12/12 — "All regions complete".

**Issue faced.** The per-month lines underneath disagreed: Paris saved 0.1 MB
a month, "2.1% covered". Measured from band 3 of every file on disk:

| region | split | radar covered |
|---|---|---|
| paris | val | **2.1%** |
| serengeti | val | **24.8%** |
| egypt_desert | train | **55.0%** — since October, unnoticed |
| congo | test | 78.6% |
| kansas | test | 94.5% |
| ireland | test | 97.9% |

The other 25 are at 100%. Paris was the only built-up validation region, so
validation had effectively lost a class.

Two faults compounded. `choose_track` picked the track with the **most
images** and never asked how much of the box they covered. A track's strip is
about 250 km wide; a box near its edge is clipped on every pass. Paris's
busiest track passes 88 times a year and covers 2% of the box — three other
tracks cover all of it. And the completeness table **counted files**, so a
month on disk at 2% coverage read the same as a full one.

It also means the 5 November claim that the new regions were "measured before
download" was half true. What was measured was *time* — a track with images in
all 12 months — not *space*.

**How it was solved.** `choose_track` now measures each track's coverage
before choosing: it burns every scene's footprint, from the catalogue, onto
the region's own pixel grid and takes the worst month. A track must cover
`config.MIN_TRACK_COVER` (90%) of the box every month to be eligible; among
those, the most images wins, because more passes means less speckle. The
footprint method was checked against the downloaded files before use: within
0.5 points for every partial region (Paris predicted 2.0%, actual 2.1%).

90% rather than 100% is deliberate. Kansas's only full-coverage track has
half the images of its current one, and Kansas is the test region for the
cropland/grass pair, where speckle hurts most — 5% of the box is the cheaper
loss. Under the rule exactly four regions change track (paris, serengeti,
egypt_desert, congo), confirmed against the live catalogue for all 31; every
region already above 94% keeps the data it has.

Changing track changes which pixels get data, not the grid — that comes from
the box. So the colour imagery and labels of the four stay valid; only their
radar is re-downloaded.

Two guards were added so this cannot recur silently:

- The completeness table reads each region's **worst-month coverage** from the
  files and flags anything under 90%, with the command that fixes it.
- Each radar file now records its track in its own tags, and a re-run refuses
  to skip a month that came from a different track — otherwise one region
  could end up with months seen from two angles.

The rule matters most for the prototype, which reuses this function for any
point a user clicks: under the old rule, a point near a swath edge would have
returned a mostly blank map.

**A related correction.** Single-pass months had been counted from catalogue
image counts, which overstate passes because some images only clip the box.
Measured per pixel after the re-fetch, seven region-months rest on one pass:
delhi April, finland April, oregon December, mongolia December, egypt_desert
December, **namib December (val)**, **tokyo December (test)**. (Congo April was
single-pass on its old track; on the new one it has two. egypt_desert December
became single-pass by moving to the thinner, full-coverage track.)

**Re-fetch result.** All four at 100% coverage, each on one track (checked from
the file tags), colour grids and egypt_desert's labels still identical to the
radar grid. Paris now reads -4.5 to -5.2 dB VV at 5 passes a pixel, as bright
as a dense city should.
Kept, not dropped: the brief asks for twelve monthly maps. Change reported in
those months is the least trustworthy.

**Lesson, generalising the October one about downloads failing silently:**
a check that counts the thing being made is not a check that the thing is
right. Count files and you learn that files exist.


## 10. All 31 labelled — and where Dynamic World is weakest — 5 November 2026

`import_labels.py` then `check_labels.py` over all 31: **every grid sound**.
Every class is present in every split:

| class | train | val | test |
|---|---|---|---|
| water | 541,245 (9.8%) | 304,731 (15.6%) | 212,502 (9.9%) |
| built-up | 1,154,425 (21.0%) | 308,716 (15.8%) | 491,694 (23.0%) |
| trees | 834,037 (15.2%) | 370,724 (19.0%) | 622,367 (29.1%) |
| cropland | 1,397,722 (25.4%) | 345,377 (17.7%) | 364,098 (17.0%) |
| grass | 378,880 (6.9%) | 259,027 (13.3%) | **98,619 (4.6%)** |
| bare | 1,197,244 (21.8%) | 360,377 (18.5%) | 352,930 (16.5%) |

**Most plausibility warnings are the check's thresholds, not label errors.**
They were set for hand labels: grass may peak at most 0.80 NDVI, built-up at
0.60. Wet pasture (Ireland, NSW, Kansas) genuinely peaks at 0.85-0.95, and
Dynamic World labels whole leafy suburbs "built", which peak at 0.60-0.75.
Left as warnings, unchanged.

**The real weakness is grass.** Ireland was chosen as the grass test region
because ESA WorldCover calls it 77% grass; Dynamic World calls it 13.7% grass,
22.6% trees and 17.5% cropland, and leaves 40% unlabelled. Kansas: WorldCover
29% grass, Dynamic World 7.5%. Hedgerow pasture reads as trees, green pasture
as crops — the inverse of the Mongolian steppe error, and the cropland rule
cannot catch it because this "cropland" does green up. So test-set grass rests
on ~99,000 pixels, the thinnest class anywhere. Grass test scores must be read
as the least certain number in the report.

Also noted, not fixed: Mekong is 35% built-up, with two zones of 30,000 and
41,000 px peaking at NDVI 0.64-0.68 — canal-side villages under orchard canopy,
which Dynamic World calls built. Mekong is a training region.

---

## 11. All nine Dynamic World classes kept, none merged — 6 November 2026

**What changed.** His call: use Dynamic World's own classes instead of merging
them. `config.CLASSES` now has nine (plus 0 = unlabelled), `NUM_CLASSES = 10`
network outputs, and `DW_CLASS_MAP` is one to one. Ids 1-6 keep their old
meaning (so the hand masks and the annotator stay valid); the three restored
classes are appended:

| id | class | was |
|---|---|---|
| 5 | grass | "grass / low vegetation" (grass + shrub) |
| 6 | bare | "bare / sparse" |
| 7 | shrub & scrub | merged into 5 |
| 8 | flooded vegetation | unlabelled |
| 9 | snow & ice | unlabelled |

The cropland-must-green rule stays: it corrects a Dynamic World mistake, it does
not merge classes. `import_labels.py`, `build_dataset.py`, `check_labels.py`,
`model.py` (default head 10) and all three notebooks read the class count from
`config.NUM_CLASSES`; the notebooks' recipe changed, so no 6-class checkpoint is
ever resumed, and a cached 7-long class count is recounted.

**Measured before anyone retrained** (the masks recomputed in memory, nothing
written), labelled pixels per split:

| class | train | val | test |
|---|---|---|---|
| shrub & scrub | 62,911 (1.1%) | 8,463 (0.4%) | 5,935 (0.3%) |
| flooded vegetation | 1,218 (0.0%) | 1,537 (0.1%) | 1,403 (0.1%) |
| snow & ice | 48,804 (0.9%) | 19 (0.0%) | 1,735 (0.1%) |

The six old classes hold 99% of the pixels. Shrub is mostly Oregon (12%) and
Kenya (8%); snow is almost all Finland (16% of its labelled pixels - an annual
average in which winter snow outweighs the summer ground). Flooded vegetation
is ~1,200 pixels per split, effectively absent.

**What that means for the scores.** Mean IoU averages over every class present
in a split. Three near-empty classes will score close to zero and pull the mean
down, even if the six main classes do exactly as well as before - so 9-class
numbers are **not comparable** to the 6-class runs (47.9 / 43.9 / 47.7% test).
Finland's "snow" is also partly label noise: the radar for July sees forest and
fields under a label that says snow.

---

# Phase 4 — Dataset assembly

*29 October 2026 - done*

## 1. Initial build

`build_dataset.py` cuts every labelled region into 128 x 128 tiles, one tile
per region-month, and writes `data/tiles/<split>/*.npz` plus a manifest.

Three channels, all radar, all decibels scaled by **fixed** ranges from
`config.py`: VV, VH, and VV-VH. Fixed rather than per-image, because scaling
each image by its own min and max would make every image's brightest pixel read
1.0 and destroy the thing the model needs - that water is absolutely dark and
cities absolutely bright, everywhere. Per-image scaling would also make the
live prototype preprocess differently from training, and fail silently.

**3,840 tiles, 284 MB**: 2,796 train, 528 val (mannheim, atacama), 516 test
(mekong, oregon). Splits are by whole region; 41% of the pixels in the kept
tiles are unlabelled and excluded from the loss.

## 2. Issues faced

**A region's last 88 pixels would have been thrown away.** 600 pixels with a
stride of 128 stops at 512. A plain `range` silently drops the far edge of
every region.

**Pixels no satellite saw would have been trained on.** A mask says "trees"
across a whole zone, but radar band 3 records how many passes saw each pixel,
and where that is zero there is no measurement - only a fill value.

## 3. How it was solved

The tile grid adds one final tile flush with the far edge, overlapping its
neighbour rather than losing the ground.

Pixels with zero passes are forced to label 0 regardless of what the mask says,
so they are excluded from the loss like any other unlabelled pixel.

`build_dataset.py` also re-runs the grid check and refuses to build if a mask
does not match its radar, so a bad mask cannot reach training even if
`check_labels.py` was skipped.

---
# Phase 5 — Training

*29 October 2026 - experiment 1 done*

## 1. Initial build

`model.py` is a U-Net written from scratch: four down-blocks, base width 32,
batch-norm, skip connections, **7.76 M parameters**. Not imported, because the
architecture is part of what is assessed, and 24 M pretrained parameters would
memorise 14 regions rather than learn from them.

**Correction, 29 Oct:** these docs first claimed 1.9 M parameters. The real count
at base 32 is **7.76 M** - the 1-2 M figure in design.md corresponds to base 16,
since every block doubles the width. Still well under a pretrained encoder's
24 M, but large enough against ~168 distinct ground positions that the
validation curve needs watching.  gives the originally intended size.

`train.py` has class-weighted cross-entropy with `ignore_index=0`, cosine
learning-rate decay, and an `--overfit N` mode that trains on N tiles with
augmentation off - the standing discipline, run before trusting any real run.

Augmentation: flips and quarter-turns, since land cover has no preferred
orientation, plus two radar-specific ones - multiplicative **speckle**, because
radar is grainy and the grain differs between passes, and a small **brightness
shift**, standing in for calibration differences between orbit tracks.

`baseline_rf.py` is experiment 1: a per-pixel Random Forest on
`[VV, VH, VV-VH]` with no spatial context at all. It runs without PyTorch,
which is why it came first - it proved the whole chain before a 2.5 GB
download.

**Result, test split (mekong, oregon):** 45.3% mean IoU, 61.2% accuracy.
Water **83.5%**, trees 58.3%, built-up 31.1%, cropland 8.4%.

## 2. Issues faced

**The scoring was wrong, and it under-reported the baseline by 15 points.**
Mekong and Oregon contain no bare and no grass. Those two classes were scored
0% and averaged into the mean anyway, giving 30.2% instead of 45.3%.

Worse than a wrong number: `train.py` already averaged only over classes
present in the ground truth, so the U-Net and the forest would have been scored
by different rules - and experiment 1 is precisely a comparison of the two.

**Cropland scored 8.4%, mistaken for trees 47% of the time.** Measured cause:
Iowa cropland swings 6.7 dB of VH across the year (-21.2 in January, -15.1 in
August) while genuinely bare desert swings 1.0 dB. In August, cropland at
-15.1 dB and Arabian desert at -15.6 dB are indistinguishable.

**Built-up was mistaken for trees 43% of the time.** Buildings double-bounce
off wall and ground; tree canopies volume-scatter. Both return bright and
rough.

## 3. How it was solved

`per_class_iou` now averages only over classes that occur in the ground truth,
matching `train.py`, and prints "absent here, NOT counted in the mean" so the
reason is visible rather than inferred.

The cropland measurement is not a bug to fix but the physical justification for
**experiment 3**, the three-month input: no single month separates cropland
from bare, but the *change between months* separates them completely. That is a
stronger argument than the cropland/grass overlap originally written down for
it, and the measurement belongs in the report.

The built-up/trees confusion is a real limit of radar and is reported as one.

Two notebooks (`notebooks/01_baseline_random_forest.ipynb`,
`notebooks/02_train_unet.ipynb`) run this work with narrative and plots. They
**import** `baseline_rf`, `train` and `model` rather than copying them, so the
submitted notebooks cannot drift away from the submitted code - the failure
mode that bit the RADE project's `webui.py`.

## 4. First run on Dynamic World, 31 regions — 5 November 2026

U-Net base 16, best epoch 20 of 28 (early stop, patience 8):

| class | validation | test |
|---|---|---|
| water | 82.8% | 62.2% |
| built-up | 85.6% | 75.7% |
| trees | 35.4% | 56.7% |
| cropland | 51.1% | 31.5% |
| grass | 64.2% | **12.5%** |
| bare | 97.2% | 48.7% |
| **mean IoU** | **69.4%** | **47.9%** |

Test 47.9% sits inside the published radar-only range (Copernicus-Bench,
45-53) — scored against Dynamic World, not ground truth.

**The validation curve is noisy:** mean IoU moved 10.6 points per epoch on
average, 17.8 at most. Validation has one region per class, so each class's
score is one region's score and swings with it. Picking the best epoch off a
curve that noisy partly picks luck, which is one reason validation (69.4%)
sits 21.5 points above test. Test is the honest number.

**Two classes flip between splits, and both point at terrain.** Bare is 97%
on validation (Namib, a flat sand sea) and 49% on test (Death Valley, basin and
mountain ranges). Trees are 35% on validation (Black Forest, mountain forest)
and 57% on test (Congo, flat rainforest). The training deserts are flat; steep
slopes change what radar returns. Hypothesis until a confusion matrix checks it.

**Grass on test, 12.5%,** was expected: ~99,000 labelled pixels, and in
Ireland Dynamic World's own grass labels are doubtful (section 10 of Phase 3).


## 5. DeepLabV3+ and SegFormer written — 5 November 2026

**Built.** Both hand-written in `training/model.py` beside the U-Net, one per
family: convolutional encoder-decoder (U-Net), dilated convolutions with
multi-scale pooling (DeepLabV3+), transformer (SegFormer). **Sized alike on
purpose** — U-Net 1.94 M, DeepLabV3+ 2.35 M, SegFormer 2.28 M parameters — so a
difference in score is the architecture, not its size. `build_model(name)`
builds any of them, and checkpoints now save the name and settings, so the
prototype rebuilds exactly the trained network.

Checked without training: all three produce (B, 7, 128, 128), every parameter
receives a gradient, a single tile works in evaluation mode (the prototype's
case), and the old `unet16_best.pt` still loads — the U-Net code is unchanged.

Two adaptations to our 128-pixel tiles, each worth a sentence in the report:

- **DeepLab's dilation rates are 2, 4, 6, not the published 6, 12, 18.** Those
  were set for 512-pixel images, where the deepest map is 32x32. Ours is 8x8:
  a rate of 12 would only sample the padding past the edge.
- **SegFormer is MiT-B0 slimmed in its last two stages** (128 and 192 wide, not
  160 and 256) to match the others' size.

**Issue found while wiring the notebook.** Resuming a run restarted the
early-stopping counter at zero. Re-running the notebook after a run had
stopped early would quietly train it another eight epochs and change its
result. The counter is now rebuilt from the saved history: a run that already
stopped is scored, not retrained. Checked against `unet16_last.pt` (stopped at
28, best at 20, patience 8): scored.

**Added:** a confusion-matrix cell (rows = label, columns = prediction, % of
each row), saved into every result file, to test the terrain hypothesis above;
and the comparison table now shows parameters and per-class test IoU.


## 6. First three-way comparison, and why it was unstable — 6 November 2026

**Built.** One notebook per model, run overnight with early stopping
(patience 8):

| | best epoch | val mIoU | test mIoU | epoch-to-epoch swing |
|---|---|---|---|---|
| U-Net | 20 of 28 | 69.4% | 47.9% | 10.6 pts |
| DeepLabV3+ | 9 of 17 | 68.6% | 43.9% | 7.4 pts |
| SegFormer | 20 of 28 | 74.9% | 47.7% | 6.2 pts |

**The confusion matrices confirm the terrain hypothesis.** On test, bare
ground (almost all Death Valley) went to built-up 28% / trees 12% / cropland
25% (DeepLab) and trees 30% / built-up 13% (SegFormer) - the bright, rough
classes, never water. Mountain slopes return radar the way rough surfaces do.
Cropland and grass swap 33-45% of their pixels each way - the hard pair.
Water lost 10-30% to grass on test.

**Issue 1: every overfit test had printed FAIL** (49-52%), and the runs went
ahead. Not a model bug - all three failed identically, and the full runs
learned. The test was wrong for the new data: it took the first ten files
alphabetically (all Amazon, January), where grass and bare covered 662 and
1,314 pixels; averaged over classes those two held the score near 50%. And it
gave 180 steps with the learning rate fading to zero - less training than one
real epoch.

**Issue 2: validation was unstable,** and early stopping made it worse.
Three causes:

- **Validation is six regions, one per class**, twelve months of the same ground
  each. Each class's score is essentially one region's, so a small shift in a
  decision boundary flips a large contiguous area. This part is structural and
  no training change removes it.
- **Runs stopped while the learning rate was still high.** The cosine schedule
  was set for 40 epochs; stopping at 17 or 28 cut it off at 60% or 20% of its
  starting rate, so every scored epoch was a large-step snapshot.
- **The best epoch was picked off a noisy curve,** so it partly picked luck -
  the validation-test gap was 21.5 (U-Net), 24.7 (DeepLab) and 27.2 (SegFormer)
  points. SegFormer's 74.9% was a single spike between 56% and 66%.

**How it was solved** (all three notebooks, identically; old runs archived to
`weights/` and `outputs/archive_early_stop/`):

- **No early stopping** (his call). All 40 epochs run; the best is kept.
- **The learning-rate fade completes**: two warm-up epochs, then cosine down to
  1% of the starting rate at epoch 40 (3.5e-6), so the last epochs settle.
- **Weight averaging (EMA, decay 0.999):** after every step the notebook keeps a
  slowly moving average of the weights - roughly the last 1,000 steps, about
  three epochs. The average is what gets scored and saved. It is the standard
  remedy for a jumpy validation curve and costs almost nothing.
- **Gradient clipping at 1.0**, so no single batch can throw the weights.
- **The overfit test** now takes one tile from each of ten regions (every class
  present), trains 1,200 steps at a constant rate, and passes on pixel accuracy
  above 90%.
- Checkpoints record their recipe; a run from another recipe is refused rather
  than resumed into a mixture.

Checked without training: notebooks differ only in `MODEL`, every cell
compiles, the schedule reaches 3.5e-6 at epoch 40, the averaged weights load
into a plain network (as the prototype needs) and survive a resume.

## 7. Nine classes, stable recipe: the results — 6 November 2026

The three notebooks were renamed when they were run: `01_train_unet`,
`02_train_deeplab`, `03_train_seg`; the monthly-change notebook is now
`04_monthly_change`.

**The curves are clean now.** Training loss falls smoothly in all three. The
averaged-weights validation score climbs and then flattens: SegFormer from
epoch ~22, DeepLab from ~30, U-Net peaks at 27 and slowly slides back. The
first epochs score near zero only because the weight average still holds the
random starting weights. It forgets them within ~3 epochs.

| | best epoch | val mIoU (9) | test mIoU (9) | val, 6 main | test, 6 main | test accuracy |
|---|---|---|---|---|---|---|
| U-Net | 27 | 42.1% | 31.1% | 62.1% | 46.6% | 65.0% |
| DeepLabV3+ | 32 | 49.0% | 30.3% | 72.9% | 45.2% | 64.1% |
| **SegFormer** | 40 | **51.3%** | 31.1% | **75.5%** | 46.6% | 66.7% |

"6 main" is the mean over water, built-up, trees, cropland, grass and bare
only: the classes the earlier runs had. It is the like-for-like comparison.

**What the numbers say.**

- **The headline dropped from ~48% to ~31%, but the model did not get worse.**
  On the six classes the old runs had, test is 45-47%. That is the same as
  the early-stopping runs (43.9-47.9%). The three restored classes score
  close to zero: shrub 0.0-0.2%, flooded vegetation 0-1.6%, snow 0%. Each
  still counts as one ninth of the mean. They are 0.1-0.4% of the pixels,
  too few to learn from (Phase 3 §11 predicted this).
- **Not better, but now trustworthy.** The run-to-run lottery is gone. The
  best epoch sits on a plateau, not a spike. The ranking on validation is
  SegFormer > DeepLab > U-Net, and on test they tie.
- **Loss keeps falling while validation flattens: the model memorises the 18
  training regions.** More epochs or a bigger network make that worse, not
  better. The limit is the data: 18 regions, one-month inputs, labels about
  75-80% right.
- **Validation is easy and test is hard.** 75.5% against 46.6% on the same
  six classes. Measured per test region (scratchpad, read only):
  - **Death Valley** is 17% of all test pixels and 99% bare. Only 19-49% of it
    is called bare. Rocky slopes and alluvial fans return radar like
    buildings, trees or ploughed fields. The training deserts are sand: smooth
    and dark.
  - **Kansas** is 76% cropland, which is found at 43-50% IoU.
  - **Ireland** mixes trees, cropland and grass about 40/30/22 and gets each at
    13-30% IoU.
  - **Congo** forest (98%) and **Tokyo** city (96%) are near perfect.
- **The confusion is where the radar is weakest.** 52-72% of all test errors
  are inside the four classes trees / cropland / grass / bare. The biggest
  single flows: trees->cropland (5.5-8.1% of all test pixels),
  bare->built-up or ->cropland (Death Valley), and cropland<->grass. One month
  of radar sees only roughness and wetness. A field and a meadow look alike
  in most months; they differ in how they change over the year.

**Two free improvements, measured on the trained weights (no retraining).**

| test, 6 main classes | one month | 12 months averaged |
|---|---|---|
| SegFormer alone | 46.6% | 48.9% |
| all three models averaged | 47.9% | 48.8% |

The second column averages each tile's predicted probabilities over its 12
months before picking a class. This gives one yearly map, matching how the
labels were made. On validation the same step gives 76.5% -> 81.6%. It is the
evidence for the next step: **the year's sequence holds information that a
single month lacks.**

## 8. Does longer training help? SegFormer for 60 epochs — 6 November 2026

**Why.** The question was whether more epochs would help. In the 40-epoch runs,
validation stopped improving well before the end while the loss kept falling:

| validation mIoU | epoch 21 | best | epoch 40 |
|---|---|---|---|
| SegFormer | 49.4% | 51.3% (epoch 40) | 51.3% |
| DeepLab | 30.8% | 49.0% (epoch 32) | 48.4% |
| U-Net | 31.3% | 42.1% (epoch 27) | 38.9% |

DeepLab and U-Net were already overfitting. SegFormer was the only one whose
best epoch was its last, so it is the one worth testing. Expected gain: about
one point on validation and little on test, because the bigger gap is between
validation (51%) and test (31%), which more epochs do not close.

**The run.** `notebooks/05_segformer_60_epochs.ipynb` is a copy of notebook 03.
Only two things differ: `EPOCHS = 60`, and the run name `segformer60`. Because
of the name, the 40-epoch weights, results and executed notebook stay
untouched. Took about 33 min (the 40-epoch run took 22). The app and notebook
04 pick the best run by validation, so if `segformer60` wins there, both
switch to it. Notebook 04 then has to be rerun to give it a noise floor.

**Result (run on the datalab A6000, 9 November, 9.9 min).**

| SegFormer, 2.3 M | best epoch | validation | test, 9 classes | test, 6 main |
|---|---|---|---|---|
| 40 epochs | 40 | **51.3%** | 31.1% | 46.6% |
| 60 epochs | 55 | 51.1% | 31.5% | 47.3% |

No gain. Validation passed 50% again only in the last few epochs, as the
learning rate faded (epoch 30: 47.7%, 40: 46.1%, 50: 47.9%, 60: 50.0%).
The longer schedule mostly spends extra epochs at a high learning rate
before settling where the 40-epoch run settled. Test moves by +0.4 / +0.7,
inside the run-to-run noise. **Longer training is not what is missing.**

## 9. Does a bigger model help? A larger SegFormer on the datalab — 8 November 2026

**Why.** All three networks were about 2 M parameters, so they would fit the
4 GB laptop card. The university datalab has an A6000 (48 GB), so size is no
longer the limit. The question is whether the model is too small, or whether
the data sets the limit.

**The run.** `notebooks/06_segformer_big.ipynb` is a copy of notebook 05.
Only the encoder changes: it is shaped like MiT-B2, the SegFormer paper's
mid-size encoder (widths 64/128/320/512, depths 3/4/6/3, decoder 768).
That is 27.4 M parameters instead of 2.3 M, about 12x. It is still trained
from scratch. Data, loss, augmentation, weight averaging and the 60 epochs
are as in notebook 05, so the gap to `segformer60` is model size alone.
Saved as `segformer_big`. `build_model` stores the size with the weights,
and the prototype rebuilds the network from that, so it can load this run
unchanged.

**Datalab setting, both notebooks 05 and 06.** `WORKERS = 4` loads tiles in
4 parallel processes. With 0, a small network waits on the disk more than it
computes. Each worker starts with a copy of the same random generator, so
without a fix all four would flip, turn and add speckle in lockstep.
`fresh_dice` reseeds each worker's generator. On Windows set `WORKERS = 0`.
There, notebook-defined datasets cannot be handed to worker processes.

**Expectation.** Small at most. The validation curves were already flat,
and the bigger gap is validation 51% vs test 31%, i.e. new regions. A larger
model fits the training regions more closely, which can widen that gap.
Either outcome is a finding: a gain means the small networks were holding
the scores back; no gain means the data is.

**Result (run on the datalab, 9 November).**

| | parameters | validation | test, 9 classes | test, 6 main |
|---|---|---|---|---|
| SegFormer | 2.3 M | **51.3%** | 31.1% | 46.6% |
| SegFormer, big | 27.4 M | 51.1% | **32.7%** | **49.0%** |
| DeepLabV3+ | 2.3 M | 49.0% | 30.3% | 45.2% |
| U-Net | 1.9 M | 42.1% | 31.1% | 46.6% |

| test IoU | water | built-up | trees | cropland | grass | bare |
|---|---|---|---|---|---|---|
| SegFormer | 57.9 | 65.4 | 65.0 | 35.6 | 11.7 | 44.2 |
| SegFormer, big | 56.4 | 71.2 | 64.1 | 34.8 | 13.4 | 54.2 |

(the three rare classes stay at 0-0.4% for every model)

- **12x the parameters, no gain on validation** (51.1% against 51.3%).
- **On test, +1.6 points (9 classes) and +2.4 (6 main),** almost all from two
  classes: bare (+10.0) and built-up (+5.8). Test bare comes mostly from one
  region, Death Valley, so this is one region moving, not a broad gain.
  Trees, cropland and water barely move or drop.
- **Conclusion: the small networks were not what held the scores back.** A
  model 12x larger learns the same classes to the same level. The limits are
  in the data:
  - cropland, grass and trees look alike to one month of radar;
  - shrub, flooded vegetation and snow are 1.16%, 0.03% and 0.87% of the
    training pixels. Flooded vegetation is about one tile's worth in all of
    training.
- The prototype still picks the run by validation, so it keeps the small
  SegFormer (51.3% > 51.1%). Choosing the big one for its test score would be
  using the test set to choose, which it must not be.
- The test confusion matrix (big model) shows two more patterns:
  - 32% of test water is called grass;
  - snow goes to cropland 83% of the time.
- **The big model peaked early.** Its best epoch was 27 of 60 (19.3 min on
  the A6000); after that, validation dipped and recovered without passing it.
  More capacity reached the same ceiling sooner, which is the usual sign that
  the data, not the model, sets the ceiling.
- With §8 (60 epochs, no gain), both "train longer" and "train bigger" are
  now measured, and neither moves validation.

## 10. More data for the three rare classes — 9 November 2026

**Why.** Every model scores about 0% on shrub & scrub, flooded vegetation and
snow & ice. The training set barely has them:

| class | training pixels | share | validation pixels | test pixels |
|---|---|---|---|---|
| shrub & scrub | 875,388 | 1.16% | 110,676 | 84,700 |
| snow & ice | 662,832 | 0.87% | 324 | 26,076 |
| flooded vegetation | 19,500 | 0.03% | 21,768 | 16,811 |

Flooded vegetation is about one tile's worth of pixels in all of training.
§8 and §9 showed that longer training and a bigger model do not help, so the
remaining lever is data.

**Regions chosen.** Thirteen candidates were scouted read-only against the
live catalogues before anything was downloaded. Each was checked for
Sentinel-1 in all 12 months of 2021 on one track (the same ≥90% coverage rule
as `fetch_s1.choose_track`) and for ESA WorldCover 2021's share of the box in
the target class. Kept, two per class, all **training only**:

| region | class | WorldCover share | passes/month |
|---|---|---|---|
| vatnajokull (Iceland) | snow & ice | 100% snow/ice | 5.0 |
| baltoro (Karakoram) | snow & ice | 87% snow/ice | 2.5 |
| sudd (South Sudan) | flooded vegetation | 91% wetland | 6.8 |
| everglades (Florida) | flooded vegetation | 98% wetland | 2.4 |
| sonoran (Arizona) | shrub & scrub | 43% shrub | 4.9 |
| kalahari (Botswana) | shrub & scrub | 40% shrub, 60% grass | 7.0 |

Rejected:
- aletsch: only 35% ice;
- louisiana: 81% open water;
- sundarbans: mangrove, which Dynamic World tends to call trees;
- svalbard: 46% ice, 34% bare;
- karoo: 72% bare;
- mallee: 60% trees;
- okavango: kept as a reserve.

WorldCover is only a proxy. The labels still come from Dynamic World, whose
"flooded vegetation" and "shrub & scrub" do not always match WorldCover's
"wetland" and "shrub". `check_labels.py` reports the real shares after
import.

**Why training only.** Validation and test stay exactly as they were, so the
new run compares directly with every earlier one. The cost is that test still
holds only 17k-85k pixels of these classes, so their test IoU will be noisy.

**The run.** `notebooks/07_segformer_rare_classes.ipynb` uses notebook 03's
SegFormer and recipe (2.3 M parameters, 40 epochs) on the rebuilt tiles,
saved as `segformer_rare`. The new name matters: under `segformer`, RESUME
would find the finished 40-epoch checkpoint and only re-score it.

**The app now skips runs whose weights are missing.**
- `app/pipeline.py` `pick_run` considers only runs with a `weights/<run>_best.pt`.
- Results are committed and weights are not. A run trained on the datalab
  that wins validation would otherwise make the app look for weights that
  are not on the laptop, and crash.

**Result.** Pending.

---
# Phase 6 — Evaluation

## 1. Initial build

### How the land changed over 12 months — 6 November 2026

**The ask.** Show the professor what changed over the year, in percent: e.g.
cropland covering 20% of a region in January and 60% in July.

**The method** (`notebooks/04_monthly_change.ipynb`). The best run, chosen on
validation, maps all 12 months of every test region from radar alone - tiles
overlapping by half, probabilities averaged so there are no seams. The class
shares are counted only over ground the radar saw in **all** 12 months, so each
month is measured on the same area. Then: a monthly table and chart, a
seasonal table (three-month averages, hemisphere-aware), the biggest swing per
class, and maps of January / April / July / October.

**Measured before building it**, with the archived 6-class U-Net (read only):
Kansas, a test region, goes cropland 22% (Jan) -> 64% (Jun-Jul) -> 8% (Dec),
grass the mirror image; the Mekong's water doubles in October, its flood
season. Training regions stay flat (Iowa 81-93% cropland all year): the model
memorised their yearly label - which is why only test regions are used.

**Three things keep it honest:**

- **What a change means.** The labels are yearly, so the model never saw a
  "winter field" label. A field reported as grass in December is a harvested
  field that *looks* like grass to the radar. The numbers are land cover as the
  radar sees it that month, not a change of land use - say so with them.
- **A noise floor.** A city cannot grow or vanish in a year, so the swing of
  the built-up share (median over test regions with >=5% built-up) is the
  model's own wobble - Mannheim's alone is 8 points. A swing under the floor is
  reported as no measurable change; over twice the floor as clear.
- **An independent check.** `labelling/gee/export_dw_monthly.js` exports
  Dynamic World's own class shares per month (mean probabilities per month,
  winner per pixel, counted at 50 m) - a few-kB CSV, not images. The notebook
  puts the radar curve beside the optical one and reports whether they rise and
  fall together (correlation of the two curves, only where both move by more
  than the noise). Months where Dynamic World saw under half the box (clouds)
  are left out of the check - and are exactly where the radar is worth having.

Checked without a trained 9-class model: a stand-in (the archived U-Net's
weights in a 10-output head) and a made-up CSV ran every cell, reproduced the
Kansas numbers above to the decimal, and drew all three figures.

### First real run — 6 November 2026

Run with SegFormer (best on validation, 51.3%). Every test region was measured
on 95-100% of its box. Noise floor: **7.5 points**.

| region | what moved over 2021 | verdict |
|---|---|---|
| Ireland | cropland 88% (Dec-Feb) -> 32% (Jul); grass 1% (Feb) -> 55% (Jul), its mirror image | clear |
| Kansas | grass 24% (Feb) -> 66% (Dec); cropland 54% (Feb) -> 19% (Dec); bare 16% in Jan-Feb, 0% Apr-Oct | clear |
| Mannheim | cropland 20% (Dec) -> 41% (Sep); trees 10% (Feb) -> 26% (Apr); shrub 8-11% in winter, under 1% in summer | clear |
| Lake Geneva | cropland 21% (Apr) -> 33% (Sep) | just above the floor |
| Congo, Tokyo | nothing beyond 4.3 points: rainforest and city hold still | no change |

## 2. Issues faced

- **Death Valley's "built-up" is not a city.** Its predicted built-up share
  swings 23% -> 41% over the year, but the label has 0.1% built-up. These are
  rocky desert slopes that the radar reads as buildings (Phase 5 §7). The
  noise floor selects its regions by the *predicted* built-up share, so this
  false city was used as a reference. Without Death Valley, the median swing
  of the real cities (Tokyo 1.5, Lake Geneva 3.7, Mannheim 11.3) is
  **3.7 points**, not 7.5.
- **Most of the "change" is the season, not a change of land use.** Irish
  pasture is called cropland in winter (wet, bare-looking soil) and grass in
  summer. The Kansas bare share in Jan-Feb is most likely frozen ground: ice
  changes how soil reflects radar. Mannheim's winter "shrub" is bare winter
  fields. The radar measures what the ground looks like that month. This is
  the honest way to present it.
- **The Dynamic World check has not run yet.** `outputs/dw_monthly_2021.csv`
  is missing; `export_dw_monthly.js` must be run in Earth Engine first.

## 3. How it was solved

---

# Phase 7 — Prototype

## 1. Initial build

### The web page — 6 November 2026

**What it does.** Pick any point on the map (or a listed place), a box size
(10 / 20 / 30 km) and a first month (Jan 2017 up to 12 months before today).
Then press **Map 12 months**.

1. **Search.** The Sentinel-1 catalogue is searched for those 12 months.
2. **Track and download.** One orbit track is chosen with `fetch_s1.choose_track`.
   Every pass is downloaded, already cropped to the box at 50 m (4 at a time),
   with `fetch_s1.download_window`.
3. **Composite.** Passes are combined into a monthly median and saved with
   `fetch_s1.write_geotiff`.
4. **Same preprocessing.** Each month is read back with
   `build_dataset.radar_channels`, the exact scaling the model trained on.
   No preprocessing is written twice (`architecture.md`).
5. **Model.** The best model by validation (SegFormer, 51.3%) maps each month,
   with tiles overlapping by half and averaged, as in notebook 04.
6. **Year map.** A thirteenth map averages each pixel's probabilities over the
   12 months. This is the step that scored best on the test regions
   (Phase 5 §7).
7. **Shares.** Each class's share is counted only over ground the radar saw in
   all 12 months, so partly covered months cannot invent change.

**The page** (`app/static/`, designed with the frontend-design skill):

- **Colours.** Neutral fog-grey chrome, so Dynamic World's own colours carry
  the meaning. One signal colour, magenta, marks "now" everywhere: the
  current month, progress, focus.
- **Map.** A Leaflet map with every training / validation / test region drawn
  dashed. Moving the box says at once whether the model has seen that ground.
- **Month strip.** Shows the passes per month, plus a play button.
- **The signature: a year dial** shaped like a radar scope. Twelve sectors,
  one per month, each split into rings by class share (area-true). A magenta
  sweep turns while the radar downloads, filling a sector as each month
  arrives. Then it plays the year: the map, the shares, the change since last
  month and the sparklines follow the sweep.
- **Panel.** Below the dial: the biggest swings over the year, judged against
  the model's own wobble (notebook 04's noise floor). Then the checks: seen
  before or not, passes per month, share of the box measured, model
  confidence, and what the score means.

**Checked without running the app** (he runs it): Python compiles, the
JavaScript parses, and a static mock of the page was clicked through in a
browser. The mock used real Mannheim radar and labels from disk and notebook
04's Mannheim shares; the progress, dial, playback, layers and phone layout
all worked. The live path (catalogue -> download -> model) has not run yet.

## 2. Issues faced

- **CARTO's base-map tiles now require an API key** (seen in the mock).
  Switched to Esri's keyless light-grey canvas.
- Streamlit, the planned framework, was not installed. FastAPI was
  (`tech.md`).

### Simplified after his first live run — 6 November 2026

**His first live run worked:** a box near Mannheim, Feb 2022 - Jan 2023, in 20
seconds, with 2-3 passes a month (Sentinel-1B had failed by then). It came out
about 42% built-up and 50% cropland.

**His verdict:** the dial, the play button and the look are right, but the
middle is too complicated. The base map, the photo mode, the Land cover /
Radar input / Whole year switch, opacity and box size are not needed. The page
should take a latitude, a longitude and the 12 months, then show the radar
picture of each month with its land-cover map.

- **The map is gone,** and Leaflet and Esri with it. The page now loads
  nothing from outside except its fonts.
- **Two square frames, side by side.** On the left is the month's radar, as a
  plain greyscale picture: VV, on the same fixed decibel scale the model
  reads. It used to be a false-colour mix of the three input channels. On the
  right is the land-cover map predicted from that radar. The month buttons
  and Play swap both together. A magenta scan line crosses the empty frames
  while the radar downloads.
- **Box size fixed at 20 km.** The "whole year" map is no longer drawn (its
  shares are still in `result.json`).
- **Old caches are redrawn, not re-downloaded.** `result.json` records a
  `pictures` version. A cache from the first version keeps its radar GeoTIFFs
  and only re-runs the model and redraws the PNGs.

Checked again on the scratchpad mock (real Mannheim radar): the frames, the
captions and Play stepping both pictures all work.

## 3. How it was solved

---

# Phase 8 — Submission

## 1. Initial build

## 2. Issues faced

## 3. How it was solved
