# Project Phases

SRH Hochschule Heidelberg — Deep Learning
Sentinel-1-only Land Use / Land Cover segmentation.

## The goal, in one paragraph

A user enters a latitude, longitude and a start month in a web page. The system
fetches Sentinel-1 radar for that location for 12 months, runs a trained model on
each month, and returns 12 land-cover maps plus the percentage of the area in each
class per month, the month-to-month change and the total change — both in
percentage points. The model sees **radar only**. No optical imagery at prediction
time.

## Status at a glance

| Phase | What it produces | State |
|---|---|---|
| 0. Setup | Working environment | **done** |
| 1. Radar acquisition | Monthly Sentinel-1 composites | **done** — 31 regions, full coverage |
| 2. Optical acquisition | Matching Sentinel-2, aligned to the radar grid | **done** |
| 2b. Data evaluation | What the radar alone can tell us, before labelling | **dropped** |
| 2c. Global area selection | Which places the model will know | **done** — 19 regions |
| 3. Labelling | Dynamic World masks | **done** — 31 regions; **6 Nov: all 9 Dynamic World classes kept**, re-import needed |
| 4. Dataset assembly | Training tiles and splits | rebuild for 9 classes (was 5,400 / 1,800 / ~2,100) |
| 5. Training | Trained model weights | three models written and compared once (6 classes); retrain all three on 9 classes, on the data-lab GPU |
| 6. Evaluation | Honest accuracy numbers | not started |
| 7. Prototype | The web page | not started |
| 8. Submission | Notebooks, README, recording | not started |

*Status as of Friday 6 November 2026. Deadline Sunday 8 November.*

## Where things stand, and what comes next

**Done.** 19 regions with radar, optical and Dynamic World labels (cropland rule
applied). Code reorganised into packages. Model training moved into notebooks.
Hand labels archived and the Random Forest dropped. 12 new validation and test
regions chosen by measurement and added to `config.py`; the Earth Engine script
exports only those 12.

**Not yet done, in order:**

1. ~~Download the 12 new regions~~ — **done 5 Nov**, but four regions came out
   partly empty because the orbit track was picked by image count, not by
   coverage (paris 2%, serengeti 25%, egypt_desert 55%, congo 79%; see
   `changes.md` §9). Fixed in `fetch_s1.py`. Delete those four radar folders
   and re-run `acquisition\run_all.py --skip-png`. Dynamic World from the Earth
   Engine script is unaffected — it works from the boxes, not the tracks.
2. `labelling\import_labels.py`, `labelling\check_labels.py`, then
   `training\build_dataset.py` to rebuild the tiles (the old ones were deleted).
3. ~~Write DeepLabV3+ and SegFormer~~ — **done 5 Nov**, in `training/model.py`;
   one notebook per model (`02_train_unet`, `02_train_deeplab`, `02_train_seg`).
   First comparison (early stopping): test 47.9 / 43.9 / 47.7%, unstable.
   **6 Nov: retrain all three** with the stabilised recipe - 40 full epochs,
   weight averaging, warm-up, clipping, fixed overfit test (`changes.md` Phase 5 §6).
4. **6 Nov: all nine Dynamic World classes kept** (`changes.md` Phase 3 §11):
   re-run `import_labels.py` and `build_dataset.py`, then retrain all three.
   Code pushed to GitHub; tiles go to the university data lab separately
   (they are ~700 MB and stay out of git), training runs there.
5. Build the prototype (`app/`, Phase 7) and record the Mannheim screen capture.

The full account of every decision and what broke is in `changes.md`.

---

## Phase 0 — Setup

**Goal.** A Python environment that can hold the whole project.

**Done.** Python 3.11 virtual environment (not 3.14 — see `changes.md`), with
numpy, rasterio, requests and matplotlib. Project laid out with `config.py`
holding every setting so no script hard-codes anything.

---

## Phase 1 — Radar acquisition ✅

**Goal.** One clean radar image per month, per area.

**What it does.** Asks Microsoft Planetary Computer's free catalogue which
Sentinel-1 images cover the area, picks a single orbit track, downloads each pass
cropped and shrunk to 50 m by the server, and combines the passes for each month
by taking the middle value per pixel.

**Output.** `data/s1_monthly/<area>/<area>_<year>-<month>_50m.tif` — a GeoTIFF with
three layers: VV, VH, and how many passes saw each pixel.

**Done when.** Twelve files per area, each covering 100% of the box.

---

## Phase 2 — Optical acquisition

**Goal.** A colour image of the same places, on the same pixel grid as the radar.

**Why.** Radar cannot be labelled by eye. A human needs a picture they can
actually interpret in order to create the answer key. Sentinel-2 provides that.
It is used **only** to build labels and never reaches the model.

**Key requirement.** The optical image must land on the **exact same grid** as the
radar — same box, same width, same height. If it does not, every label is shifted
relative to the radar it is meant to describe, and the model learns nonsense.

**The constraint.** Clouds. Over Mannheim in June 2021 there were 11 optical
images and only 1 was clear enough to use. Winter is far worse. So optical is
collected only for the months where it exists, which is precisely why the final
model must run on radar.

**Output.** `data/s2_monthly/<area>/...` — spectral bands plus a true-colour image
for human viewing, on the radar grid.

**Done when.** At least one clear optical composite per area, aligned and checked.

**Done.** 31 monthly composites across the three areas — 10 for `mannheim`, 10
for `karlsruhe`, 11 for `odenwald`. Fewer than twelve each because some months
are too cloudy to use at all. Alignment with the radar verified.

---

## Phase 2b — Data evaluation

**Goal.** Understand what the radar alone can and cannot do, **before** spending
days on labelling.

**Why it has its own phase.** Two of its checks can change the plan:

- How many classes does the data actually support? If the answer is four, then
  labelling six is wasted effort — and the class list is the hardest decision to
  undo once labelling has started.
- Do crops reveal themselves across twelve months? If not, the three-month input
  experiment planned for Phase 5 has no basis and should be dropped.

A third sets numbers that would otherwise be guessed: how far apart two pixels
must be before they are independent, which fixes the tile size and justifies
splitting the data by whole area rather than by tile.

**Dropped, 3 November 2026.** The notebook (`evaluation/data_eval.ipynb`) was
written as sections but never run, and was then removed. Its purpose was to
decide how many classes the radar could support *before* investing in hand
labels. Once the professor approved external labels, the class list became
fixed by the mapping from the external product, so the question it answered
no longer needed asking.

**Needs.** `scikit-learn`, not yet installed. Sections 3 and 10 also need
individual satellite passes rather than monthly composites, so `fetch_s1.py`
needs a small option to keep single scenes.

**Done when.** The findings table at the end of the notebook is filled in, and
the class count for Phase 3 has been decided from evidence rather than
assumption.

---

## Phase 2c — Global area selection ✅

**Goal.** Decide which places the model will know about, before any labelling
locks that choice in.

**Why it comes first.** The prototype takes any latitude and longitude on Earth.
Three areas in the Rhine valley cannot support that. Labelling is what commits us
to a set of places, so the set has to be chosen before labelling starts — see
**The deployment gap** in `design.md`.

**Verified already.** Sentinel-1 RTC exists worldwide, so any area is reachable.
But acquisition density varies about sevenfold: Europe gets 170–210 scenes a
year, the Amazon, the Sahel and inland Australia get about 30. Thin regions
produce noisier composites, which is a second domain shift on top of the
landscape one.

### The candidate list — 3 regions per class, 18 total

The sampling rule: **three regions per class, where that class is prominent,
spread across climates and both hemispheres.** Every region still contains most
of the other classes, so 18 regions gives every class both dedicated examples
and many incidental ones.

All 18 were checked before committing. The figure that matters is **passes per
month on a single orbit track**, not the total over the box — a box overlaps
several tracks but only one is ever used, so the raw total flatters.

| Class | Region | lon, lat | Passes/month | Clear optical months |
|---|---|---|---|---|
| water | IJsselmeer, NL | 5.40, 52.70 | 7.4 | 12 |
| water | Mekong delta, VN | 105.80, 10.00 | 5.1 | 11 |
| water | Nile delta, EG | 31.00, 30.80 | 4.9 | 11 |
| built-up | Randstad, NL | 4.60, 52.10 | 7.3 | 12 |
| built-up | Cairo, EG | 31.25, 30.05 | 4.9 | 12 |
| built-up | Delhi, IN | 77.10, 28.60 | 4.8 | 12 |
| trees | Finland boreal | 25.70, 62.20 | 6.0 | 12 |
| trees | Oregon, US | −122.50, 44.00 | 4.8 | 12 |
| trees | Amazon, BR | −60.00, −3.10 | 2.6 | 11 |
| cropland | Pampas, AR | −60.50, −34.00 | 4.8 | 12 |
| cropland | Punjab, IN | 75.80, 30.90 | 2.6 | 12 |
| cropland | Iowa, US | −93.60, 41.90 | 2.5 | 12 |
| grass | NSW, AU | 147.40, −35.10 | 4.8 | 12 |
| grass | Mongolia steppe | 106.00, 47.00 | 4.3 | 12 |
| grass | Kenya savanna | 36.80, −1.30 | 2.5 | 12 |
| bare | Atacama, CL | −69.50, −23.50 | 9.7 | 12 |
| bare | Egypt desert | 29.00, 26.00 | 4.7 | 12 |
| bare | Arabia, SA | 45.00, 24.00 | 4.7 | 12 |

**Every one is usable.** The weakest are 2.5 passes a month, comparable to
Germany in 2023, which still produces a usable composite. Optical coverage is
good everywhere — even the Amazon has eleven months with at least one image
under 40% cloud, which was the risk worth checking, since a region that cannot
be labelled is useless regardless of how good its radar is.

Five regions are in the southern hemisphere — Pampas, NSW, Atacama, Kenya,
Amazon — which matters for the hemisphere trap described in `design.md`.

**Picking rules applied.**

- **Transition zones, not monocultures.** A tile that is 95% open ocean or
  uniform desert teaches almost nothing. The boundaries are where the learning
  is, so each region sits where its class meets others: deltas, city edges,
  forest margins, desert fringes. Aim for the dominant class at roughly 40–60%.
- **Class balance is handled in the loss, not only by sampling.** Choosing
  water-heavy regions ensures enough water pixels exist; class-weighted loss
  stops the model over-predicting it. Both are needed.

**To do.**

1. Fine-tune each bounding box so the dominant class lands near 40–60%, and
   check what the region actually looks like before downloading all twelve
   months of it.
2. Reserve **3 whole regions** as the held-out test set, from climates not used
   in training.
3. Decide the labelling budget per region. **Sparse labelling across many
   regions beats dense labelling of few** — unlabelled pixels are excluded from
   the loss, so the same number of clicks buys far more geographic coverage.
   This is what makes 18 regions affordable at all.
4. Download. Roughly 31 MB per region-year, so about **560 MB** for all 18.

**Done when.** The boxes are fixed, the train/test split by region is written
down, and the data is on disk.

---

## Phase 3 — Labelling

> **Changed on 3 November 2026.** The professor confirmed that an existing
> labelled dataset is acceptable and that self-labelling is not graded. Training
> labels now come from an external land-cover product; the hand masks below
> become the **independent test set** and are never trained on. See "Revised
> plan" at the end of this phase. Everything above that block is the original
> plan, kept as the record of how the project got here.

**Goal.** Real, human-made land-cover masks. This is the phase the professor
explicitly asked for: *find data and label it yourself if it is unlabelled.*

**Approach.** Label **regions, not points**. Open the colour image, click inside a
field or a forest block, have an assisting model outline that whole region, then
assign it a class. A few hundred clicks produces a dense per-pixel mask.

Semantic segmentation needs a class for **every pixel**, so scattered dots are not
enough on their own.

**Classes (6 plus nodata).** water, built-up, trees, cropland, grass / low
vegetation, bare / sparse.

**Tooling.** The annotation tool being built for the AI Vision course. A crude
fallback script must exist first so this phase cannot block on that tool.

**Two separate label sets are required:**
1. **Training masks** — the bulk, over most areas.
2. **A test set** — a different area, labelled independently, never used in
   training. Without it every accuracy number is circular.

**Output.** `data/labels/<area>/...` — single-layer GeoTIFFs, same grid, values 0–6.

**Done when.** Training masks exist for at least two areas, a test mask exists for
a third, and all align with the radar.

**Time-box this phase.** It is required — the professor asked for self-made labels
— but it is not what a Deep Learning course grades. Labelling can absorb unlimited
time, and every hour spent perfecting masks is an hour not spent on the model
experiments in Phase 5, which is where the marks are. Label enough area to train
on, keep the quality honest, and move on. If more training data turns out to be
needed, the Random Forest can extend coverage — declared plainly in the report.

### Status, 29 October 2026 — 14 of 19 regions

2.8 M labelled pixels. Masks: amazon, arabia, atacama, cairo, delhi,
egypt_desert, finland, ijsselmeer, iowa, kenya, mannheim, mekong, nsw, oregon.
All verified on the radar grid by `check_labels.py`.

| Class | Labelled px | Share |
|---|---|---|
| bare / sparse | 779,241 | 27.7% |
| trees | 649,908 | 23.1% |
| built-up | 626,398 | 22.3% |
| water | 487,970 | 17.4% |
| cropland | 264,527 | 9.4% |
| **grass / low vegetation** | **2,037** | **0.1%** |

**The grass class is the one real gap.** 2,037 pixels against bare's 779,241
cannot be learned, and class weighting does not rescue a class that is barely
present. Mongolia is 97% grass and unlabelled; NSW and Kenya were labelled
without it. Until one of those is done, expect grass IoU near zero, and report
it as a data result rather than a model one.

**Mistakes caught and fixed**, all by `check_labels.py` and the annual-NDVI
check. Worth recording because each was invisible to the eye:

- **Arabia** — 26,612 px of desert labelled built-up. No optical index
  separates bare from built-up (desert NDBI 0.08-0.16, built-up 0.06-0.15), so
  that pair always needs a human looking for street-grid structure. Cleared and
  redone.
- **Finland** — 41,167 px of built-up that was **snow lying on boreal forest**:
  January NDVI 0.02 at 5.8x the summer brightness, July NDVI 0.71. The tool's
  own legend said built-up "looks pale, cyan, white", and so does snow. Cleared
  and redone. **Never label a snowy region from a winter month.**
- **Delhi** — 5,071 px of "water" on the Yamuna's farmed floodplain, only 18%
  of it overlapping real water. The deeper cause was the month: the clearest
  month shows 132 water pixels, September shows 8,487. **Pick the month for the
  feature being labelled, not once per region.**
- **Amazon** — doubted on its percentages (27% water, 25% built-up in a
  rainforest), then confirmed correct: the region is **Manaus**. Check the place
  before doubting the labels.


---

### Revised again, 5 November 2026 — Dynamic World only

**Every label is Dynamic World; the hand labels are archived** in
`data/archive/hand_labels/`, out of the pipeline but not deleted.

| split | regions |
|---|---|
| train | the 18 original regions other than mannheim |
| validation | one per class: chesapeake, paris, black_forest, beauce, serengeti, namib |
| test | one per class: lake_geneva, tokyo, congo, kansas, ireland, death_valley — plus mannheim |

The 12 new regions were **measured before download**: Sentinel-1 in all 12
months of 2021 on one orbit track, and ESA WorldCover 2021 confirming the
class dominates the box. *That measured time, not space* — a track with images
every month, not one covering the box. The download showed the gap: see
`changes.md` §9. That rejected two picks that sounded right — a
"steppe" box that was 81% lake, and a Lake Victoria box with no shoreline.
Kansas sits in test deliberately: 61% cropland and 29% grass, it scores the
model on the pair radar finds hardest. Mannheim is test so the screen
recording shows a city the model never trained on.

**Consequence:** with no hand labels, every score is agreement with Dynamic
World, not with ground truth.

### Revised plan, 3 November 2026

> **Superseded on 5 November 2026.** The hand labels were then archived and
> the Random Forest baseline dropped; every label is Dynamic World, and
> validation and test use new regions, one dominated by each class. Kept
> below as the record of how the project got there.

**Train on external labels, test on hand labels.**

| | Source | Used for |
|---|---|---|
| `data/labels_external/` 🔜 | Dynamic World, 10 m → 50 m by majority vote | training |
| `data/labels/` ✅ | 14 regions labelled by hand | **validation and testing** |

**Why the swap.** Hand labelling hit three limits that more clicking could not
fix in the time left: grass at 2,037 px (unlearnable), 41% of each tile
unlabelled, and overfitting after ~6 epochs on only ~233 distinct pieces of
ground. External labels solve the first two and make the third cheap.

**Why the hand labels go to the test side, not the bin.** Testing on external
labels would only measure how well radar copies another model. Testing on hand
labels measures agreement with careful human judgement the model never saw.

**Rule:** a region is train or test, never both, whatever labels it carries.

**Source: Google Dynamic World V1**, chosen 3 November over Impact Observatory
io-lulc and ESA WorldCover (compared in `tech.md`). Exported from Earth Engine
by `labelling/gee/export_dynamic_world.js`, imported by `import_labels.py`.

**Test split changed:** mongolia left `TEST_AOIS`. The test set is now scored
against hand labels and mongolia has none, so it could not be scored — and it
is ~97% grass, the one class the hand labels barely cover.

**Done when.** External masks exist for every training region on the radar
grid; an agreement report compares them with the hand masks; the U-Net is
retrained on them and scored on the hand-labelled test regions alongside the
hand-trained model.

---

## Phase 4 — Dataset assembly

**Goal.** Turn full-size images into something a model can train on.

**Steps.** Cut the radar and the masks into matching tiles (128×128 pixels).
Discard tiles that are mostly nodata. Convert radar to decibels and scale it.
Stack VV, VH and their ratio as three channels so an ImageNet-pretrained encoder
can be used unchanged. Split by **whole area**, never by tile — neighbouring tiles
are nearly identical, so a random tile split would leak and inflate the score.

**Output.** `data/tiles/` plus a manifest listing which tile belongs to which split.

**Done when.** Tile counts and class balance are reported, and a visual check
confirms radar and mask line up.

### Status — done, 29 October 2026

`build_dataset.py` produced **3,840 tiles**, 284 MB:

| Split | Tiles | Regions |
|---|---|---|
| train | 2,796 | everything not listed below |
| val | 528 | mannheim, atacama |
| test | 516 | mekong, oregon (mongolia still unlabelled) |

41% of the pixels inside the kept tiles carry no label and are excluded from
the loss — sparse labelling working as intended, not a shortfall.

One tile is one region-month: the annual mask paired with each month's radar in
turn. That is what forces the model to recognise a class across the whole
seasonal cycle instead of one appearance of it.


---

## Phase 5 — Training

**This is a Deep Learning course, so this phase and Phase 6 are the heart of the
submission.** The data pipeline is necessary groundwork, but the grader is looking
at the model: the architecture choice, the training procedure, and whether the
experiments supporting them are sound. Budget time accordingly.

**Goal.** A model that turns radar into a land-cover map, plus the experiments
that justify its design.

**Approach.** A **U-Net written from scratch in PyTorch** — roughly four
down-blocks, base width 32, batch normalisation, skip connections, about 1–2
million parameters. Not imported from a library: in a Deep Learning course the
architecture is the thing being assessed, and 24 million pretrained parameters
would badly overfit 168 distinct ground positions anyway. See `tech.md`.

Class-weighted loss because water and built-up are rare compared with farmland.
Nodata pixels excluded from the loss entirely. Augmentation includes flips and
rotations plus two radar-specific ones: artificial speckle and a small random
brightness shift, which together buy month-to-month stability.

**Before trusting any training run:** take ten tiles and deliberately overfit
them. A correct implementation should drive the loss to near zero on ten
examples within a couple of minutes. If it cannot, the bug is in the code, not
the data or the hyperparameters — and this catches it in minutes rather than
after a day of confusing results. Do this first, every time the model changes.

**Hardware.** RTX 3050, 4 GB. At 50 m with 128×128 tiles this is comfortable, and
a full run takes well under an hour — so several experiments fit in a day.

### The experiments

> **Changed on 5 November 2026:** the Random Forest baseline was dropped, and
> the plan below was replaced by **three architectures trained from scratch
> and compared on the same validation and test regions**, the best taken
> forward. The original plan is kept as the record.

A deep learning submission needs more than one trained model and a loss curve.
Each of these answers a real question and costs roughly one training run.

**1. A non-deep baseline.** A per-pixel Random Forest on `[VV, VH, VV/VH]`,
classifying each pixel with no spatial context at all.

This is the most valuable single number in the report. It isolates exactly what
the U-Net's ability to see *neighbourhoods* is worth. If the U-Net barely beats
it, the texture around a pixel does not matter much for this problem — itself a
finding. If it beats it clearly, that gap is the justification for using deep
learning here at all.

**2. Our model vs a pretrained one.** Build the same U-Net shape with an
ImageNet-pretrained ResNet-18 encoder, via `segmentation_models_pytorch`, and
compare.

ImageNet weights were learned on ordinary photographs. Radar is not photographs.
Does that transfer? The answer is genuinely not obvious, and it cuts both ways:
pretrained features usually help when data is scarce, but 24 million parameters
on 168 distinct ground positions invites memorisation.

Whichever wins, it is a result worth reporting — and reporting that our own
smaller model matched or beat a pretrained one on this data would be a stronger
finding than the reverse.

Note this comparison needs `[VV, VH, VV/VH]` as exactly three channels, since the
pretrained first layer expects three. Our own model has no such constraint.

**3. Single-month input vs a three-month window.** Feed months (m−1, m, m+1) as
nine channels and predict month m.

This targets the one weakness we have already measured: cropland and grass overlap
71% at 10 m and still 61% at 50 m from a single date, because the thing that
separates them is the sowing–growth–harvest cycle, which is invisible in one
snapshot. If anything fixes that pair, this will.

Note that the output stays **one map per month**, as the brief requires — the
extra months are context, not a change in what is predicted.

**Output.** `weights/`, training curves, and a results table comparing the runs.

**Done when.** Loss falls, validation score plateaus, per-class scores are
physically sensible — water and built-up well above the vegetation classes — and
all three experiments have numbers.

### Status — experiment 1 done, 29 October 2026

**Random Forest baseline** (notebook 01), per pixel, no spatial context:

| | Validation | Test (unseen regions) |
|---|---|---|
| mean IoU | 30.6% | **45.3%** |
| accuracy | 51.7% | 61.2% |

Per class on test: water **83.5%**, trees 58.3%, built-up 31.1%, cropland 8.4%.
Grass and bare do not occur in the test regions and are excluded from the mean.

Three findings already worth reporting:

- **Water at 83.5%, from three numbers per pixel**, confirms the project's
  premise. Radar sees water as a mirror — measured at IJsselmeer, VV reads
  -16.7 dB over water against -8.8 dB over land.
- **Cropland fails at 8.4%**, mistaken for trees 47% of the time. Measured
  cause: Iowa cropland swings 6.7 dB across the year while genuinely bare
  ground swings 1.0 dB, and in August the two are indistinguishable. No
  single-month model can resolve that — which is the physical justification for
  experiment 3, stronger than the one written above it.
- **Built-up is mistaken for trees 43% of the time.** Buildings double-bounce
  off wall and ground, canopies volume-scatter; both return bright and rough. A
  genuine limitation of radar, not of the model.

**Compare like with like.** An earlier version of this section said "the U-Net
must beat 45.3%" — but 45.3% is the forest's *test* score, while the U-Net's
52.9% is a *validation* score: two different pairs of regions. On the same
regions the gap is far larger than first reported:

| validation (mannheim, atacama) | mean IoU |
|---|---|
| Random Forest, per pixel | 30.6% |
| U-Net, base 16 | **52.9%** |
| **gap — what spatial context is worth** | **~22 points** |

The U-Net has no test score yet: `train.py` only scored validation until
3 November. It now reports both, and every model in the comparison must be
quoted on the **test** split. Both figures above predate the Oregon relabel and
the move to external training labels, so they are re-run as part of that.

The notebooks in `notebooks/` run this and the U-Net with plots, importing the
same modules the scripts use so the two cannot drift apart.


---

## Phase 6 — Evaluation

**Goal.** Numbers that are actually true, and an honest account of where the model
fails.

**What it reports.**

- **Per-class IoU** on the held-out area, against the independently hand-labelled
  test set — never against the labels used for training.
- **A confusion matrix**, which should show the cropland/grass confusion
  explicitly rather than burying it in a mean.
- **The experiment comparison** from Phase 5, as a single table.
- **A stability check / noise floor.** Run the model on two radar passes from the
  same month and measure how much the output changes. Land cover did not change in
  those six days, so whatever moves is noise. Any month-to-month change smaller
  than this figure is meaningless, and saying so is what makes the prototype's
  change numbers trustworthy.
- **Failure analysis with pictures.** Where does it go wrong, and why? Water in
  high wind, field boundaries, built-up edges. A deep learning report that shows
  its failure cases and explains them reads far better than one reporting a single
  mean score.

**Expected.** Water and built-up well above average. Cropland versus grass poor,
because radar physically struggles to separate them. Report this honestly with
the reason rather than hiding it — the measurement is already in `changes.md`.

---

## Phase 7 — Prototype

**Goal.** The web page described in the brief.

**Flow.** User picks a point on a map and a start month → the app works out a
20 km box → fetches radar for 12 months using the **same code as Phase 1** →
runs the model → counts pixels → shows maps, a table and charts.

**Critical details.**
- The same preprocessing recipe as training. If they differ, the model fails
  silently.
- Pixels no satellite saw must be excluded from the percentage denominator, or
  partial months deflate every class and invent change.
- Results cached per location and month, and pre-warmed before recording.
- Only offer start months at least 12 months in the past, or the data will not
  exist yet.

**Output.** `app/` and a working page.

---

## Phase 8 — Submission

Everything the brief asks for:

| Required | Where it comes from |
|---|---|
| Dataset + download link | Our generated tiles and masks, on Zenodo, plus the exact queries in `data/README.md` |
| Notebooks with results shown | Data exploration, labelling, training, evaluation |
| Prototype source, README, requirements | `app/`, `README.md`, `requirements.txt` |
| Trained weights | `weights/` |
| Screen recording, urban area | Full run over Mannheim, cache pre-warmed |

`changes.md` feeds the written report directly.

---

## Open questions for the professor

1. ~~Does "do not train on high-resolution data" rule out Sentinel-2 for
   *generating* labels?~~ **Answered 3 Nov:** external labelled data is fine,
   which settles this — a label source is not model input.
2. ~~Is ESA WorldCover permitted for validation only?~~ **Answered 3 Nov:**
   external labels are allowed for training as well, and self-labelling is not
   graded.
3. Should the 12 months run forward from the start month, or backward? Forward is
   assumed.
