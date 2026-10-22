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
| 1. Radar acquisition | Monthly Sentinel-1 composites | **done** — 36 files |
| 2. Optical acquisition | Matching Sentinel-2, aligned to the radar grid | **done** — 31 files |
| 2b. Data evaluation | What the radar alone can tell us, before labelling | skeleton written, not run |
| 3. Labelling | Hand-made land-cover masks | not started |
| 4. Dataset assembly | Training tiles and splits | not started |
| 5. Training | Trained model weights | not started |
| 6. Evaluation | Honest accuracy numbers | not started |
| 7. Prototype | The web page | not started |
| 8. Submission | Notebooks, README, recording | not started |

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

**Where.** `evaluation/data_eval.ipynb` — ten checks, written out as sections
with notes. No code yet.

**Needs.** `scikit-learn`, not yet installed. Sections 3 and 10 also need
individual satellite passes rather than monthly composites, so `fetch_s1.py`
needs a small option to keep single scenes.

**Done when.** The findings table at the end of the notebook is filled in, and
the class count for Phase 3 has been decided from evidence rather than
assumption.

---

## Phase 3 — Labelling

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

---

## Phase 5 — Training

**This is a Deep Learning course, so this phase and Phase 6 are the heart of the
submission.** The data pipeline is necessary groundwork, but the grader is looking
at the model: the architecture choice, the training procedure, and whether the
experiments supporting them are sound. Budget time accordingly.

**Goal.** A model that turns radar into a land-cover map, plus the experiments
that justify its design.

**Approach.** U-Net with a ResNet-18 or ResNet-34 encoder. Class-weighted loss
because water and built-up are rare compared with farmland. Nodata pixels
excluded from the loss entirely. Augmentation includes flips and rotations plus
two radar-specific ones: artificial speckle and a small random brightness shift,
which together buy month-to-month stability.

**Hardware.** RTX 3050, 4 GB. At 50 m with 128×128 tiles this is comfortable, and
a full run takes well under an hour — so several experiments fit in a day.

### The experiments

A deep learning submission needs more than one trained model and a loss curve.
Each of these answers a real question and costs roughly one training run.

**1. A non-deep baseline.** A per-pixel Random Forest on `[VV, VH, VV/VH]`,
classifying each pixel with no spatial context at all.

This is the most valuable single number in the report. It isolates exactly what
the U-Net's ability to see *neighbourhoods* is worth. If the U-Net barely beats
it, the texture around a pixel does not matter much for this problem — itself a
finding. If it beats it clearly, that gap is the justification for using deep
learning here at all.

**2. Pretrained encoder vs trained from scratch.** ImageNet weights were learned
on ordinary photographs. Radar is not photographs. Does that transfer?

A genuinely interesting question rather than a box-ticking ablation, and the
answer is not obvious. Stacking `[VV, VH, VV/VH]` as three channels is what makes
the comparison possible without modifying the first layer.

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

1. Does "do not train on high-resolution data" also rule out using Sentinel-2 to
   *generate* labels? The plan assumes not, since Sentinel-2 is a label source
   rather than model input.
2. Is ESA WorldCover permitted **for validation only**? Currently avoided.
3. Should the 12 months run forward from the start month, or backward? Forward is
   assumed.
