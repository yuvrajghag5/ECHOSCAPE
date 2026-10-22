# Design

What is being built, for whom, and the decisions behind it.
(`architecture.md` covers *how* the pieces fit together. `tech.md` covers *which*
tools. This file is the *why*.)

## What is being assessed

This is a **Deep Learning** course. The model is the subject: its architecture,
its training procedure, and the experiments that justify both.

That shapes where effort goes. The data pipeline in Phases 1–4 is necessary — and
the professor explicitly required self-made labels — but it is groundwork. The
submission stands or falls on Phases 5 and 6: a model trained properly, compared
against a sensible baseline, with honest per-class numbers and a clear account of
where it fails.

Three experiments are planned rather than a single trained model: a non-deep
baseline to show what spatial context is worth, a pretrained-versus-scratch
comparison to test whether ImageNet features transfer to radar, and a
single-month-versus-three-month input test aimed at the one weakness already
measured. See `phase.md`.

## The problem

Land cover maps — where the water, buildings, forest and farmland are — are
normally made from optical satellite imagery, because a camera image is easy to
interpret. But cameras cannot see through cloud.

Over Mannheim in 2021 we measured this directly. At a 40% cloud limit:

- **October: 0 usable optical images out of 6**
- **November: 0 usable optical images out of 8**

In those same months, Sentinel-1 radar made 7 and 8 passes, every one usable.
Radar brings its own signal, so cloud and darkness are irrelevant to it.

So the problem this project solves is: **produce land-cover maps in the months
when optical mapping is impossible.**

That is also exactly what the assignment asks for — a model using Sentinel-1 as
its only input.

## Who uses it, and what they do

One user: someone who wants to know what the ground looks like at a location
across a year.

They enter a latitude, longitude and a start month. They get back twelve monthly
land-cover maps, the percentage of the area in each class each month, how those
percentages changed month to month, and the total change across the year.

They do not need an account, an API key, or any knowledge of satellites.

## The core difficulty

**Radar cannot be read by eye.**

In an optical image, a forest is green and a city is grey — anyone can label it.
In a radar image, water is black and cities are bright white, and *everything else
is grey static*. A wheat field, a meadow and a potato field look the same.

This is not a skill problem. The information is genuinely not visible.

Which creates the central design problem: the professor requires that we label our
own data, but the data we must label is unreadable.

## The key decision: an optical teacher, discarded before use

The brief constrains the model's **input at prediction time**. It says nothing
about where the training labels came from.

So:

```
Sentinel-2 (colour)  ->  human labels regions  ->  land-cover masks
                                                        |
Sentinel-1 (radar)  ------------------------------------+--> model learns
                                                        |
                        at prediction time: radar only, no optical anywhere
```

The colour imagery is a translator. It lets a human say what is on the ground; the
radar provides the signal the model actually learns from. Once training is done,
the optical imagery is never used again — which is the point, because in October
and November it would not exist.

**This is the project's argument in one line: train where optical works, deploy
where it does not.**

## The deployment gap — the project's biggest validity problem

**Status: open. To be worked through before Phase 3 labelling begins.**

The prototype accepts **any latitude and longitude on Earth**. The model, as
currently planned, would be trained on three areas in the Rhine valley.

A model trained on temperate German river plain will produce confident, wrong
maps over a desert, a rainforest or an alpine valley. It will not fail loudly —
it will return a plausible-looking land-cover map that is simply incorrect. That
is worse than refusing to answer.

This is the single largest threat to the project being *valid* rather than merely
*finished*, and it has to be addressed before labelling starts, because labelling
is what locks in which places the model knows.

### What we verified

Sentinel-1 RTC is available **worldwide** — wherever raw GRD exists, the
terrain-corrected product exists at essentially the same count. No need to fall
back to processing raw scenes ourselves.

But acquisition density varies about sevenfold:

| Region | Scenes per year | Passes per month |
|---|---|---|
| Finland, Spain, Egypt, Germany | 170–212 | plentiful |
| Argentina, Vietnam, India | 90–118 | workable |
| Iowa, Shanghai | ~57 | thin |
| **Amazon, Sahel, New South Wales** | **~30** | **1–3 total** |

ESA prioritises Europe. Much of the southern hemisphere gets a fraction of the
coverage.

**This matters beyond convenience.** Our monthly composites over Germany average
7–8 passes and come out clean. A composite over the Sahel might rest on a single
pass and be far noisier. So a globally deployed model faces *two* shifts at once:
different landscapes, and different noise levels.

### The approach to take

**1. Many areas, sparsely labelled — the key move.**

Semantic segmentation does not require every pixel of a training image to be
labelled. Unlabelled pixels are simply excluded from the loss. The model still
sees the full radar tile as input; it just is not scored on the parts we skipped.

So the choice is not "3 areas labelled fully" versus "15 areas labelled fully".
It is **the same number of clicks, spread across five times as many places.**
Geographic coverage is what this model lacks, and this buys it at no extra
labelling cost.

**2. Choose areas deliberately, not conveniently.**

Stratify by climate zone and by the classes we need. Every class should appear in
several different climates — "built-up" in a European city, a Nile delta town and
a Brazilian city are not the same radar signature. Avoid regions with ~30 scenes
a year unless they are the only source of a land cover type we need.

**3. Train with the noise the deployment will actually have.**

Our training composites are built from 7–8 passes. Much of the world gets 1–3.
If the model only ever sees clean data it will degrade badly on noisy data for
reasons that look like a modelling failure.

Fix: during training, sometimes build composites from fewer passes, and augment
with extra speckle. Make the model see the full range of data quality it will
meet.

**4. Tell the user when we are guessing.**

Keep the worldwide input, but measure whether the incoming radar resembles the
training data, and say so on screen when it does not.

This is honest, cheap, and a genuine deep learning contribution rather than a
disclaimer — out-of-distribution detection is a real topic and suits the course.
A prototype that says "this location is unlike anything I was trained on, treat
with caution" is more defensible than one that silently guesses.

**5. The hemisphere trap.**

If Phase 5 uses a three-month input window to catch the crop cycle, that window
encodes a *northern* growing season. Deployed in Argentina or Australia, the
seasons are inverted and the model would be exactly wrong.

Two ways out: include southern-hemisphere areas in training, or feed the month
relative to the local growing season rather than as a calendar month. Decide
before building that experiment.

**6. A clean experiment this creates.**

Does standardising each scene by its own statistics improve transfer to unseen
regions? It should reduce the effect of regional differences in absolute
brightness, at the cost of discarding information — water is dark in an absolute
sense, and that is useful.

That is a well-formed ablation with a real answer, and it belongs in Phase 5.

### What this does not fix

Even done well, a student project cannot claim a globally validated model. The
honest end state is: trained across *n* regions, tested on held-out regions,
with measured performance per climate zone and an explicit statement of where it
has not been tested. That is a much stronger claim than an unqualified one, and
it is true.

## Decisions and their trade-offs

### Label regions, not points

Semantic segmentation needs a class for *every* pixel. Scattered labelled dots do
not provide that.

Clicking inside a field and having a model outline the whole region gives dense,
genuinely human-made masks for a few hundred clicks. Slower per area than dotting,
but it satisfies "label it yourself" literally, with no argument to have later
about whether a Random Forest did the labelling.

### 50 m pixels — not 10 m, and not 100 m

Required by the professor to be coarse, but 50 m was chosen by measurement, not
by feel. The same area and month was downloaded at three resolutions:

| | 25 m | **50 m** | 100 m |
|---|---|---|---|
| Image size | 1159 × 1113 | **579 × 557** | 290 × 278 |
| Training tiles available (128 px, 3 areas × 12 months) | 9,792 | **2,016** | **324** |
| Noise, as effective looks (higher is cleaner) | 26.7 | **60.1** | 113.6 |
| Size per month | 10.2 MB | **2.6 MB** | 0.6 MB |

**100 m fails on data volume.** The whole 29 × 28 km area becomes 290 × 278
pixels, yielding only 324 training tiles in total — far too few for a U-Net
separating six classes. No choice of tile size fixes this: 100 m data simply
contains a quarter of the measurements.

**The noise argument for going coarser is already spent.** A single raw pass has
about 4 effective looks. By 25 m the monthly median over 7–8 passes has already
brought that to 26.7. Compositing does nearly all the noise removal before
resolution is involved at all, so going from 50 to 100 m buys cleanliness that
was not needed and pays for it with three quarters of the data.

**25 m was rejected** for being only 2.5× coarser than Sentinel-1's native 10 m,
which is a weak answer to "do not train on high-resolution data". 50 m is 5×
coarser and comfortably satisfies it.

The cost is losing field boundaries and street-level structure. For area
percentages over a 20 km box, that does not matter.

### Build the model, do not import it

The U-Net is written from scratch in PyTorch rather than pulled from a library
with pretrained weights.

The presentational reason is real — this is a Deep Learning course and the
architecture is what is being assessed — but it is not the main one. The main
one is **size**. A pretrained ResNet-34 U-Net carries about 24 million
parameters. Our dataset is roughly 2,000 tiles from 168 distinct ground
positions. A model that large on data that small memorises rather than learns.
A hand-built U-Net at 1–2 million parameters is simply better matched to the
problem.

It also frees the input design. Pretrained weights force exactly three input
channels; our own model can take two, or four with a texture measure, or nine
for a three-month window. The architecture follows the data rather than the
other way round.

**The trade-off:** writing the training loop means owning its bugs, and a
silently wrong implementation can look like a data problem for days. The
mitigation is a discipline, not a library: before trusting any run, deliberately
overfit ten tiles. A correct model drives the loss to near zero on ten examples
in minutes. If it cannot, the fault is in the code.

The pretrained version is still built — as Phase 5's second experiment, to
measure ours against.

### One orbit track per area

Radar brightness depends on the angle the satellite looked from. Mixing ascending
and descending passes makes identical ground read differently, which the model
would learn as change that is not there.

Cost: fewer passes available. Benefit: the month-to-month change numbers mean
something.

### Year 2021

Sentinel-1A and 1B both flying gave 6–8 passes **every month**, evenly. 2022–2024
have roughly half that after Sentinel-1B failed; 2025's first four months are thin.

Even coverage matters because uneven noise between months is indistinguishable
from real change.

### No account, for anyone

Every data source used is anonymous. The person marking this can clone the repo
and run it. Any design requiring them to register for a cloud service and wait for
approval is a risk on grading day for no benefit.

## What the output honestly means

Land cover does not really change month to month. A forest in March is a forest in
April.

So what are twelve monthly maps showing?

- **The class is stable.** The radar signal is not — it swings with crop growth,
  harvest, soil moisture and frost.
- Most month-to-month variation is therefore *signal* variation, not land-cover
  change.
- Genuine change does appear: harvest turning cropland to bare soil, flooding,
  snow.

The honest framing, which belongs in the report: the seasonal signal is what the
model classifies *with*, and only changes larger than the measured noise floor
should be read as real. Phase 6 measures that noise floor by running the model on
two passes from the same month.

## Known limitations, stated up front

| Limitation | Why | Response |
|---|---|---|
| Cropland vs grass is unreliable | Their radar signatures overlap 71% at 10 m, 61% at 50 m — physically, not a modelling flaw | Report per-class scores honestly; consider multi-month input |
| Winter was never supervised | The optical teacher cannot see in October–November | Document it; it is also the project's whole point |
| Water fails in high wind | Wind roughens the surface so it stops reflecting away | Note it as a known failure mode |
| Model trained on German landscapes | Only three areas, all Rhine valley | Do not claim it generalises worldwide |

## Non-goals

- Not a global model. Three areas in one region.
- Not real-time. Radar data appears in the catalogue days after acquisition.
- Not detecting individual buildings or fields — 50 m pixels cannot support that.
- Not competing with published benchmarks. The dataset is self-labelled by design.
