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
