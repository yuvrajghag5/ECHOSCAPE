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

---

# Phase 3 — Labelling

## 1. Initial build

## 2. Issues faced

## 3. How it was solved

---

# Phase 4 — Dataset assembly

## 1. Initial build

## 2. Issues faced

## 3. How it was solved

---

# Phase 5 — Training

## 1. Initial build

## 2. Issues faced

## 3. How it was solved

---

# Phase 6 — Evaluation

## 1. Initial build

## 2. Issues faced

## 3. How it was solved

---

# Phase 7 — Prototype

## 1. Initial build

## 2. Issues faced

## 3. How it was solved

---

# Phase 8 — Submission

## 1. Initial build

## 2. Issues faced

## 3. How it was solved
