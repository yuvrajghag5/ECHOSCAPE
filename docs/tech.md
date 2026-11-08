# Technology Choices

Every choice here has a reason. The report has to justify them, so the reason is
written down next to the choice.

## Summary

| Layer | Choice | Phase |
|---|---|---|
| Language | Python 3.11 | all |
| Satellite data source | Microsoft Planetary Computer (STAC + data API) | 1, 2, 7 |
| Geospatial file handling | rasterio | 1–7 |
| Arrays and maths | numpy | all |
| HTTP | requests | 1, 2, 7 |
| Clustering for data evaluation | scikit-learn (KMeans, silhouette) | 2b |
| Labels | Google Dynamic World V1, via Earth Engine | 3 |
| Region checks | ESA WorldCover 2021, via Planetary Computer | 2c |
| Deep learning | PyTorch — **U-Net written from scratch** | 5, 6 |
| Pretrained comparison only | segmentation_models_pytorch | 5 |
| Web prototype | FastAPI + uvicorn, hand-written page | 7 |
| Plots | matplotlib | all |
| Notebooks | Jupyter | 8 |

---

## Python 3.11, not 3.14

The machine has Python 3.14 installed as default, but **PyTorch has no CUDA build
for 3.14**. Without it the graphics card cannot be used and training would run on
the processor, which is far too slow.

3.11 is the version with the widest support across PyTorch, rasterio and the rest.
This was found the hard way — see `changes.md`.

## Microsoft Planetary Computer

**What it is.** A free, public catalogue of satellite data with a search interface
(STAC) and a service that can crop and resize images before sending them.

**Why this and not Google Earth Engine or Copernicus Open Hub:**

| | Planetary Computer | Google Earth Engine | Copernicus Data Space |
|---|---|---|---|
| Account needed | **none** | Google Cloud project, manual approval | registration |
| Crops server-side | **yes** | yes | no, whole scenes |
| Resizes server-side | **yes** | yes | no |
| Works from plain Python | **yes** | needs their SDK and auth | yes |

The deciding factor is the **account**. The brief says the README must document
"required accounts or API keys". With Planetary Computer there are none — the
person marking this can run the prototype immediately. Anything requiring the
evaluator to register for a cloud service and wait for approval is a risk on
grading day for no benefit.

The Sentinel-1 RTC collection is also licensed **CC-BY-4.0**, which means the
dataset we build from it can be redistributed with attribution. The brief requires
submitting the dataset, so this matters.

## Sentinel-1 RTC, not raw GRD

RTC means Radiometrically Terrain Corrected. Hills distort radar badly — a slope
facing the satellite looks far brighter than the same ground facing away. RTC has
already corrected for this using an elevation model.

Doing that correction ourselves would need ESA's SNAP toolbox, a large install and
a slow processing step per scene. Using RTC removes an entire phase of work.

## rasterio

Satellite images are not ordinary pictures. Every pixel has a real-world
coordinate attached. rasterio reads and writes that properly.

The alternative, GDAL directly, is the same engine with a much harder interface.
rasterio installs from a normal wheel on Windows, which matters — this machine has
no Visual Studio build tools, so anything needing compilation fails.

## Server-side cropping instead of downloading scenes

One full Sentinel-1 scene is about 1 GB and covers 250 × 170 km. The study area is
29 × 28 km, roughly 2% of it.

| Approach | Download size for one area, one year |
|---|---|
| Whole scenes, crop locally | ~84 GB |
| Ask the server for the crop at 50 m | **31 MB** |

The data API accepts a bounding box and an output size and returns a correctly
georeferenced GeoTIFF. This is the single biggest practical decision in the
project — it is the difference between a feasible and an infeasible download.

## PyTorch + segmentation_models_pytorch

**The main model is written from scratch in plain PyTorch**, not imported from a
library. This is a Deep Learning course: the architecture is the thing being
assessed, and a model we built and can account for line by line is worth more
than a one-line import.

There are three substantive reasons beyond that, not just presentation:

**1. Size.** A ResNet-34 U-Net carries roughly 24 million parameters. Our dataset
is about 2,000 tiles drawn from 168 distinct ground positions. That is enormous
overcapacity, and overcapacity on a small dataset means memorisation rather than
learning. A hand-built U-Net at around 1–2 million parameters is far better
matched, and on 4 GB of video memory it also trains faster, which means more
experiments.

**2. ImageNet is photographs. Radar is not.** Pretrained encoders learn edges and
textures from natural images. Whether that transfers to backscatter is a genuine
open question here, not a given — so it belongs in the experiments section as
something to test, not in the design as an assumption.

**3. It frees the input.** Using pretrained weights forces exactly three input
channels. Building our own means the input can be whatever serves the problem —
two channels, or four with a texture measure added, or nine for a three-month
window. The architecture follows the data instead of the data being bent to fit
the architecture.

**U-Net** as the shape, because it is the standard for semantic segmentation on
small datasets: an encoder that compresses, a decoder that expands, and skip
connections carrying fine detail across so boundaries survive. Roughly four
down-blocks, base width 32, batch normalisation throughout.

**A transformer only as one of three.** Written when the plan was one model, 14
regions and a 4 GB card, this said "not a transformer". Since 5 November the
plan compares three architectures on 31 regions with a university GPU, so a
small SegFormer (2.3 M parameters, shaped like the published MiT-B0) is one of
them — sized to match the U-Net, not the 3.7 M original. Whether attention
earns its keep on this little data is now a question the comparison answers.

**`segmentation_models_pytorch` is still installed** — but only to construct the
pretrained comparison model in Phase 5, as something to measure ours against.

## Streamlit for the prototype

The brief wants a prototype taking a latitude, longitude and month. Streamlit
turns a Python script into a web page with no front-end code, and
`streamlit-folium` gives a real clickable map — which makes the latitude/longitude
input obvious on the screen recording.

The alternative, FastAPI plus a hand-written front end, would be more work for no
extra marks here.

**Changed 6 November 2026: FastAPI and a hand-written page instead.** Streamlit
was never installed, and FastAPI and uvicorn already were. The page also had
to animate the year as it plays, which Streamlit's rerun-on-every-click model
does poorly. Only the fonts come from a public CDN. The app needs the internet
anyway, to fetch the radar. No service it uses needs an API key.

## Hardware

RTX 3050 with 4 GB. Low by modern standards, but after downsampling to 50 m the
tiles are small and this stops being the limiting factor. No cloud training —
everything runs locally and reproducibly.

## Deliberately not used

| Not using | Why |
|---|---|
| SEN12MS, BigEarthNet and similar | Their labels are MODIS-derived at 500 m, ten times coarser than our grid. (The earlier objection — that labels had to be self-made — was withdrawn by the professor on 3 November.) |
| Google Earth Engine — *for anything but one export* | Forces the evaluator to register for a cloud account. **Used once since 3 November**, to export Dynamic World labels; the exported files ship with the dataset, so nobody needs Earth Engine to train or evaluate — only to regenerate the labels from scratch. |
| ESA SNAP | Heavy install; RTC products already provide what it would compute. |
| Sentinel-2 at prediction time | The brief forbids it. It is used only to build labels, then discarded. |

---

## What is actually installed, 29 October 2026

| Package | State | Used by |
|---|---|---|
| numpy, rasterio, requests, matplotlib | installed | everything |
| scipy | installed | region growing, zone analysis in `check_labels.py` |
| scikit-learn | installed | unused since 5 Nov, when the baseline was dropped |
| **torch (CUDA)** | **not installed** | Phase 5 U-Net — a ~2.5 GB download |
| segmentation_models_pytorch | not installed | experiment 2 only |
| jupyter, ipykernel | not installed | running the submitted notebooks |

PyTorch is deliberately kept out of `requirements.txt`: it must come from the
CUDA index, not PyPI, or pip quietly installs a CPU-only build that works and
is far too slow to notice until training.

```bash
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install jupyter ipykernel
```

---

## The training-label source — Dynamic World V1, chosen 3 November

Decided 3 November: train on an external land-cover product, test on the hand
labels. **Dynamic World was chosen.** Its decisive advantages here are a
per-pixel confidence, which lets uncertain pixels be dropped rather than
trained on, a separate grass class, and that it is made from Sentinel-2 alone —
a purely optical teacher. Its cost is that it lives only on Earth Engine.

The three candidates that were weighed, all 10 m, all CC-BY-4.0:

| | Impact Observatory `io-lulc-annual-v02` | **ESA WorldCover 2021** (v200) | Google Dynamic World V1 |
|---|---|---|---|
| Where | Planetary Computer, anonymous | Planetary Computer, anonymous | **Earth Engine only** — Google login |
| 2021 | annual map, ready | annual map, ready | per-image; you build the composite |
| Grass | merged with shrub as "rangeland" | **its own class** | **its own class** |
| Confidence per pixel | no | no | **yes** — can drop unsure pixels |
| Accuracy (2024 comparison) | lowest of the three | **highest — 83.8%** | middle |
| Made from | Sentinel-2 | **Sentinel-1 and Sentinel-2** | Sentinel-2 |
| Fits `fetch_s2.py` code | yes | yes | no — Drive export workflow |

**The WorldCover catch.** It is the most accurate and its classes map onto ours
almost one-for-one — but it was produced partly *from Sentinel-1*. A radar model
trained on labels a radar-aware model drew is not learning from a purely optical
teacher, which blunts the claim "radar alone can do this". It must be
declared, and it is the one reason to prefer a purely optical source — one of
the reasons Dynamic World, made from Sentinel-2 alone, was chosen. (WorldCover
is still used, but only to *check* that new regions are dominated by their
class before downloading them.)

**Class mapping**, for whichever is chosen. Until 6 Nov 2026 this merged down to
six (table below); since then Dynamic World's nine classes are all kept, one to
one, as ids 1-9 (`config.DW_CLASS_MAP`) - see `changes.md` Phase 3 §11.

| ours | io-lulc | WorldCover | Dynamic World |
|---|---|---|---|
| water | Water | Permanent water | water |
| built-up | Built area | Built-up | built |
| trees | Trees | Tree cover, Mangroves | trees |
| cropland | Crops | Cropland | crops |
| grass / low veg | Rangeland | Grassland, Shrubland | grass, shrub & scrub |
| bare / sparse | Bare ground | Bare / sparse vegetation | bare |
| *unlabelled* | Flooded veg, Snow, Clouds | Snow, Herbaceous wetland, Moss & lichen | flooded veg, snow & ice |

Every product is downsampled 10 m → 50 m by **majority vote** onto the radar
grid. Earth Engine should export at 10 m and be resampled locally, so the grid
match stays in code that `check_labels.py` can verify.

### How Dynamic World becomes a training mask

Dynamic World is one map per Sentinel-2 pass, not an annual product, and its
export has to be made to fit a grid it knows nothing about.

| Step | Where | Why |
|---|---|---|
| mean of the nine class probabilities over 2021, then the winner | Earth Engine | Keeps how sure each pass was. Taking the most frequent label instead lets a run of uncertain passes outvote a few confident ones. |
| export the label **+ 1** | Earth Engine | Dynamic World's class 0 is **water**, and Earth Engine writes masked pixels as 0. Without the shift, every pixel with no cloud-free view arrives silently as water. |
| export confidence and a view count too | Earth Engine | So the threshold can be tuned locally without re-exporting. |
| nearest-neighbour onto an exact 5x-finer copy of the radar grid | `import_labels.py` | Makes the final grid identical to the radar's by construction. Nearest, not averaging — these are class codes. |
| drop 10 m pixels under 0.35 confidence | `import_labels.py` | Dynamic World's own advice. Kept moderate, because it warns crops and arid ground score low — two of our classes, four of our regions. |
| vote in 5 x 5 blocks: >= 50% usable, winner >= 60% | `import_labels.py` | Refuses mixed pixels — field edges, shorelines — which is where external labels are most often wrong. |
| **cropland must green up**: "cropland" whose yearly peak NDVI stays under 0.40 becomes grass (or bare) | `import_labels.py` | Dynamic World labelled the Mongolian steppe 46% cropland at a peak of 0.23; real cropland peaks at 0.71–0.81. The rule catches 99% of that and under 1% of real cropland. |
