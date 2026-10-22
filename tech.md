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
| Label helper model | scikit-learn (Random Forest) | 3 |
| Deep learning | PyTorch + segmentation_models_pytorch | 5, 6 |
| Web prototype | Streamlit + streamlit-folium | 7 |
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

## Random Forest for label generation

Not deep learning. A Random Forest on Sentinel-2 spectral bands is:

- fast to train on a few hundred hand-labelled examples
- not prone to overfitting on small data the way a neural network is
- able to report a confidence per pixel, so uncertain pixels can be dropped
- easy to explain in a report

Its job is to extend human labels across an area, not to be the final model.

## PyTorch + segmentation_models_pytorch

`segmentation_models_pytorch` provides U-Net with pretrained encoders in a few
lines. It is a thin, well-maintained wrapper over PyTorch rather than a separate
framework, so the model stays fully inspectable and modifiable — which matters for
a Deep Learning course, where the model itself is under scrutiny and not just the
pipeline feeding it.

**U-Net** because it is the standard architecture for semantic segmentation on
small datasets. It keeps fine detail through skip connections and trains on a few
thousand tiles, which a transformer would not.

**ResNet-18 or ResNet-34 encoder, ImageNet-pretrained.** Radar is not photographs,
but the early layers of an ImageNet encoder learn edges and textures, which
transfer usefully. Stacking VV, VH and their ratio as three channels lets the
pretrained weights be used unchanged rather than discarded.

**Not a transformer.** Too little data, and 4 GB of video memory.

## Streamlit for the prototype

The brief wants a prototype taking a latitude, longitude and month. Streamlit
turns a Python script into a web page with no front-end code, and
`streamlit-folium` gives a real clickable map — which makes the latitude/longitude
input obvious on the screen recording.

The alternative, FastAPI plus a hand-written front end, would be more work for no
extra marks here.

## Hardware

RTX 3050 with 4 GB. Low by modern standards, but after downsampling to 50 m the
tiles are small and this stops being the limiting factor. No cloud training —
everything runs locally and reproducibly.

## Deliberately not used

| Not using | Why |
|---|---|
| SEN12MS, BigEarthNet and similar | Pre-labelled. The professor requires self-made labels. Their labels are also 500 m, far coarser than the imagery. |
| ESA WorldCover as training labels | Same reason. Possibly allowed for validation — awaiting an answer. |
| Google Earth Engine | Forces the evaluator to register for a cloud account. |
| ESA SNAP | Heavy install; RTC products already provide what it would compute. |
| Sentinel-2 at prediction time | The brief forbids it. It is used only to build labels, then discarded. |
