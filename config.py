"""
All the knobs for the project live here.

Change things in this file, not inside the scripts.
"""

# ---------------------------------------------------------------- resolution
# How much ground one pixel covers, in metres.
# The professor said "do not train on high-resolution data", so we work at 50 m
# instead of Sentinel-1's native 10 m. This also removes most of the radar
# speckle noise. Changing this number changes the whole pipeline.
TARGET_RES_M = 100

# ---------------------------------------------------------------- study areas
# Each area is (west, south, east, north) in degrees.
# "urban" marks the one used for the required screen recording.
AOIS = {
    # Main area: Mannheim + Ludwigshafen + Heidelberg.
    # Contains city, the Rhine and Neckar rivers, farmland and forest.
    # Best Sentinel-1 coverage of anywhere we checked.
    "mannheim": {
        "bbox": (8.40, 49.30, 8.80, 49.55),
        "urban": True,
        "note": "main area - used for the screen recording",
    },
    # Extra areas for training variety. Add more as needed.
    "karlsruhe": {
        "bbox": (8.20, 48.90, 8.60, 49.15),
        "urban": True,
        "note": "second city, Rhine plain",
    },
    "odenwald": {
        "bbox": (8.80, 49.40, 9.20, 49.65),
        "urban": False,
        "note": "mostly forest - gives the model forest examples",
    },
}

# Which area to download when you do not say one on the command line.
DEFAULT_AOI = "mannheim"

# ---------------------------------------------------------------- time
# 2021 is the best year for this project: Sentinel-1A and 1B were both flying,
# so every single month has 6-8 passes. That matters because the assignment asks
# for month-to-month change - if some months had 3 passes and others 8, the
# noisier months would look like change that is not really there.
#
# Avoid 2022-2024 entirely: Sentinel-1B failed in Dec 2021, leaving half as many
# images. 2025 works too, but its first four months are thin (2-3 passes) because
# Sentinel-1C was still ramping up.
YEAR = 2021

# ---------------------------------------------------------------- satellite
# Radiometrically Terrain Corrected Sentinel-1. Already corrected for hills,
# licensed CC-BY-4.0 so we are allowed to redistribute what we build from it,
# and it needs no account or API key.
COLLECTION = "sentinel-1-rtc"

# Radar looks different depending on which direction the satellite was flying.
# Mixing them makes the same field look like two different things, so we pick
# one and stick to it. "auto" picks whichever has the most images.
ORBIT = "auto"          # "auto", "ascending" or "descending"

# ---------------------------------------------------------------- classes
# What the model will learn to recognise. Six classes that radar can actually
# tell apart, give or take the cropland/grass pair which is genuinely hard.
CLASSES = {
    0: "nodata",
    1: "water",
    2: "built-up",
    3: "trees",
    4: "cropland",
    5: "grass / low vegetation",
    6: "bare / sparse",
}

# ---------------------------------------------------------------- folders
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
S1_MONTHLY = DATA / "s1_monthly"     # radar - the model's input
S2_MONTHLY = DATA / "s2_monthly"     # colour - only used to make labels
LABELS = DATA / "labels"             # the answer key we create in Phase 3
