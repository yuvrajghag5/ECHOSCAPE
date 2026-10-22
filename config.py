"""
All the knobs for the project live here.

Change things in this file, not inside the scripts.
"""

import math
from pathlib import Path

# ---------------------------------------------------------------- resolution
# How much ground one pixel covers, in metres.
# The professor said "do not train on high-resolution data", so we work at 50 m
# instead of Sentinel-1's native 10 m. This also removes most of the radar
# speckle noise.
#
# 50 m was chosen by measurement, not by feel - see design.md. 100 m leaves only
# ~324 training tiles in total, far too few, and the noise it saves was already
# removed by the monthly median.
TARGET_RES_M = 50

# How big each study area is, in kilometres on each side.
# 30 km at 50 m gives 600 x 600 pixels, which is 16 non-overlapping training
# tiles of 128 px per image. Bigger boxes mean more data but a much longer
# download; 19 regions x 12 months is already about 600 MB.
AREA_SIZE_KM = 30

# ---------------------------------------------------------------- time
# 2021 is the best year available: Sentinel-1A and 1B were both flying, so every
# month has plenty of passes, evenly. Sentinel-1B failed in Dec 2021, leaving
# 2022-2024 with about half as many. 2025 recovers but its first months are thin.
#
# Even coverage matters because the assignment asks for month-to-month change:
# if some months had 3 passes and others 8, the noisier months would look like
# change that is not really there.
YEAR = 2021

# ---------------------------------------------------------------- satellite
# Radiometrically Terrain Corrected Sentinel-1. Already corrected for hills,
# licensed CC-BY-4.0 so we may redistribute what we build from it, and it needs
# no account or API key.
COLLECTION = "sentinel-1-rtc"

# Radar brightness depends on the angle the satellite looked from, so mixing
# flight paths makes the same field read differently. "auto" picks whichever
# single path has the most images over that area.
ORBIT = "auto"          # "auto", "ascending" or "descending"

# ---------------------------------------------------------------- classes
CLASSES = {
    0: "nodata",
    1: "water",
    2: "built-up",
    3: "trees",
    4: "cropland",
    5: "grass / low vegetation",
    6: "bare / sparse",
}

# ---------------------------------------------------------------- study areas
# Three regions per class, where that class is prominent, spread across climates
# and both hemispheres. Every region still contains most of the other classes,
# so this gives each class dedicated examples plus many incidental ones.
#
# All were checked before being listed. The figure that matters is passes per
# month on a SINGLE orbit path, not the total over the box - a box overlaps
# several paths but only one is ever used. See phase.md, Phase 2c.
#
# Regions sit at transitions - deltas, city edges, forest margins, desert
# fringes - not in the middle of uniform terrain. Boundaries are where the
# learning happens; a square of open ocean teaches nothing.
AOIS = {
    # ---- water ----------------------------------------------------------
    "ijsselmeer":  {"centre": (5.40, 52.70),    "class": "water",
                    "passes": 7.4, "note": "Dutch lake and reclaimed land"},
    "mekong":      {"centre": (105.80, 10.00),  "class": "water",
                    "passes": 5.1, "note": "tropical delta, rice and water"},
    "nile_delta":  {"centre": (31.00, 30.80),   "class": "water",
                    "passes": 4.9, "note": "arid delta, water meets irrigation"},

    # ---- built-up -------------------------------------------------------
    "randstad":    {"centre": (4.60, 52.10),    "class": "built-up",
                    "passes": 7.3, "note": "dense low-rise European city"},
    "cairo":       {"centre": (31.25, 30.05),   "class": "built-up",
                    "passes": 4.9, "note": "city against desert, sharp edges"},
    "delhi":       {"centre": (77.10, 28.60),   "class": "built-up",
                    "passes": 4.8, "note": "dense city, monsoon climate"},

    # ---- trees ----------------------------------------------------------
    "finland":     {"centre": (25.70, 62.20),   "class": "trees",
                    "passes": 6.0, "note": "boreal forest and lakes, snow"},
    "oregon":      {"centre": (-122.50, 44.00), "class": "trees",
                    "passes": 4.8, "note": "temperate conifer, clear-cut patches"},
    "amazon":      {"centre": (-60.00, -3.10),  "class": "trees",
                    "passes": 2.6, "note": "tropical rainforest, southern"},

    # ---- cropland -------------------------------------------------------
    "pampas":      {"centre": (-60.50, -34.00), "class": "cropland",
                    "passes": 4.8, "note": "southern hemisphere, seasons inverted"},
    "punjab":      {"centre": (75.80, 30.90),   "class": "cropland",
                    "passes": 2.6, "note": "intensive irrigated, double cropping"},
    "iowa":        {"centre": (-93.60, 41.90),  "class": "cropland",
                    "passes": 2.5, "note": "large-field continental maize/soy"},

    # ---- grass / low vegetation -----------------------------------------
    "nsw":         {"centre": (147.40, -35.10), "class": "grass",
                    "passes": 4.8, "note": "Australian pasture, southern"},
    "mongolia":    {"centre": (106.00, 47.00),  "class": "grass",
                    "passes": 4.3, "note": "dry continental steppe"},
    "kenya":       {"centre": (36.80, -1.30),   "class": "grass",
                    "passes": 2.5, "note": "tropical savanna, southern"},

    # ---- bare / sparse --------------------------------------------------
    "atacama":     {"centre": (-69.50, -23.50), "class": "bare",
                    "passes": 9.7, "note": "hyper-arid, almost no vegetation"},
    "egypt_desert": {"centre": (29.00, 26.00),  "class": "bare",
                    "passes": 4.7, "note": "sand and rock desert"},
    "arabia":      {"centre": (45.00, 24.00),   "class": "bare",
                    "passes": 4.7, "note": "sand sea and gravel plain"},

    # ---- demo area ------------------------------------------------------
    # Not part of the 3-per-class design. Included because the brief requires a
    # screen recording over an urban area, and a local one is more convincing -
    # the professor can check it against his own window.
    "mannheim":    {"centre": (8.60, 49.42),    "class": "mixed",
                    "passes": 7.0, "urban": True,
                    "note": "DEMO - Mannheim/Heidelberg, for the screen recording"},
}

# Which area to use when a script is not told one.
DEFAULT_AOI = "mannheim"

# Regions held back entirely for testing. The model never sees these during
# training, so the score on them is an honest measure of whether it generalises
# to places it has not seen. Chosen from climates used in training, but
# different continents.
#
# TODO Phase 2c: confirm this choice once the areas have been looked at.
TEST_AOIS = ["mekong", "oregon", "mongolia"]


def bbox_for(name):
    """Work out the (west, south, east, north) box for an area.

    Areas are stored as a centre point plus a size in kilometres, not as a box,
    because a degree of longitude shrinks as you move away from the equator. A
    fixed box in degrees would cover 30 km in Kenya and 15 km in Finland. This
    keeps every region the same size on the ground.
    """
    lon, lat = AOIS[name]["centre"]
    half_lat = (AREA_SIZE_KM / 2) / 111.32
    half_lon = half_lat / max(math.cos(math.radians(lat)), 0.01)
    return (round(lon - half_lon, 6), round(lat - half_lat, 6),
            round(lon + half_lon, 6), round(lat + half_lat, 6))


def areas_by_class():
    """Group the area names by which class they were chosen for."""
    grouped = {}
    for name, info in AOIS.items():
        grouped.setdefault(info["class"], []).append(name)
    return grouped


# ---------------------------------------------------------------- folders
ROOT = Path(__file__).parent
DATA = ROOT / "data"
S1_MONTHLY = DATA / "s1_monthly"     # radar - the model's input
S2_MONTHLY = DATA / "s2_monthly"     # colour - only used to make labels
LABELS = DATA / "labels"             # the answer key we create in Phase 3
