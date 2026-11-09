"""
All the knobs for the project live here.

Change things in this file, not inside the scripts.
"""

import math
import re
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
# flight paths makes the same field read differently. "auto" picks one single
# path: among those that cover enough of the box, the one with the most images.
ORBIT = "auto"          # "auto", "ascending" or "descending"

# A path must cover at least this share of the box in EVERY month to be used.
# Before 5 Nov the path was picked by image count alone, and a path can pass
# every six days while only clipping the corner of the box: paris came out 2%
# covered, serengeti 25%, egypt_desert 55%, congo 79%. Measured from scene
# footprints in the catalogue, before anything is downloaded. 0.90 rather
# than 1.0 so a region missing a thin edge strip (kansas 95%, ireland 98%)
# keeps its busier path - more passes means less speckle.
MIN_TRACK_COVER = 0.90

# ---------------------------------------------------------------- classes
# All nine Dynamic World classes, none merged (since 6 Nov 2026; before that,
# shrub was folded into grass and flooded vegetation and snow were dropped).
# Ids 1-6 kept their old meaning; the three restored classes come after
# them as 7-9.
CLASSES = {
    0: "nodata",
    1: "water",
    2: "built-up",
    3: "trees",
    4: "cropland",
    5: "grass",
    6: "bare",
    7: "shrub & scrub",
    8: "flooded vegetation",
    9: "snow & ice",
}
NUM_CLASSES = len(CLASSES)      # network outputs, including 0 = unlabelled

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

    # ---- the three rare classes (added 9 Nov 2026, training only) -------
    # In the first build shrub & scrub was 1.16% of the training pixels, snow
    # & ice 0.87% and flooded vegetation 0.03% - about one tile's worth - and
    # every model scored ~0% on all three. Two regions each, training only,
    # so validation and test stay exactly as before and the runs compare.
    # Scouted read-only before downloading (9 Nov): every one has Sentinel-1
    # in all 12 months of 2021 on one track covering >=95% of the box, and
    # "share" is ESA WorldCover 2021's share of the box in that class.
    # (Rejected: aletsch 35% ice, louisiana 81% open water, sundarbans
    # mangrove - Dynamic World tends to call it trees - karoo 72% bare,
    # mallee 60% trees.)
    "vatnajokull": {"centre": (-16.70, 64.42),  "class": "snow & ice",
                    "passes": 5.0, "note": "Icelandic ice cap, 100% snow/ice"},
    "baltoro":     {"centre": (76.45, 35.75),   "class": "snow & ice",
                    "passes": 2.5, "note": "Karakoram glaciers, 87% snow/ice"},
    "sudd":        {"centre": (30.60, 7.50),    "class": "flooded vegetation",
                    "passes": 6.8, "note": "Nile swamp, South Sudan, 91% wetland"},
    "everglades":  {"centre": (-80.80, 25.70),  "class": "flooded vegetation",
                    "passes": 2.4, "note": "sawgrass marsh, Florida, 98% wetland"},
    "sonoran":     {"centre": (-113.00, 32.40), "class": "shrub & scrub",
                    "passes": 4.9, "note": "Arizona desert scrub, 43% shrub"},
    "kalahari":    {"centre": (21.50, -24.50),  "class": "shrub & scrub",
                    "passes": 7.0, "note": "Botswana bushland, 40% shrub + 60% grass"},

    # ---- demo area ------------------------------------------------------
    # Not part of the 3-per-class design. Included because the brief requires a
    # screen recording over an urban area, and a local one is more convincing -
    # the professor can check it against his own window.
    "mannheim":    {"centre": (8.60, 49.42),    "class": "mixed",
                    "passes": 7.0, "urban": True,
                    "note": "DEMO - Mannheim/Heidelberg, for the screen recording"},

    # ---- validation and test regions (added 5 Nov 2026) -------------------
    # One region dominated by each class for VALIDATION and one for TEST, so
    # every class is scored in both. Measured before downloading: every one
    # has Sentinel-1 in all 12 months of 2021 on a single orbit track, and
    # "share" is how much of the box ESA WorldCover 2021 puts in that class -
    # an independent check that the region really is what it is meant to be.
    # (That check rejected the Kazakh steppe, which turned out to be 81% lake,
    # and Lake Victoria, whose box held no shoreline at all.)
    #
    # validation
    "chesapeake":   {"centre": (-76.25, 38.20), "class": "water", "passes": 2.5,
                     "split": "val", "note": "estuary, 82% water"},
    "paris":        {"centre": (2.35, 48.86),   "class": "built-up", "passes": 7.3,
                     "split": "val", "note": "dense European city, 60% built-up"},
    "black_forest": {"centre": (8.15, 48.35),   "class": "trees", "passes": 7.4,
                     "split": "val", "note": "temperate mountain forest, 79% trees"},
    "beauce":       {"centre": (1.65, 48.20),   "class": "cropland", "passes": 7.4,
                     "split": "val", "note": "French wheat plain, 91% cropland"},
    "serengeti":    {"centre": (34.90, -2.60),  "class": "grass", "passes": 5.0,
                     "split": "val", "note": "East African savanna, 100% grass"},
    "namib":        {"centre": (15.20, -24.50), "class": "bare", "passes": 2.3,
                     "split": "val", "note": "coastal sand sea, 96% bare"},
    # test
    "lake_geneva":  {"centre": (6.55, 46.45),   "class": "water", "passes": 7.2,
                     "split": "test", "note": "lake with forest and pasture, 39% water"},
    "tokyo":        {"centre": (139.70, 35.70), "class": "built-up", "passes": 4.7,
                     "split": "test", "note": "megacity on the bay, 79% built-up"},
    "congo":        {"centre": (24.50, 0.60),   "class": "trees", "passes": 2.4,
                     "split": "test", "note": "tropical rainforest, 90% trees"},
    "kansas":       {"centre": (-98.60, 38.40), "class": "cropland", "passes": 4.8,
                     "split": "test",
                     "note": "Great Plains, 61% cropland + 29% grass: the hard pair"},
    "ireland":      {"centre": (-7.40, 53.40),  "class": "grass", "passes": 7.2,
                     "split": "test", "note": "lush pasture, 77% grass"},
    "death_valley": {"centre": (-116.90, 36.40), "class": "bare", "passes": 4.8,
                     "split": "test", "note": "desert basin and ranges, 94% bare"},
}

# Which area to use when a script is not told one.
DEFAULT_AOI = "mannheim"

# Regions held back entirely for testing. The model never sees these during
# training, so the score on them is an honest measure of whether it generalises
# to places it has not seen.
#
# Since 5 Nov 2026 every label comes from Dynamic World, and validation and test
# are new regions picked per class (see the end of AOIS). mannheim is in test as
# well: it is the screen-recording city, so the recording shows ground the model
# never trained on. Everything not listed here or in VAL_AOIS is training data.
TEST_AOIS = [name for name, a in AOIS.items() if a.get("split") == "test"] + ["mannheim"]


def month_of(path):
    """Pull the YYYY-MM out of a data filename.

    Deliberately NOT by splitting on underscores. Two region names contain one
    - nile_delta and egypt_desert - so "nile_delta_2021-01_50m" splits into four
    parts instead of three and every position shifts. That made the completeness
    report claim both regions had no data at all when all 12 months were sitting
    on disk.

    Match the shape of the thing instead of counting separators.
    """
    found = re.search(r"(\d{4}-\d{2})", Path(path).stem)
    if not found:
        raise ValueError(f"no YYYY-MM found in {path}")
    return found.group(1)


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
LABELS = DATA / "labels"             # Dynamic World masks - the ONLY labels (5 Nov)
DW_RAW = DATA / "dw_raw"             # Earth Engine exports, downloaded from Drive


# ================================================================ Phase 4 / 5
# Everything below is for turning labelled regions into training tiles and
# training on them. Added 29 Oct 2026.

TILES = DATA / "tiles"               # Phase 4 - training tiles cut from LABELS
WEIGHTS = ROOT / "weights"           # Phase 5 - trained models
OUTPUTS = ROOT / "outputs"           # maps, percentages, curves, figures

# ---------------------------------------------------------------- tiling
# How big one training example is, in pixels. 128 at 50 m is 6.4 km on the
# ground - wide enough for the model to see a field next to a town, which is
# the context a per-pixel classifier cannot use.
TILE_PX = 128

# How far to step between tiles. Equal to TILE_PX means no overlap.
#
# Overlapping tiles look like more data but are not: two tiles sharing half
# their pixels are nearly the same example. Splits here are by whole region so
# a leak cannot happen anyway, but keeping them separate means the tile count
# is an honest count.
TILE_STRIDE = 128

# Throw a tile away if less than this fraction of it carries a label.
#
# Sparse labelling is the plan, so most tiles are partly empty and that is
# fine - unlabelled pixels are excluded from the loss. But a tile that is 2%
# labelled costs a full forward and backward pass to learn from a handful of
# pixels.
MIN_LABELLED_FRACTION = 0.10

# ---------------------------------------------------------------- radar scaling
# Radar brightness spans orders of magnitude, so it is converted to decibels
# before the model sees it. These are the limits used to squash decibels into
# roughly 0-1.
#
# FIXED NUMBERS ON PURPOSE. Scaling each image by its own min and max would
# make every image's brightest pixel read as 1.0, destroying the thing the
# model needs most: that water is absolutely dark and cities are absolutely
# bright, everywhere, always. Per-image scaling would also mean the live
# prototype preprocessed differently from training, which fails silently.
VV_DB_RANGE = (-25.0, 5.0)
VH_DB_RANGE = (-32.0, 0.0)
RATIO_DB_RANGE = (0.0, 15.0)         # VV minus VH, in decibels

# ---------------------------------------------------------------- splits
# Whole regions held back for deciding when to stop training and which model
# won. Never trained on, but looked at often enough that the score stops being
# independent - which is what TEST_AOIS is for. One region per class.
VAL_AOIS = [name for name, a in AOIS.items() if a.get("split") == "val"]


# ================================================================ external labels
# Training labels from Google Dynamic World V1 (since 3 Nov 2026). Exported by
# gee/export_dynamic_world.js, turned into masks on the radar grid by
# import_labels.py. See docs/tech.md for why this source.

# Dynamic World class + 1, as exported (0 = no data), to our id. One to one:
# every Dynamic World class is kept as its own class, only renumbered.
DW_CLASS_MAP = {
    1: 1,   # water               -> water
    2: 3,   # trees               -> trees
    3: 5,   # grass               -> grass
    4: 8,   # flooded_vegetation  -> flooded vegetation
    5: 4,   # crops               -> cropland
    6: 7,   # shrub_and_scrub     -> shrub & scrub
    7: 2,   # built               -> built-up
    8: 6,   # bare                -> bare
    9: 9,   # snow_and_ice        -> snow & ice
}

# A 10 m pixel counts only if Dynamic World's average certainty over the year
# was at least this. Its own documentation recommends thresholding, and warns
# that crops and arid surfaces score low - which is two of our classes and four
# of our regions - so this is deliberately moderate rather than strict.
DW_MIN_CONFIDENCE = 0.35

# One 50 m pixel is 5 x 5 = 25 Dynamic World pixels. Label it only if enough of
# them are usable and enough of those agree. Mixed pixels - field edges,
# shorelines, the edge of a town - are where labels go wrong, and an unlabelled
# pixel costs nothing: it is simply left out of the loss.
DW_MIN_VALID = 0.5      # at least half the 25 must be usable
DW_MIN_PURITY = 0.6     # and the winning class must hold 60% of those

# Cropland must green up. It is sown, grown and harvested, so at some point in
# the year it carries a full canopy. Dynamic World judges from single images,
# and dry seasonal grassland fools it: it labelled the Mongolian steppe 46%
# cropland, at a yearly peak NDVI of 0.23 - where real cropland peaks at 0.71
# (Iowa) to 0.81 (NSW). So a Dynamic World "cropland" pixel whose year never
# reaches DW_CROP_MIN_PEAK is not cropland: it becomes grass if
# anything grows there at all (peak >= DW_VEG_MIN_PEAK), and bare if not.
#
# Measured before choosing 0.40: it catches 99.1% of Mongolia's false cropland
# and under 1% of the real cropland in Iowa, Pampas, Punjab and the Nile Delta.
# These are the same thresholds check_labels.py checks every mask against
# (cropland >= 0.40, grass >= 0.15).
#
# "Peak" is the 85th percentile of the monthly NDVI, not the maximum: one month
# brightened by thin cloud would otherwise decide it.
DW_CROP_MIN_PEAK = 0.40
DW_VEG_MIN_PEAK = 0.15
