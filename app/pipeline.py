"""
The prototype's engine: a point and a start month in, twelve land-cover maps out.

    point + start month
      -> a square box around the point
      -> Sentinel-1 for the 12 months, fetched with the SAME code as Phase 1
         (acquisition/fetch_s1.py: one orbit track, server-side crop to 50 m,
         monthly median)
      -> one GeoTIFF per month, read back with the SAME scaling as training
         (training/build_dataset.py: radar_channels)
      -> the best trained model, overlapping tiles averaged, as in notebook 04
      -> class shares per month, counted only over ground seen in every month
      -> PNG maps and one result.json, cached per point / month / size

Nothing here is a second copy of the preprocessing. If training and the app
prepared radar differently the model would fail silently, so both call the
same functions.
"""

import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from acquisition import fetch_s1
from training.build_dataset import radar_channels
from training.model import build_model

CACHE = config.DATA / "app_cache"          # data/ is gitignored, so this is too
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE = 64                                # tiles overlap by half, as in notebook 04
BATCH = 16
DOWNLOAD_WORKERS = 4                       # passes fetched in parallel
EARLIEST_START = (2017, 1)                 # Sentinel-1 RTC is dense from 2017 on
PICTURES = 2                               # bump when the PNGs change; old caches are redrawn

# Dynamic World's own colours, in our class order (index 0 = not mapped)
COLOURS = ["#000000", "#419bdf", "#c4281b", "#397d49", "#e49635",
           "#88b053", "#a59b8f", "#dfc35a", "#7a87c6", "#b39fe1"]
CLASS_IDS = list(range(1, config.NUM_CLASSES))


# ------------------------------------------------------------------ months
def month_list(start):
    """'2021-03' -> the 12 months from March 2021 to February 2022."""
    year, month = (int(v) for v in start.split("-"))
    out = []
    for i in range(12):
        y, m = year + (month - 1 + i) // 12, (month - 1 + i) % 12 + 1
        out.append(f"{y}-{m:02d}")
    return out


def latest_start(today=None):
    """The last start month whose 12 months are all in the past."""
    today = today or date.today()
    # the 12th month must have ended: start <= today - 12 months, minus the current month
    y, m = today.year - 1, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y}-{m:02d}"


def start_months():
    first_y, first_m = EARLIEST_START
    last = latest_start()
    out, y, m = [], first_y, first_m
    while f"{y}-{m:02d}" <= last:
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ------------------------------------------------------------------ geometry
def box_around(lat, lon, size_km):
    """(west, south, east, north), size_km on each side - as config.bbox_for."""
    half_lat = (size_km / 2) / 111.32
    half_lon = half_lat / max(math.cos(math.radians(lat)), 0.01)
    return (round(lon - half_lon, 6), round(lat - half_lat, 6),
            round(lon + half_lon, 6), round(lat + half_lat, 6))


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def familiarity(bbox, months):
    """Did the model see this ground in training? Says which region, if any.

    A training region is the one place the model's answer is not a real test:
    it learned that ground's yearly label by heart.
    """
    for name in config.AOIS:
        if overlaps(bbox, config.bbox_for(name)):
            split = ("test" if name in config.TEST_AOIS
                     else "validation" if name in config.VAL_AOIS else "training")
            same_year = all(m.startswith(str(config.YEAR)) for m in months)
            return {"status": split, "region": name, "same_year": same_year}
    return {"status": "unseen", "region": None, "same_year": False}


# ------------------------------------------------------------------ model
#
# Every trained run whose weights are on this computer is offered, so the model
# is a choice on the page rather than something the app decides silently. What
# makes a run usable is the INPUT it was trained to read, because the pipeline
# has to be able to build it from the radar it downloads:
#
#   3 channels   the month being mapped: VV, VH, VV-VH          "month"
#   9 channels   that, plus the year's mean and standard         "month + year"
#                deviation per channel (training's `mstat`)
#
# A run with any other input is listed by nothing and cannot be selected.
# Output classes may differ too - a six-class run has seven channels out - and
# everything downstream reads the count off the model, so the page's legend,
# the shares and the change table all follow.
_models = {}
_listing = {}
_model_lock = threading.Lock()

WEIGHT_DIRS = [config.WEIGHTS, config.WEIGHTS / "exp08", config.WEIGHTS / "sixclass"]
FEATURES = {3: "month", 9: "month + year"}


def _scores_for(name):
    """(validation mIoU, test mIoU, parameters) from whichever results file holds them.

    Three notebooks write three shapes, and a run is worth offering even when
    its results file is missing entirely - the score is then simply unknown.
    """
    for path in (config.OUTPUTS / f"{name}_result.json",
                 config.OUTPUTS / "exp08" / f"{name}.json",
                 config.OUTPUTS / "sixclass" / f"{name}_result.json"):
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        val, test = r.get("val") or {}, r.get("test") or {}
        if "scores" in r:                                  # notebook 09's shape
            val = (r["scores"].get("val") or {}).get("plain", {})
            test = (r["scores"].get("test") or {}).get("plain", {})

        def pick(d):
            for key in ("mean_iou", "miou", "miou_9"):
                if key in d:
                    return d[key]
            return None

        return pick(val), pick(test), r.get("parameters")
    return None, None, None


def available_models(refresh=False):
    """Runs this computer can actually run, best validation score first."""
    if _listing and not refresh:
        return _listing["runs"]
    found = {}
    for folder in WEIGHT_DIRS:
        for path in sorted(folder.glob("*_best.pt")):
            name = path.stem.replace("_best", "")
            if name in found:
                continue                                   # weights/ wins over its subfolders
            try:
                saved = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue                                   # unreadable checkpoint: not offered
            kwargs = dict(saved.get("kwargs") or {})
            channels = kwargs.get("in_channels", saved.get("in_channels", 3))
            classes = kwargs.get("classes", saved.get("classes", config.NUM_CLASSES))
            if channels not in FEATURES:
                continue
            val, test, parameters = _scores_for(name)
            found[name] = {"name": name, "path": str(path), "architecture": saved.get("model"),
                           "in_channels": channels, "classes": classes,
                           "features": FEATURES[channels], "epoch": saved.get("epoch"),
                           "parameters": parameters, "val_miou": val, "test_miou": test,
                           "recipe": saved.get("recipe")}
    runs = sorted(found.values(), key=lambda m: (m["val_miou"] is None, -(m["val_miou"] or 0)))
    _listing["runs"] = runs
    return runs


def default_model():
    """The best VALIDATION score among the FULL class list, never the best test score.

    Two rules, and both matter.

    Choosing by test would mean the test set had been used to make a decision,
    and it would stop being an independent estimate.

    And a mean IoU is only comparable across models that answer the same
    question. A six-class run scores 75.4% on validation against this
    nine-class model's 51.3%, but it is not three classes better - it simply
    was not asked about the three hardest ones. Ranking them together would
    quietly hand the prototype a model that cannot name a wetland or a
    glacier. So the default is drawn from the runs that predict every class;
    anything narrower has to be chosen by hand on the page.
    """
    runs = available_models()
    if not runs:
        raise RuntimeError("No trained weights found - put a *_best.pt file in "
                           "weights/ (see the README).")
    full = [r for r in runs if r["classes"] == config.NUM_CLASSES]
    return (full or runs)[0]["name"]


def model(name=None):
    name = name or default_model()
    with _model_lock:
        if name not in _models:
            spec = next((m for m in available_models() if m["name"] == name), None)
            if spec is None:
                raise RuntimeError(f"No usable weights for {name!r} on this computer.")
            saved = torch.load(spec["path"], map_location=DEVICE, weights_only=False)
            kwargs = dict(saved.get("kwargs") or {})
            kwargs.setdefault("in_channels", spec["in_channels"])
            kwargs.setdefault("classes", spec["classes"])
            net = build_model(saved["model"], **kwargs).to(DEVICE).eval()
            net.load_state_dict(saved["state"])
            noise = None
            change_file = config.OUTPUTS / f"monthly_change_{name}.json"
            if change_file.exists():
                noise = json.loads(change_file.read_text()).get("noise_floor_pts")
            spec = {**spec, "parameters": spec["parameters"]
                    or sum(p.numel() for p in net.parameters())}
            _models[name] = {**spec, "net": net, "noise_floor": noise}
        return _models[name]


CARD = ("name", "architecture", "parameters", "val_miou", "test_miou", "epoch",
        "noise_floor", "classes", "in_channels", "features")


def card(info):
    """What the page is told about the model it is looking at."""
    return {k: info.get(k) for k in CARD}


def year_statistics(months_x, months_seen):
    """The year's mean and standard deviation per channel.

    The same arithmetic as the `mstat` tiles in notebook 08: months a satellite
    did not see are left out of both, and the deviation is scaled by four
    because the deviation of a 0-1 channel is small. If this differed from
    training by so much as that factor, the model would fail quietly.
    """
    stack = np.stack(months_x)                                  # (months, 3, H, W)
    weight = np.stack(months_seen)[:, None].astype(np.float32)  # (months, 1, H, W)
    counted = np.clip(weight.sum(0), 1.0, None)
    mean = (stack * weight).sum(0) / counted
    std = np.sqrt((((stack - mean) ** 2) * weight).sum(0) / counted)
    return mean.astype(np.float32), np.clip(std * 4.0, 0.0, 1.0).astype(np.float32)


def month_input(months_x, year_mean, year_std, i, channels):
    """The tensor the chosen model reads for month i."""
    if channels == 3:
        return months_x[i]
    return np.concatenate([months_x[i], year_mean, year_std], axis=0)


def starts(total, tile=config.TILE_PX, stride=STRIDE):
    s = list(range(0, total - tile + 1, stride))
    if s[-1] != total - tile:
        s.append(total - tile)
    return s


@torch.no_grad()
def probabilities(x, info):
    """(C, H, W) radar -> (classes - 1, H, W) probabilities, overlapping tiles averaged."""
    net = info["net"]
    _, H, W = x.shape
    T = config.TILE_PX
    score = np.zeros((info["classes"] - 1, H, W), np.float32)
    hits = np.zeros((H, W), np.float32)
    spots = [(r, c) for r in starts(H) for c in starts(W)]
    for i in range(0, len(spots), BATCH):
        chunk = spots[i:i + BATCH]
        batch = torch.from_numpy(np.stack([x[:, r:r + T, c:c + T] for r, c in chunk])).to(DEVICE)
        prob = torch.softmax(net(batch)[:, 1:], dim=1).cpu().numpy()   # class 0 is never an answer
        for (r, c), p in zip(chunk, prob):
            score[:, r:r + T, c:c + T] += p
            hits[r:r + T, c:c + T] += 1
    return score / np.maximum(hits, 1)


# ------------------------------------------------------------------ pictures
def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


PALETTE = np.array([_hex(c) for c in COLOURS], np.uint8)


def save_classes_png(classes, seen, path):
    rgba = np.zeros(classes.shape + (4,), np.uint8)
    rgba[..., :3] = PALETTE[classes]
    rgba[..., 3] = np.where(seen & (classes > 0), 255, 0)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def save_radar_png(x, seen, path):
    """The month's radar as a plain greyscale picture: VV backscatter, on the
    same fixed decibel scale the model reads (dark = smooth, bright = rough)."""
    grey = (np.clip(x[0], 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([grey, grey, grey, np.where(seen, 255, 0).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


# ------------------------------------------------------------------ radar
def search_range(bbox, first_month, last_month):
    """fetch_s1.search_scenes, for any 12 months instead of one calendar year."""
    y, m = (int(v) for v in last_month.split("-"))
    last_day = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    body = {"collections": [config.COLLECTION], "bbox": list(bbox),      # same form as fetch_s1
            "datetime": f"{first_month}-01/{last_day.isoformat()}", "limit": 1000}
    reply = fetch_s1.SESSION.post(fetch_s1.STAC, json=body, timeout=120)
    reply.raise_for_status()
    scenes = reply.json().get("features", [])
    wanted = set(month_list(first_month))
    return [s for s in scenes if s["properties"]["datetime"][:7] in wanted]


class Job:
    """One run. The web page polls `state()` while it works."""

    def __init__(self, lat, lon, start, size_km, model_name=None):
        self.lat, self.lon, self.start, self.size_km = lat, lon, start, size_km
        self.model_name = model_name or default_model()
        # Two folders, because the radar and the maps have different owners.
        #
        # The RADAR belongs to the place: twelve months of Sentinel-1 for a
        # point are the same bytes whichever model reads them, and downloading
        # them again to compare two models would waste minutes and bandwidth on
        # files already on disk.
        #
        # The MAPS belong to the model: switching model and getting the first
        # one's PNGs back would be worse than slow, it would be wrong.
        #
        # The default model's maps stay in the radar folder, so every cache made
        # before the model could be chosen is still found exactly as it was.
        place = f"{lat:.3f}_{lon:.3f}_{start}_{size_km}km"
        tag = "" if self.model_name == default_model() else f"_{self.model_name}"
        self.key = place + tag
        self.radar = CACHE / place
        self.folder = CACHE / self.key
        self.months = month_list(start)
        self.stage, self.message, self.error = "queued", "Waiting to start", None
        self.months_done, self.passes = [], {}
        self.started = time.time()
        self.result = None

    def state(self):
        return {"key": self.key, "model": self.model_name,
                "stage": self.stage, "message": self.message,
                "error": self.error, "months": self.months, "months_done": self.months_done,
                "passes": self.passes, "elapsed": round(time.time() - self.started, 1),
                "done": self.stage in ("done", "failed")}

    def say(self, stage, message):
        self.stage, self.message = stage, message

    # ---------------------------------------------------------------- run
    def run(self):
        try:
            cached = self.folder / "result.json"
            if cached.exists():
                result = json.loads(cached.read_text())
                # The model's name is checked as well as the picture version: a
                # cache written by a different model describes a different map.
                if (result.get("pictures") == PICTURES
                        and (result.get("model") or {}).get("name") == self.model_name):
                    self.result = result
                    self.months_done = list(self.months)
                    self.say("done", "Loaded from the cache")
                    return
                # pictures drawn by an older version: redraw them from the radar
                # already on disk - _fetch skips every month it finds there

            self.radar.mkdir(parents=True, exist_ok=True)
            self.folder.mkdir(parents=True, exist_ok=True)
            self._fetch()
            self._map()
            self.say("done", "Done")
        except Exception as err:                       # shown on the page, not swallowed
            self.error = f"{type(err).__name__}: {err}"
            self.say("failed", "Stopped")

    def _fetch(self):
        bbox = box_around(self.lat, self.lon, self.size_km)
        width, height = fetch_s1.pixel_grid(bbox, config.TARGET_RES_M)
        self.bbox, self.width, self.height = bbox, width, height

        self.say("searching", "Asking the Sentinel-1 catalogue what flew over this box")
        scenes = search_range(bbox, self.months[0], self.months[-1])
        if not scenes:
            raise RuntimeError("No Sentinel-1 images over this point in these 12 months.")
        try:
            (direction, track), track_scenes, _, cover = fetch_s1.choose_track(
                scenes, config.ORBIT, bbox, width, height)
        except SystemExit as stop:                     # fetch_s1 exits on "no scenes"
            raise RuntimeError(str(stop))
        self.track = {"direction": direction, "relative_orbit": track,
                      "worst_month_cover": round(100 * cover[(direction, track)], 1)}

        by_month = {m: [] for m in self.months}
        for s in track_scenes:
            by_month[s["properties"]["datetime"][:7]].append(s)
        self.passes = {m: len(v) for m, v in by_month.items()}
        empty = [m for m, v in by_month.items() if not v]
        if empty:
            raise RuntimeError(f"Track {track} has no pass in {', '.join(empty)}; "
                               f"try another start month.")

        todo = [m for m in self.months if not (self.radar / f"s1_{m}.tif").exists()]
        self.months_done = [m for m in self.months if m not in todo]
        total = sum(self.passes[m] for m in todo)
        got = {m: [] for m in todo}
        self.say("downloading", f"Downloading {total} radar passes on track {track}")
        with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
            jobs = {pool.submit(fetch_s1.download_window, s["id"], bbox, width, height): m
                    for m in todo for s in by_month[m]}
            left = {m: self.passes[m] for m in todo}
            for future in as_completed(jobs):
                month = jobs[future]
                left[month] -= 1
                try:
                    got[month].append(future.result())
                except Exception:
                    pass                               # a failed pass is a pass less, as in fetch_s1
                if left[month] == 0:
                    self._compose(month, got.pop(month), (direction, track))
                    self.months_done.append(month)
                    self.months_done.sort()
                    done = len(self.months_done)
                    self.say("downloading", f"{done} of 12 months composited")

    def _compose(self, month, passes, track):
        """The monthly median - fetch_s1.monthly_composite, from passes already in memory."""
        if not passes:
            raise RuntimeError(f"Every pass of {month} failed to download.")
        with np.errstate(all="ignore"):
            vv = np.nanmedian(np.stack([p[0] for p in passes]), axis=0)
            vh = np.nanmedian(np.stack([p[1] for p in passes]), axis=0)
        observations = np.sum(np.stack([p[2] for p in passes]), axis=0).astype("float32")
        fetch_s1.write_geotiff(self.radar / f"s1_{month}.tif", vv, vh, observations,
                               passes[0][3], track)

    def _map(self):
        info = model(self.model_name)
        class_ids = list(range(1, info["classes"]))
        self.say("mapping", f"Running {info['name']} on 12 months of radar")

        # Read all twelve first. A model that reads the year's statistics needs
        # every month before it can map any single one.
        months_x, months_seen = [], []
        for month in self.months:
            x, seen = radar_channels(self.radar / f"s1_{month}.tif")
            months_x.append(x)
            months_seen.append(seen)
        year_mean, year_std = (year_statistics(months_x, months_seen)
                               if info["in_channels"] > 3 else (None, None))

        seen_all, classes, probs_sum, conf = None, [], None, []
        radar_stats = []
        for i, month in enumerate(self.months):
            x, seen = months_x[i], months_seen[i]
            p = probabilities(month_input(months_x, year_mean, year_std, i,
                                          info["in_channels"]), info)
            c = (p.argmax(0) + 1).astype(np.uint8)
            classes.append(c)
            probs_sum = p if probs_sum is None else probs_sum + p
            conf.append(float(p.max(0)[seen].mean()) if seen.any() else 0.0)
            seen_all = seen if seen_all is None else seen_all & seen
            save_radar_png(x, seen, self.folder / f"radar_{month}.png")
            save_classes_png(c, seen, self.folder / f"classes_{month}.png")
            radar_stats.append([float(np.median(x[0][seen])) if seen.any() else 0.0,
                                float(np.median(x[1][seen])) if seen.any() else 0.0])
            self.message = f"Mapped {i + 1} of 12 months"

        year = (probs_sum.argmax(0) + 1).astype(np.uint8)    # kept as numbers only

        counted = np.stack(classes)[:, seen_all]
        n = max(counted.shape[1], 1)
        shares = [[round(100 * float((row == c).sum()) / n, 2) for c in class_ids]
                  for row in counted]
        year_share = [round(100 * float((year[seen_all] == c).sum()) / n, 2) for c in class_ids]
        changes = [[round(b - a, 2) for a, b in zip(shares[i - 1], shares[i])]
                   for i in range(1, 12)]
        total = [round(b - a, 2) for a, b in zip(shares[0], shares[-1])]

        H, W = seen_all.shape
        self.result = {
            "key": self.key,
            "point": {"lat": self.lat, "lon": self.lon},
            "size_km": self.size_km,
            "bbox": list(self.bbox),
            "pixels": [W, H],
            "months": self.months,
            "passes": self.passes,
            "track": self.track,
            "measured_pct": round(100 * float(seen_all.mean()), 1),
            "familiarity": familiarity(self.bbox, self.months),
            "model": card(info),
            "classes": [{"id": c, "name": config.CLASSES[c], "colour": COLOURS[c]}
                        for c in class_ids],
            "shares": shares,                 # [month][class] in %, same pixels every month
            "year_share": year_share,         # one map from the 12 months' averaged probabilities
            "change": changes,                # month-to-month, percentage points
            "total_change": total,            # last month minus first, percentage points
            "confidence": [round(100 * c, 1) for c in conf],
            "radar_median": radar_stats,      # VV, VH channel medians, 0-1 scale
            "elapsed_s": round(time.time() - self.started, 1),
            "pictures": PICTURES,
        }
        (self.folder / "result.json").write_text(json.dumps(self.result, indent=1))
