"""
ECHOSCAPE prototype - the web server.

Run from the project root:

    .venv\\Scripts\\python.exe app\\server.py

then open http://127.0.0.1:8010 in a browser.

Needs the internet (Sentinel-1 comes live from Microsoft Planetary Computer;
the base map and fonts come from public CDNs). A first run over a new point
takes a few minutes - mostly downloading radar passes. The result is cached
in data/app_cache/, so the same point and month opens instantly afterwards.
"""

import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import pipeline

PORT = 8010
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="ECHOSCAPE")
jobs = {}                       # key -> pipeline.Job
one_at_a_time = threading.Lock()   # one GPU, one download pool: runs queue up


class Request(BaseModel):
    lat: float = Field(ge=-80, le=80)       # Sentinel-1 RTC stops short of the poles
    lon: float = Field(ge=-180, le=180)
    start: str
    size_km: int = Field(default=20, ge=10, le=40)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    m = pipeline.model()
    return {
        "model": {k: m[k] for k in ("name", "architecture", "parameters", "val_miou",
                                    "test_miou", "epoch", "noise_floor")},
        "start_months": pipeline.start_months(),
        "default_start": f"{pipeline.config.YEAR}-01",
        "classes": [{"id": c, "name": pipeline.config.CLASSES[c], "colour": pipeline.COLOURS[c]}
                    for c in pipeline.CLASS_IDS],
        "regions": [{"name": n, "bbox": pipeline.config.bbox_for(n),
                     "split": ("test" if n in pipeline.config.TEST_AOIS
                               else "validation" if n in pipeline.config.VAL_AOIS else "training")}
                    for n in pipeline.config.AOIS],
    }


def _work(job):
    with one_at_a_time:
        job.run()


@app.post("/api/run")
def run(req: Request):
    if req.start not in pipeline.start_months():
        raise HTTPException(400, f"Start month must be between {pipeline.start_months()[0]} "
                                 f"and {pipeline.latest_start()}, so all 12 months exist.")
    job = pipeline.Job(round(req.lat, 3), round(req.lon, 3), req.start, req.size_km)
    running = jobs.get(job.key)
    if running and not (running.stage == "failed"):
        return running.state()                      # same request twice: follow the first
    jobs[job.key] = job
    threading.Thread(target=_work, args=(job,), daemon=True).start()
    return job.state()


@app.get("/api/job/{key}")
def job_state(key: str):
    if key not in jobs:
        raise HTTPException(404, "No such run - start it again.")
    return jobs[key].state()


@app.get("/api/result/{key}")
def result(key: str):
    job = jobs.get(key)
    if job and job.result:
        return job.result
    cached = pipeline.CACHE / key / "result.json"
    if cached.exists():
        return FileResponse(cached)
    raise HTTPException(404, "Not finished yet.")


@app.get("/api/image/{key}/{name}")
def image(key: str, name: str):
    path = (pipeline.CACHE / key / name).resolve()
    if path.parent != (pipeline.CACHE / key).resolve() or path.suffix != ".png" or not path.exists():
        raise HTTPException(404, "No such image.")
    return FileResponse(path, headers={"Cache-Control": "max-age=86400"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    print("loading the model ...")
    m = pipeline.model()
    print(f"   {m['name']} ({m['architecture']}), validation mIoU {100 * m['val_miou']:.1f}%, "
          f"on {pipeline.DEVICE}")
    print(f"\nECHOSCAPE is at  http://127.0.0.1:{PORT}\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
