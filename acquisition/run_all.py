"""
Collect everything, in one command.

For each region, in order:
    1. download the Sentinel-1 radar  (the model's input)
    2. download the Sentinel-2 colour (only used to make labels)
    3. write PNGs you can actually look at

It works region by region rather than step by step, for two reasons. The colour
imagery has to be downloaded onto the grid the radar defines, so the radar must
come first for that region. And if one region fails, the rest still finish -
you get usable data for 18 regions instead of nothing.

Anything already downloaded is skipped, so stopping and restarting is safe.

    .venv\\Scripts\\python.exe acquisition\\run_all.py                  # everything, ~600 MB
    .venv\\Scripts\\python.exe acquisition\\run_all.py --dry-run        # just check what exists
    .venv\\Scripts\\python.exe acquisition\\run_all.py --aoi mannheim   # one region
    .venv\\Scripts\\python.exe acquisition\\run_all.py --skip-png       # no pictures
"""

import argparse
import subprocess
import sys
import time

import rasterio

from pathlib import Path

# Every script runs from its own folder, where Python cannot see config.py
# or the other packages. Putting the project root first fixes that the same
# way for every script, however it is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# Our prints are buffered while subprocess output is not, so without this the
# headers appear after the output they introduce.
print = __import__('functools').partial(print, flush=True)

PYTHON = sys.executable


def run(script, arguments):
    """Run one script and let its output stream to the screen as it goes."""
    command = [PYTHON, str(config.ROOT / script)] + arguments
    result = subprocess.run(command, cwd=config.ROOT)
    return result.returncode == 0


def human_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def months_present(folder, name, year):
    """Which months actually ended up on disk for this region."""
    if not folder.exists():
        return set()
    return {config.month_of(p) for p in folder.glob(f"{name}_{year}-*.tif")}


def radar_coverage(folder, name, year):
    """The worst month's share of the box that the radar actually saw.

    Read from band 3 (passes per pixel) of the files on disk. A file can exist
    and cover almost nothing: before 5 Nov paris had all 12 months on disk at
    2% coverage each, and this table still said 12/12.
    """
    worst = None
    for path in folder.glob(f"{name}_{year}-*.tif"):
        with rasterio.open(path) as src:
            share = float((src.read(3) > 0).mean())
        worst = share if worst is None else min(worst, share)
    return worst


def completeness_report(areas, year):
    """Say exactly what is missing, per region - months AND ground.

    Downloads fail occasionally. A failed month prints one line and scrolls away,
    and the region still finishes - so without this check a gap is invisible
    until something much later behaves oddly. Counting files is not enough
    either: a month can be on disk and cover a sliver of the box, so the
    radar's real coverage is read from the files too.
    """
    expected = {f"{year}-{m:02d}" for m in range(1, 13)}
    gaps, thin = [], []
    print()
    print(f"{'region':14}{'radar':>8}{'colour':>9}{'covered':>10}   problems")
    print("-" * 74)
    for name in areas:
        radar = months_present(config.S1_MONTHLY / name, name, year)
        colour = months_present(config.S2_MONTHLY / name, name, year)
        worst = radar_coverage(config.S1_MONTHLY / name, name, year)
        missing = sorted(expected - radar)
        notes = []
        if missing:
            gaps.append((name, missing))
            notes.append("missing months " + ", ".join(m[-2:] for m in missing))
        if worst is not None and worst < config.MIN_TRACK_COVER:
            thin.append((name, worst))
            notes.append("RADAR COVERS TOO LITTLE OF THE BOX")
        covered = "-" if worst is None else f"{100 * worst:.1f}%"
        print(f"{name:14}{len(radar):>5}/12{len(colour):>6}/12{covered:>10}   "
              f"{'; '.join(notes)}")

    if gaps:
        print()
        print("  Radar gaps found. Colour gaps are usually real cloud, but radar")
        print("  should be complete - these are most likely failed downloads.")
        print("  Re-run to fill them in (finished months are skipped):")
        for name, missing in gaps:
            print(f"     run_all.py --aoi {name}")
    if thin:
        print()
        print(f"  Radar covering under {100 * config.MIN_TRACK_COVER:.0f}% of the box "
              "(worst month shown above).")
        print("  Usually a track that only clips the box. If its months came from an")
        print("  older run, delete them and re-run - fetch_s1 now picks by coverage:")
        for name, _ in thin:
            print(f"     delete data\\s1_monthly\\{name}  then  run_all.py --aoi {name}")
    if not gaps and not thin:
        print("\n  All regions complete, and every month covers the box.")
    return gaps + [(name, ["coverage"]) for name, _ in thin]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default="all",
                        choices=["all"] + list(config.AOIS))
    parser.add_argument("--year", type=int, default=config.YEAR)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is available, download nothing")
    parser.add_argument("--skip-png", action="store_true",
                        help="skip making the picture files")
    parser.add_argument("--skip-optical", action="store_true",
                        help="radar only, no colour imagery")
    args = parser.parse_args()

    areas = list(config.AOIS) if args.aoi == "all" else [args.aoi]

    print("=" * 74)
    print(f"ECHOSCAPE data collection")
    print(f"  regions     {len(areas)}")
    print(f"  year        {args.year}")
    print(f"  resolution  {config.TARGET_RES_M} m")
    print(f"  area size   {config.AREA_SIZE_KM} x {config.AREA_SIZE_KM} km each")
    if not args.dry_run:
        print(f"  expect      roughly {len(areas) * 32} MB and "
              f"{len(areas) * 4} minutes")
    print("=" * 74)

    started = time.time()
    done, failed = [], []

    for position, name in enumerate(areas, start=1):
        info = config.AOIS[name]
        print()
        print("=" * 74)
        print(f"[{position}/{len(areas)}]  {name.upper()}  "
              f"[{info['class']}]  {info['note']}")
        print("=" * 74)

        common = ["--aoi", name, "--year", str(args.year)]
        if args.dry_run:
            common.append("--dry-run")

        ok = run("acquisition/fetch_s1.py", common)
        if not ok:
            print(f"\n  !! radar failed for {name}, moving on")
            failed.append(name)
            continue

        if not args.skip_optical:
            if not run("acquisition/fetch_s2.py", common):
                print(f"\n  !! colour failed for {name} - radar is still fine")

        if not (args.dry_run or args.skip_png):
            run("visualisation/to_png.py", ["--aoi", name])

        done.append(name)

    elapsed = time.time() - started
    print()
    print("=" * 74)
    print(f"FINISHED in {human_time(elapsed)}")
    print(f"  completed  {len(done)}/{len(areas)}")
    if failed:
        print(f"  FAILED     {', '.join(failed)}")
        print(f"  re-run just those with:  --aoi <name>")
    print("=" * 74)

    if not args.dry_run:
        completeness_report(areas, args.year)
        print()
        print("  next:")
        print(f"     python acquisition\\check_alignment.py --aoi "
              f"{done[0] if done else config.DEFAULT_AOI}")
        print(f"     look at  data\\png\\<region>\\")


if __name__ == "__main__":
    main()
