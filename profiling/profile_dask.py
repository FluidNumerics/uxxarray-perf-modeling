"""Profile what dask's scheduling machinery consumes during scattered sampling.

The workload is the stage-03 finding boiled down: a field is `.persist()`ed into
RAM (so there is *zero* disk I/O), and we then issue the same vectorised
`isel(...).compute()` Parcels performs, in a loop. Any time spent is therefore
pure dask client-side overhead -- graph build, optimization, and scheduling --
not array math and not disk.

Two profilers:

  --tool cprofile   (default, stdlib): deterministic. Dumps a .prof file (open with
                    `snakeviz` / `gprof2dot`) and prints the top functions by self
                    time and by cumulative time, so you can read off exactly which
                    dask functions dominate.

  --tool viztracer  timeline TRACE. Records every function call/return on a
                    timeline and writes a Chrome-trace JSON you scrub in Perfetto:
                    you can see one isel().compute() expand into graph-build ->
                    optimize -> schedule, call by call. Needs `viztracer`
                    (`uv pip install --python <env-python> viztracer`). Keep
                    --ncalls small (the trace records every call).

  --tool none       Just run the loop (use when launching under py-spy, which
                    samples the process externally).

Usage:
    python profile_dask.py --ncalls 200 --tool cprofile
    python profile_dask.py --ncalls 20  --tool viztracer
    # then:  vizviewer profiling/results/dask_trace.json      (opens Perfetto UI)
    #   or:  snakeviz   profiling/results/dask_sampling.prof
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def make_workload(npart, ntime, n, seed=0):
    """An in-RAM (persisted) dask DataArray plus a scattered-sampling closure."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((ntime, n, n)).astype("float64")
    xda = xr.DataArray(
        da.from_array(arr, chunks=(1, n, n)),
        dims=("time", "lat", "lon"),
    ).persist()  # resident in memory -> any cost below is scheduling, not I/O

    ti = rng.integers(0, ntime - 1, npart)
    yi = rng.integers(0, n, npart)
    xi = rng.integers(0, n, npart)
    sel = dict(
        time=xr.DataArray(ti, dims="p"),
        lat=xr.DataArray(yi, dims="p"),
        lon=xr.DataArray(xi, dims="p"),
    )

    def sample_once():
        # exactly the Parcels gather: vectorised isel + compute
        return xda.isel(**sel).compute().values

    sample_once()  # warm up
    return sample_once


def run_loop(sample_once, ncalls):
    for _ in range(ncalls):
        sample_once()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npart", type=int, default=2000)
    p.add_argument("--ncalls", type=int, default=200)
    p.add_argument("--ntime", type=int, default=60)
    p.add_argument("--n", type=int, default=1000, help="lat == lon grid edge")
    p.add_argument("--tool", choices=["cprofile", "viztracer", "none"], default="cprofile")
    p.add_argument("--restrict", default="dask|xarray|toolz|tlz|threading|slicing",
                   help="regex to filter the printed function list")
    args = p.parse_args()

    sample_once = make_workload(args.npart, args.ntime, args.n)

    if args.tool == "none":
        run_loop(sample_once, args.ncalls)
        print(f"ran {args.ncalls} scattered isel().compute() calls (no profiler)")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.tool == "viztracer":
        from viztracer import VizTracer  # noqa: PLC0415

        out = RESULTS / "dask_trace.json"
        if args.ncalls > 50:
            print(f"note: tracing {args.ncalls} calls -- the trace records every "
                  f"function call; consider --ncalls 20 for a navigable timeline.")
        # trace only the sampling loop (setup/persist/warmup already happened)
        tracer = VizTracer(output_file=str(out), ignore_frozen=True, min_duration=1)
        tracer.start()
        run_loop(sample_once, args.ncalls)
        tracer.stop()
        tracer.save()
        print(f"\ntraced {args.ncalls} scattered isel().compute() calls -> {out}")
        print("view the timeline:  vizviewer profiling/results/dask_trace.json")
        print("  (opens a Perfetto UI; zoom into one compute() to see "
              "graph-build -> optimize -> schedule)")
        return
    prof_path = RESULTS / "dask_sampling.prof"

    pr = cProfile.Profile()
    pr.enable()
    run_loop(sample_once, args.ncalls)
    pr.disable()
    pr.dump_stats(prof_path)

    total = pstats.Stats(pr).total_tt
    print(f"\n{args.ncalls} scattered isel().compute() calls on an in-RAM field "
          f"({args.ntime},{args.n},{args.n})")
    print(f"total profiled time: {total:.2f} s  "
          f"({1000*total/args.ncalls:.1f} ms / call)\n")

    for sort_key, title in [("tottime", "TOP BY SELF TIME (where the CPU actually is)"),
                            ("cumulative", "TOP BY CUMULATIVE TIME (call hierarchy)")]:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_key)
        ps.print_stats(args.restrict, 18)
        print(f"=== {title} -- filtered to /{args.restrict}/ ===")
        # keep only the data rows (drop pstats preamble noise)
        for line in s.getvalue().splitlines():
            if line.strip() and ("function)" in line or line.lstrip()[:1].isdigit()):
                print(line)
        print()

    print(f"saved {prof_path}")
    print("visualize:  snakeviz profiling/results/dask_sampling.prof")
    print("       or:  gprof2dot -f pstats profiling/results/dask_sampling.prof | dot -Tsvg -o callgraph.svg")


if __name__ == "__main__":
    main()
