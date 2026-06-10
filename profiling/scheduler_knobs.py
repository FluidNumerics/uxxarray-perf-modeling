"""How far can dask-ONLY knobs reduce the scheduling overhead?

Same in-RAM (`.persist()`ed) scattered gather as dask_overhead.py / profile_dask.py,
timed under a series of dask-internal configurations. Nothing here leaves dask or
touches disk -- it isolates what the scheduler/graph machinery costs and which
knobs move it.

Knobs measured (see the repo discussion / 03_parcels/README.md):
  * scheduler="synchronous"   -- drop the thread pool (and its lock/condition
                                 storm, which dominated the cProfile self-time);
                                 a win for tiny graphs of cheap tasks.
  * optimize_graph=False      -- skip the optimizer pass on trivial graphs.
  * fewer/larger chunks       -- overhead is ~per-task, so fewer tasks = less
                                 build/tokenize/dispatch.
  * batching                  -- many gathers in one dask.compute() amortizes the
                                 fixed per-compute costs.

Usage:
    python scheduler_knobs.py --npart 2000 --iters 100
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import dask  # noqa: E402
import dask.array as da  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npart", type=int, default=2000)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--ntime", type=int, default=48)
    p.add_argument("--n", type=int, default=600, help="lat == lon edge")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    base = rng.standard_normal((args.ntime, args.n, args.n)).astype("float64")

    def mksel():
        return dict(
            time=xr.DataArray(rng.integers(0, args.ntime - 1, args.npart), dims="p"),
            y=xr.DataArray(rng.integers(0, args.n, args.npart), dims="p"),
            x=xr.DataArray(rng.integers(0, args.n, args.npart), dims="p"),
        )

    sel = mksel()

    def timeit(fn, it=None):
        it = it or args.iters
        fn()  # warm up
        t = time.perf_counter()
        for _ in range(it):
            fn()
        return (time.perf_counter() - t) / it * 1e3  # ms/call

    def persist(chunks):
        return xr.DataArray(da.from_array(base, chunks=chunks),
                            dims=("time", "y", "x")).persist()

    xnp = xr.DataArray(base, dims=("time", "y", "x"))
    t_np = timeit(lambda: xnp.isel(**sel).values)

    xda = persist((1, args.n, args.n))             # ntime chunks (like the demo)
    xda1 = persist((args.ntime, args.n, args.n))   # one chunk -> minimal tasks
    ntasks = len(xda.isel(**sel).data.__dask_graph__())
    ntasks1 = len(xda1.isel(**sel).data.__dask_graph__())

    rows = [
        ("threads (default)",
         lambda: xda.isel(**sel).compute()),
        ('scheduler="synchronous"',
         lambda: xda.isel(**sel).compute(scheduler="synchronous")),
        ("threads, optimize_graph=False",
         lambda: xda.isel(**sel).compute(optimize_graph=False)),
        ("synchronous + no-optimize",
         lambda: xda.isel(**sel).compute(scheduler="synchronous", optimize_graph=False)),
        (f"synchronous + single-chunk ({ntasks1} tasks)",
         lambda: xda1.isel(**sel).compute(scheduler="synchronous")),
    ]

    results = [(name, timeit(fn)) for name, fn in rows]

    # batching: 10 gathers in ONE compute, report per-gather cost
    sels = [mksel() for _ in range(10)]
    batch_ms = timeit(
        lambda: dask.compute(*[xda.isel(**s) for s in sels], scheduler="synchronous"),
        it=max(args.iters // 5, 10),
    )
    results.append(("synchronous, batched x10 (per gather)", batch_ms / 10))

    print(f"scattered gather of {args.npart} points from an in-RAM "
          f"({args.ntime},{args.n},{args.n}) field (no disk I/O)")
    print(f"default graph: {ntasks} tasks (chunks (1,n,n)); single-chunk: {ntasks1} tasks\n")
    print(f"  {'configuration':40s}{'ms/call':>10}{'x numpy':>10}{'x default':>11}")
    print("  " + "-" * 70)
    print(f"  {'numpy baseline':40s}{t_np:10.3f}{1:>9.0f}x{'':>11}")
    default = results[0][1]
    for name, ms in results:
        print(f"  {name:40s}{ms:10.3f}{ms/t_np:>9.0f}x{default/ms:>10.1f}x")

    print("\nfloor: every .compute() still rebuilds + tokenizes a graph, so even fully")
    print("tuned this stays ~50x numpy. Closing the gap to 1x needs the windowed")
    print("NumPy approach (stage 05) -- stop calling compute per gather.")


if __name__ == "__main__":
    main()
