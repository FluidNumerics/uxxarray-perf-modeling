"""Prototype: fewer .compute() calls in Parcels' velocity sampling.

Today Parcels samples each velocity component separately -- `XLinear_Velocity`
calls `XLinear` for U, then V, then W, and each `XLinear` ends with its own
`value.compute()` (src/parcels/interpolators/_xinterpolators.py). So a 3-D RK2
step issues ~6 `.compute()` calls (3 components x 2 RK stages).

The components share the same particle positions and the same instant, so there's
no reason to materialize them in separate graphs. This probe reproduces the A-grid
corner gather for N components and times two strategies on an in-RAM
(`.persist()`ed) field, so the difference is purely the number of graphs built
and scheduled -- no disk, no array math of consequence:

  separate : [gather(c).compute() for c in components]      <- current behavior
  batched  : dask.compute(*[gather(c) for c in components]) <- proposed

Usage:
    python batched_compute.py --ncomp 3 --npart 2000 --iters 100
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings

warnings.filterwarnings("ignore")

import dask  # noqa: E402
import dask.array as da  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ncomp", type=int, default=3, help="velocity components (U,V[,W])")
    p.add_argument("--npart", type=int, default=2000)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--ntime", type=int, default=12)
    p.add_argument("--n", type=int, default=600, help="lat == lon edge")
    p.add_argument("--scheduler", default="threads",
                   help='dask scheduler ("threads" or "synchronous")')
    args = p.parse_args()

    rng = np.random.default_rng(0)
    ntime, n = args.ntime, args.n
    # one persisted (in-RAM) dask field per component, slab-chunked like real output
    comps = ["U", "V", "W"][: args.ncomp]
    fields = {
        c: xr.DataArray(
            da.from_array(rng.standard_normal((ntime, 1, n, n)).astype("float64"),
                          chunks=(1, 1, n, n)),
            dims=("time", "depth", "lat", "lon"),
        ).persist()
        for c in comps
    }

    # scattered particle cells, and the 2(time) x 2(lat) x 2(lon) corner stencil
    ti = rng.integers(0, ntime - 1, args.npart)
    yi = rng.integers(0, n - 1, args.npart)
    xi = rng.integers(0, n - 1, args.npart)
    T = np.stack([ti, ti + 1]); Y = np.stack([yi, yi + 1]); X = np.stack([xi, xi + 1])
    combos = list(itertools.product([0, 1], [0, 1], [0, 1]))  # 8 corners
    tt = np.concatenate([T[a] for a, _, _ in combos])
    yy = np.concatenate([Y[b] for _, b, _ in combos])
    xx = np.concatenate([X[c] for _, _, c in combos])
    zz = np.zeros_like(tt)
    sel = dict(
        time=xr.DataArray(tt, dims="points"), depth=xr.DataArray(zz, dims="points"),
        lat=xr.DataArray(yy, dims="points"), lon=xr.DataArray(xx, dims="points"),
    )

    def gather(c):                       # lazy corner gather for one component
        return fields[c].isel(**sel).data

    ntasks_each = len(gather("U").__dask_graph__())

    def separate():                      # current Parcels: one compute per component
        return [gather(c).compute(scheduler=args.scheduler) for c in comps]

    def batched():                       # proposed: one combined graph
        return dask.compute(*[gather(c) for c in comps], scheduler=args.scheduler)

    def timeit(fn):
        fn()
        t = time.perf_counter()
        for _ in range(args.iters):
            fn()
        return (time.perf_counter() - t) / args.iters * 1e3  # ms/call

    t_sep = timeit(separate)
    t_bat = timeit(batched)

    print(f"{args.ncomp}-component velocity gather, {args.npart} particles, "
          f"in-RAM field ({ntime},1,{n},{n}), scheduler={args.scheduler}")
    print(f"(each component gather is a {ntasks_each}-task graph)\n")
    print(f"  separate  ({args.ncomp} x .compute())   : {t_sep:8.2f} ms   "
          f"[{args.ncomp} graphs built/scheduled]")
    print(f"  batched   (1 x dask.compute())  : {t_bat:8.2f} ms   [1 graph]")
    print(f"  -> {t_sep / t_bat:.2f}x faster per velocity evaluation\n")

    for scheme, stages in [("AdvectionRK2", 2), ("AdvectionRK4", 4)]:
        sep_calls, bat_calls = args.ncomp * stages, stages
        print(f"  {scheme}: {sep_calls} compute() -> {bat_calls} per step "
              f"({sep_calls/bat_calls:.0f}x fewer); "
              f"per-step sample time {t_sep*stages:.1f} -> {t_bat*stages:.1f} ms")

    print("\nfewer calls, but still 1 graph per RK stage (stages are sequential -- "
          "the midpoint depends on the previous velocity). Removing the last graphs "
          "needs the in-RAM window (stage 05).")


if __name__ == "__main__":
    main()
