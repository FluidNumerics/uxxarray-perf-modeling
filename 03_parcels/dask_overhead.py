"""Isolate dask's *per-compute scheduling overhead* from disk I/O.

Stages 01-02 show that random sampling drags far more bytes off disk than it
needs. But stage 03 reveals a second, independent tax: even when the field is
already resident in memory (so there is no disk I/O at all), a dask-backed
Parcels run is dramatically slower than a numpy one and is pinned at ~100% CPU.

That cost is dask's task-graph machinery. Every ``field.data.isel(...).compute()``
rebuilds and schedules a fresh task graph -- and Parcels issues one such call per
velocity component, per Runge-Kutta stage, per timestep. None of that work touches
the disk; it is pure Python/dask overhead, paid again on every step.

This probe measures the per-call cost with the data *persisted in RAM* (via
``.persist()``), so disk is entirely out of the picture and only the scheduling
machinery remains.

Usage:
    python dask_overhead.py --npart 2000 --ncalls 50
"""

from __future__ import annotations

import argparse
import time

import dask.array as da
import numpy as np
import xarray as xr


def per_call(fn, n: int) -> float:
    fn()  # warm up (build any caches once)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npart", type=int, default=2000)
    p.add_argument("--ncalls", type=int, default=50)
    p.add_argument("--ntime", type=int, default=60)
    p.add_argument("--n", type=int, default=1000, help="lat == lon grid size")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    shape = (args.ntime, args.n, args.n)
    chunks = (1, args.n, args.n)  # one slab per time level, like flow.nc surface

    np_arr = rng.standard_normal(shape).astype("float64")
    xda_np = xr.DataArray(np_arr, dims=("time", "lat", "lon"))
    # dask-backed AND resident in RAM -> any cost is scheduling, not I/O
    xda_da = xr.DataArray(da.from_array(np_arr, chunks=chunks),
                          dims=("time", "lat", "lon")).persist()

    ti = rng.integers(0, args.ntime - 1, args.npart)
    yi = rng.integers(0, args.n, args.npart)
    xi = rng.integers(0, args.n, args.npart)
    sel = dict(
        time=xr.DataArray(ti, dims="p"),
        lat=xr.DataArray(yi, dims="p"),
        lon=xr.DataArray(xi, dims="p"),
    )

    ntasks = len(xda_da.isel(**sel).data.__dask_graph__())

    f_np = lambda: xda_np.isel(**sel).values            # noqa: E731
    f_da = lambda: xda_da.isel(**sel).compute().values  # noqa: E731

    t_np = per_call(f_np, max(args.ncalls, 200))
    t_da = per_call(f_da, args.ncalls)

    print(f"sampling {args.npart:,} scattered points from an in-RAM "
          f"{shape} field (no disk I/O)\n")
    print(f"  numpy  isel().values    : {t_np * 1e6:10.1f} us / call")
    print(f"  dask   isel().compute() : {t_da * 1e3:10.3f} ms / call  "
          f"({ntasks} tasks in the graph)")
    print(f"  dask overhead factor    : {t_da / t_np:10.0f} x\n")

    # Parcels issues ~ (2 RK stages) x (U, V) = 4 such calls per timestep.
    calls_per_step = 4
    for nsteps in (480,):
        overhead = (t_da - t_np) * calls_per_step * nsteps
        print(f"  projected pure-dask overhead for {nsteps} steps "
              f"x {calls_per_step} calls/step:")
        print(f"    = ({t_da*1e3:.2f} - {t_np*1e3:.3f}) ms x {calls_per_step*nsteps} "
              f"calls = {overhead:.1f} s of scheduling alone")


if __name__ == "__main__":
    main()
