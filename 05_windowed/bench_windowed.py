"""Benchmark: rolling NumPy time-window vs naive lazy-dask sampling.

Models a Parcels run over a long span (longer than fits in RAM all at once) on the
20 GB Atlantic dataset (stage 04). The field is hourly; the integration timestep
is sub-hourly, so each pair of bracketing time levels serves several steps -- the
realistic case.

Two strategies, identical scattered positions and identical time interpolation:

  naive   : per step, sample the dask array at the bracketing levels via
            isel().compute(). Re-reads the same slabs every sub-step and pays the
            per-compute scheduling overhead each time.
  window  : WindowSampler -- read each time level once (bulk sequential) into
            NumPy, sample NumPy in the hot loop, evict behind the clock, optional
            --prefetch of the next level on a background thread.

Usage:
    python bench_windowed.py --dir ../04_atlantic/data/atlantic \
        --days 7 --dt-min 10 --npart 500 --mode both --prefetch --check
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

import dask  # noqa: E402

from windowed_sampler import WindowSampler  # noqa: E402


def build_clock(times, days, dt_min):
    t0 = times[0]
    nsteps = int(days * 24 * 60 / dt_min)
    step_dt = np.timedelta64(int(dt_min * 60), "s")
    clock = t0 + np.arange(nsteps) * step_dt
    # don't run past the available data
    clock = clock[clock <= times[-1]]
    return clock


def bracket(times, t):
    i = int(np.searchsorted(times, np.datetime64(t), side="right")) - 1
    return min(max(i, 0), len(times) - 2)


def naive_step(U, V, times, t, yi, xi, depth):
    """One step of the naive path: dask isel().compute() at both bracketing levels."""
    i = bracket(times, t)
    tau = (np.datetime64(t) - times[i]) / (times[i + 1] - times[i])
    n = len(yi)
    tt = np.concatenate([np.full(n, i), np.full(n, i + 1)])
    yy = np.concatenate([yi, yi])
    xx = np.concatenate([xi, xi])
    selU = U.isel(time=xr.DataArray(tt, dims="p"), depth=depth,
                  latitude=xr.DataArray(yy, dims="p"), longitude=xr.DataArray(xx, dims="p"))
    selV = V.isel(time=xr.DataArray(tt, dims="p"), depth=depth,
                  latitude=xr.DataArray(yy, dims="p"), longitude=xr.DataArray(xx, dims="p"))
    uu, vv = dask.compute(selU, selV)  # one combined graph
    uu, vv = uu.values, vv.values
    u = (1 - tau) * uu[:n] + tau * uu[n:]
    v = (1 - tau) * vv[:n] + tau * vv[n:]
    return u, v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("../04_atlantic/data/atlantic"))
    p.add_argument("--days", type=float, default=7.0)
    p.add_argument("--dt-min", type=float, default=10.0)
    p.add_argument("--npart", type=int, default=500)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--mode", choices=["naive", "window", "both"], default="both")
    p.add_argument("--check", action="store_true", help="verify both agree on first steps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    files = sorted(str(f) for f in args.dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"no NetCDF files in {args.dir} -- generate stage 04 first")
    ds = xr.open_mfdataset(files, chunks={"time": 1}, combine="by_coords")
    U, V, times = ds["uo"], ds["vo"], ds["time"].values

    clock = build_clock(times, args.days, args.dt_min)
    nsteps = len(clock)
    nlat, nlon = U.sizes["latitude"], U.sizes["longitude"]
    slab_per_var = nlat * nlon * U.dtype.itemsize
    levels_spanned = bracket(times, clock[-1]) - bracket(times, clock[0]) + 2

    rng = np.random.default_rng(args.seed)
    pos = [(rng.integers(0, nlat, args.npart), rng.integers(0, nlon, args.npart))
           for _ in range(nsteps)]

    print(f"=== windowed vs naive sampling on {len(files)} files "
          f"({sum(Path(f).stat().st_size for f in files)/1024**3:.0f} GiB) ===")
    print(f"grid {nlat}x{nlon}, slab {2*slab_per_var/1e6:.1f} MB/level (U+V)")
    print(f"run {args.days:g} d @ dt={args.dt_min:g} min -> {nsteps} steps, "
          f"spanning {levels_spanned} hourly levels, {args.npart} particles "
          f"(prefetch={args.prefetch})\n")

    if args.check:
        ws = WindowSampler({"uo": U, "vo": V}, times, depth=args.depth)
        worst = 0.0
        for k in range(min(20, nsteps)):
            yi, xi = pos[k]
            un, vn = naive_step(U, V, times, clock[k], yi, xi, args.depth)
            s = ws.sample(clock[k], yi, xi)
            worst = max(worst, float(np.abs(un - s["uo"]).max()),
                        float(np.abs(vn - s["vo"]).max()))
        ws.close()
        print(f"correctness check: max |naive - window| over 20 steps = {worst:.2e}  "
              f"({'OK' if worst < 1e-5 else 'MISMATCH'})\n")

    res = {}

    if args.mode in ("naive", "both"):
        start = time.perf_counter()
        for k in range(nsteps):
            yi, xi = pos[k]
            naive_step(U, V, times, clock[k], yi, xi, args.depth)
        secs = time.perf_counter() - start
        gb = nsteps * 2 * 2 * slab_per_var / 1024**3  # 2 levels x 2 vars per step
        res["naive"] = secs
        print(f"  naive  (dask isel().compute()/step) : {secs:9.2f} s  "
              f"({nsteps/secs:7.1f} steps/s, ~{gb:5.1f} GiB read, "
              f"{2*nsteps} compute graphs)")

    if args.mode in ("window", "both"):
        ws = WindowSampler({"uo": U, "vo": V}, times, prefetch=args.prefetch, depth=args.depth)
        start = time.perf_counter()
        for k in range(nsteps):
            yi, xi = pos[k]
            ws.sample(clock[k], yi, xi)
        secs = time.perf_counter() - start
        ws.close()
        res["window"] = secs
        print(f"  window (NumPy + rolling reload)     : {secs:9.2f} s  "
              f"({nsteps/secs:7.1f} steps/s, {ws.bytes_read/1024**3:5.1f} GiB read, "
              f"{ws.levels_loaded} level loads)")

    if "naive" in res and "window" in res:
        print(f"\n  window is {res['naive']/res['window']:.0f}x faster, and reads "
              f"{(nsteps*2*2*slab_per_var)/(ws.levels_loaded*2*slab_per_var):.0f}x "
              f"less data off disk.")


if __name__ == "__main__":
    main()
