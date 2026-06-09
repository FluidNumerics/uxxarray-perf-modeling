"""Sequential vs random sampling of a chunked xarray/dask dataset.

This is the bridge between the raw-storage fio numbers (stage 01) and Parcels
itself (stage 03). It reproduces, in isolation, the access pattern that Parcels'
field interpolation performs every kernel step:

    field.data.isel(<scattered per-particle indices>).compute()

(see ``src/parcels/interpolators/_xinterpolators.py`` -- ``_get_corner_data_Agrid``
builds vectorised per-particle index arrays and the interpolator ends with
``value.compute()``.)

The access pattern is modelled *faithfully*: all particles share the simulation's
advancing time level (Parcels integrates every particle with the same clock), and
interpolation needs two adjacent time levels. Only the horizontal positions are
scattered -- the particles have drifted apart. The crucial question is then: how
many on-disk chunks must be fetched to serve those scattered positions, and how
many bytes does that drag off disk per useful value?

Two regimes are compared, delivering the same values:

  1. RANDOM / dask  -- scattered ``isel(...).compute()`` once per timestep, straight
                       off the chunked file. This is what Parcels does today.
  2. LOAD-ONCE / RAM -- read the needed field into memory once (a big sequential
                       read), then index it in RAM. This is the obvious fix.

Run it against both a slab-chunked and a tile-chunked file to see why chunk shape
is the whole game (mirrors fio's random-1M vs random-4K split):

    python make_dataset.py --gb 10 --chunk slab  --out data/ocean_slab_10g.nc
    python make_dataset.py --gb 10 --chunk tiled --out data/ocean_tiled_10g.nc
    python bench_sampling.py --file data/ocean_slab_10g.nc
    python bench_sampling.py --file data/ocean_tiled_10g.nc
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr


def drop_caches() -> bool:
    """Best-effort flush of the OS page cache (needs passwordless sudo)."""
    try:
        subprocess.run(["sync"], check=True)
        subprocess.run(
            ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
            input=b"3", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=Path("data/ocean_slab_10g.nc"))
    p.add_argument("--npart", type=int, default=10000, help="number of particles")
    p.add_argument("--nsteps", type=int, default=50, help="number of timesteps")
    p.add_argument("--drop-caches", action="store_true",
                   help="flush OS page cache before each regime (needs sudo -n)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    ds = xr.open_dataset(args.file, chunks={})
    U = ds["U"]
    ntime, ndepth, nlat, nlon = U.shape
    cT, cZ, cY, cX = U.data.chunksize
    chunk_bytes = cT * cZ * cY * cX * U.dtype.itemsize
    itemsize = U.dtype.itemsize

    print(f"file        : {args.file}")
    print(f"shape       : {U.shape}  (time, depth, lat, lon)")
    print(f"on-disk chunk: {(cT, cZ, cY, cX)}  ({chunk_bytes / 1e6:.3f} MB each, "
          f"{int(np.ceil(nlat / cY) * np.ceil(nlon / cX))} tiles/slab)")
    print(f"workload    : {args.npart:,} particles, {args.nsteps} timesteps "
          f"(2 time levels / step for interpolation)\n")

    cache_note = ""
    if args.drop_caches:
        cache_note = " [cache flushed]" if drop_caches() else \
                     " [WARNING: cache NOT flushed -- wall-clock is cache-warm]"

    # ---- build the per-step scattered horizontal positions -------------------
    # All particles at the surface (depth 0). Each step they have drifted to new
    # random (lat, lon) cells. Time advances one level per step; interpolation
    # uses level ti and ti+1.
    steps = []
    for s in range(args.nsteps):
        ti = min(s, ntime - 2) if ntime >= 2 else 0
        yi = rng.integers(0, nlat, args.npart)
        xi = rng.integers(0, nlon, args.npart)
        steps.append((ti, yi, xi))

    # useful values actually consumed: 2 time levels x npart, per step
    useful_vals = 2 * args.npart * args.nsteps
    useful_bytes = useful_vals * itemsize

    # ---- chunk accounting (cache-independent): how many distinct on-disk chunks
    #      must be fetched, counting re-reads across independent timesteps --------
    chunks_touched = 0
    for ti, yi, xi in steps:
        cyi = yi // cY
        cxi = xi // cX
        tiles = np.unique(np.stack([cyi, cxi], axis=1), axis=0).shape[0]
        chunks_touched += 2 * tiles  # two time levels, each its own chunk
    eff_bytes = chunks_touched * chunk_bytes

    # =========================================================================
    print(f"=== 1. RANDOM pointwise sampling off dask/disk -- the Parcels pattern{cache_note} ===")
    if args.drop_caches:
        drop_caches()
    start = time.perf_counter()
    for ti, yi, xi in steps:
        t_idx = np.concatenate([np.full(args.npart, ti), np.full(args.npart, ti + 1)])
        y_idx = np.concatenate([yi, yi])
        x_idx = np.concatenate([xi, xi])
        sel = U.isel(
            time=xr.DataArray(t_idx, dims="p"),
            depth=xr.DataArray(np.zeros_like(t_idx), dims="p"),
            lat=xr.DataArray(y_idx, dims="p"),
            lon=xr.DataArray(x_idx, dims="p"),
        )
        _ = sel.compute().values  # triggers chunk loads, exactly like Parcels
    rnd = time.perf_counter() - start
    print(f"    wall time            : {rnd:10.3f} s")
    print(f"    useful values        : {useful_vals:13,d}  ({useful_bytes/1e6:8.2f} MB)")
    print(f"    chunks fetched        : {chunks_touched:13,d}  "
          f"({chunks_touched/rnd:,.0f} chunk-reads/s)")
    print(f"    data dragged off disk: {eff_bytes/1e6:10.1f} MB  ({eff_bytes/1e6/rnd:8.1f} MB/s)")
    print(f"    READ AMPLIFICATION   : {eff_bytes/useful_bytes:10.1f} x")
    print(f"    throughput (useful)  : {useful_bytes/1e6/rnd:10.3f} MB/s\n")

    # =========================================================================
    print("=== 2. LOAD-ONCE into RAM, then sample in RAM -- the fix ===")
    if args.drop_caches:
        drop_caches()
    start = time.perf_counter()
    surf = U.isel(depth=0).values  # one sequential read of (time, lat, lon)
    load = time.perf_counter() - start
    load_bytes = surf.nbytes
    start = time.perf_counter()
    for ti, yi, xi in steps:
        _ = surf[ti, yi, xi]
        _ = surf[ti + 1, yi, xi]
    idx = time.perf_counter() - start
    print(f"    sequential load      : {load:10.3f} s  "
          f"({load_bytes/1e6:.0f} MB at {load_bytes/1e6/load:,.0f} MB/s)")
    print(f"    in-RAM indexing      : {idx:10.4f} s")
    print(f"    total                : {load+idx:10.3f} s\n")

    # =========================================================================
    print("=== summary ===")
    print(f"  random-off-disk is {rnd/(load+idx):6.1f}x slower than load-once-then-sample")
    print(f"  read amplification of the random pattern: {eff_bytes/useful_bytes:.1f}x")
    print(f"  ({chunks_touched:,} scattered chunk reads vs "
          f"{int(np.ceil(ntime/cT)*np.ceil(nlat/cY)*np.ceil(nlon/cX)):,} "
          f"sequential reads to stream the surface once)")


if __name__ == "__main__":
    main()
