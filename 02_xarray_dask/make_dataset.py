"""Generate a large, chunked NetCDF dataset that looks like an ocean model output.

The file is written straight from a lazy dask array, so generating a 10 GiB file
never materialises more than one chunk at a time in RAM.

Dimensions mimic a gridded hydrodynamic model: (time, depth, lat, lon). The
on-disk NetCDF4 chunking is one full horizontal slab per (time, depth) -- i.e.
``(1, 1, lat, lon)`` -- which is an extremely common layout for reanalysis /
model output. That layout is the crux of the demo: the smallest unit the storage
layer will ever hand back is one whole horizontal slab, even if you only want a
single grid point.

Usage:
    python make_dataset.py --gb 10 --out data/ocean_10g.nc
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gb", type=float, default=10.0, help="approx target size in GiB")
    p.add_argument("--out", type=Path, default=Path("data/ocean_10g.nc"))
    p.add_argument("--lat", type=int, default=400)
    p.add_argument("--lon", type=int, default=1000)
    p.add_argument("--depth", type=int, default=26)
    p.add_argument("--chunk", choices=["slab", "tiled"], default="slab",
                   help="on-disk chunk layout: 'slab' = one full horizontal "
                        "plane per (time, depth); 'tiled' = small horizontal tiles")
    p.add_argument("--tile", type=int, default=128, help="tile edge for --chunk tiled")
    args = p.parse_args()

    bytes_per = 8  # float64
    slab = args.lat * args.lon * args.depth * bytes_per
    ntime = max(1, round(args.gb * 1024**3 / slab))

    shape = (ntime, args.depth, args.lat, args.lon)
    if args.chunk == "slab":
        # One full horizontal slab per (time, depth) -- typical reanalysis layout.
        chunks = (1, 1, args.lat, args.lon)
    else:
        # Small horizontal tiles -- cloud/zarr style. Scattered access then has
        # to fetch many tiny chunks from scattered offsets (the 4 KiB-random analog).
        chunks = (1, 1, min(args.tile, args.lat), min(args.tile, args.lon))
    nbytes = int(np.prod(shape)) * bytes_per

    print(f"shape       : {shape}  (time, depth, lat, lon)")
    print(f"on-disk chunk: {chunks}  ({np.prod(chunks) * bytes_per / 1e6:.1f} MB each)")
    print(f"total size  : {nbytes / 1024**3:.2f} GiB")
    print(f"writing to  : {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Lazy data: cheap to build, streamed to disk chunk-by-chunk. Values are a
    # smooth-ish field so they are realistic, but the content is irrelevant to I/O.
    lat = np.linspace(-80, 80, args.lat)
    lon = np.linspace(0, 360, args.lon, endpoint=False)
    depth = np.linspace(0, 5000, args.depth)
    t0 = np.datetime64("2000-01-01")
    times = t0 + np.arange(ntime) * np.timedelta64(1, "D")

    data = da.random.default_rng(0).standard_normal(shape, chunks=chunks)

    ds = xr.Dataset(
        {"U": (("time", "depth", "lat", "lon"), data)},
        coords={"time": times, "depth": depth, "lat": lat, "lon": lon},
    )
    ds["U"].encoding = {"chunksizes": chunks, "zlib": False}

    start = time.perf_counter()
    ds.to_netcdf(args.out, engine="netcdf4", format="NETCDF4")
    dt = time.perf_counter() - start
    actual = args.out.stat().st_size
    print(f"wrote {actual / 1024**3:.2f} GiB in {dt:.1f}s "
          f"({actual / 1e6 / dt:.0f} MB/s)")


if __name__ == "__main__":
    main()
