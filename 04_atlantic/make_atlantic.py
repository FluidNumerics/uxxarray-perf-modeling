"""Generate a fictitious, production-scale Atlantic flow-field dataset.

The goal is a reproducer that *looks like* a real Copernicus Marine / GLORYS12
download: a 1/12-degree regular lon/lat Atlantic basin, surface eastward/northward
velocities (``uo``/``vo``), written as **one NetCDF file per day with hourly time
levels** -- so the full time series lives on disk but cannot all fit in RAM at
once (the exact situation users hit in production).

The flow itself is an idealized, time-dependent **double gyre** (Shadden et al.
2005): a divergence-free, periodically-meandering pair of counter-rotating gyres
-- a reasonable caricature of the Atlantic subtropical + subpolar gyre system.
The values are fictitious but smooth, bounded, and physically plausible (~1.5 m/s
peak), so Parcels advection behaves sensibly.

The files are written with CF-compliant coordinate metadata, so they load through
Parcels' real ingestion path:

    ds = xr.open_mfdataset("data/atlantic/*.nc", chunks={"time": 1})
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields={"U": ds.uo, "V": ds.vo})
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)

Usage:
    python make_atlantic.py --target-gb 20 --out data/atlantic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# Atlantic basin bounding box (regional-model-like extent).
LON0, LON1 = -100.0, 20.0   # 120 deg of longitude
LAT0, LAT1 = -45.0, 65.0    # 110 deg of latitude
RES = 1.0 / 12.0            # 1/12 degree, GLORYS12 resolution

U0 = 1.5                    # peak speed [m/s]
EPS = 0.25                  # gyre meander amplitude
GYRE_PERIOD_DAYS = 10.0     # meander period


def double_gyre(lon, lat, t_seconds):
    """Time-dependent double-gyre velocities (m/s) on a (time, lat, lon) grid.

    Stream function psi = (U0/pi) sin(pi f) sin(pi y), with the classic
    meandering f(x,t) = a(t) x^2 + b(t) x mapped onto the basin box.
    """
    x = (lon - LON0) / (LON1 - LON0) * 2.0          # -> [0, 2]
    y = (lat - LAT0) / (LAT1 - LAT0) * 1.0          # -> [0, 1]
    omega = 2 * np.pi / (GYRE_PERIOD_DAYS * 86400.0)

    a = EPS * np.sin(omega * t_seconds)             # (time,)
    b = 1.0 - 2.0 * EPS * np.sin(omega * t_seconds) # (time,)

    # separable in lon/lat per time level -> build via broadcasting
    a = a[:, None]; b = b[:, None]                  # (time, 1)
    f = a * x[None, :] ** 2 + b * x[None, :]        # (time, lon)
    dfdx = 2 * a * x[None, :] + b                    # (time, lon)

    cos_piy = np.cos(np.pi * y)[None, :, None]       # (1, lat, 1)
    sin_piy = np.sin(np.pi * y)[None, :, None]       # (1, lat, 1)

    u = -U0 * np.sin(np.pi * f)[:, None, :] * cos_piy
    v = U0 * np.cos(np.pi * f)[:, None, :] * dfdx[:, None, :] * sin_piy
    return u.astype("float32"), v.astype("float32")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-gb", type=float, default=20.0, help="approx total size, GiB")
    p.add_argument("--out", type=Path, default=Path("data/atlantic"))
    p.add_argument("--hours-per-file", type=int, default=24, help="time levels per daily file")
    p.add_argument("--start", default="2010-01-01")
    args = p.parse_args()

    lon = np.arange(LON0, LON1, RES, dtype="float64")
    lat = np.arange(LAT0, LAT1, RES, dtype="float64")
    depth = np.array([0.494], dtype="float64")  # GLORYS first level
    nx, ny = lon.size, lat.size

    bytes_per_level = nx * ny * 4 * 2  # uo + vo, float32
    ndays = max(1, round(args.target_gb * 1024**3 / (bytes_per_level * args.hours_per_file)))

    print(f"grid        : {ny} lat x {nx} lon  @ {RES:.5f} deg "
          f"({LON0}..{LON1} E, {LAT0}..{LAT1} N)")
    print(f"per level   : {bytes_per_level/1e6:.1f} MB (uo+vo, float32)")
    print(f"files       : {ndays} daily files x {args.hours_per_file} h "
          f"= {ndays*args.hours_per_file} time levels")
    print(f"total       : ~{ndays*args.hours_per_file*bytes_per_level/1024**3:.2f} GiB")

    args.out.mkdir(parents=True, exist_ok=True)
    chunks = (1, 1, ny, nx)  # one horizontal slab per (time, depth) on disk
    enc = {"chunksizes": chunks, "zlib": False, "_FillValue": None}

    t0 = np.datetime64(args.start)
    for d in range(ndays):
        day0 = t0 + np.timedelta64(d, "D")
        times = day0 + np.arange(args.hours_per_file) * np.timedelta64(1, "h")
        t_sec = (times - t0) / np.timedelta64(1, "s")

        u, v = double_gyre(lon, lat, t_sec.astype("float64"))
        u = u[:, None, :, :]  # add depth axis -> (time, depth, lat, lon)
        v = v[:, None, :, :]

        ds = xr.Dataset(
            {
                "uo": (("time", "depth", "latitude", "longitude"), u,
                       {"standard_name": "eastward_sea_water_velocity", "units": "m s-1"}),
                "vo": (("time", "depth", "latitude", "longitude"), v,
                       {"standard_name": "northward_sea_water_velocity", "units": "m s-1"}),
            },
            coords={
                "time": ("time", times),
                "depth": ("depth", depth,
                          {"standard_name": "depth", "units": "m", "axis": "Z", "positive": "down"}),
                "latitude": ("latitude", lat,
                             {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}),
                "longitude": ("longitude", lon,
                              {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}),
            },
        )
        ds["time"].attrs["standard_name"] = "time"
        fname = args.out / f"atlantic_{str(day0)[:10]}.nc"
        ds.to_netcdf(fname, engine="netcdf4", format="NETCDF4",
                     encoding={"uo": enc, "vo": enc})
        if d == 0 or (d + 1) % 10 == 0 or d == ndays - 1:
            print(f"  [{d+1:3d}/{ndays}] wrote {fname.name}")

    total = sum(f.stat().st_size for f in args.out.glob("*.nc"))
    print(f"done. {ndays} files, {total/1024**3:.2f} GiB on disk in {args.out}/")


if __name__ == "__main__":
    main()
