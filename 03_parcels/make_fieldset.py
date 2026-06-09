"""Build a synthetic, SGRID-compliant structured-grid dataset for Parcels.

The flow is a steady solid-body rotation, so particles orbit the domain centre
and, crucially for this demo, sweep across many grid cells over the run -- which
forces the field interpolation to touch many different chunks.

The variables are written as an A-grid (U and V on the same nodes) and the file
is chunked one horizontal slab per (time, depth) on disk -- the same realistic
layout used in stage 02. That is what makes the dask-backed Parcels run pay the
random-read tax in stage 03.

Usage:
    python make_fieldset.py --nx 1000 --ny 1000 --nt 60 --out data/flow.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

import parcels._sgrid as sgrid

# Physical domain: a 1000 km x 1000 km box, surface only, two months of hourly-ish
# snapshots. Units are metres / seconds (mesh="flat").
LX = LY = 1.0e6  # m
DAYS = 60


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=1000)
    p.add_argument("--ny", type=int, default=1000)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--nt", type=int, default=DAYS)
    p.add_argument("--out", type=Path, default=Path("data/flow.nc"))
    args = p.parse_args()

    X, Y, Z, T = args.nx, args.ny, args.nz, args.nt

    xg = np.linspace(0, LX, X)
    yg = np.linspace(0, LY, Y)
    zg = np.linspace(0, 1000.0, Z)
    time = np.datetime64("2000-01-01") + np.arange(T) * np.timedelta64(1, "D")

    # Solid-body rotation about the domain centre; one full orbit in ~30 days.
    omega = 2 * np.pi / (30 * 86400.0)  # rad/s
    xc, yc = LX / 2, LY / 2
    XX, YY = np.meshgrid(xg, yg)  # (Y, X)
    u2d = (-omega * (YY - yc)).astype("float64")
    v2d = (omega * (XX - xc)).astype("float64")

    chunks = (1, 1, Y, X)  # one horizontal slab per (time, depth)
    U = da.from_array(np.broadcast_to(u2d, (T, Z, Y, X)), chunks=chunks)
    V = da.from_array(np.broadcast_to(v2d, (T, Z, Y, X)), chunks=chunks)

    ds = xr.Dataset(
        {
            "U": (["time", "ZG", "YG", "XG"], U),
            "V": (["time", "ZG", "YG", "XG"], V),
        },
        coords={
            "XG": (["XG"], xg, {"axis": "X", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], xg + 0.5 * LX / X, {"axis": "X"}),
            "YG": (["YG"], yg, {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "YC": (["YC"], yg + 0.5 * LY / Y, {"axis": "Y"}),
            "ZG": (["ZG"], zg, {"axis": "Z", "c_grid_axis_shift": -0.5}),
            "ZC": (["ZC"], zg + 0.5, {"axis": "Z"}),
            "lon": (["XG"], xg),
            "lat": (["YG"], yg),
            "depth": (["ZG"], zg),
            "time": (["time"], time, {"axis": "T"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.HIGH),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.HIGH),
            ),
            node_coordinates=("lon", "lat"),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "ZG", sgrid.Padding.HIGH),),
        ),
    )

    ds["U"].encoding = {"chunksizes": chunks, "zlib": False}
    ds["V"].encoding = {"chunksizes": chunks, "zlib": False}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbytes = 2 * T * Z * Y * X * 8
    print(f"grid (T,Z,Y,X) = {(T, Z, Y, X)}   ~{nbytes / 1e6:.0f} MB (U+V)")
    print(f"on-disk chunk  = {chunks}  ({np.prod(chunks) * 8 / 1e6:.1f} MB/slab)")
    ds.to_netcdf(args.out, engine="netcdf4", format="NETCDF4")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
