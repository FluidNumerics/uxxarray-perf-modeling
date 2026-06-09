"""Minimal Parcels structured-grid advection, timed dask-backed vs in-memory.

Same FieldSet, same particles, same kernel -- the only difference is whether the
velocity fields are left lazy (dask, read off disk on demand) or eagerly loaded
into RAM first.

Every Runge-Kutta sub-step, Parcels evaluates U and V at the scattered particle
positions via ``field.data.isel(<per-particle indices>).compute()`` (see
``src/parcels/interpolators/_xinterpolators.py``). With a dask-backed field that
``.compute()`` is the random-read pattern measured in stages 01-02, now paid
hundreds of times (once per timestep) over the life of the simulation.

Usage:
    python run_parcels.py --file data/flow.nc --npart 2000 \
        --runtime-days 20 --dt-hours 1 --mode both
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")  # silence the v4-alpha API warning

import parcels  # noqa: E402


def build_fieldset(path: Path, load: bool) -> parcels.FieldSet:
    """Open the SGRID file as either a dask-backed or fully in-memory FieldSet."""
    if load:
        ds = xr.open_dataset(path).load()  # eager: everything in RAM
    else:
        ds = xr.open_dataset(path, chunks={})  # lazy: dask, on-disk chunking
    return parcels.FieldSet.from_sgrid_conventions(ds, mesh="flat")


def run(path: Path, npart: int, runtime_days: float, dt_hours: float, load: bool):
    fieldset = build_fieldset(path, load=load)

    ds = xr.open_dataset(path)
    lon_max = float(ds["lon"].max())
    lat_max = float(ds["lat"].max())
    z0 = float(ds["depth"].values[0])
    t0 = ds["time"].values[0]

    rng = np.random.default_rng(0)
    lon = rng.uniform(0.2 * lon_max, 0.8 * lon_max, npart)
    lat = rng.uniform(0.2 * lat_max, 0.8 * lat_max, npart)

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.Particle,
        time=np.repeat(t0, npart),
        z=np.repeat(z0, npart),
        lat=lat,
        lon=lon,
    )

    label = "in-memory (numpy)" if load else "dask-backed (on-disk)"
    start = time.perf_counter()
    pset.execute(
        [parcels.kernels.AdvectionRK2],
        runtime=np.timedelta64(int(runtime_days * 24 * 3600), "s"),
        dt=np.timedelta64(int(dt_hours * 3600), "s"),
    )
    secs = time.perf_counter() - start

    nsteps = int(runtime_days * 24 / dt_hours)
    print(f"  {label:<24}: {secs:8.2f} s  "
          f"({npart} particles x {nsteps} steps = "
          f"{npart * nsteps:,} advection steps, "
          f"{npart * nsteps / secs:,.0f} steps/s)")
    return secs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=Path("data/flow.nc"))
    p.add_argument("--npart", type=int, default=2000)
    p.add_argument("--runtime-days", type=float, default=20.0)
    p.add_argument("--dt-hours", type=float, default=1.0)
    p.add_argument("--mode", choices=["dask", "memory", "both"], default="both")
    args = p.parse_args()

    print(f"=== Parcels structured-grid advection ({args.file}) ===")
    results = {}
    if args.mode in ("memory", "both"):
        results["memory"] = run(args.file, args.npart, args.runtime_days,
                                args.dt_hours, load=True)
    if args.mode in ("dask", "both"):
        results["dask"] = run(args.file, args.npart, args.runtime_days,
                              args.dt_hours, load=False)

    if "memory" in results and "dask" in results:
        print(f"\n  dask-backed is {results['dask'] / results['memory']:.1f}x "
              f"slower than in-memory for the identical simulation.")


if __name__ == "__main__":
    main()
