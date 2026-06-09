"""Production-scale Parcels reproducer on the fictitious Atlantic dataset.

Loads the multi-file, ~20 GB Atlantic dataset exactly the way a user would load a
Copernicus Marine download, and runs an identical advection two ways:

  --mode dask    : open the whole multi-file series lazily (dask). This is what
                   you are forced to do when the full time series does not fit in
                   RAM. Every kernel sub-step samples the field via
                   isel().compute() -> the per-compute dask overhead (see
                   ../03_parcels/README.md) is paid on every step.

  --mode window  : load only the time window the run actually needs into RAM
                   (numpy), then run. This is the recommended mitigation: a
                   simulation only ever needs the two time levels bracketing the
                   current clock, and a few days of surface fields fit easily in
                   memory even when the full series does not.

Usage:
    python run_atlantic.py --runtime-days 2 --npart 200 --mode both
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

import parcels  # noqa: E402


def build_fieldset(files, load_window=None):
    """Open the Atlantic files and build a FieldSet via the Copernicus path.

    If ``load_window`` (a (start, stop) np.datetime64 pair) is given, the dataset
    is sliced to that time range and eagerly loaded into RAM (numpy-backed).
    Otherwise it stays lazy (dask-backed).
    """
    ds = xr.open_mfdataset(files, chunks={"time": 1}, combine="by_coords")
    if load_window is not None:
        ds = ds.sel(time=slice(*load_window)).load()
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields={"U": ds["uo"], "V": ds["vo"]})
    return parcels.FieldSet.from_sgrid_conventions(ds_fset), ds


def run(files, npart, runtime_days, dt_hours, load):
    # peek at the time axis to choose the load window and the start time
    probe = xr.open_mfdataset(files, chunks={"time": 1}, combine="by_coords")
    t0 = probe["time"].values[0]
    runtime = np.timedelta64(int(runtime_days * 24 * 3600), "s")
    # +1 day buffer so the bracketing time level for interpolation is present
    window = (t0, t0 + runtime + np.timedelta64(1, "D")) if load else None
    probe.close()

    fieldset, ds = build_fieldset(files, load_window=window)

    # seed particles in the subtropical-gyre region, at the surface
    rng = np.random.default_rng(0)
    lon = rng.uniform(-60.0, -20.0, npart)
    lat = rng.uniform(10.0, 40.0, npart)
    z = np.full(npart, float(ds["depth"].values[0]))

    pset = parcels.ParticleSet(
        fieldset=fieldset, pclass=parcels.Particle,
        time=np.repeat(t0, npart), z=z, lat=lat, lon=lon,
    )

    label = "window (numpy, in RAM)" if load else "full series (dask, lazy)"
    start = time.perf_counter()
    pset.execute(
        [parcels.kernels.AdvectionRK2],
        runtime=runtime,
        dt=np.timedelta64(int(dt_hours * 3600), "s"),
    )
    secs = time.perf_counter() - start
    nsteps = int(runtime_days * 24 / dt_hours)
    print(f"  {label:<26}: {secs:9.2f} s  "
          f"({npart} particles x {nsteps} steps, {npart*nsteps/secs:,.0f} steps/s)")
    return secs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("data/atlantic"))
    p.add_argument("--npart", type=int, default=200)
    p.add_argument("--runtime-days", type=float, default=2.0)
    p.add_argument("--dt-hours", type=float, default=1.0)
    p.add_argument("--mode", choices=["dask", "window", "both"], default="both")
    args = p.parse_args()

    files = sorted(str(f) for f in args.dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"no NetCDF files in {args.dir} -- run make_atlantic.py first")
    total_gb = sum(Path(f).stat().st_size for f in files) / 1024**3
    print(f"=== Atlantic reproducer: {len(files)} files, {total_gb:.1f} GiB total ===")

    res = {}
    if args.mode in ("window", "both"):
        res["window"] = run(files, args.npart, args.runtime_days, args.dt_hours, load=True)
    if args.mode in ("dask", "both"):
        res["dask"] = run(files, args.npart, args.runtime_days, args.dt_hours, load=False)
    if "window" in res and "dask" in res:
        print(f"\n  lazy dask over the full series is {res['dask']/res['window']:.1f}x slower "
              f"than loading just the {args.runtime_days:g}-day window into RAM.")


if __name__ == "__main__":
    main()
