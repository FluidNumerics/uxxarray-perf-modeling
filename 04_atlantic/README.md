# Stage 04 — production-scale Atlantic reproducer

A fictitious but **production-shaped** dataset: a 1/12° Atlantic basin with an
idealized time-dependent flow, written as daily hourly NetCDF files totalling
~20 GB — sized and laid out like a Copernicus Marine / GLORYS12 download. The
point is to reproduce the situation users actually hit: **the full time series
fits on disk but not in RAM, so the fields must be opened lazily with dask.**

## What gets generated

```bash
python make_atlantic.py --target-gb 20 --out data/atlantic
```

| property            | value |
|---------------------|-------|
| domain              | 100°W–20°E, 45°S–65°N (Atlantic basin) |
| resolution          | 1/12° (GLORYS12), **1320 × 1440** grid points |
| variables           | `uo`, `vo` (surface eastward/northward velocity), `float32` |
| time                | hourly, **1416 levels over 59 days** |
| file layout         | **one file per day**, 24 time levels each (`atlantic_YYYY-MM-DD.nc`) |
| on-disk chunking    | one horizontal slab per `(time, depth)` — `(1, 1, 1320, 1440)`, 7.6 MB |
| total size          | **~20 GiB** (59 files × ~365 MB) |
| coordinate metadata | CF-compliant (`longitude`/`latitude`/`depth`/`time`), so it loads through Parcels' real Copernicus path |

The flow is an idealized, divergence-free, periodically-meandering **double gyre**
(Shadden et al. 2005) mapped onto the basin — a caricature of the subtropical +
subpolar gyre system, with ~1.5 m/s peak speeds. The numbers are fictitious but
smooth and bounded, so advection behaves sensibly. It is *not* a physical ocean
model; it exists to exercise the I/O and dask code paths at production scale.

## How it loads (the real Copernicus path)

The files carry the same coordinate metadata as a Copernicus download, so the
ingestion is verbatim the quickstart code:

```python
import xarray as xr, parcels
ds = xr.open_mfdataset("data/atlantic/*.nc", chunks={"time": 1})   # lazy, dask-backed
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields={"U": ds.uo, "V": ds.vo})
fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)        # U.data is dask-backed
```

## The reproducer

```bash
python run_atlantic.py --runtime-days 2 --npart 200 --mode both
```

It runs the same advection two ways:

- **`dask`** — open the whole 20 GB series lazily (what you must do when it
  doesn't fit in RAM) and let Parcels sample it per step via `isel().compute()`.
- **`window`** — load only the time window the run needs (a couple of days ≈ 1 GB,
  which *does* fit) into RAM as numpy, then run. This is the mitigation discussed
  in [`../03_parcels/README.md`](../03_parcels/README.md): a simulation only ever
  needs the two time levels bracketing the current clock.

### Result

200 particles, 2 days, `dt = 1 h` (48 steps), reading the full 20 GiB series:

| field backing                        | wall time | steps/s | slowdown |
|---------------------------------------|----------:|--------:|---------:|
| window loaded into RAM (numpy)        | **0.30 s**|  31 720 |     1×   |
| full series, lazy (dask)              | **8.11 s**|   1 183 | **26.8×**|

> Loading just the 2-day window the run needs (≈1 GB, easily resident) is **26.8×
> faster** than the lazy-dask path forced by the 20 GB series — even though the
> window approach still only ever holds a sliver of the data in memory.

The gap grows with run length and particle count (the longer 480-step run in
stage 03 reached 327×), because the per-`compute()` overhead is paid on every
step. The takeaway is the same as stage 03, now at production scale: the lazy-dask path
pays the per-`compute()` scheduling tax on every step, while loading just the
needed window converts the problem into one sequential read plus fast numpy
indexing. The full series never has to be resident — only the rolling window.

## Notes

- ~20 GB of generated data lives in `data/atlantic/` and is git-ignored.
- The `dask` mode is intentionally the slow path; scale `--runtime-days` /
  `--npart` down for a quick look, up to feel a real production run.
- Real Copernicus files are usually **compressed** (and sometimes int16-packed);
  this generator writes uncompressed float32 so on-disk size equals data size and
  the I/O numbers are clean. Compression trades disk bytes for decompression CPU.
