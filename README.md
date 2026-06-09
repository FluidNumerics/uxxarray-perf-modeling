# Dask I/O performance for large datasets in Parcels

This demo investigates a concrete hypothesis:

> Parcels simulations that read their flow fields lazily from dask-backed
> `xarray` datasets are slow because, at the storage layer, **particle field
> sampling amounts to random reads from disk** — and random reads are far slower
> than sequential reads.

It builds the argument in three stages, each runnable on its own:

| stage | what it measures | tool |
|-------|------------------|------|
| [`01_fio/`](01_fio/) | raw storage: sequential vs random read of a 10 GiB file | `fio` |
| [`02_xarray_dask/`](02_xarray_dask/) | xarray/dask: sequential streaming vs scattered `isel().compute()` | mini-app |
| [`03_parcels/`](03_parcels/) | a real Parcels advection, dask-backed vs in-memory (+ the dask scheduling-overhead writeup) | Parcels |
| [`04_atlantic/`](04_atlantic/) | production-scale (~20 GB, 1/12° Atlantic, GLORYS-like) reproducer: lazy dask vs in-RAM time window | Parcels |

See **[Setup](#setup)** for the environment. Run everything with
[`run_all.sh`](run_all.sh) (`SMALL=1 ./run_all.sh` for a quick ~5 GB laptop pass),
or run each stage individually as documented below. All numbers below were
measured on the development box (NVMe SSD, 38 GiB RAM, Parcels v4 alpha, dask
2026.1, xarray 2026.2).

> **Note on the page cache.** With 38 GiB of RAM the OS happily caches a 10 GiB
> file, which would make every read look fast. The fio stage uses `O_DIRECT` to
> bypass the cache entirely. The xarray stage cannot easily do that, so it
> *also* reports **read amplification** — bytes pulled off disk per useful byte —
> which is a property of the chunk layout and access pattern and does **not**
> depend on the cache. Read amplification is the number to trust.

---

## Where the random reads come from (the mechanism)

Every Runge–Kutta sub-step, for every particle, Parcels interpolates `U`/`V` at
the particle's position. On a structured grid the interpolator gathers the
surrounding grid corners with a *vectorised* `isel` over per-particle index
arrays, then forces evaluation with `.compute()`:

```python
# src/parcels/interpolators/_xinterpolators.py  (_get_corner_data_Agrid)
selection_dict[axis_dim["X"]] = xr.DataArray(xi, dims=("points"))
selection_dict[axis_dim["Y"]] = xr.DataArray(yi, dims=("points"))
...
return data.isel(selection_dict).data.reshape(lenT, lenZ, 2, 2, npart)
# ... and at the end of XLinear():
return value.compute() if is_dask_collection(value) else value
```

`xi`, `yi`, … are the *scattered* cell indices of the particles. When `data` is a
dask array, that `.compute()` must materialise **every on-disk chunk that any
particle lands in** — and it does so afresh on every timestep, with no reuse of
chunks loaded on the previous step. That is the random-read pattern, repeated
hundreds of times per run.

---

## Stage 01 — raw storage (`fio`)

```bash
cd 01_fio && ./run.sh          # SIZE=10G by default; fio pulled via `pixi exec`
```

Sequential and random reads of a 10 GiB file, all with `O_DIRECT` (numbers below
are the system disk, `nvme1`/`rl-home`; the box's second disk `nvme0` is faster on
small-random — full side-by-side in [`01_fio/results/two_disk_comparison.md`](01_fio/results/two_disk_comparison.md)):

| pattern            |     MB/s |   IOPS | vs sequential |
|--------------------|---------:|-------:|--------------:|
| sequential, 1 MiB  | **2094** |  1 997 |        1.00×  |
| random, 1 MiB      |   1912   |  1 824 |        0.91×  |
| random, 4 KiB      | **36**   |  8 793 |        0.02×  |

**Takeaway.** On this SSD, *randomness itself* is nearly free at a large block
size (1 MiB random ≈ 0.91× of sequential). What destroys throughput is **small
I/O**: 4 KiB random reads run at 36 MB/s — **58× slower** than sequential. The
disk is happy to seek; it is *not* happy to do so for only 4 KiB at a time.

This reframes the hypothesis: the enemy is not "random" per se, it is **many
small, scattered reads**. Chunk size is therefore the dominant lever — which is
exactly what stage 02 shows.

---

## Stage 02 — xarray / dask sampling mini-app

```bash
cd 02_xarray_dask
python make_dataset.py --gb 10 --chunk slab  --out data/ocean_slab_10g.nc
python make_dataset.py --gb 10 --chunk tiled --out data/ocean_tiled_10g.nc
python bench_sampling.py --file data/ocean_slab_10g.nc
python bench_sampling.py --file data/ocean_tiled_10g.nc
```

`bench_sampling.py` reproduces the Parcels access pattern (all particles share
the advancing time level; two time levels per step for interpolation; horizontal
positions scattered) and compares it to loading the field once and indexing in
RAM. Two on-disk chunk layouts of the *same* 10 GiB field:

- **slab** — one full horizontal plane per `(time, depth)` → `(1,1,400,1000)`, 3.2 MB chunks (typical reanalysis output)
- **tiled** — small horizontal tiles → `(1,1,128,128)`, 0.13 MB chunks (cloud / zarr style)

Workload: 10 000 particles × 50 timesteps.

| metric (random `isel().compute()`) | slab (3.2 MB chunks) | tiled (0.13 MB chunks) |
|------------------------------------|---------------------:|-----------------------:|
| chunks fetched off disk            |                  100 |                  3 200 |
| data dragged off disk              |              320 MB  |               419 MB  |
| **read amplification**             |          **40×**     |           **52×**     |
| effective read bandwidth           |          103 MB/s    |        **34 MB/s**    |
| wall time                          |             3.1 s    |             12.4 s    |
| vs. load-once-then-sample          |             2.6×     |             1.4×*     |

\* the tiled "load once" is itself slow (sequential surface load ran at 46 MB/s
vs 345 MB/s for slab) because even a *contiguous* logical read is fragmented into
thousands of tiny chunk reads — so the dask penalty looks smaller only because
the baseline got dragged down too.

**Takeaways.**
- To deliver **8 MB** of useful values, the random pattern reads **320–420 MB**
  off disk — a **40–52× read amplification** that no cache trick removes. Each
  timestep re-reads chunks the previous step already touched.
- The **tiled** field samples at **34 MB/s effective** — essentially the same as
  fio's 4 KiB random read (36 MB/s). Small chunks turn field sampling into the
  worst-case storage pattern. The **slab** field, with big chunks, behaves like
  fio's random-1 MiB (fast per-read, but you still haul a whole 3.2 MB plane to
  read a handful of points).
- Tiny chunks also cost *elsewhere*: the tiled file is **14 GiB on disk** (vs
  10 GiB) and was written at 68 MB/s (vs 549 MB/s) — chunk metadata overhead.
- **Loading the field once and indexing in RAM is the fix** whenever the field
  fits in memory: it converts thousands of scattered reads into one sequential
  pass.

---

## Stage 03 — minimal Parcels advection, dask vs in-memory

```bash
cd 03_parcels
python make_fieldset.py --nx 1000 --ny 1000 --nt 60 --out data/flow.nc
python run_parcels.py --file data/flow.nc --npart 2000 --runtime-days 20 --mode both
```

`make_fieldset.py` writes a synthetic, SGRID-compliant A-grid field (steady
solid-body rotation so particles orbit and sweep many cells), chunked one
horizontal slab per `(time, depth)` on disk. `run_parcels.py` then runs the
*identical* simulation twice — once with the field left lazy (dask, read on
demand) and once after `.load()`-ing it into RAM.

Workload: 2 000 particles, 20 days, `dt = 1 h` → 480 steps (960 000 advection steps).

| field backing            |   wall time | advection steps/s | slowdown |
|--------------------------|------------:|------------------:|---------:|
| in-memory (numpy)        |  **6.6 s**  |           145 637 |     1×   |
| dask-backed (on-disk)    | **2156 s**  |               445 | **327×** |

> **dask-backed is 327× slower** than the identical in-memory run.

Crucially, this run is **not** disk-bound — the 960 MB field is cache-resident and
the process is pinned at ~100 % CPU. The dominant cost is **dask's per-`compute()`
scheduling overhead**, paid on every Runge–Kutta sub-step. That second, disk-
independent tax — what it is, how it was measured in isolation, and the full set of
options for reducing it (with references to the dask/xarray docs) — is written up
in **[`03_parcels/README.md`](03_parcels/README.md)**.

---

## Stage 04 — production-scale Atlantic reproducer

```bash
cd 04_atlantic
python make_atlantic.py --target-gb 20 --out data/atlantic
python run_atlantic.py  --runtime-days 2 --npart 200 --mode both
```

A fictitious but production-shaped dataset — 1/12° Atlantic basin (1320 × 1440),
hourly surface `uo`/`vo`, **59 daily NetCDF files totalling ~20 GiB** with
CF/Copernicus metadata — built to reproduce the real bind: *the full time series
fits on disk but not in RAM, so it must be opened lazily with dask.* It loads
through Parcels' actual `copernicusmarine_to_sgrid` path. The reproducer contrasts
the forced lazy-dask run against loading just the **time window the run needs**
into RAM (a couple of days ≈ 1 GB). Details and numbers:
**[`04_atlantic/README.md`](04_atlantic/README.md)**.

---

## Conclusions & recommendations

1. **The hypothesis holds, with a sharpening.** Dask-backed Parcels field
   sampling *is* a random-read workload, but the cost is dominated by **small,
   repeated, scattered chunk reads**, not by randomness alone (stage 01).
2. **Chunk shape is the dominant lever.** Few-but-large chunks (full horizontal
   slabs) keep per-read sizes high; many-small tiles collapse throughput to
   4 KiB-random territory (stage 02).
3. **Re-reads are the silent killer.** Parcels re-`compute()`s every timestep,
   so chunks are fetched again and again with no reuse. This multiplies the
   already-large read amplification by the number of timesteps.
4. **There is a second, disk-independent tax: dask scheduling overhead.** Even
   with the field fully in RAM, each `.compute()` rebuilds and reschedules a task
   graph (~200 µs–1 ms per task), making the stage-03 run 327× slower than NumPy.
   The detailed background and the options for reducing it (with references) are
   in **[`03_parcels/README.md`](03_parcels/README.md)**.
5. **If the field fits in RAM, load it.** `ds.load()` before building the
   `FieldSet` turns the whole problem into a single sequential pass and sidesteps
   both the read amplification and the scheduling overhead (stage 03).
6. **If it does not fit, chunk for the access pattern and cache.** Prefer large
   horizontal chunks aligned to storage; `.persist()` and pre-load only the time
   window in flight so chunks read on one timestep are reused on the next.

## Setup

You need a Python environment with **Parcels v4** (alpha) plus `xarray`, `dask`,
`netcdf4`, and `fio`. Two options:

```bash
# (a) conda/mamba -- best-effort, installs Parcels v4 from git
conda env create -f environment.yml
conda activate uxxarray-perf-modeling

# (b) reuse the pixi env of an existing Parcels v4 checkout (most reliable):
export PARCELS_PYTHON=/path/to/parcels/.pixi/envs/default/bin/python
```

`run_all.sh` uses `$PARCELS_PYTHON` if set, otherwise the `python` on `PATH`.
`fio` is in `environment.yml`; `01_fio/run.sh` also falls back to a system `fio`
or `pixi exec fio` if available.

## Reproducing

```bash
./run_all.sh          # full ~50 GB run
SMALL=1 ./run_all.sh  # quick ~5 GB laptop pass
```

- Each stage can also be run on its own — see the per-stage commands above and
  the stage READMEs.
- **Generated data is large** (~50 GB at full size) and git-ignored. It lands
  under each stage's `data/`; on a small home disk, point `data/` at a scratch
  disk (e.g. `ln -s /mnt/bigdisk/uxxarray-data 02_xarray_dask/data`) or pass the
  `--out` flag to the `make_*.py` scripts.
- Committed `results/` are the numbers measured on the development box (NVMe SSD,
  38 GiB RAM, Parcels v4 alpha, dask 2026.1, xarray 2026.2); yours will differ
  with hardware. Stage 01 also has a two-disk comparison in
  [`01_fio/results/two_disk_comparison.md`](01_fio/results/two_disk_comparison.md).
