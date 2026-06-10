# Stage 05 — rolling time-window prototype (the fix)

Stages 03–04 showed the lazy-dask path is slow for two independent reasons:
read amplification (re-reading chunks every step) and dask's per-`compute()`
scheduling overhead. This stage prototypes the Tier-1 mitigation and measures it.

**Idea:** a Lagrangian run only ever needs the **two time levels bracketing the
current clock**, even when the full series is far larger than RAM. So use dask for
what it's good at — one **bulk, sequential** read of a time level — and keep
**NumPy** in the hot loop:

1. hold only the bracketing level(s) resident, as NumPy arrays;
2. read a new level with a single `.values` pull (one sequential slab);
3. sample those NumPy arrays directly (no dask in the per-step path);
4. evict levels behind the clock; optionally **prefetch** the next level on a
   background thread so its read overlaps integration.

[`windowed_sampler.py`](windowed_sampler.py) is the `WindowSampler` (field- and
Parcels-agnostic, so the strategy can be benchmarked in isolation and later folded
into a Parcels Field cache). [`bench_windowed.py`](bench_windowed.py) drives the
Parcels access pattern (advancing clock, scattered horizontal positions, linear
time interpolation) on the 20 GB Atlantic dataset two ways and **checks both
produce identical values**.

## Result

7-day run, `dt = 10 min` (hourly fields → ~6 sub-steps per level), 500 particles,
reading the 20 GB Atlantic series:

1008 steps spanning 169 hourly levels. Both paths verified identical
(`max |naive − window| = 0`).

| strategy                         | wall time | steps/s | data read off disk | dask graphs |
|----------------------------------|----------:|--------:|-------------------:|------------:|
| naive (dask `isel().compute()`/step) | **64.1 s** |   15.7 | **28.6 GiB**       | 2016        |
| window (NumPy + rolling reload)  | **11.3 s** |   89.0 | **2.4 GiB**        | 169 loads   |

> **6× faster, reading 12× less data off disk** — for byte-identical results.
> The window did 169 one-shot level loads instead of 2016 scattered per-step
> gathers. The gap widens as the integration `dt` shrinks (more sub-steps per
> level), with float32, or with spatial/depth subsetting.

Why it wins:

- **Reads far less off disk.** The naive path re-reads both bracketing slabs every
  sub-step; the window reads each level **once** and reuses it. The reduction
  scales with sub-steps-per-level (≈ field-dt / integration-dt).
- **No dask in the hot loop.** Per-step sampling is NumPy fancy-indexing, so the
  per-`compute()` scheduling tax (stage 03 / `profiling/`) disappears entirely —
  this part of the win holds even when the data is fully cache-resident.
- **Bounded memory.** Only ~2 levels are ever resident, regardless of series
  length. Shrink further with float32, depth subsetting, or a spatial bbox.

```bash
python bench_windowed.py --dir ../04_atlantic/data/atlantic \
    --days 7 --dt-min 10 --npart 500 --mode both --prefetch --check
```

## From prototype to Parcels

`WindowSampler` deliberately stops at the sampling layer. Two integration paths:

- **User-level, today:** time-blocking — `ds.sel(time=window).load()` a block,
  build a (NumPy-backed) `FieldSet`, `execute` for the block, carry particle state
  to the next block. No Parcels changes; coarser than a true rolling window.
- **Upstream:** a window-cache layer inside the structured-grid `Field` so
  `field.eval` samples a resident NumPy window and refreshes/prefetches on time
  advance — transparent to users, and the real fix. This prototype is the
  reference for what that layer should do (and the before/after to justify it).
