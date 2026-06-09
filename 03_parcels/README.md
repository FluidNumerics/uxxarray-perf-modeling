# Stage 03 — Parcels advection: dask-backed vs in-memory

This stage runs a real (if minimal) Parcels simulation on a structured grid and
times it twice — once with the velocity fields left **lazy** (dask, read from the
NetCDF file on demand) and once after **loading them into RAM** (`.load()`). The
two runs are otherwise identical: same grid, same particles, same kernel.

```bash
python make_fieldset.py --nx 1000 --ny 1000 --nt 60 --out data/flow.nc
python run_parcels.py    --file data/flow.nc --npart 2000 --runtime-days 20 --mode both
```

`make_fieldset.py` writes a synthetic, SGRID-compliant A-grid field (a steady
solid-body rotation, so particles orbit the domain centre and continuously sweep
into new grid cells), chunked one horizontal slab per `(time, depth)` on disk.

## Result

2 000 particles, 20 days, `dt = 1 h` → 480 timesteps (960 000 advection steps):

| field backing          |   wall time | advection steps/s | slowdown |
|------------------------|------------:|------------------:|---------:|
| in-memory (numpy)      |  **6.59 s** |           145 637 |     1×   |
| dask-backed (on-disk)  | **2156 s**  |               445 | **327×** |

The dask-backed run is **327× slower** for a byte-for-byte identical simulation.
Strikingly, this is **not** a disk-bandwidth problem: the field is only 960 MB
and is fully resident in the OS page cache after the first read, so the process
sits at ~100 % CPU with a small (~640 MB) resident set the entire time. The cost
is almost entirely **dask's per-`compute()` scheduling machinery**, paid again on
every Runge–Kutta sub-step. The rest of this document explains why.

---

## Background: how dask executes work

[Dask](https://docs.dask.org/) parallelises array/dataframe code by splitting an
array into **chunks** and recording every operation as a node in a **task
graph** — a dictionary of `{key: (function, *args)}` tuples — rather than running
it immediately. Nothing actually computes until you call `.compute()` (or
`.load()`, `.persist()`, `.values`, …); at that point the graph is handed to a
**scheduler** that runs the tasks. xarray inherits this directly:

> "Xarray operations on Dask-backed arrays are lazy. This means computations are
> not executed immediately, but are instead queued up as tasks in a Dask graph."
> — [xarray: Parallel computing with Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html)

Two consequences matter here:

1. **Every task carries fixed overhead.** Building, optimising, serialising and
   dispatching a task is not free. The dask docs put it at:

   > "Every task comes with some overhead. This is somewhere between 200us and
   > 1ms." — [Dask best practices](https://docs.dask.org/en/stable/best-practices.html)

   The default **threaded scheduler** is the cheapest option and still costs
   "around 50us per task"
   ([Dask scheduling](https://docs.dask.org/en/stable/scheduling.html)), and it
   only parallelises code that releases the GIL (i.e. NumPy internals), not the
   Python-level graph manipulation itself.

2. **A fresh `.compute()` is a fresh graph.** dask does not, by default, cache
   results between separate `.compute()` calls. Each call rebuilds and reschedules
   its graph from scratch, and any chunk it reads is dropped afterwards unless you
   explicitly `.persist()` it.

Parcels' interpolator (`src/parcels/interpolators/_xinterpolators.py`) ends every
field evaluation with exactly such a call:

```python
return value.compute() if is_dask_collection(value) else value
```

and it does so for **each velocity component, for each RK stage, on every
timestep**. So the fixed per-task overhead above is multiplied by (tasks per
gather) × (gathers per step) × (number of steps) — hundreds of thousands of
times over the run. That is the 327×.

### Measuring the overhead in isolation

[`dask_overhead.py`](dask_overhead.py) strips the problem down to the bare
mechanism. It builds a field, **`.persist()`s it into RAM so there is zero disk
I/O**, and then times a single scattered gather — the same vectorised
`isel(...).compute()` Parcels performs — repeatedly:

```
sampling 2,000 scattered points from an in-RAM (60, 1000, 1000) field (no disk I/O)

  numpy  isel().values    :      312.4 us / call
  dask   isel().compute() :     63.412 ms / call   (120 tasks in the graph)
  dask overhead factor    :        203 x
```

So a single gather that NumPy does in **0.3 ms** takes dask **63 ms** — a **203×
penalty with no disk involved at all.** The 120-task graph at ~0.5 ms/task lands
squarely in dask's own documented 200 µs–1 ms range, confirming the cost is the
scheduler, not the data. Multiply 63 ms by the many gathers Parcels issues per
step across 480 steps and you recover the multi-thousand-second runtime.

### Where the time actually goes

[`../profiling/`](../profiling/) profiles this loop (cProfile + a VizTracer
timeline trace). With the field in RAM, the self-time is almost entirely the
**threaded scheduler's synchronization** — `threading` lock/condition
`acquire`/`release`/`wait`/`notify` — as dask dispatches the ~120-task graph of
each gather through its thread pool, plus per-`compute()` graph construction
(`__dask_graph__`, `__dask_tokenize__`, `start_state_from_dask`). There is no
array math in the hot path. See [`../profiling/README.md`](../profiling/README.md).

---

## Reducing the dask scheduling overhead

Roughly in order of impact for a Parcels-style workload:

### 1. Don't use dask when the field fits in RAM — `.load()` it
This is the fix demonstrated above (6.6 s vs 2156 s). dask's own array docs are
explicit:

> "If your data fits comfortably in RAM and you are not performance bound, then
> using NumPy directly may be a better choice. Dask adds another layer of
> complexity which may get in the way."
> — [Dask array best practices](https://docs.dask.org/en/stable/array-best-practices.html)

In xarray terms: `ds.load()` (or `ds.compute()`) before building the `FieldSet`.
This converts the whole problem into one sequential read followed by free NumPy
indexing.

### 2. If it doesn't fit, `.persist()` what you can and pre-load the time window
`.persist()` keeps chunks in (distributed) memory so they are not re-read or
re-scheduled on the next step:

> "You can also use `Dataset.persist()` for quickly accessing intermediate
> outputs." — [xarray: Parallel computing with Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html)

Because a simulation only ever needs the two time levels bracketing the current
clock, pre-loading just that rolling window (rather than the whole field) keeps
memory bounded while still eliminating per-step re-reads.

### 3. Use bigger chunks — fewer tasks per gather
Task count, and therefore overhead, is set by chunk count. The guidance is to err
large:

> "In general, chunks should be large in order to reduce the number of chunks that
> Dask has to think about (which affects overhead) … it is rare to see chunk sizes
> below 100 MB."
> — [Dask array best practices](https://docs.dask.org/en/stable/array-best-practices.html)

> "A good rule of thumb is to create arrays with a minimum chunk size of at least
> one million elements." — [xarray: Parallel computing with Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html)

And critically, **align dask chunks to the on-disk chunks** (make each a multiple
of the storage chunk) so a read does not repeatedly pull the same bytes — this is
the read-amplification lever quantified in [stage 02](../02_xarray_dask/). Tiny
storage tiles are the worst case (see stage 02: the tiled field samples at
~34 MB/s, matching fio's 4 KiB random read).

### 4. Batch work into one `.compute()` instead of many
Calling compute in a tight loop serialises the client and forbids result sharing:

> "Calling `compute` … blocks the execution … Instead … call `dask.compute(*results)`
> … this allows Dask to share intermediate results."
> — [Dask best practices](https://docs.dask.org/en/stable/best-practices.html)

Parcels' per-particle, per-step `.compute()` is the opposite of this. Restructuring
to gather all particles' samples for a step (or several steps) in a single graph
amortises the fixed overhead. (This is an upstream Parcels design lever, not a
user knob, but it is the root cause.)

### 5. Fuse / `map_blocks` to shrink graphs; pick the right scheduler
For custom array ops, `da.map_blocks` collapses many tasks into one
([best practices](https://docs.dask.org/en/stable/best-practices.html)). For
debugging and for very small graphs the **synchronous scheduler**
(`scheduler="synchronous"`) removes thread-pool overhead and makes profiling
sane ([scheduling docs](https://docs.dask.org/en/stable/scheduling.html)).

### What does *not* help
- **More threads / a distributed cluster.** The bottleneck is Python-level graph
  construction per `compute()`, much of which is GIL-bound and runs per call
  regardless of worker count. Parallelism cannot remove a cost that is paid
  serially before any task runs.
- **Faster storage.** The stage-03 field is cache-resident; the SSD is idle. NVMe
  would not move the 327×.

---

## References

- Dask — Best Practices: <https://docs.dask.org/en/stable/best-practices.html>
- Dask — Array Best Practices (chunks, NumPy-vs-dask): <https://docs.dask.org/en/stable/array-best-practices.html>
- Dask — Scheduling (schedulers, per-task overhead, GIL): <https://docs.dask.org/en/stable/scheduling.html>
- Dask — Array Slicing (fancy/`vindex` indexing caveats): <https://docs.dask.org/en/stable/array-slicing.html>
- xarray — Parallel computing with Dask (`load`/`persist`/chunks): <https://docs.xarray.dev/en/stable/user-guide/dask.html>
- Parcels interpolator that issues the per-step `.compute()`: `src/parcels/interpolators/_xinterpolators.py`
