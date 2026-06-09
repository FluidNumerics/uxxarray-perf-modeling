# Profiling the dask scheduling overhead

Stage 03 showed that a dask-backed Parcels run is ~327× slower than an in-memory
one **even though no disk I/O happens** — the field is cache-resident and the
process is CPU-bound. This directory profiles *where* that CPU goes, to confirm it
is dask's client-side graph machinery (build → optimize → schedule), not array
math.

[`profile_dask.py`](profile_dask.py) runs the exact Parcels gather —
`field.isel(<scattered indices>).compute()` — in a loop against a field that has
been **`.persist()`ed into RAM**, so every microsecond measured is pure scheduling
overhead.

## Timeline trace (VizTracer) — recommended

```bash
uv pip install --python "$PARCELS_PYTHON" viztracer   # one-time, into the parcels env
python profile_dask.py --ncalls 20 --tool viztracer
vizviewer profiling/results/dask_trace.json           # opens a Perfetto UI in the browser
```

VizTracer records every function call/return on a timeline. In the viewer, zoom
into a single `compute()` and you can watch it expand into the dask pipeline call
by call — graph construction (`HighLevelGraph`, `blockwise`/slicing), optimization
(`optimize`, `cull`, `fuse`), and the scheduler (`get_async` / `execute_task`).
That repeating block, once per gather, *is* the overhead. Keep `--ncalls` small
(~20) — the trace records everything, so it grows fast.

The JSON is a standard Chrome trace, so you can also open it at
<https://ui.perfetto.dev> or `chrome://tracing` instead of `vizviewer`.

## Function breakdown (cProfile)

```bash
python profile_dask.py --ncalls 200 --tool cprofile
snakeviz profiling/results/dask_sampling.prof          # interactive icicle
# or: gprof2dot -f pstats profiling/results/dask_sampling.prof | dot -Tsvg -o callgraph.svg
```

Deterministic, attributes wall time to named functions. The script prints the top
functions by **self time** (where the CPU actually is) and by **cumulative time**
(the call hierarchy), filtered to dask/xarray. See
[`results/cprofile_top.txt`](results/) for a captured run.

## Sampling trace (py-spy) — zero-overhead, captures native frames

cProfile and VizTracer only see Python frames; py-spy samples the whole process
(including NumPy/Cython C frames) with negligible overhead and emits a speedscope
trace:

```bash
pixi exec --spec py-spy py-spy record --format speedscope \
    -o profiling/results/dask_sampling.speedscope.json \
    -- "$PARCELS_PYTHON" profile_dask.py --ncalls 500 --tool none
# then drag the JSON onto https://www.speedscope.app
```

(`--tool none` makes the script just run the loop; py-spy profiles it externally.)

## Reading the result

All three converge on the same story: with the data in RAM, time is spent
**constructing and scheduling a fresh task graph on every `.compute()`** — there
is no array work to speak of. That is why the fix is structural (load the needed
window into NumPy, or batch many gathers into one graph), not faster storage. See
[`../03_parcels/README.md`](../03_parcels/README.md) for the options.
