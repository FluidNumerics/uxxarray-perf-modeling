"""Chunk size vs (a) scheduler overhead and (b) read amplification.

Larger chunks mean fewer tasks per gather -> less dask scheduler/graph overhead.
But for SCATTERED sampling they also mean each touched chunk drags more bytes off
disk per useful value -> more read amplification. This sweep measures both on the
same scattered gather so the tradeoff is explicit.

  * ms/call is on an in-RAM (.persist()ed) field  -> pure scheduler/graph cost
    (this is the side that IMPROVES with bigger chunks).
  * read amplification is analytic from the chunk layout + which chunks the
    scattered indices touch (this is the side that WORSENS with bigger chunks,
    and is what bites when the data is actually on disk).
"""
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np, dask.array as da, xarray as xr

ntime, n, npart, iters = 12, 1200, 5000, 60
rng = np.random.default_rng(0)
base = rng.standard_normal((ntime, n, n)).astype("float64")
ti = rng.integers(0, ntime-1, npart); yi = rng.integers(0, n, npart); xi = rng.integers(0, n, npart)
# two bracketing time levels, scattered horizontally (the Parcels pattern)
tt = np.concatenate([ti, ti+1]); yy = np.concatenate([yi, yi]); xx = np.concatenate([xi, xi])
sel = dict(time=xr.DataArray(tt,dims="p"), y=xr.DataArray(yy,dims="p"), x=xr.DataArray(xx,dims="p"))
useful_bytes = npart * 2 * 8  # one value per particle per time level

def timeit(fn):
    fn()
    t=time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter()-t)/iters*1e3

print(f"scattered gather: {npart} particles x 2 time levels on ({ntime},{n},{n}), in-RAM\n")
print(f"  {'chunk (t,y,x)':18}{'#chunks':>9}{'#tasks':>8}{'ms/call':>9}{'read-amp':>10}")
print("  " + "-"*56)
for c in [64, 128, 300, 600, 1200]:
    chunks = (1, c, c)
    xda = xr.DataArray(da.from_array(base, chunks=chunks), dims=("time","y","x")).persist()
    nch = ntime * int(np.ceil(n/c))**2
    ntasks = len(xda.isel(**sel).data.__dask_graph__())
    ms = timeit(lambda: xda.isel(**sel).compute(scheduler="synchronous"))
    # chunks touched by the scattered selection -> bytes dragged off disk
    cy, cx = yy//c, xx//c
    keys = np.unique(np.stack([tt, cy, cx], axis=1), axis=0)
    chunk_bytes = 1*min(c,n)*min(c,n)*8
    amp = keys.shape[0]*chunk_bytes/useful_bytes
    print(f"  {str(chunks):18}{nch:>9}{ntasks:>8}{ms:>9.2f}{amp:>9.0f}x")
del base
