"""Transparent dask->NumPy time-window cache behind the xarray `.isel` API.

Assumption (true in Parcels): every particle shares the simulation clock, so any
field access touches only the time level(s) bracketing "now" -- typically {ti}
or {ti, ti+1}. Under that assumption we can make the rolling window completely
transparent: wrap the lazy (dask-backed) DataArray in `WindowedArray`, and its
`.isel(...)` will

  1. find the unique time indices requested,
  2. ensure those levels are resident as NumPy (load the missing ones with a
     single bulk read each; "ensure" is the dask->NumPy step),
  3. evict (retire) any cached level below the current minimum -- stale, since
     the clock only moves forward,
  4. perform the actual (vectorised) gather on the small NumPy block.

Because the result is NumPy, Parcels' interpolator -- which does
``data.isel(sel).data.reshape(...)`` and then ``value.compute() if
is_dask_collection(value) else value`` -- transparently skips `.compute()`
entirely. So a drop-in `field.data = WindowedArray(field.data)` removes BOTH the
per-step re-reads (read amplification) AND the dask scheduling tax, with no change
to the interpolator.

Everything except `isel`/`sel` is forwarded to the wrapped DataArray, so
`.dims`, `.shape`, `.coords`, `.dtype`, etc. behave as before.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


class WindowedArray:
    def __init__(self, da: xr.DataArray, time_dim: str = "time", max_levels: int | None = None):
        if da.dims[0] != time_dim:
            raise ValueError(f"expected {time_dim!r} as the leading dim, got {da.dims}")
        self._da = da
        self._tdim = time_dim
        self._cache: dict[int, np.ndarray] = {}   # time index -> NumPy slab (other dims)
        self._max = max_levels                     # optional hard cap (else evict < min req)
        # accounting
        self.loads = 0
        self.bytes_read = 0
        self._slab_bytes = int(da.isel({time_dim: 0}).size) * da.dtype.itemsize

    # forward everything else (dims, shape, coords, dtype, name, ...)
    def __getattr__(self, name):
        return getattr(self._da, name)

    def __repr__(self):
        return f"WindowedArray(cached_levels={sorted(self._cache)}, loads={self.loads})\n{self._da!r}"

    def _ensure(self, levels: np.ndarray) -> None:
        for lvl in levels:
            lvl = int(lvl)
            if lvl not in self._cache:
                self._cache[lvl] = self._da.isel({self._tdim: lvl}).values  # bulk dask->NumPy
                self.loads += 1
                self.bytes_read += self._slab_bytes
        # retire stale levels (clock only moves forward)
        lo = int(levels.min())
        for old in [k for k in self._cache if k < lo]:
            del self._cache[old]
        # optional hard cap: keep the most recent `max_levels`
        if self._max is not None and len(self._cache) > self._max:
            for old in sorted(self._cache)[: len(self._cache) - self._max]:
                del self._cache[old]

    def isel(self, indexers: dict | None = None, **kw):
        kw = {**(indexers or {}), **kw}
        if self._tdim not in kw:
            return self._da.isel(**kw)  # no time selection -> nothing to window

        t_ind = kw[self._tdim]
        t_vals = np.asarray(t_ind.values if isinstance(t_ind, xr.DataArray) else t_ind)
        levels = np.unique(np.atleast_1d(t_vals))
        self._ensure(levels)

        # stack the needed levels into one small NumPy block and remap indices to it
        block = np.stack([self._cache[int(l)] for l in levels])  # (nlevels, *rest)
        local = xr.DataArray(np.searchsorted(levels, t_vals), dims=getattr(t_ind, "dims", ()))
        nda = xr.DataArray(block, dims=self._da.dims)            # NumPy-backed, full logical shape
        return nda.isel({**kw, self._tdim: local})               # vectorised gather in NumPy

    def sel(self, indexers: dict | None = None, **kw):
        kw = {**(indexers or {}), **kw}
        if self._tdim in kw:  # translate time labels -> positional, keep windowing
            idx = self._da.indexes[self._tdim]
            t = kw.pop(self._tdim)
            tv = np.asarray(t.values if isinstance(t, xr.DataArray) else t)
            pos = idx.get_indexer(np.atleast_1d(tv))
            kw[self._tdim] = xr.DataArray(pos, dims=getattr(t, "dims", ()))
        # remaining label-based selection is rare in the hot path; delegate via isel
        return self.isel(**kw)


# ---------------------------------------------------------------------------
def _demo() -> None:
    """Verify transparency, correctness, and bounded memory on a synthetic field."""
    import warnings; warnings.filterwarnings("ignore")
    import dask.array as da

    ntime, ndepth, nlat, nlon, npart = 20, 1, 200, 300, 1000
    rng = np.random.default_rng(0)
    base = rng.standard_normal((ntime, ndepth, nlat, nlon)).astype("float64")
    lazy = xr.DataArray(da.from_array(base, chunks=(1, ndepth, nlat, nlon)),
                        dims=("time", "depth", "lat", "lon"))
    win = WindowedArray(lazy)

    worst = 0.0
    max_cache = 0
    # walk the clock forward; several sub-steps per time level (like dt < field-dt)
    for step in range(60):
        ti = min(step // 3, ntime - 2)                 # advances every 3 steps
        yi = rng.integers(0, nlat, npart)
        xi = rng.integers(0, nlon, npart)
        zi = np.zeros(npart, dtype=int)
        n = npart
        sel = dict(
            time=xr.DataArray(np.concatenate([np.full(n, ti), np.full(n, ti + 1)]), dims="p"),
            depth=xr.DataArray(np.concatenate([zi, zi]), dims="p"),
            lat=xr.DataArray(np.concatenate([yi, yi]), dims="p"),
            lon=xr.DataArray(np.concatenate([xi, xi]), dims="p"),
        )
        got = win.isel(**sel).data                     # NumPy (transparent)
        ref = lazy.isel(**sel).data.compute()          # ground truth via dask
        worst = max(worst, float(np.abs(got - ref).max()))
        max_cache = max(max_cache, len(win._cache))

    print(f"steps: 60 over {ntime} time levels, {npart} particles")
    print(f"correctness : max |windowed - dask| = {worst:.2e}  "
          f"({'OK' if worst == 0 else 'MISMATCH'})")
    print(f"transparency: same .isel(...) API; result is NumPy "
          f"(is_dask_collection -> .compute() skipped)")
    print(f"loads       : {win.loads} level loads for 60 steps "
          f"(vs {60*2} dask gathers naively)")
    print(f"memory      : max {max_cache} time levels resident at once "
          f"(stale levels retired)")


if __name__ == "__main__":
    _demo()
