"""A rolling time-window field sampler: dask for bulk loading, NumPy for sampling.

This is the prototype of the Tier-1 mitigation from the top-level README. The key
observation is that a Lagrangian simulation only ever needs the **two time levels
bracketing the current clock**, even when the full time series is far larger than
RAM. So instead of sampling a dask array per step (which re-reads chunks and pays
the per-`compute()` scheduling tax every step), we:

  1. keep only the bracketing time level(s) resident, as plain NumPy arrays;
  2. read a new level with ONE bulk, sequential `.values` pull (dask used here, for
     what it is good at -- streaming a contiguous slab off disk);
  3. sample those NumPy arrays directly (no dask in the hot path);
  4. evict levels behind the clock, and optionally PREFETCH the next level on a
     background thread so the read overlaps integration.

`WindowSampler` is field-agnostic and Parcels-agnostic on purpose -- it isolates
the loader strategy so it can be benchmarked (see bench_windowed.py) and later
adapted into a Parcels Field cache.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import xarray as xr


class WindowSampler:
    """Hold the bracketing time levels of a set of fields as NumPy, sampled in time.

    Parameters
    ----------
    fields : dict[str, xr.DataArray]
        Lazy (dask-backed) DataArrays sharing a ``time`` axis, dims
        ``(time, depth, lat, lon)``.
    times : np.ndarray[datetime64]
        The time coordinate values.
    prefetch : bool
        If True, asynchronously load level i+2 while the clock is in [i, i+1].
    depth : int
        Depth index to sample (this prototype samples a single level).
    """

    def __init__(self, fields, times, *, prefetch=False, depth=0):
        self.fields = fields
        self.names = list(fields)
        self.times = np.asarray(times)
        self.prefetch = prefetch
        self.depth = depth
        # cache: level_index -> {name: ndarray(lat, lon)}
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self._pending: dict[int, Future] = {}
        self._pool = ThreadPoolExecutor(max_workers=1) if prefetch else None

        # accounting
        self.levels_loaded = 0
        self.bytes_read = 0
        one = next(iter(fields.values())).isel(time=0, depth=depth)
        self._slab_bytes = int(one.size) * one.dtype.itemsize * len(fields)

    # -- internals -----------------------------------------------------------
    def _read_level(self, i: int) -> dict[str, np.ndarray]:
        """Bulk, sequential read of one time level into NumPy (the dask part)."""
        out = {}
        for name, da in self.fields.items():
            out[name] = da.isel(time=i, depth=self.depth).values  # (lat, lon)
        return out

    def _bracket(self, t) -> int:
        i = int(np.searchsorted(self.times, np.datetime64(t), side="right")) - 1
        return min(max(i, 0), len(self.times) - 2)

    def _ensure(self, i: int) -> None:
        for lvl in (i, i + 1):
            if lvl in self._cache:
                continue
            if lvl in self._pending:  # finish a prefetch we already kicked off
                self._cache[lvl] = self._pending.pop(lvl).result()
            else:
                self._cache[lvl] = self._read_level(lvl)
            self.levels_loaded += 1
            self.bytes_read += self._slab_bytes
        # prefetch the next level so its read overlaps this interval's integration
        if self.prefetch:
            nxt = i + 2
            if nxt < len(self.times) and nxt not in self._cache and nxt not in self._pending:
                self._pending[nxt] = self._pool.submit(self._read_level, nxt)
        # evict everything behind the clock
        for lvl in [l for l in self._cache if l < i]:
            del self._cache[lvl]

    # -- public --------------------------------------------------------------
    def sample(self, t, yi: np.ndarray, xi: np.ndarray) -> dict[str, np.ndarray]:
        """Linear-in-time sample of every field at integer (yi, xi) positions."""
        i = self._bracket(t)
        self._ensure(i)
        t0, t1 = self.times[i], self.times[i + 1]
        tau = (np.datetime64(t) - t0) / (t1 - t0)
        lo, hi = self._cache[i], self._cache[i + 1]
        return {
            name: (1 - tau) * lo[name][yi, xi] + tau * hi[name][yi, xi]
            for name in self.names
        }

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
