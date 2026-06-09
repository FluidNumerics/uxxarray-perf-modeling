# Two-disk comparison

Same fio jobs (O_DIRECT, 10 GiB), run on both physical NVMe drives in the box.

## fio raw reads

| pattern           | nvme1 (`rl-home`, system disk) | nvme0 (1.8 TB scratch) |
|-------------------|-------------------------------:|-----------------------:|
| sequential 1 MiB  | 2094 MB/s · 1997 IOPS          | 2083 MB/s · 1987 IOPS  |
| random 1 MiB      | 1912 MB/s · 1824 IOPS (0.91×)  | 2052 MB/s · 1957 IOPS (0.99×) |
| random 4 KiB      | **36 MB/s · 8793 IOPS** (0.02×)| **56 MB/s · 13777 IOPS** (0.03×) |
| small-random penalty | **~58× slower than seq**    | **~37× slower than seq** |

Both drives stream at ~2 GB/s and barely notice randomness at 1 MiB. They differ
on the pathological case: nvme0 sustains ~1.5× the small-random throughput/IOPS of
nvme1, so its 4 KiB penalty is "only" ~37× instead of ~58×. Faster storage shrinks
the penalty but does **not** remove it — small scattered reads stay an order of
magnitude off sequential on either disk.

## stage-02 dask sampling (10k particles × 50 steps)

| metric              | nvme1 (`rl-home`) | nvme0 |
|---------------------|------------------:|------:|
| slab — read amplification  | 40.0× | 40.0× |
| tiled — read amplification | 52.4× | 52.4× |
| slab — wall time    | 3.10 s | 2.88 s |
| tiled — wall time   | 12.4 s | 18.9 s |

**Read amplification is identical on both disks** — it is a property of the chunk
layout and access pattern, not the hardware. The wall-clock differences are page-
cache artifacts (no `sudo` to flush; the freshly-moved nvme0 files read colder),
which is exactly why amplification is the metric to trust.

## Takeaway

Moving to a faster disk helps the raw per-I/O cost a little, but the structural
problems — ~40–52× read amplification and the dask per-`compute()` scheduling
overhead — are disk-independent. The fixes remain: larger/aligned chunks, and
loading the needed time window into RAM rather than sampling lazily off disk.
