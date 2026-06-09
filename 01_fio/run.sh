#!/usr/bin/env bash
#
# Sequential vs random read benchmark using fio.
#
# Produces three numbers that motivate the whole demo:
#   1. sequential 1 MiB read bandwidth      (best case)
#   2. random     1 MiB read bandwidth      (randomness cost, same block size)
#   3. random     4 KiB read bandwidth/IOPS (small scattered reads, worst case)
#
# All jobs use O_DIRECT (direct=1) so we measure the storage device rather than
# the OS page cache. With ~38 GiB of RAM a 10 GiB file would otherwise be served
# entirely from cache and every number would look identical.
#
# fio is pulled in on-demand via `pixi exec` -- nothing is installed system-wide.
#
# Override defaults with env vars:
#   TESTFILE=/path/to/file  SIZE=10G  ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${HERE}/results"
mkdir -p "${RESULTS}"

export TESTFILE="${TESTFILE:-${HERE}/fio_testfile.bin}"
export SIZE="${SIZE:-10G}"

# `pixi exec` runs fio in an ephemeral environment. Allow an already-installed
# fio to take precedence if present.
if command -v fio >/dev/null 2>&1; then
    FIO=(fio)
else
    FIO=(pixi exec --spec fio fio)
fi

echo "=== fio sequential vs random read benchmark ==="
echo "test file : ${TESTFILE}"
echo "size      : ${SIZE}"
echo "fio       : ${FIO[*]}"
echo

# --- lay out the test file once (sequential write, so it is contiguous) -------
if [[ ! -f "${TESTFILE}" ]]; then
    echo "Creating ${SIZE} test file (one-time)..."
    "${FIO[@]}" --name=layout --rw=write --bs=1M --direct=1 \
        --filename="${TESTFILE}" --size="${SIZE}" \
        --output-format=normal >/dev/null
    echo "done."
    echo
fi

run_job () {
    local jobfile="$1" tag="$2"
    echo ">>> ${tag}"
    "${FIO[@]}" "${jobfile}" --output-format=json \
        > "${RESULTS}/${tag}.json"
    # pull the headline numbers out of the JSON
    python3 - "${RESULTS}/${tag}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
j = d["jobs"][0]["read"]
bw_mb = j["bw_bytes"] / 1e6
iops = j["iops"]
lat = j.get("lat_ns", {}).get("mean", 0) / 1000.0  # us
print(f"    bandwidth : {bw_mb:8.1f} MB/s")
print(f"    IOPS      : {iops:10.0f}")
print(f"    mean lat  : {lat:8.1f} us")
PY
    echo
}

run_job "${HERE}/seqread.fio"    "seq_read_1M"
run_job "${HERE}/randread_1M.fio" "rand_read_1M"
run_job "${HERE}/randread_4k.fio" "rand_read_4k"

echo "Raw fio JSON saved under ${RESULTS}/"
echo "Summary table:"
python3 - "${RESULTS}" <<'PY'
import json, os, sys
res = sys.argv[1]
rows = []
for tag, label in [("seq_read_1M", "sequential 1 MiB"),
                   ("rand_read_1M", "random 1 MiB"),
                   ("rand_read_4k", "random 4 KiB")]:
    p = os.path.join(res, f"{tag}.json")
    if not os.path.exists(p):
        continue
    j = json.load(open(p))["jobs"][0]["read"]
    rows.append((label, j["bw_bytes"] / 1e6, j["iops"]))
seq = rows[0][1] if rows else 0
print(f"{'pattern':<18}{'MB/s':>12}{'IOPS':>14}{'vs seq':>10}")
print("-" * 54)
for label, bw, iops in rows:
    ratio = f"{bw/seq:5.2f}x" if seq else "  -  "
    print(f"{label:<18}{bw:12.1f}{iops:14.0f}{ratio:>10}")
PY
