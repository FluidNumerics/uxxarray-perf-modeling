#!/usr/bin/env bash
#
# Run the whole four-stage demo end to end. Generates ~50 GB of scratch data
# under the stage data/ folders (git-ignored); set SMALL=1 for a quick
# laptop-friendly pass (~5 GB). Point the data elsewhere with the per-stage
# --out flags if your home disk is small (see README "Setup").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python interpreter that has parcels (v4) + xarray + dask + netcdf4 installed.
# Override with PARCELS_PYTHON=/path/to/python (e.g. a Parcels pixi env);
# otherwise the `python` on PATH is used -- see README "Setup".
PY="${PARCELS_PYTHON:-python}"
echo "using python: ${PY}"

if [[ "${SMALL:-0}" == "1" ]]; then
    FIO_SIZE=2G; GB=2; LAT=200; LON=500; NPART=4000; NSTEPS=25
    NX=600; NY=600; NT=40; PNPART=1000; RUN_DAYS=10
else
    FIO_SIZE=10G; GB=10; LAT=400; LON=1000; NPART=10000; NSTEPS=50
    NX=1000; NY=1000; NT=60; PNPART=2000; RUN_DAYS=20
fi

echo "##################  STAGE 01: fio  ##################"
SIZE="${FIO_SIZE}" "${HERE}/01_fio/run.sh"

echo "##################  STAGE 02: xarray / dask  ##################"
cd "${HERE}/02_xarray_dask"
"${PY}" make_dataset.py --gb "${GB}" --lat "${LAT}" --lon "${LON}" --chunk slab  --out data/ocean_slab_${GB}g.nc
"${PY}" make_dataset.py --gb "${GB}" --lat "${LAT}" --lon "${LON}" --chunk tiled --out data/ocean_tiled_${GB}g.nc
mkdir -p results
"${PY}" bench_sampling.py --file data/ocean_slab_${GB}g.nc  --npart "${NPART}" --nsteps "${NSTEPS}" --drop-caches | tee results/slab.txt
"${PY}" bench_sampling.py --file data/ocean_tiled_${GB}g.nc --npart "${NPART}" --nsteps "${NSTEPS}" --drop-caches | tee results/tiled.txt

echo "##################  STAGE 03: parcels  ##################"
cd "${HERE}/03_parcels"
"${PY}" make_fieldset.py --nx "${NX}" --ny "${NY}" --nt "${NT}" --out data/flow.nc
mkdir -p results
# Isolate dask's per-compute scheduling overhead from disk I/O (data persisted in RAM).
"${PY}" dask_overhead.py --npart "${PNPART}" --ncalls 50 | tee results/dask_overhead.txt
# The full run. NOTE: the dask-backed pass is intentionally slow (~30+ min at full
# size) -- that is the result. Use SMALL=1 or --mode memory to skip the slow pass.
"${PY}" run_parcels.py --file data/flow.nc --npart "${PNPART}" --runtime-days "${RUN_DAYS}" --mode both | tee results/parcels.txt

echo "##################  STAGE 04: production-scale Atlantic  ##################"
cd "${HERE}/04_atlantic"
ATL_GB=$([[ "${SMALL:-0}" == "1" ]] && echo 3 || echo 20)
"${PY}" make_atlantic.py --target-gb "${ATL_GB}" --out data/atlantic
mkdir -p results
"${PY}" run_atlantic.py --runtime-days 2 --npart 200 --mode both | tee results/atlantic.txt

echo "done. See README.md for interpretation."
