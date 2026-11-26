#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python milestone28/run.py 16 | tee milestone28/results/run.log
python milestone28/render.py
python -m pytest -q milestone28/tests milestone27/tests | tee milestone28/results/test_report.txt
if [[ ${1:-} == --full-regression ]]; then
 python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28}/tests | tee milestone28/results/full_regression.txt
fi
