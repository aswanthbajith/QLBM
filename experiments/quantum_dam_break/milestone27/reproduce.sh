#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python milestone27/run.py reference
python milestone27/run.py reference --zero-gravity
python milestone27/run.py gate
python milestone27/compare.py
python milestone27/render.py all
python milestone27/geometry.py
python -m pytest -q milestone27/tests | tee milestone27/results/test_report.txt
if [[ ${1:-} == --full-regression ]]; then
 python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}/tests | tee milestone27/results/full_regression.txt
fi

python milestone27/verify_outputs.py
