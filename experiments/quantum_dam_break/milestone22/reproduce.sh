#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python milestone22/run.py
python milestone22/stability.py
python milestone22/plot.py
python -m pytest -q milestone18/tests milestone19/tests milestone20/tests milestone21/tests milestone22/tests | tee milestone22/results/focused_regression.txt
if [[ ${1:-} == --full-regression ]]; then
 python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}/tests | tee milestone22/results/full_regression.txt
fi
