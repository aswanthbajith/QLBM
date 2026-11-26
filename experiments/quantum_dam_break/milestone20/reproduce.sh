#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python milestone20/run.py
python milestone20/plot.py
python -m pytest -q milestone18/tests milestone19/tests milestone20/tests | tee milestone20/results/test_report.txt
if [[ ${1:-} == --full-regression ]]; then
 python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}/tests | tee milestone20/results/full_regression.txt
fi
