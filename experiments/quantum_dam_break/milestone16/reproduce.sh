#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
python milestone16/run.py
python milestone16/run_3d.py
python milestone16/plot.py
python -m pytest -q milestone16/tests | tee milestone16/results/test_report.txt
if [[ ${1:-} == --full-regression ]]; then
  python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}/tests | tee milestone16/results/full_regression.txt
fi
