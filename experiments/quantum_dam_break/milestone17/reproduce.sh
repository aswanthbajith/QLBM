#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
python milestone17/run.py
python milestone17/run_coherent.py | tee milestone17/results/coherent_execution.txt
python milestone17/plot.py
python -m pytest -q milestone17/tests milestone16/tests | tee milestone17/results/test_report.txt
if [[ ${1:-} == --full-regression ]]; then
  python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}/tests | tee milestone17/results/full_regression.txt
fi
