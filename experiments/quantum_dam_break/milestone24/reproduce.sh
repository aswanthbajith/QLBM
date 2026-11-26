#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python milestone24/screen.py | tee milestone24/results/screen.log
python milestone24/run.py | tee milestone24/results/run.log
python milestone24/plot.py
python -m pytest -q milestone{18,19,20,21,22,23,24}/tests | tee milestone24/results/focused_regression.txt
if [[ ${1:-} == --full-regression ]]; then
 python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}/tests | tee milestone24/results/full_regression.txt
fi
