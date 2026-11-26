#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python milestone26/capture.py
python milestone26/render.py all
python milestone26/geometry.py
python -m pytest -q milestone26/tests | tee milestone26/results/test_report.txt
if [[ ${1:-} == --full-regression ]]; then
  python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}/tests | tee milestone26/results/full_regression.txt
fi

python milestone26/verify_outputs.py
