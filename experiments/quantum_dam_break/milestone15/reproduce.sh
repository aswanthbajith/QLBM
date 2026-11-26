#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
python milestone15/run.py
python milestone15/run.py --scale 64 --out milestone15/results/precision64
python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15}/tests | tee milestone15/results/test_report.txt
