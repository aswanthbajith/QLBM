#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for nu in .02 .01 .005 .0025 .001; do python milestone25/run.py "nu${nu}" --nu "$nu" | tee "milestone25/results/nu${nu}.log"; done
python milestone25/run.py gate --nu .01 --mode gate --vtk | tee milestone25/results/gate.log
python milestone25/run.py refined --nu .01 --refine 2 | tee milestone25/results/refined.log
python milestone25/run.py slow106 --nu .005 --slow 2 | tee milestone25/results/slow106.log
python milestone25/run.py slow212 --nu .0025 --slow 2 | tee milestone25/results/slow212.log
python milestone25/run.py slow530 --nu .001 --slow 4 | tee milestone25/results/slow530.log
python milestone25/run.py slow212_gate --nu .0025 --slow 2 --mode gate --vtk | tee milestone25/results/slow212_gate.log
python milestone25/analyze.py | tee milestone25/results/analysis.log
python -m pytest -q milestone25/tests | tee milestone25/results/test_report.txt
if [[ ${1:-} == --full-regression ]]; then
 python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}/tests | tee milestone25/results/full_regression.txt
fi
