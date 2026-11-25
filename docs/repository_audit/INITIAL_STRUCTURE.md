# Initial Repository Structure Assessment

**Date**: 2026-10-29  
**Repository**: `~/Research/qlbm`  
**Total Tracked/Untracked Files (excl. virtualenvs & git internal)**: 591  
**Total Storage Volume**: 33.51 MB  
**Total Source/Text Lines**: 63,512 lines  

---

## 1. Top-Level Directory Distribution

| Directory / Scope | Files | Size (KB) | Text Lines | Primary Role / Description |
|---|---|---|---|---|
| `demos/` | 243 | 845.3 KB | 2,389 | Demonstration notebooks, Paraview visualization outputs, simulation benchmarks, lattice JSONs |
| `qlbm/` | 108 | 868.1 KB | 24,198 | Core Python library package (classical solvers, quantum components, lattice encodings, compiler, runners) |
| `results/` | 80 | 31496.9 KB | 9,865 | Simulation output metrics, diagnostic plots, animated GIFs, NPZ snapshot data for dam break |
| `test/` | 79 | 475.5 KB | 14,083 | Legacy framework test suite (unit, integration, regression, e2e, and Paraview artifacts) |
| `docs/` | 41 | 360.1 KB | 5,768 | Sphinx documentation, research monographs (00-15), dam-break audits, research diary |
| `tests/` | 17 | 165.1 KB | 3,668 | Research test suite (classical reference, Shan-Chen, Laplace, MS, quantum VOF, dam-break validation) |
| `root/` | 9 | 38.8 KB | 894 | Root project configuration, packaging, license, and standalone helper scripts |
| `research_experiments/` | 9 | 54.5 KB | 2,585 | Standalone experiment scripts (Space-time reinitialization, LQLGA D1Q3) |
| `Docker/` | 5 | 7.2 KB | 62 | Container configuration files for CPU and GPU execution environments |
| **TOTAL** | **591** | **34311.5 KB** | **63,512** | Entire repository snapshot |

---

## 2. File Format & Extension Breakdown

| Extension | Category | Count | Total Size (KB) | Description |
|---|---|---|---|---|
| `.py` | Python Source Code | 207 | 1588.6 KB | Core library, test suites, experiment scripts, runners |
| `.vti` | Paraview Volume Data (Generated) | 165 | 262.5 KB | VTK image data storing 2D/3D flowfield velocity/density distributions |
| `.json` | Configuration / Metrics | 61 | 250.5 KB | Lattice geometry descriptors, benchmark configurations, run metrics |
| `.md` | Markdown Documentation | 28 | 339.6 KB | Research notes, architectural specifications, validation reports |
| `.png` | Raster Figures (Generated) | 22 | 2851.2 KB | Diagnostic plots, comparison figures, snapshots |
| `.gif` | Animated Media (Generated) | 19 | 20340.9 KB | Flowfield animations, free-surface evolution videos |
| `.stl` | 3D Geometry (Generated) | 18 | 21.4 KB | Obstacle boundary meshes for Paraview visualization |
| `.npz` | NumPy Binary Data (Generated) | 18 | 8489.2 KB | Compressed hydrodynamic field snapshots and verification frames |
| `.ipynb` | Jupyter Notebooks | 15 | 90.3 KB | Interactive research experiments, component visualizations, benchmarks |
| `.rst` | reStructuredText | 10 | 29.5 KB | Sphinx documentation source files |
| `.csv` | Tabular History (Generated) | 6 | 4.8 KB | Time-series simulation diagnostics (mass, front, energy) |
| `.nblink` | Jupyter Sphinx Links | 5 | 0.3 KB | Notebook linking extensions for Sphinx HTML builder |
| `no-ext` | Build / Meta Files | 4 | 21.5 KB | Makefile, LICENSE, Dockerfiles |
| `.conf` | Logging Configuration | 4 | 1.8 KB | Python logging configuration files for benchmarks |
| `.dockerfile` | Miscellaneous | 2 | 0.8 KB | Other data files |
| `.dockerignore` | Docker Meta | 2 | 5.3 KB | Exclusion lists for container builds |
| `.cff` | Miscellaneous | 1 | 1.1 KB | Other data files |
| `.bat` | Miscellaneous | 1 | 0.8 KB | Other data files |
| `.css` | Miscellaneous | 1 | 0.7 KB | Other data files |
| `.bib` | Miscellaneous | 1 | 5.9 KB | Other data files |
| `.toml` | Miscellaneous | 1 | 4.8 KB | Other data files |

---

## 3. Current Structural Patterns & Identified Pain Points

1. **Dual Test Suite Split (`test/` vs `tests/`)**:
   - `test/` contains the legacy framework tests organized around circuit modules.
   - `tests/research/` contains the newer research test suite (16 test files, 125 tests) verifying physical and mathematical invariants.
   - *Observation*: Pytest discovers both, but running `pytest` without path filters runs across both suites. In addition, `test/artifacts/` contains 10 generated `.vti` and `.stl` files.

2. **Generated Simulation Outputs Stored Inside Demo/Source Trees**:
   - `demos/simulation/qlbm-output/` contains 95 generated `.vti` and `lattice.json` files.
   - `demos/visualization/sample-flowfield/` contains 82 generated `.vti` and `.stl` files.
   - `results/` contains 75 generated `.csv`, `.json`, `.npz`, `.gif`, and `.png` files for the dam-break experiment.
   - *Observation*: Having generated Paraview outputs scattered across `demos/` clutters source code discovery.

3. **Root Directory Clutter**:
   - Standalone scripts: `make_vti_gif.py`, `visualize_spacetime.py`, `visualize_vti.py`.
   - Generated GIFs: `qlbm_evolution.gif`, `qlbm_evolution1.gif`, `qlbm_evolution2.gif`, `qlbm_ms_evolution.gif`, `qlbm_spactime.gif` reside in the repository root.

4. **Accidental / Typo Files**:
   - `research_experiments/spacetime_reinitialization/nano decode_real_counts.py` was created accidentally via a shell typo.
   - `test/unit/collision/eqc_permutation_test.py` is an empty 0-byte file.
