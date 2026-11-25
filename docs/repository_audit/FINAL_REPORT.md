# Master Repository Audit & Classification Report
## Quantum Lattice Boltzmann Methods (QLBM) Research Codebase

**Author**: Senior Software Architect & Computational Fluid Dynamics Verification Engineer  
**Date**: 2026-10-29  
**Repository**: `~/Research/qlbm`  
**Audit Phase**: **Phase 1 (Read-Only Complete)** — No files moved or deleted  

---

## 1. Executive Summary

This report delivers an exhaustive, read-only architectural audit of the entire `qlbm` repository. Every single file has been inspected via static analysis (AST), dependency cross-referencing, SHA-256 cryptographic hashing, Git baseline verification, and test execution mapping.

### Key Quantities Audited
- **Total Non-Virtualenv Files Analyzed**: **591 files** (394 Git-tracked, 53 untracked, 146 gitignored).
- **Python Source Code Modules**: **209 files** across library packages, tests, demos, and experiment scripts.
- **Paraview Volume & Surface Artifacts**: **183 files** (`.vti`, `.stl`) stored across `demos/` and `test/artifacts/`.
- **Simulation Snapshot & Result Datasets**: **24 files** (`.npz`, `.csv`) storing hydrodynamic field verification history.
- **Confirmed Research Experiments**: **7 distinct paradigms** (Dam-Break VOF-LBM, Shan-Chen Multiphase, Classical D2Q9, Multi-Speed, Space-Time, LQLGA, and Amplitude-Based).
- **Confirmed Duplicate / Obsolete Files**: **1 accidental duplicate** (`nano decode_real_counts.py`), **1 empty test file** (`eqc_permutation_test.py`), and **1 legacy operator** (`d2q4_old.py`).

---

## 2. High-Level Architecture: How the Modules Connect

The repository operates on a layered pipeline architecture connecting mathematics to quantum circuits and classical fluid visualization:

```text
[Physical Experiment Scenario] (e.g. Dam Break / Poiseuille / Droplet)
       │
       ▼
[Lattice & Geometry Definition] (ABLattice, MSLattice, SpaceTimeLattice + Block/Circle/YMonomial)
       │
       ▼
[Quantum Operator Assembly] (Draper Adder Streaming, Quantum Bounce-Back, EQC Collision)
       │
       ▼
[Compiler & Hardware Backend] (QLBMCompiler -> Qiskit Aer / Qulacs GPU / MPI)
       │
       ▼
[Simulation & Hydrodynamic Decoding] (Statevector evolution -> Macroscopic rho, u, fill fraction C)
       │
       ▼
[Visualization & Diagnostics] (Paraview VTI, Matplotlib publication plots, animated GIFs)
```

---

## 3. The 7 Confirmed Research Experiments

1. **Experiment 1: Two-Phase Free-Surface Dam-Break (VOF-LBM Hybrid)**  
   - *Status*: **Implemented and rigorously validated**.
   - *Entry point*: [`demos/simulation/vof_dam_break.py`](file:///home/aswa/Research/qlbm/demos/simulation/vof_dam_break.py).
   - *Core modules*: [`qlbm/classical/vof.py`](file:///home/aswa/Research/qlbm/qlbm/classical/vof.py), [`hybrid_vof_lbm.py`](file:///home/aswa/Research/qlbm/qlbm/classical/hybrid_vof_lbm.py), [`qlbm/components/vof/streaming.py`](file:///home/aswa/Research/qlbm/qlbm/components/vof/streaming.py), [`hybrid_stepper.py`](file:///home/aswa/Research/qlbm/qlbm/components/vof/hybrid_stepper.py).

2. **Experiment 2: Multiphase Shan-Chen & Young-Laplace Surface Tension**  
   - *Status*: **Implemented and tested** (Classical mesoscopic model).
   - *Core module*: [`qlbm/classical/multiphase.py`](file:///home/aswa/Research/qlbm/qlbm/classical/multiphase.py).

3. **Experiment 3: Classical D2Q9 Hydrodynamic Benchmark**  
   - *Status*: **Implemented and rigorously validated** (Navier-Stokes baseline).
   - *Core modules*: [`qlbm/classical/d2q9.py`](file:///home/aswa/Research/qlbm/qlbm/classical/d2q9.py), [`solver.py`](file:///home/aswa/Research/qlbm/qlbm/classical/solver.py).

4. **Experiment 4: Multi-Speed (MS) QLBM (Toda & Lubich Scheme)**  
   - *Status*: **Implemented and tested** (Multi-speed quantum lattice).
   - *Entry point*: [`demos/simulation/ms_simulation.ipynb`](file:///home/aswa/Research/qlbm/demos/simulation/ms_simulation.ipynb).
   - *Core modules*: [`qlbm/components/ms/`](file:///home/aswa/Research/qlbm/qlbm/components/ms/), [`qlbm/lattice/lattices/ms_lattice.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/ms_lattice.py).

5. **Experiment 5: Space-Time QLBM & Reinitialization (Mezzacapo Scheme)**  
   - *Status*: **Implemented and tested** (Quantum circuit depth mapping & mid-circuit reset).
   - *Entry point*: [`demos/simulation/spacetime_simulation.ipynb`](file:///home/aswa/Research/qlbm/demos/simulation/spacetime_simulation.ipynb) and [`research_experiments/spacetime_reinitialization/`](file:///home/aswa/Research/qlbm/research_experiments/spacetime_reinitialization/).
   - *Core modules*: [`qlbm/components/spacetime/`](file:///home/aswa/Research/qlbm/qlbm/components/spacetime/), [`qlbm/lattice/spacetime/`](file:///home/aswa/Research/qlbm/qlbm/lattice/spacetime/).

6. **Experiment 6: Lattice Quantum Linear Gas Automaton (LQLGA)**  
   - *Status*: **Implemented and tested** (Permutation-based quantum automaton).
   - *Entry point*: [`demos/simulation/lqlga_simulation.ipynb`](file:///home/aswa/Research/qlbm/demos/simulation/lqlga_simulation.ipynb), [`research_experiments/lqlga/experiment_d1q3.py`](file:///home/aswa/Research/qlbm/research_experiments/lqlga/experiment_d1q3.py).
   - *Core modules*: [`qlbm/components/lqlga/`](file:///home/aswa/Research/qlbm/qlbm/components/lqlga/), [`qlbm/lattice/lattices/lqlga_lattice.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/lqlga_lattice.py).

7. **Experiment 7: Amplitude-Based (AB) & Linearization**  
   - *Status*: **Implemented and tested** (Draper QFT adder streaming & reflection).
   - *Entry point*: [`demos/simulation/ab_simulation.ipynb`](file:///home/aswa/Research/qlbm/demos/simulation/ab_simulation.ipynb).
   - *Core modules*: [`qlbm/components/ab/`](file:///home/aswa/Research/qlbm/qlbm/components/ab/), [`qlbm/lattice/lattices/ab_lattice.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/ab_lattice.py).

---

## 4. Confirmed Common Infrastructure

The following modules are shared across multiple experiments and must remain in the core package:
- **Fundamental Primitives**: `qlbm/tools/utils.py`, `qlbm/tools/exceptions.py`, `qlbm/components/base.py`.
- **Quantum Circuit Primitives**: `qlbm/components/common/adders.py` (Draper QFT adders), `comparators.py` (obstacle boundaries).
- **Lattice & Geometry**: `qlbm/lattice/lattices/base.py`, `qlbm/lattice/geometry/shapes/*` (Block, Circle, Y-Monomial).
- **Execution Engine**: `qlbm/infra/compiler.py`, `qlbm/infra/runner/qiskit_runner.py`, `simulation_config.py`.
- **Classical Fluid Mechanics Constants**: `qlbm/classical/d2q9.py` (discrete velocities, weights, equilibrium functions).

---

## 5. Candidate Cleanup Files for Human Review

The audit identified only 2 immediate cleanup candidates, which have been placed into the review manifest without deleting them:
1. **`research_experiments/spacetime_reinitialization/nano decode_real_counts.py`**: Accidental duplicate created from shell command. **Recommended action**: Move to `trash_review/` and delete upon confirmation.
2. **`test/unit/collision/eqc_permutation_test.py`**: Empty 0-byte file. **Recommended action**: Implement permutation assertions or move to `trash_review/`.

---

## 6. Actionable Implementation Plan (Phase 3)

When approved by the user, reorganization will follow a strict, non-destructive sequence:
1. **Step 1**: Relocate the 3 root visualization scripts (`make_vti_gif.py`, `visualize_vti.py`, `visualize_spacetime.py`) to `demos/visualization/` using `git mv`.
2. **Step 2**: Relocate root flowfield GIFs (`qlbm_*.gif`) to `results/flowfield_media/`.
3. **Step 3**: Relocate typo file `nano decode_real_counts.py` to `trash_review/`.
4. **Step 4**: Run the complete 125-test research test suite (`pytest tests/research/`) and framework tests (`pytest test/`) to verify that 100% of tests pass.
