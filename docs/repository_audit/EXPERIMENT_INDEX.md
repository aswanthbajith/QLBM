# Master Experiment Index & Implementation Status

**Date**: 2026-10-29  

This document tracks all **7 confirmed experimental tracks** in `qlbm`. For each experiment, we record its scientific purpose, exact implementation status, entry points, tests, documentation, limitations, and verified reproduction commands.

---

## Experiment 1: Two-Phase Free-Surface Dam-Break (VOF-LBM Hybrid)

- **Scientific Purpose**: Simulates 2D rectangular water column collapse under gravity against solid walls using the Volume-of-Fluid (VOF) method coupled to quantum Draper-QFT streaming.
- **Implementation Status**: **Implemented and rigorously validated** (Option A Modular Hybrid Architecture).
- **Primary Source Modules**: `qlbm/classical/vof.py`, `qlbm/classical/hybrid_vof_lbm.py`, `qlbm/components/vof/` (`streaming.py`, `boundary.py`, `collision.py`, `hybrid_stepper.py`)
- **Shared Core Dependencies**: `qlbm/classical/d2q9.py`, `qlbm/components/common/adders.py`, `qlbm/lattice/lattices/ab_lattice.py`, `qlbm/tools/utils.py`
- **Demonstration & Runner Scripts**: `demos/simulation/vof_dam_break.py`, `demos/visualization/visualize_dam_break.py`, `demos/simulation/make_dam_break_gif.py`
- **Test Suites**: `tests/research/test_dam_break_validation.py` (12 tests), `test_quantum_vof.py` (8 tests), `test_vof_lbm_hybrid.py` (8 tests), `test_vof_reconstruction.py` (8 tests)
- **Research Monographs & Documentation**: `docs/research/dam_break/INDEPENDENT_AUDIT.md`, `IMPLEMENTATION.md`, `VALIDATION_REPORT.md`, `RESULTS.md`, `11_vof_*.md`, `12_vof_*.md`
- **Generated Results**: `results/dam_break/`, `results/dam_break_final/`, `results/test_modes/`
- **Known Limitations**: Statevector simulation of Draper adder on CPU is ~3,500x slower than classical streaming. Quantum bounce-back operator is defined but bypassed in hybrid step.
- **Verified Execution Command**:
  ```bash
  python demos/simulation/vof_dam_break.py --mode hybrid --nx 16 --ny 8 --steps 20 --output-dir results/dam_break
  ```

---

## Experiment 2: Multiphase Shan-Chen & Young-Laplace Surface Tension

- **Scientific Purpose**: Simulates mesoscopic phase separation, pseudopotential inter-particle attraction, and Laplace pressure jump across static fluid droplets.
- **Implementation Status**: **Implemented and tested** (Classical reference).
- **Primary Source Modules**: `qlbm/classical/multiphase.py`
- **Shared Core Dependencies**: `qlbm/classical/d2q9.py`, `qlbm/tools/utils.py`
- **Demonstration & Runner Scripts**: Embedded in test benchmarks
- **Test Suites**: `tests/research/test_shan_chen.py` (8 tests), `test_laplace_surface_tension.py` (6 tests)
- **Research Monographs & Documentation**: `docs/research/08_multiphase_lbm_survey.md`, `09_shan_chen_model.md`, `10_surface_tension_laplace.md`
- **Generated Results**: Recorded in test diagnostics
- **Known Limitations**: Classical diffuse-interface model; does not have an active quantum circuit implementation.
- **Verified Execution Command**:
  ```bash
  pytest tests/research/test_shan_chen.py tests/research/test_laplace_surface_tension.py -v
  ```

---

## Experiment 3: Classical D2Q9 Hydrodynamic Benchmark

- **Scientific Purpose**: Standard 9-velocity single-phase LBM solver with BGK collision, Guo second-order forcing, and Chapman-Enskog expansion to incompressible Navier-Stokes.
- **Implementation Status**: **Implemented and rigorously validated** (Reference baseline).
- **Primary Source Modules**: `qlbm/classical/d2q9.py`, `qlbm/classical/solver.py`, `qlbm/classical/__init__.py`
- **Shared Core Dependencies**: `qlbm/tools/utils.py`
- **Demonstration & Runner Scripts**: Benchmarked through research test suite
- **Test Suites**: `tests/research/test_classical_d2q9.py` (12 tests), `test_d2q9_bgk_relaxation.py` (8 tests), `test_d2q9_gravity_forcing.py` (6 tests), `test_incompressible_limit.py` (6 tests)
- **Research Monographs & Documentation**: `docs/research/04_classical_lbm_reference.md`, `05_d2q9_bgk.md`, `06_moments_and_incompressible_limit.md`, `07_gravity_and_forcing.md`
- **Generated Results**: Embedded in validation tests
- **Known Limitations**: Single-phase classical reference.
- **Verified Execution Command**:
  ```bash
  pytest tests/research/test_classical_d2q9.py tests/research/test_d2q9_bgk_relaxation.py -v
  ```

---

## Experiment 4: Multi-Speed (MS) QLBM (Toda & Lubich Scheme)

- **Scientific Purpose**: Multi-speed quantum lattice Boltzmann algorithm for multiple velocity tiers with subcycling streaming and specular wall reflections.
- **Implementation Status**: **Implemented and tested** (Quantum circuit simulation).
- **Primary Source Modules**: `qlbm/components/ms/` (`msqlbm.py`, `streaming.py`, `specular_reflection.py`, `bounceback_reflection.py`), `qlbm/lattice/lattices/ms_lattice.py`
- **Shared Core Dependencies**: `qlbm/components/base.py`, `qlbm/lattice/lattices/base.py`, `qlbm/infra/compiler.py`
- **Demonstration & Runner Scripts**: `demos/simulation/ms_simulation.ipynb`
- **Test Suites**: `tests/research/test_ms_register_encoding.py` (7 tests), `test_ms_streaming_mathematics.py` (10 tests), `test_ms_boundary_conditions.py` (5 tests), `test/e2e/test_ms_e2e.py`
- **Research Monographs & Documentation**: `docs/research/01_ms_lattice_encoding.md`, `02_ms_streaming_mathematics.md`, `03_ms_boundary_conditions.md`
- **Generated Results**: `demos/simulation/qlbm-output/ms-4x4-16x16-1-obstacle-qiskit/`, `qlbm_ms_evolution.gif`
- **Known Limitations**: Collisionless streaming/reflection circuits; collision is treated as a separate operator.
- **Verified Execution Command**:
  ```bash
  pytest tests/research/test_ms_*.py test/e2e/test_ms_e2e.py -v
  ```

---

## Experiment 5: Space-Time QLBM & Reinitialization (Mezzacapo Scheme)

- **Scientific Purpose**: Space-time quantum lattice Boltzmann model where time evolution is mapped directly into circuit depth, with mid-circuit state reset and reinitialization.
- **Implementation Status**: **Implemented and tested** (Quantum circuit simulation & hardware experiment decoding).
- **Primary Source Modules**: `qlbm/components/spacetime/` (`spacetime.py`, `streaming.py`, `collision/`, `initial/`, `reflection/`), `qlbm/lattice/spacetime/`, `qlbm/infra/reinitialize/spacetime_reinitializer.py`
- **Shared Core Dependencies**: `qlbm/components/base.py`, `qlbm/lattice/lattices/base.py`, `qlbm/infra/compiler.py`
- **Demonstration & Runner Scripts**: `demos/simulation/spacetime_simulation.ipynb`, `demos/visualization/spacetime_components.ipynb`, `visualize_spacetime.py`, `research_experiments/spacetime_reinitialization/*`
- **Test Suites**: `test/e2e/test_spacetime_e2e.py`, `test/unit/spacetime/` (11 unit tests)
- **Research Monographs & Documentation**: `docs/research/00_repository_audit.md`, `docs/source/code/comps_other.rst`
- **Generated Results**: `demos/simulation/qlbm-output/spacetime-d2q4-8x8-qiskit/`, `qlbm_spactime.gif`
- **Known Limitations**: Circuit depth scales rapidly with physical time steps without mid-circuit reset.
- **Verified Execution Command**:
  ```bash
  pytest test/e2e/test_spacetime_e2e.py test/unit/spacetime/ -v
  ```

---

## Experiment 6: Lattice Quantum Linear Gas Automaton (LQLGA)

- **Scientific Purpose**: Quantum cellular automaton formulation with Fermi-Dirac-like exclusion, permutation-based quantum collision operators, and particle-conserving streaming circuits.
- **Implementation Status**: **Implemented and tested** (Quantum circuit simulation).
- **Primary Source Modules**: `qlbm/components/lqlga/` (`lqlga.py`, `collision.py`, `streaming.py`, `reflection.py`), `qlbm/lattice/lattices/lqlga_lattice.py`
- **Shared Core Dependencies**: `qlbm/components/base.py`, `qlbm/lattice/lattices/base.py`, `qlbm/infra/compiler.py`
- **Demonstration & Runner Scripts**: `demos/simulation/lqlga_simulation.ipynb`, `demos/visualization/lqlga_components.ipynb`, `research_experiments/lqlga/experiment_d1q3.py`
- **Test Suites**: `test/e2e/test_lqlga_e2e.py`, `test/unit/lqlga/` (6 unit tests)
- **Research Monographs & Documentation**: `docs/source/code/comps_lga.rst`
- **Generated Results**: `demos/simulation/qlbm-output/lqlga-d1q3-7-qiskit/`
- **Known Limitations**: Restricted to 1D and simple lattices; linear gas dynamics.
- **Verified Execution Command**:
  ```bash
  pytest test/e2e/test_lqlga_e2e.py test/unit/lqlga/ -v
  ```

---

## Experiment 7: Amplitude-Based (AB) Collisionless & Linearization

- **Scientific Purpose**: Draper QFT adder streaming on continuous coordinate registers, specular and curved (Y-monomial) boundary reflections, and quantum collision linearization.
- **Implementation Status**: **Implemented and tested** (Streaming & reflection verified; collision linearization experimental).
- **Primary Source Modules**: `qlbm/components/ab/` (`ab.py`, `streaming.py`, `encodings.py`, `initial.py`, `reflection/`), `qlbm/lattice/lattices/ab_lattice.py`, `oh_lattice.py`
- **Shared Core Dependencies**: `qlbm/components/base.py`, `qlbm/components/common/adders.py`, `qlbm/infra/compiler.py`
- **Demonstration & Runner Scripts**: `demos/simulation/ab_simulation.ipynb`, `ab_simulation_parallel_bcs.ipynb`, `ab_simulation_y_monomial.ipynb`, `demos/visualization/amplitude_components.ipynb`
- **Test Suites**: `test/e2e/test_ab_e2e.py`, `test/unit/ab/` (10 unit tests), `tests/research/test_quantum_collision.py`, `tests/research/test_cross_paradigm_synthesis.py`
- **Research Monographs & Documentation**: `docs/research/14_quantum_collision_linearization.md`
- **Generated Results**: `demos/simulation/qlbm-output/ab-*`, `qlbm_evolution.gif`, `qlbm_evolution1.gif`, `qlbm_evolution2.gif`
- **Known Limitations**: Averaged collision operator is non-unitary; Carleman linearization requires state space truncation.
- **Verified Execution Command**:
  ```bash
  pytest test/e2e/test_ab_e2e.py test/unit/ab/ tests/research/test_quantum_collision.py -v
  ```

---
