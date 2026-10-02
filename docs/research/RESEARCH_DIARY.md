# Master's Thesis Research Diary
**Project:** Quantum Lattice Boltzmann Method (QLBM) for Two-Phase Gas-Liquid Flow and Dam-Break Problems  
**Repository:** `~/Research/qlbm`  
**Branch:** `research/multiphase-vof`  

---

## Phase 0: Project Audit & Codebase Inspection

### Date
2026-10-02

### Objective
Systematically inspect the complete `qlbm` repository structure, catalog all lattice, encoding, streaming, collision, boundary condition, geometry, and infrastructure components, search for collision implementations and multiphase/relaxation/equilibrium keywords, classify functionality into IMPLEMENTED / PARTIALLY IMPLEMENTED / ONLY DOCUMENTED / ABSENT, and produce `docs/research/00_repository_audit.md`.

### Files Inspected
- `pyproject.toml`, `Makefile`, `README.md`
- `docs/source/refs.bib`, `docs/source/index.rst`, `docs/source/code/*.rst`
- `qlbm/lattice/lattices/base.py`, `ms_lattice.py`, `ab_lattice.py`, `oh_lattice.py`, `spacetime_lattice.py`, `lqlga_lattice.py`
- `qlbm/lattice/spacetime/properties_base.py`, `d1q2.py`, `d2q4.py`, `d3q6.py`
- `qlbm/lattice/geometry/shapes/base.py`, `block.py`, `circle.py`, `ymonomial.py`
- `qlbm/lattice/geometry/encodings/ms.py`, `spacetime.py`, `lqlga.py`
- `qlbm/lattice/eqc/eqc.py`, `eqc_generator.py`
- `qlbm/components/base.py`, `cqlbm.py`
- `qlbm/components/ms/msqlbm.py`, `streaming.py`, `primitives.py`, `bounceback_reflection.py`, `specular_reflection.py`
- `qlbm/components/ab/ab.py`, `streaming.py`, `averaged_collision.py`, `initial.py`, `reflection/standard_reflection.py`, `reflection/agnosotic_reflection.py`
- `qlbm/components/spacetime/spacetime.py`, `collision/eqc_collision.py`, `collision/d2q4_old.py`
- `qlbm/components/lqlga/lqlga.py`, `collision.py`
- `qlbm/components/common/cbse_collision/cbse_collision.py`, `cbse_permutation.py`, `cbse_redistribution.py`
- `qlbm/components/common/adders.py`, `comparators.py`, `primitives.py`
- `qlbm/infra/compiler.py`, `runner/qiskit_runner.py`, `runner/qulacs_runner.py`, `simulation_config.py`
- `demos/simulation/ms_simulation.ipynb`
- `test/e2e/test_ms_e2e.py`, `test_ab_e2e.py`

### Files Created
- `docs/research/00_repository_audit.md`
- `docs/research/RESEARCH_DIARY.md`

### Commands Executed
- `git status && git branch --show-current`
- `git branch -a`
- `git checkout -b research/multiphase-vof`
- `.venv/bin/python --version && .venv/bin/pytest --version`
- `.venv/bin/pip list`
- `find qlbm -maxdepth 4`
- `find test demos docs -maxdepth 3`
- `grep -R "class .*Collision" -n qlbm`
- `grep -rnE "(BGK|relaxation|equilibrium|D2Q4|D2Q9|D3Q19|multiphase|...)" ...`
- `.venv/bin/pytest` (full test suite execution in progress)

### Tests Executed
Full repository test suite executed on `.venv/bin/pytest`:
- **939 passed**, 493 warnings in 255.07s (0:04:15).
- Confirms baseline integrity of all existing unit, integration, e2e, and regression tests.

### Results
1. Collision implementation search revealed 5 classes:
   - `ABEAveragedCollisionOperator`: Truncated QFT on velocity qubits in D1Q3 only (marked `WIP`). Not BGK.
   - `EQCCollisionOperator`: Permutation-Redistribution-Permutation unitary mixing using DFT blocks for equivalence classes of identical mass/momentum (Georgescu et al. 2026). Not BGK.
   - `GenericSpaceTimeCollisionOperator`: Applies `EQCCollisionOperator` to space-time velocity qubits.
   - `SpaceTimeD2Q4CollisionOperator`: Pairwise $R_y(\theta)$ rotation between opposite states $|0101\rangle \leftrightarrow |1010\rangle$.
   - `GenericLQLGACollisionOperator`: Applies `EQCCollisionOperator` to local QLGA velocity qubits.
2. Search for classical LBM / multiphase concepts:
   - `BGK`: ABSENT
   - `relaxation`: ABSENT
   - `equilibrium`: ABSENT
   - `multiphase`, `phase field`, `pseudopotential`, `free energy`, `Shan-Chen`, `color gradient`, `entropic`, `volume of fluid`, `VOF`, `surface tension`: ALL ABSENT.
3. Multi-Speed QLBM (`MSQLBM`) and Amplitude-Based QLBM (`ABQLBM`) are verified to be collisionless transport algorithms composed of streaming, boundary reflection, and ancilla reset.
4. Working branch `research/multiphase-vof` created to ensure safe research progression without modifying `main`.

### Scientific Interpretation
The current state of the repository represents state-of-the-art quantum transport and lattice gas automata building blocks (multi-speed streaming, QFT adders, comparator-based boundary reflections, and unitary equivalence-class mixing). However, it does not yet have:
- Classical LBM baseline with BGK collision and D2Q9 equilibrium.
- A quantum formulation of non-unitary or dissipative BGK relaxation.
- Any multiphase physics or interface tracking mechanism (VoF).

Therefore, building toward a two-phase gas-liquid dam-break solver requires a disciplined, step-by-step path:
First deeply mastering the existing collisionless MS building blocks, establishing a transparent classical D2Q9 reference, designing a validated quantum collision prototype, selecting a technically compatible multiphase model, coupling it to classical/hybrid VoF, and testing on a small, reproducible dam-break proof-of-concept.

### What is Validated
- Repository structure, dependencies, and environment (Python 3.14.4, Qiskit 2.1.0, Aer 0.17.2, Qulacs 0.6.14).
- The exact inventory of all lattice models, encodings, streaming operators, boundary reflections, and collision components.
- The fundamental fact that MSQLBM and ABQLBM are collisionless algorithms.

### What is Not Validated
- Full collision dynamics for D2Q9 (absent).
- Multiphase models, surface tension, and Volume-of-Fluid methods (absent).

### Problems
None at this stage. The codebase is clean, well-tested, and its algorithmic scope is now thoroughly understood and documented.

### Next Step
Proceed to **PHASE 1 — UNDERSTAND MSLattice** (Completed).

---

## Phase 1: MSLattice Register Allocation & Encoding Formulation

### Date
2026-10-02

### Objective
Deeply inspect `MSLattice` register structure, define the exact qubit layout, endianness conventions, and basis-state mathematical mapping for 2D and 3D configurations, author `docs/research/01_ms_lattice_encoding.md`, and implement/validate computational basis state diagnostic tests in `tests/research/test_ms_register_encoding.py`.

### Files Inspected
- `qlbm/lattice/lattices/ms_lattice.py`
- `qlbm/lattice/lattices/base.py`
- `test/e2e/utils.py`
- `test/unit/lattice/ms_lattice_test.py`
- `test/e2e/test_ms_e2e.py`

### Files Created
- `docs/research/01_ms_lattice_encoding.md`
- `tests/research/test_ms_register_encoding.py`

### Commands Executed
- `.venv/bin/pytest tests/research/test_ms_register_encoding.py -v`
- `.venv/bin/pytest test/unit/lattice/ms_lattice_test.py test/e2e/test_ms_e2e.py -v`

### Tests Executed
1. `tests/research/test_ms_register_encoding.py`:
   - `TestMSLatticeRegisterSizing::test_2d_8x8_no_obstacle_counts` PASSED
   - `TestMSLatticeRegisterSizing::test_2d_8x8_specular_obstacle_counts` PASSED
   - `TestMSLatticeRegisterSizing::test_3d_8x8x8_specular_obstacle_counts` PASSED
   - `TestMSLatticeIndexing::test_2d_8x8_index_partition` PASSED
   - `TestMSLatticeIndexing::test_indexing_exceptions` PASSED
   - `TestMSBasisStateEncoding::test_encode_and_decode_single_particles` PASSED
   - `TestMSEndianConvention::test_grid_register_bit_endianness` PASSED
   - Summary: **7 passed in 3.02s**.
2. Existing MS Unit & End-to-End Tests:
   - `test/unit/lattice/ms_lattice_test.py` (31 tests) PASSED
   - `test/e2e/test_ms_e2e.py` (9 tests) PASSED
   - Summary: **40 passed in 2.25s**.

### Results
1. **Exact Register Layout & Sizing:**
   - Registers are allocated in contiguous sequence: $(a_v, a_o, a_c, g_0 \dots g_{d-1}, v_0 \dots v_{d-1}, vd_0 \dots vd_{d-1})$.
   - For an $8 \times 8$ 2D lattice with 4 velocities/dim:
     - No obstacles or bounce-back only: $a_v$ (2) + $a_o$ (1) + $a_c$ (2) + $g_x$ (3) + $g_y$ (3) + $v_x$ (1) + $v_y$ (1) + $vd_x$ (1) + $vd_y$ (1) = **15 qubits**.
     - With specular obstacle: $a_o$ expands from 1 to 2 ($d=2$) $\implies$ **16 qubits**.
   - For an $8 \times 8 \times 8$ 3D lattice with 4 velocities/dim:
     - Specular obstacle: $a_v$ (3) + $a_o$ (3) + $a_c$ (4) + $g$ (9) + $v$ (3) + $vd$ (3) = **25 qubits**.
2. **Qubit Index Mapping:**
   - 2D (8x8, no obstacle):
     $a_v \in [0, 1]$, $a_o \in [2]$, $a_c \in [3, 4]$, $g_x \in [5, 6, 7]$, $g_y \in [8, 9, 10]$, $v_x \in [11]$, $v_y \in [12]$, $vd_x \in [13]$, $vd_y \in [14]$.
3. **Endianness:**
   - Little-endian within each register: qubit index $k = 0$ corresponds to bit 0 ($2^0$, LSB).
   - In global Qiskit statevector index: qubit 0 is LSB, qubit $N_{\text{total}}-1$ is MSB.
4. **Physical Interpretation:**
   - $v \in \{0, 1\}$ represents speed tiers (0 = slow speed, 1 = fast speed). MS has NO rest state ($c=0$).
   - $vd \in \{0, 1\}$ represents direction sign (1 = positive $+$, 0 = negative $-$).

### Scientific Interpretation
`MSLattice` is an amplitude-encoding data structure designed for Cartesian tensor-product multi-speed transport, not isotropic $D_d Q_q$ velocity stencils. A 2D 4-velocity lattice represents 16 discrete velocities $(\pm 1, \pm 3) \times (\pm 1, \pm 3)$. The encoding is completely contiguous, deterministic, and reversible.

### What is Validated
- Contiguous register partitioning without gaps or overlaps.
- Bit-level Little-Endian representation and integer basis-state decoding.
- Dynamic expansion of obstacle ancillae ($a_o$) for specular vs bounceback.
- 100% test pass on both new diagnostic tests and existing MS tests.

### What is Not Validated
- Multi-speed collision (MSQLBM remains strictly collisionless).
- D2Q9 lattice representation in MS (D2Q9 is handled separately by `ABLattice` / `OHLattice`).

### Problems
None. Diagnostic suite runs in ~3 seconds with zero failures.

### Next Step
Proceed to **PHASE 2 — MSQLBM STREAMING MATHEMATICS**:
1. Deeply study `qlbm/components/ms/streaming.py` (`StreamingAncillaPreparation`, `ControlledIncrementer`, `MSStreamingOperator`).
2. Derive mathematical formulation of QFT-based position shifts and CFL velocity scheduling.
3. Author `docs/research/02_ms_streaming_mathematics.md`.
4. Implement deterministic validation tests for 2D streaming directions $(\pm 1, \pm 1)$ and velocity tiers.

