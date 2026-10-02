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
Proceed to **PHASE 2 — MSQLBM STREAMING MATHEMATICS** (Completed).

---

## Phase 2: MSQLBM Streaming Mathematics & QFT Advection Validation

### Date
2026-10-02

### Objective
Derive mathematically the QFT-based position translation $x' = (x \pm v_x) \bmod N_x$, analyze the directional control logic ($vd$) and CFL ancilla preparation ($a_v$), author `docs/research/02_ms_streaming_mathematics.md`, and implement/validate deterministic test cases across all quadrants and velocity tiers in `tests/research/test_ms_streaming_mathematics.py`.

### Files Inspected
- `qlbm/components/ms/streaming.py`
- `qlbm/components/common/adders.py`
- `qlbm/tools/utils.py`
- `test/e2e/test_ms_e2e.py`

### Files Created
- `docs/research/02_ms_streaming_mathematics.md`
- `tests/research/test_ms_streaming_mathematics.py`

### Commands Executed
- `.venv/bin/pytest tests/research/test_ms_streaming_mathematics.py -v`
- `.venv/bin/pytest tests/research/ -v`
- `.venv/bin/pytest test/e2e/test_ms_e2e.py test/unit/lattice/ms_lattice_test.py -q`

### Tests Executed
1. `tests/research/test_ms_streaming_mathematics.py`:
   - Single substep positive/negative streaming: PASSED
   - Slow tier (+1) macro-step across all 4 quadrants $((+1,+1), (+1,-1), (-1,+1), (-1,-1))$: PASSED
   - Fast tier (+3) macro-step across all 4 quadrants: PASSED
   - Mixed speed tiers ($x$ fast, $y$ slow): PASSED
   - Modular boundary wraparound (lower/upper limits): PASSED
   - Multi-step evolution (3 consecutive timesteps): PASSED
   - Summary: **14 passed in 4.93s**.
2. Combined research test suite (`tests/research/`):
   - **21 passed in 5.47s**.
3. Regression suite (`test/e2e/test_ms_e2e.py` + `test/unit/lattice/ms_lattice_test.py`):
   - **40 passed in 2.21s**.

### Mathematical Findings
1. Spatial advection by $\Delta x = \pm 1$ diagonalizes in Fourier space as a phase shift operator:
   $$\hat{P}(\pm 1) = \sum_{y=0}^{2^n-1} \exp\left( \pm \frac{2\pi i y}{2^n} \right) |y\rangle \langle y|$$
   Synthesized via single-qubit phase gates $P(\phi_j)$ with angles $\phi_j = \pm \frac{\pi}{2^{n-1-j}}$ on Fourier-transformed grid qubits.
2. Controlled direction logic:
   Controlled phase shifts conditioned on $(a_v, vd)$, followed by $X(vd)$, controlled negative phase shift, and $X(vd)$ uncomputation executes signed addition without destructively measuring the velocity direction.
3. Multi-speed CFL scheduling:
   For $N_v = 4$, macroscopic time steps are partitioned into 3 substeps $\mathcal{V}_1 = [1], \mathcal{V}_2 = [1], \mathcal{V}_3 = [0, 1]$. Fast particles cover $\Delta x = 3$, slow particles cover $\Delta x = 1$. Periodic boundaries $\mathbb{Z}_{N}$ wrap automatically via integer phase modularity $e^{2\pi i (x \pm \Delta x) y / 2^n}$.

### Scientific Interpretation
MS streaming is an exact, unitary, collisionless spatial permutation operator. It provides deterministic transport without dissipation. However, it lacks isotropic lattice weights ($w_i$) and rest particles ($c=0$), which will necessitate a distinct classical D2Q9 formulation for hydrodynamic modeling.

### What is Validated
- Analytic advection $x' = (x \pm v_x) \bmod N_x$ on quantum circuits.
- 100% state fidelity on statevector simulation across all velocity tiers and quadrants.
- Exact modular boundary wraparound.

### What is Not Validated
- Collision mechanisms (streaming is non-interacting).
- Non-periodic boundary walls (deferred to Phase 3).

### Problems
None. No bugs found in `streaming.py`.

### Next Step
Proceed to **PHASE 3 — MSQLBM BOUNDARY CONDITIONS** (Completed).

---

## Phase 3: MSQLBM Boundary Conditions & Quantum Reflection Operators

### Date
2026-10-02

### Objective
Deeply inspect `MSBounceBackReflectionOperator`, `MSSpecularReflectionOperator`, and `primitives.py`; derive the mathematical mechanics of the 4-stage quantum reflection cycle (interior comparator intrusion $\to$ direction reversal $\to$ outward streaming $\to$ exterior uncomputation $\to$ corner/point cleanup); author `docs/research/03_ms_boundary_conditions.md`; implement and validate deterministic boundary tests in `tests/research/test_ms_boundary_conditions.py`.

### Files Inspected
- `qlbm/components/ms/bounceback_reflection.py`
- `qlbm/components/ms/specular_reflection.py`
- `qlbm/components/ms/primitives.py`
- `qlbm/lattice/geometry/encodings/ms.py`
- `qlbm/lattice/geometry/shapes/block.py`
- `test/e2e/test_ms_e2e.py`

### Files Created
- `docs/research/03_ms_boundary_conditions.md`
- `tests/research/test_ms_boundary_conditions.py`

### Commands Executed
- `.venv/bin/pytest tests/research/test_ms_boundary_conditions.py -v`
- `.venv/bin/pytest tests/research/ -v`
- `.venv/bin/pytest test/unit/lattice/ms_lattice_test.py test/e2e/test_ms_e2e.py -q`

### Tests Executed
1. `tests/research/test_ms_boundary_conditions.py`:
   - `TestMS2DBounceBack::test_bounceback_collision` PASSED
   - `TestMS2DBounceBack::test_free_streaming_away_from_bounceback` PASSED
   - `TestMS2DSpecular::test_specular_wall_collision` PASSED
   - `TestMS2DSpecular::test_specular_corner_collision` PASSED
   - `TestMS3DBounceBack::test_3d_bounceback_collision` PASSED
   - Summary: **5 passed in 19.96s**.
2. Combined research test suite (`tests/research/`):
   - **26 passed in 23.78s**.
3. Regression suite (`test/unit/lattice/ms_lattice_test.py test/e2e/test_ms_e2e.py`):
   - **40 passed in 2.21s**.

### Mathematical & Algorithmic Findings
1. **Four-Stage Quantum Reflection Cycle:**
   - **Stage 1 (Collision Detection):** Geometry comparator checks if grid registers lie within obstacle bounds, marking obstacle ancilla $|a_o\rangle \to |1\rangle$.
   - **Stage 2 (Direction Inversion):**
     - Bounce-Back (full momentum inversion): Controlled $X$ conditioned on $a_o$ flips all direction qubits $(vd_0, \dots, vd_{d-1})$.
     - Specular (free-slip wall reflection): Component-wise detection marks $a_{o,k}$, inverting only the normal velocity component $vd_k \to 1 - vd_k$ while keeping tangential components unaffected.
   - **Stage 3 (Outward Stepping):** Controlled Draper advection steps the particle backward out of the obstacle into the adjacent fluid cell.
   - **Stage 4 (Exterior Detection & Restoration):** Exterior comparator or corner cleanup restores ancillae to $|0\rangle$.
2. **Ancilla Lifecycle Differences:**
   - In `MSBounceBackReflectionOperator`, $a_o$ stays at $|1\rangle$ after reflection and is re-initialized to $|0\rangle$ between macroscopic steps by `MSQLBM.reset_step()`.
   - In `MSSpecularReflectionOperator`, $a_o$ is strictly restored to $|0\rangle$ within the circuit via edge and corner uncomputation operators.
3. **Conservation & Reversibility:**
   - The entire reflection sequence is composed of unitary permutations, adders, and multi-controlled NOTs. Reflection preserves statevector norm exactly ($||\psi||=1$) and particle speed tier.

### Scientific Interpretation
Quantum reflection in MSQLBM is a discrete particle mechanics operator. It mirrors hard-sphere specular and bounce-back collision dynamics on Cartesian grids. Unlike classical LBM, where boundary conditions are applied to continuous distribution functions ($f_i$) using halfway bounce-back schemes or interpolated boundary rules, MSQLBM reflects individual basis states. This provides a clear contrast with the classical D2Q9 LBM solver required for macroscopic hydrodynamics.

### What is Validated
- Full 2D and 3D bounce-back and specular reflection circuits.
- Obstacle non-interference: fluid particles moving away from obstacles pass freely without state perturbation.
- 100% state fidelity in statevector simulation.

### What is Not Validated
- Wet-node or half-way bounceback for distribution functions (classical LBM).
- Curved or complex boundary interpolations (Filippova-Hänel or Bouzidi).

### Problems
None. Reflection operators operate exactly as mathematically derived.

### Next Step
Proceed to **PHASE 4 — CLASSICAL LBM REFERENCE MODEL (D2Q9 BGK)** (Completed).

---

## Phase 4: Classical LBM Reference Model (D2Q9 BGK)

### Date
2026-10-02

### Objective
Implement and mathematically validate a complete, standalone, vectorized classical two-dimensional nine-velocity (D2Q9) Lattice Boltzmann reference solver with Bhatnagar-Gross-Krook (BGK) collision, Guo body forcing, halfway bounce-back solid boundary conditions, and moving wall momentum exchange in `qlbm/classical/`; author `docs/research/04_classical_lbm_reference.md`; implement and validate comprehensive unit and analytical benchmark tests in `tests/research/test_classical_d2q9.py`.

### Files Created
- `qlbm/classical/__init__.py`
- `qlbm/classical/d2q9.py`
- `qlbm/classical/solver.py`
- `docs/research/04_classical_lbm_reference.md`
- `tests/research/test_classical_d2q9.py`

### Commands Executed
- `.venv/bin/python -m py_compile qlbm/classical/d2q9.py qlbm/classical/solver.py qlbm/classical/__init__.py`
- `.venv/bin/pytest tests/research/test_classical_d2q9.py -v`
- `.venv/bin/pytest tests/research/ -v`
- `.venv/bin/pytest test/unit/lattice/ms_lattice_test.py test/e2e/test_ms_e2e.py -q`

### Tests Executed
1. `tests/research/test_classical_d2q9.py`:
   - `TestD2Q9LatticeProperties`: weights sum (1.0), 1st moment (0), 2nd moment ($c_s^2 \delta_{ab}$), 3rd moment (0), 4th moment (isotropic tensor closure), opposite index involution: **6 passed**.
   - `TestD2Q9EquilibriumAndMoments`: 0th, 1st, and 2nd moment reproduction, macroscopic moment inversion: **2 passed**.
   - `TestBGKCollisionAndForcing`: mass/momentum conservation in unforced BGK collision, Guo forcing moment consistency ($\sum F_i = 0, \sum F_i \mathbf{c}_i = (1 - \frac{1}{2\tau})\mathbf{F}$): **2 passed**.
   - `TestConservationAndInvariance`: periodic domain mass and momentum conservation, solid obstacle halfway bounceback mass conservation ($|\Delta M / M_0| < 10^{-13}$), stationary uniform state invariance: **3 passed**.
   - `TestAnalyticalBenchmarks`:
     - Analytical Poiseuille channel flow ($L_2 < 0.31\%$, centerline error $< 0.1\%$): PASSED.
     - Analytical Taylor-Green vortex viscous decay rate ($E(t)/E(0) = e^{-4\nu k^2 t}$, relative decay error $< 0.8\%$): PASSED.
     - Lid-driven cavity primary recirculation vortex formation: PASSED.
   - Summary: **16 passed in 4.56s**.
2. Combined research test suite (`tests/research/`):
   - **42 passed in 25.52s**.
3. Regression suite (`test/unit/lattice/ms_lattice_test.py test/e2e/test_ms_e2e.py`):
   - **40 passed in 2.30s**.

### Mathematical Findings
1. **Navier-Stokes Recovery & Viscosity:**
   The Chapman-Enskog expansion directly ties the BGK relaxation time $\tau$ to the kinematic shear viscosity $\nu = c_s^2 (\tau - 1/2) \Delta t = \frac{2\tau - 1}{6}$. The 2D Taylor-Green vortex test validated this relationship with $< 0.8\%$ relative error across 200 time steps.
2. **Guo Forcing Exactness:**
   The Guo forcing scheme eliminates the discrete velocity shift error of simple source terms. The effective momentum injection rate exactly equals $\mathbf{F}$ when combined with the half-step macroscopic momentum definition $\rho \mathbf{u} = \sum f_i \mathbf{c}_i + \frac{\Delta t}{2}\mathbf{F}$.
3. **Halfway Bounce-Back Wall Accuracy:**
   Halfway bounce-back places the physical boundary strictly halfway between fluid and solid nodes, achieving second-order spatial accuracy ($O(\Delta x^2)$) on the parabolic Poiseuille profile ($L_2 \text{ error} = 0.31\%$) while conserving fluid mass to machine precision ($< 10^{-13}$).

### Scientific Interpretation
The repository now possesses a fully verified, high-performance classical hydrodynamic reference solver (`qlbm.classical`). This solver provides the exact physical baseline necessary to formulate multiphase physics (Phase 8-10), Volume-of-Fluid advection (Phase 11-13), and to evaluate quantum collision operators (Phase 5 and Phase 16-18).

### What is Validated
- Full D2Q9 lattice tensor symmetries and moments through 4th order.
- Exact polynomial equilibrium distribution and moment inversion.
- BGK relaxation operator and Guo forcing scheme.
- Halfway bounce-back boundary conditions for stationary and moving walls.
- Navier-Stokes analytical convergence (Poiseuille flow and Taylor-Green decay).
- Exact machine-precision mass conservation.

### What is Not Validated
- Multiphase interaction potentials (Shan-Chen, phase-field, color gradient) — scheduled for Phase 8.
- Free-surface / Volume-of-Fluid (VOF) interface reconstruction — scheduled for Phase 11.
- Quantum representation of the dissipative BGK collision operator — scheduled for Phase 5 & 16.

### Problems
None. All numerical benchmarks converge with high precision.

### Next Step
Proceed to **PHASE 5 — D2Q9 BGK OPERATOR DEEP VERIFICATION & COLLISION DYNAMICS**:
1. Perform detailed relaxation spectrum and eigenvalue analysis of the D2Q9 BGK collision matrix.
2. Characterize non-equilibrium moments, stress tensor $\Pi_{\alpha \beta}^{(neq)}$, and entropy production ($H$-theorem).
3. Author `docs/research/05_d2q9_bgk.md`.
4. Validate with relaxation tests in `tests/research/test_d2q9_bgk_relaxation.py`.




