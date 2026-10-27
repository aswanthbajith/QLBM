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
Proceed to **PHASE 5 — D2Q9 BGK OPERATOR DEEP VERIFICATION & COLLISION DYNAMICS** (Completed).

---

## Phase 5: D2Q9 BGK Operator Deep Verification & Collision Dynamics

### Date
2026-10-02

### Objective
Perform detailed mathematical and numerical analysis of the D2Q9 BGK collision operator; construct the 9x9 Gram-Schmidt orthogonal moment basis matrix $M$; derive the collision matrix spectrum ($\lambda_k \in \{1.0, 1 - \omega\}$) and stability boundaries ($\tau > 0.5$); analyze the non-equilibrium stress tensor $\Pi_{\alpha \beta}^{(neq)}$ in Couette flow; evaluate discrete entropy and prove the quadratic Lyapunov functional $\Phi(f) = \sum \frac{(f_i - f_i^{(eq)})^2}{w_i}$; author `docs/research/05_d2q9_bgk.md`; implement and validate in `tests/research/test_d2q9_bgk_relaxation.py`.

### Files Modified / Created
- `qlbm/classical/d2q9.py`: added `M_MATRIX`, `M_INV`, `compute_moments`, `compute_distributions_from_moments`, `compute_nonequilibrium_stress`, `compute_entropy_h`.
- `qlbm/classical/solver.py`: added `get_nonequilibrium_stress()`, `get_entropy_h()`.
- `qlbm/classical/__init__.py`: updated exports.
- `docs/research/05_d2q9_bgk.md`: authored theoretical and benchmark report.
- `tests/research/test_d2q9_bgk_relaxation.py`: 18 validation tests.

### Commands Executed
- `.venv/bin/pytest tests/research/test_d2q9_bgk_relaxation.py -v`
- `.venv/bin/pytest tests/research/ -v`

### Tests Executed
1. `tests/research/test_d2q9_bgk_relaxation.py`:
   - `TestMomentMatrixProperties`: $M M^T$ diagonal orthogonality, analytical $M^{-1}$ inversion, roundtrip conservation ($\mathbf{f} \to \mathbf{m} \to \mathbf{f}$ within $10^{-14}$), conserved moment mapping ($m_0, m_3, m_5$): **4 passed**.
   - `TestBGKEigenvaluesAndStability`: exact eigenvalue spectrum across 5 relaxation rates ($\tau \in \{0.6, 0.8, 1.0, 1.5, 2.0\}$), exponential decay for $\tau > 0.5$, exponential instability for $\tau < 0.5$: **7 passed**.
   - `TestNonEquilibriumStressAndCouette`: zero stress at equilibrium, analytical Couette shear stress recovery $\Pi_{xy}^{(neq)} = -\frac{\tau}{3}\rho\dot{\gamma}$ and $\sigma_{xy} = -(1 - \frac{1}{2\tau})\Pi_{xy}^{(neq)} = \rho \nu \dot{\gamma}$ ($< 1.5\%$ error): **2 passed**.
   - `TestDiscreteHTheoremAndLyapunov`: monotonic quadratic Lyapunov decay $\Phi(f^*) = (1 - \omega)^2 \Phi(f)$ across multiple relaxation rates, logarithmic $H$-convergence to $H_{eq}$: **5 passed**.
   - Summary: **18 passed in 3.81s**.

### Mathematical Findings
1. **Spectral Structure:**
   The linearized post-collision operator $C = (1 - \omega)I + \omega P_{eq}$ projects onto the 3 conserved hydrodynamic moments ($\lambda = 1.0$) and contracts all 6 non-conserved kinetic modes by $\lambda = 1 - \omega = 1 - 1/\tau$.
2. **Stability Criterion:**
   Numerical stability requires $|\lambda| < 1 \implies |1 - \omega| < 1 \implies 0 < \omega < 2 \implies \tau > 0.5$. Over-relaxation ($0.5 < \tau < 1.0$) exhibits oscillatory damping, while under-relaxation ($\tau > 1.0$) is strictly monotonic.
3. **Discrete Stress Factor:**
   Kinetic non-equilibrium flux $\Pi_{xy}^{(neq)}$ differs from Navier-Stokes viscous stress $\sigma_{xy}$ by the factor $-(1 - \frac{1}{2\tau})$, which directly yields $\nu = \frac{2\tau - 1}{6}$.
4. **Quadratic Lyapunov Functional vs $H$-Theorem:**
   Standard polynomial LBM does NOT strictly satisfy Boltzmann's logarithmic $H$-theorem at every discrete step due to polynomial truncation of the exponential Maxwellian. However, it strictly satisfies a discrete quadratic Lyapunov theorem: $\Phi(f^*) = (1 - \omega)^2 \Phi(f) < \Phi(f)$ for all stable $\tau > 0.5$.

### Scientific Interpretation
Understanding the non-unitary, dissipative projection mechanics of the classical BGK operator provides the exact scientific boundary condition for designing quantum collision operators: quantum computing requires either linear embedding (Carleman linearization / block encoding) or hybrid classical-quantum execution.

### What is Validated
- Full orthogonal moment basis and analytical transformations.
- Complete eigenvalue spectrum and stability limits.
- Kinetic-to-macroscopic shear stress connection.
- Exact quadratic Lyapunov monotonic decay.

### What is Not Validated
- Low-Mach density fluctuation scaling and incompressible limit — scheduled for Phase 6.
- External gravity forcing in stratified/two-phase flow — scheduled for Phase 7.
- Multiphase interface mechanics — scheduled for Phase 8-10.

### Problems
None. All 18 tests pass with machine precision.

### Next Step
Proceed to **PHASE 6 — MACROSCOPIC MOMENTS & INCOMPRESSIBLE LIMIT** (Completed).

---

## Phase 6: Macroscopic Moments & Incompressible Limit

### Date
2026-10-02

### Objective
Mathematically characterize and numerically validate the athermal equation of state $p = c_s^2 \rho$, the artificial compressibility mechanism of LBM, the quadratic scaling of density variations and velocity divergence errors with Mach number ($\delta \rho / \rho_0 \sim O(\text{Ma}^2)$, $||\nabla \cdot \mathbf{u}|| \sim O(\text{Ma}^2)$), physical acoustic wave speed $c_s = 1/\sqrt{3}$, and the Torricelli low-Mach constraint ($g H \le 1.667 \times 10^{-3}$) for two-phase dam-break simulations; author `docs/research/06_moments_and_incompressible_limit.md`; implement and validate in `tests/research/test_incompressible_limit.py`.

### Files Modified / Created
- `qlbm/classical/solver.py`: added `get_pressure(reference_density=None)` and `get_divergence()`.
- `docs/research/06_moments_and_incompressible_limit.md`: authored technical report.
- `tests/research/test_incompressible_limit.py`: 6 validation tests.

### Commands Executed
- `.venv/bin/pytest tests/research/test_incompressible_limit.py -v`
- `.venv/bin/pytest tests/research/ -v`

### Tests Executed
1. `tests/research/test_incompressible_limit.py`:
   - `TestEquationOfState`: absolute pressure ($p = c_s^2 \rho$) and gauge pressure ($p_{gauge} = c_s^2 (\rho - \rho_0)$) exactness: **2 passed**.
   - `TestIncompressibleScaling`:
     - Density fluctuation scaling $\Delta \rho / u_0^2 = \text{const}$ to within $0.3\%$ across 4 Mach numbers: **passed**.
     - Velocity divergence error scaling $||\nabla \cdot \mathbf{u}||_{L_2} / u_0^2 = \text{const}$ to within $0.1\%$: **passed**.
   - `TestAcousticWavePropagation`: sound pulse propagation speed matches $c_s = 1/\sqrt{3} \approx 0.57735$ to within $0.4\%$ error: **passed**.
   - `TestDamBreakLowMachCriterion`: Torricelli bound $\text{Ma}_{torr} \le 0.1 \implies g \le 0.01 / (6 H)$: **passed**.
   - Summary: **6 passed in 2.60s**.

### Mathematical Findings
1. **Artificial Compressibility Mechanism:**
   The Lattice Boltzmann Method does not solve strictly incompressible Navier-Stokes directly; rather, it solves weakly compressible Navier-Stokes with an isothermal equation of state $p = c_s^2 \rho$. Continuity satisfies $\nabla \cdot \mathbf{u} = -\frac{1}{\rho_0} \frac{D\rho}{Dt} \sim O(\text{Ma}^2)$.
2. **Quadratic Scaling Invariance:**
   Both density fluctuations and divergence error scale strictly as $O(\text{Ma}^2)$. Keeping $\text{Ma} \le 0.1$ guarantees density variations remain below $1\%$, achieving hydrodynamic incompressibility.
3. **Dam-Break Gravity Parameter Bound:**
   Liquid column collapse from height $H$ under gravity $g$ induces maximum ejection velocity $U_{\text{peak}} \approx \sqrt{2 g H}$. Enforcing $\text{Ma}_{\text{peak}} \le 0.1$ requires $g H \le 1.667 \times 10^{-3}$. For $H=20$ lattice nodes, gravity must satisfy $g \le 8.3 \times 10^{-5}$ lattice units.

### Scientific Interpretation
The low-Mach operational constraint is the foundational design rule for simulating two-phase gravity flows and dam breaks in LBM. Exceeding this limit corrupts the free surface with unphysical acoustic compressions and leads to Volume-of-Fluid volume loss.

### What is Validated
- Absolute and gauge pressure recovery.
- Exact quadratic Mach number scaling of density and divergence.
- Numerical sound speed $c_s = 1/\sqrt{3}$.
- Dam-break gravity scaling bound.

### What is Not Validated
- Stratified two-phase gravity forcing — scheduled for Phase 7.
- Interfacial surface tension and Shan-Chen / phase-field coupling — scheduled for Phase 8-10.

### Problems
None. All 6 tests passed cleanly.

### Next Step
Proceed to **PHASE 7 — GRAVITY & EXTERNAL FORCING IN D2Q9** (Completed).

---

## Phase 7: Gravity & External Forcing in D2Q9

### Date
2026-10-02

### Objective
Incorporate vertical gravitational body forcing $\mathbf{F}(\mathbf{x}, t) = \rho(\mathbf{x}, t) \mathbf{g}$ into the classical D2Q9 solver using the discrete Guo forcing scheme; verify hydrostatic balance $\partial p / \partial y = -\rho g$ and the analytical barometric density distribution $\rho(y) = \rho(0) \exp\left( -3 g y \right)$ in an enclosed fluid column; validate uniform free-fall acceleration at exact rate $g$; support dynamic callable force callbacks; author `docs/research/07_gravity_and_forcing.md`; validate in `tests/research/test_d2q9_gravity_forcing.py`.

### Files Modified / Created
- `qlbm/classical/solver.py`: added `gravity` support, `_get_effective_force()`, callable dynamic forcing, and half-step momentum update with gravity.
- `docs/research/07_gravity_and_forcing.md`: authored technical report.
- `tests/research/test_d2q9_gravity_forcing.py`: 7 validation tests.

### Commands Executed
- `.venv/bin/pytest tests/research/test_d2q9_gravity_forcing.py -v`
- `.venv/bin/pytest tests/research/ -v`

### Tests Executed
1. `tests/research/test_d2q9_gravity_forcing.py`:
   - `TestGravityConfiguration`: scalar downward gravity, 2D vector gravity, dynamic callable force callback: **3 passed**.
   - `TestHydrostaticEquilibrium`:
     - Analytical exponential density profile $\rho(y) = \rho_{\text{mid}} e^{-3 g (y - y_{\text{mid}})}$: relative $L_2 \text{ error} = 7.34 \times 10^{-6}$ ($< 0.01\%$), spurious velocity $||\mathbf{u}||_\infty < 3.2 \times 10^{-6}$: **passed**.
     - Hydrostatic pressure gradient $\frac{dp}{dy} = -\rho g$ verified to $< 0.03\%$ error: **passed**.
     - Closed domain mass conservation under gravity ($|\Delta M / M_0| < 10^{-13}$): **passed**.
   - `TestGravityDynamicAcceleration`: uniform free-fall acceleration $u_y(t) = -(t + 0.5) g$ matching theoretical prediction within $< 10^{-12}$: **passed**.
   - Summary: **7 passed in 5.73s**.

### Mathematical Findings
1. **Guo Forcing with Spatially Variable Density:**
   Evaluating $\mathbf{F}(\mathbf{x}) = \rho(\mathbf{x}) \mathbf{g}$ locally and coupling to Guo forcing yields exact hydrostatic pressure head $\Delta p = \rho g h$ and cancels spurious discrete velocity shifts.
2. **Barometric Density Profile:**
   Because pressure follows the isothermal equation of state $p = c_s^2 \rho = \frac{1}{3}\rho$, the hydrostatic equation $\frac{dp}{dy} = -\rho g$ yields an exact exponential barometric distribution $\rho(y) = \rho_0 e^{-3 g y}$.
3. **Acoustic Ringing & Settling:**
   Gravitational sudden onset excites acoustic standing waves with fundamental frequency $\omega = \pi c_s / H$. Critical relaxation ($\tau = 1.0$) damps these acoustic oscillations exponentially, achieving steady equilibrium to $10^{-8}$ accuracy.

### Scientific Interpretation
The single-phase hydrodynamic reference engine is now complete. It supports:
- Navier-Stokes laminar and sheared viscous flow (Phases 4–5).
- Weakly compressible low-Mach limits (Phase 6).
- Body forcing, gravity, and hydrostatic columns (Phase 7).
This sets the stage for introducing multiphase physics and Volume-of-Fluid coupling.

### What is Validated
- Full gravity configuration and dynamic callable forcing.
- Analytical barometric density profile and hydrostatic pressure gradient.
- Machine-precision mass conservation and uniform free-fall acceleration.

### What is Not Validated
- Multiphase interface mechanics (Shan-Chen, phase-field, color gradient) — scheduled for Phase 8-10.
- Volume-of-Fluid advection and interface reconstruction — scheduled for Phase 11-13.
- Two-phase dam-break simulation — scheduled for Phase 14-15.

### Problems
None. All 7 tests passed with extreme precision.

### Next Step
Proceed to **PHASE 8 — MULTIPHASE LBM MODEL SURVEY & ARCHITECTURAL SELECTION** (Completed).

---

## Phase 8: Multiphase LBM Model Survey & Architectural Selection

### Date
2026-10-02

### Objective
Conduct a comprehensive, scientifically rigorous comparative analysis of the four primary multiphase LBM paradigms (Pseudopotential / Shan-Chen, Free Energy / Swift, Color-Gradient / Gunstensen, Phase-Field / He-Chen-Zhang) and the Volume-of-Fluid (VOF) interface tracking method; evaluate maximum density ratio, spurious currents, thermodynamic consistency, mass conservation under violent splashing, and quantum compatibility; author `docs/research/08_multiphase_lbm_survey.md`; formalize the architectural selection for the Master's thesis two-phase dam-break solver.

### Files Created
- `docs/research/08_multiphase_lbm_survey.md`: comprehensive 5-model comparative survey and architectural manifesto.

### Key Comparative Findings
1. **Pseudopotential (Shan-Chen):** Excellent algorithmic simplicity and natural phase separation, but suffers from high spurious interfacial currents ($O(10^{-2})$) and numerical instability at air-water density ratios ($\rho_L / \rho_G \approx 1000$). Best utilized as a kinetic reference benchmark.
2. **Free Energy (Swift et al.):** Thermodynamically rigorous Ginzburg-Landau derivation, but unstable for density ratios $\rho_L / \rho_G > 20$ and requires wide numerical gradient stencils near solid walls.
3. **Color-Gradient:** Sharp interfaces (2–3 nodes) and low spurious currents, but the non-kinetic recoloring step breaks Galilean invariance and induces lattice-pinning artifacts.
4. **Phase-Field (Conservative Allen-Cahn / Cahn-Hilliard):** Robust at $\rho_L / \rho_G = 1000$ with minimal spurious currents, but requires dual distribution function sets ($2 \times 9 = 18$ populations) and non-local chemical potential stencils.
5. **Hybrid VOF-LBM (Körner / Free-Surface formulation):**
   - Treats the gas phase as a physical pressure boundary condition, achieving an **infinite effective density ratio** ($\rho_L / \rho_G = \infty$).
   - Strict discrete mass/volume flux conservation prevents droplet evaporation and numerical mass drift during violent wave impact and splashing.
   - Significant memory reduction ($> 50\%$) by deactivating pure gas nodes.

### Architectural Decision
**Primary Physics Solver:** Hybrid Volume-of-Fluid + Lattice Boltzmann Method (FS-VOF-LBM).
**Kinetic Benchmark:** Conservative Pseudopotential (Shan-Chen) implemented in Phase 9 to validate thermodynamic phase separation and Laplace surface tension against classical benchmarks.

### What is Validated
- Full survey and mathematical classification of multiphase LBM models.
- Formal justification for selecting the Hybrid VOF-LBM architecture for high-density-ratio dam-break flows.

### What is Not Validated
- Numerical implementation of Shan-Chen pseudo-potential — scheduled for Phase 9.
- Laplace surface tension validation — scheduled for Phase 10.
- VOF geometric advection and interface reconstruction — scheduled for Phase 11-13.

### Problems
None. The scientific architectural roadmap is clearly demarcated.

### Next Step
Proceed to **PHASE 9 — SHAN-CHEN PSEUDOPOTENTIAL MULTIPHASE IMPLEMENTATION** (Completed).

---

## Phase 9: Shan-Chen Pseudopotential Multiphase Implementation

### Date
2026-10-02

### Objective
Implement the single-component Shan-Chen pseudopotential multiphase model on the validated D2Q9 lattice; derive the microscopic inter-particle interaction force, non-ideal equation of state, and spinodal instability condition; author `docs/research/09_shan_chen_model.md`; implement and validate comprehensive thermodynamic and kinetic benchmarks in `tests/research/test_shan_chen.py`.

### Files Created/Modified
- `qlbm/classical/multiphase.py`: Shan-Chen pseudopotentials (exponential, Carnahan-Starling), interaction force calculation, non-ideal pressure EOS, and `ShanChenMultiphaseSolver` subclassing `D2Q9BGKSolver`.
- `qlbm/classical/__init__.py`: exported Shan-Chen components.
- `docs/research/09_shan_chen_model.md`: comprehensive mathematical derivation and benchmark analysis.
- `tests/research/test_shan_chen.py`: 8 unit and physical benchmark tests.

### Commands Executed
- `.venv/bin/pytest tests/research/test_shan_chen.py -v`
- `.venv/bin/pytest tests/research/ -v`

### Tests Executed
1. `tests/research/test_shan_chen.py`:
   - `TestPseudopotentialAndForceFormulation::test_exponential_pseudopotential_properties` PASSED
   - `TestPseudopotentialAndForceFormulation::test_uniform_density_force_isotropy` PASSED ($\max \|F\| < 10^{-15}$)
   - `TestPseudopotentialAndForceFormulation::test_attractive_force_direction` PASSED ($F_x(0) > 0, F_x(7) < 0$ inward pull)
   - `TestPseudopotentialAndForceFormulation::test_carnahan_starling_pseudopotential` PASSED
   - `TestPhaseSeparationAndThermodynamics::test_subcritical_stability_g_greater_than_critical` PASSED (decay below $0.005$)
   - `TestPhaseSeparationAndThermodynamics::test_supercritical_spinodal_decomposition` PASSED ($\rho_l \approx 1.89, \rho_v \approx 0.118$, ratio $> 16$)
   - `TestPhaseSeparationAndThermodynamics::test_shan_chen_equation_of_state_spinodal_loop` PASSED ($\partial p / \partial \rho < 0$)
   - `TestPhaseSeparationAndThermodynamics::test_gravity_induced_phase_stratification` PASSED (liquid core settles, gas rises)
   - Summary: **8 passed in 4.06s**.
2. Combined research test suite (`tests/research/`):
   - **81 passed in 30.91s**.

### Mathematical & Physical Findings
1. **Critical Interaction Threshold:**
   For the standard exponential potential $\psi(\rho) = 1 - e^{-\rho}$, the critical point occurs at $\rho_c = \ln(2) \approx 0.69315$, yielding $G_c = -4.0$. For $G > -4.0$, perturbations dissipate; for $G < -4.0$, the fluid spontaneously undergoes spinodal decomposition into dense liquid ($\rho_l \approx 1.894$) and dilute gas ($\rho_v \approx 0.118$).
2. **Force Isotropy and Cohesive Action:**
   On uniform density fields, the discrete interaction force vanishes to machine precision ($< 10^{-15}$). Across high-density slabs, the cohesive force strictly pulls both interior boundary nodes and surrounding vapor inward toward the dense center.
3. **Mass Conservation:**
   Coupled with Guo forcing source terms $\sum_i F_i = 0$, total mass is conserved to machine precision ($|\Delta M| < 10^{-11}$) throughout intense spinodal phase separation.
4. **Hydrostatic Phase Stratification:**
   Under gravitational acceleration, dense liquid sedimented into the lower domain while dilute vapor displaced to the upper domain, maintaining a stable hydrostatic liquid-gas stratification.

### Scientific Interpretation
The Shan-Chen pseudopotential model confirms our ability to simulate non-ideal thermodynamics, spontaneous phase separation, and fluid-fluid interfaces on a discrete D2Q9 lattice. However, parasitic spurious currents and numerical instabilities at density ratios exceeding $\approx 20$ corroborate our Phase 8 architectural decision: the final two-phase dam-break solver must be built upon the sharp-interface Free-Surface / Volume-of-Fluid (VOF) LBM formulation.

### What is Validated
- Exponential and Carnahan-Starling pseudopotentials.
- Discrete inter-particle interaction force and Guo forcing integration.
- Subcritical damping vs supercritical spinodal decomposition.
- Exact discrete mass conservation during phase separation.
- Gravity-driven multiphase stratification.

### What is Not Validated
- Quantitative Laplace law $\Delta p = \sigma / R$ and surface tension calibration — scheduled for Phase 10.
- Volume-of-Fluid (VOF) interface reconstruction and mass flux advection — scheduled for Phases 11–13.

### Problems
None encountered. All unit and benchmark tests passed cleanly.

### Next Step
Proceed to **PHASE 10 — SURFACE TENSION & LAPLACE LAW BENCHMARKING** (Completed).

---

## Phase 10: Surface Tension & Young-Laplace Law Validation

### Date
2026-10-02

### Objective
Derive the 2D Young-Laplace pressure relation $\Delta p = \frac{\sigma}{R}$ across a curved fluid-fluid interface; implement stationary circular droplet simulations across multiple radii; calibrate numerical surface tension $\sigma(G)$; analyze the spatial distribution and scaling of parasitic spurious currents; author `docs/research/10_surface_tension_laplace.md`; implement and validate comprehensive physical tests in `tests/research/test_laplace_surface_tension.py`.

### Files Created/Modified
- `qlbm/classical/multiphase.py`: added `LaplaceFitResult` dataclass, `initialize_circular_droplet`, `measure_droplet_properties`, `fit_laplace_law`, and convenience methods `initialize_droplet` and `measure_droplet` on `ShanChenMultiphaseSolver`.
- `qlbm/classical/__init__.py`: exported Laplace utilities and functions.
- `docs/research/10_surface_tension_laplace.md`: comprehensive theoretical derivation, regression tables, spurious currents spatial mapping, and VOF architectural bridge.
- `tests/research/test_laplace_surface_tension.py`: 6 rigorous unit and physical benchmark tests.

### Commands Executed
- `.venv/bin/pytest tests/research/test_laplace_surface_tension.py -v`
- `.venv/bin/pytest tests/research/ -v`

### Tests Executed
1. `tests/research/test_laplace_surface_tension.py`:
   - `TestDropletInitializationAndConservation::test_circular_droplet_tanh_profile` PASSED (smooth hyperbolic tangent profile within bounds)
   - `TestDropletInitializationAndConservation::test_droplet_mass_conservation_during_relaxation` PASSED ($|\Delta M| < 10^{-11}$)
   - `TestYoungLaplaceLaw::test_laplace_linear_scaling_and_surface_tension` PASSED ($R^2 = 0.9869 > 0.98$, $\sigma = 0.0321 \pm 0.001$, intercept near zero)
   - `TestYoungLaplaceLaw::test_surface_tension_monotonicity_with_g` PASSED ($\sigma$ increases $> 6\times$ from $G=-4.5$ to $G=-5.5$)
   - `TestSpuriousCurrentsAndParasiticVortices::test_spurious_currents_spatial_localization` PASSED (spurious currents localized strictly at interface, $< 10^{-14}$ in bulk phases)
   - `TestSpuriousCurrentsAndParasiticVortices::test_spurious_currents_scaling_with_interaction_strength` PASSED ($u_{\text{max}}$ grows from $0.0016$ to $0.0199$)
   - Summary: **6 passed in 10.01s**.
2. Combined research test suite (`tests/research/`):
   - **87 passed in ~40s**.

### Mathematical & Physical Findings
1. **2D Young-Laplace Law Distinction:**
   Because in-plane cylindrical droplets have zero out-of-plane curvature ($R_2 \to \infty$), the 2D pressure jump is $\Delta p = \frac{\sigma}{R}$ without the spherical factor of 2.
2. **High Linear Precision:**
   Across radii $R_0 \in [8, 16]$, the product $\Delta p \cdot R_{\text{eff}}$ remained tightly bounded between $0.0312$ and $0.0344$, with linear regression yielding $R^2 = 0.9869$ and intercept $8.7 \times 10^{-5} \approx 0$.
3. **Surface Tension Monotonicity:**
   Increasing interaction strength $|G|$ deepened the spinodal separation, widening density contrast from $6.1$ to $36.1$ and multiplying surface tension from $0.0152$ to $0.0936$.
4. **Spatial Confinement & Spurious Current Bottleneck:**
   Parasitic velocities were proved to be strictly confined to the diffuse interface, dropping to machine zero ($< 10^{-14}$) in the bulk liquid core and far-field vapor. However, spurious currents grew exponentially with $|G|$, reaching $u_{\text{max}} \approx 0.02$ at $G=-5.5$. This provides conclusive empirical justification for adopting Volume-of-Fluid (VOF) / Free-Surface LBM for the high-density-ratio dam-break problem.

### Scientific Interpretation
The successful validation of Laplace's law establishes that the classical LBM suite correctly represents interfacial mechanics, normal stress balance, and capillary effects. Furthermore, the quantitative analysis of spurious currents demonstrates the exact physical boundary where diffuse-interface models break down and sharp-interface VOF methods become mandatory.

### What is Validated
- 2D Young-Laplace linear scaling $\Delta p \propto 1/R$.
- Numerical surface tension calibration $\sigma(G)$.
- Mass conservation during multiphase droplet relaxation.
- Spatial localization and exponential growth of spurious currents.

### What is Not Validated
- Volume-of-Fluid (VOF) interface reconstruction (Youngs gradient, PLIC) — scheduled for Phase 11.
- Geometric mass flux advection and interface cell state transitions — scheduled for Phase 12.
- Free-surface pressure boundary conditions and gas decoupling — scheduled for Phase 13.
- Two-phase liquid column collapse dam-break simulation — scheduled for Phases 14–15.

### Problems
None. All tests passed.

### Next Step
Proceed to **PHASE 11 — VOF GEOMETRIC INTERFACE RECONSTRUCTION** (Completed).

---

## Phase 11: Volume-of-Fluid (VOF) Geometric Interface Reconstruction

### Date
2026-10-02

### Objective
Implement the geometric foundation of the sharp-interface Free-Surface / Volume-of-Fluid (VOF) LBM solver: discrete cell state classification (`GAS`, `INTERFACE`, `LIQUID`, `SOLID`), 2nd-order 9-point Youngs gradient operator for interface normal vector estimation, exact closed-form Piecewise Linear Interface Calculation (PLIC) forward area and inverse line constant determination across all 4 quadrants (Scardovelli & Zaleski 2000), and the `VOFInterface` manager; author `docs/research/11_vof_interface_reconstruction.md`; implement and validate comprehensive tests in `tests/research/test_vof_reconstruction.py`.

### Files Created/Modified
- `qlbm/classical/vof.py`: `CellType` IntEnum, `classify_cells`, `compute_normal_youngs`, `compute_normal_central`, `plic_forward_area`, `plic_inverse_alpha`, and `VOFInterface` class.
- `qlbm/classical/__init__.py`: exported VOF geometric components.
- `docs/research/11_vof_interface_reconstruction.md`: mathematical formulation of cell states, Youngs gradient, Scardovelli-Zaleski closed-form PLIC, and dam-break topology.
- `tests/research/test_vof_reconstruction.py`: 10 unit and geometric benchmark tests.

### Commands Executed
- `.venv/bin/pytest tests/research/test_vof_reconstruction.py -v`
- `.venv/bin/pytest tests/research/ -q`

### Tests Executed
1. `tests/research/test_vof_reconstruction.py`:
   - `TestCellClassification::test_cell_type_assignment` PASSED
   - `TestNormalVectorEstimation::test_horizontal_interface_normal` PASSED ($\|\mathbf{n} - (0, 1)\| < 10^{-12}$)
   - `TestNormalVectorEstimation::test_vertical_interface_normal` PASSED ($\|\mathbf{n} - (1, 0)\| < 10^{-12}$)
   - `TestNormalVectorEstimation::test_diagonal_45_deg_interface_normal` PASSED (angular error $< 10^{-10}$ deg)
   - `TestNormalVectorEstimation::test_oblique_interfaces_youngs_accuracy` PASSED (angular error $\le 2.09^\circ < 2.5^\circ$)
   - `TestNormalVectorEstimation::test_circular_droplet_radial_normals` PASSED (mean error $0.64^\circ$, all $\mathbf{n} \cdot \mathbf{n}_{\text{exact}} > 0.95$)
   - `TestPiecewiseLinearInterfaceCalculation::test_plic_limiting_cases` PASSED (exact analytical $C=0, 1$ limits)
   - `TestPiecewiseLinearInterfaceCalculation::test_plic_inverse_roundtrip_all_quadrants` PASSED (1000 random orientations, $|C_{\text{calc}} - C_{\text{true}}| < 10^{-14}$)
   - `TestPiecewiseLinearInterfaceCalculation::test_plic_axis_aligned_interfaces` PASSED ($\alpha = C$ exact)
   - `TestVOFInterfaceManager::test_dam_break_column_initialization` PASSED (rectangular dam-break column topology and volume matching)
   - Summary: **10 passed in 2.42s**.
2. Combined research test suite (`tests/research/`):
   - **97 passed in ~42s**.

### Mathematical & Algorithmic Findings
1. **Machine-Precision PLIC Inversion:**
   The Scardovelli & Zaleski analytical formulas eliminate iterative root-finding, providing an instantaneous closed-form mapping between volume fraction $C$ and line constant $\alpha$ that matches forward area to machine precision ($< 10^{-14}$) across all four quadrants.
2. **Youngs Stencil vs Central Difference:**
   The 9-point diagonally-weighted Youngs stencil achieves sub-degree angular accuracy ($0.64^\circ$ mean on circular droplets, $0.00^\circ$ at $0^\circ, 45^\circ, 90^\circ, 135^\circ$), preventing grid-pinning artifacts.
3. **Decoupled Gas Architecture:**
   Partitioning into GAS, INTERFACE, LIQUID, and SOLID states eliminates the need to compute lattice Boltzmann populations inside the ambient gas, allowing unbounded density ratios ($\rho_L / \rho_G = \infty$) essential for dam-break simulations.

### Scientific Interpretation
Phase 11 establishes the geometric backbone of the sharp-interface VOF framework. By replacing diffuse-interface approximations with exact PLIC geometric reconstruction, we eliminate the spurious-current bottleneck identified in Phases 9 and 10.

### What is Validated
- Discrete cell state classification.
- Youngs normal vector estimation on planar, oblique, and circular interfaces.
- Scardovelli & Zaleski closed-form PLIC forward area and inverse $\alpha$.
- Dam-break rectangular liquid column topological initialization.

### What is Not Validated
- Geometric mass flux advection across lattice cell faces — scheduled for Phase 12.
- Cell state transitions (interface to liquid/gas conversion) — scheduled for Phase 12.
- Free-surface hydrodynamic boundary conditions — scheduled for Phase 13.
- Full two-phase liquid column collapse dam-break simulation — scheduled for Phases 14–15.

### Problems
None. All 10 tests passed cleanly.

### Next Step
Proceed to **PHASE 12 — HYBRID VOF-LBM FREE-SURFACE SOLVER (COMPLETED)**.

---

## Phase 12: Hybrid VOF-LBM Free-Surface Solver (`VOFHydroSolver`)

### Date
2026-10-23

### Objective
Complete and validate the full classical reference solver for two-phase gas-liquid flows (`VOFHydroSolver`):
1. Pairwise anti-symmetric mass flux advection: $\Delta M_i(\mathbf{x}) = -\Delta M_{\bar{i}}(\mathbf{x}+\mathbf{c}_i)$.
2. Decoupled ambient gas phase with boundary reconstruction: $f_{\bar{i}} = f_i^{\text{eq}} + f_{\bar{i}}^{\text{eq}} - f_i^*$.
3. Topological neighborhood completeness: $\mathcal{N}(\text{LIQUID}) \cap \text{GAS} = \emptyset$.
4. Free-surface capillary pressure jump: $p_{\text{iface}} = p_{\text{gas}} + \sigma \kappa$.
5. Validate mass conservation, droplet stability, Young-Laplace scaling, and column collapse.

### Files Validated
- `qlbm/classical/hybrid_vof_lbm.py`: `VOFHydroSolver` implementation.
- `docs/research/12_vof_lbm_hybrid.md`: theoretical formulation and benchmarks.
- `tests/research/test_vof_lbm_hybrid.py`: 8 comprehensive benchmark tests.

### Tests Executed
- `pytest tests/research/test_vof_lbm_hybrid.py -v`:
  - `TestPairwiseAntiSymmetryAndConservation::test_local_flux_anti_symmetry` PASSED ($< 10^{-14}$)
  - `TestPairwiseAntiSymmetryAndConservation::test_global_mass_conservation_over_time` PASSED ($< 10^{-12}$)
  - `TestInterfaceAdvection::test_translating_droplet_advection_x` PASSED
  - `TestInterfaceAdvection::test_volume_conservation_under_shear` PASSED
  - `TestStaticDropletEquilibrium::test_static_droplet_stability` PASSED
  - `TestYoungLaplacePressureScaling::test_young_laplace_scaling_across_radii` PASSED
  - `TestDamBreakColumnCollapse::test_dam_break_collapse_dynamics` PASSED
  - `TestDamBreakColumnCollapse::test_dam_break_mass_conservation_strict` PASSED
  - Result: **8 passed in 6.07s**.

---

---
## Phase 13: OpenLB Dam-Break Transfer & Hybrid Quantum VOF Architecture

### Date
2026-10-23

### Objective
Transfer the physical problem and numerical architecture from OpenLB's `FreeSurfaceBreakingDam2D` into QLBM:
1. Deconstruct and map every OpenLB component (`SmagorinskyForcedBGKdynamics`, `boundary::BounceBack`, `FreeSurface2DSetup`, `GRAVITY`) to QLBM classical and quantum equivalents.
2. Establish a reproducible classical dam-break experiment (`demos/simulation/vof_dam_break.py`) tracking Martin & Moyce column collapse phenomenology (dimensionless front advancement  = x_{\text{front}} / L$, center of mass descent {\text{cm}}(t)$, kinetic energy, and strict mass conservation).
3. Design and implement the modular quantum VOF subpackage (`qlbm/components/vof/`):
   - `QuantumVOFComponent`: base class for free-surface quantum components.
   - `QuantumCollisionOperator`: abstract research interface for candidate quantum collision models.
   - `QuantumD2Q9BounceBackOperator`: unitary opposite-velocity reflection circuit for solid walls.
   - `QuantumD2Q9StreamingOperator`: QFT Draper adder streaming circuit for D2Q9 spatial registers.
   - `HybridVOFStep`: hybrid classical-quantum orchestrator delegating streaming and wall reflections to quantum circuit simulation while preserving classical geometric VOF tracking.
4. Implement rigorous tests in `tests/research/test_quantum_vof.py` verifying exact numerical equivalence between quantum circuit evolution and classical LBM arrays.

### Files Created/Modified
- `demos/simulation/vof_dam_break.py`: CLI and programmatic reproducible dam-break benchmark.
- `qlbm/components/vof/__init__.py`: exported quantum VOF components and hybrid stepper.
- `qlbm/components/vof/base.py`: base classes and `QuantumCollisionOperator` interface.
- `qlbm/components/vof/boundary.py`: `QuantumD2Q9BounceBackOperator`.
- `qlbm/components/vof/streaming.py`: `QuantumD2Q9StreamingOperator`.
- `qlbm/components/vof/hybrid_stepper.py`: `HybridVOFStep`.
- `qlbm/components/__init__.py`: updated package exports.
- `tests/research/test_quantum_vof.py`: 8 unit and integration tests.

### Tests Executed
1. `demos/simulation/vof_dam_break.py --steps 100 --nx 64 --ny 32`:
   - $y_{\text{cm}}$ descent: $12.50 \to 11.59$ (PASS)
   - Surge front advancement: $16.00 \to 17.00$ (PASS)
   - Global mass error: $0.00 \times 10^0 < 10^{-12}$ (PASS)
   - Low-Mach stability: max $\|\mathbf{u}\| = 0.0557 < 0.25$ (PASS)
   - Execution rate: 135 steps/s.
2. `pytest tests/research/test_quantum_vof.py -v`:
   - All 8 unit and integration tests PASSED in 7.02s.
3. Combined suite (`test_vof_lbm_hybrid.py` and `test_quantum_vof.py`):
   - **16 passed in 10.02s**.

### Scientific & Architectural Findings
1. **Option A Hybrid Feasibility**:
   Unitary streaming and solid wall reflection can be executed directly on quantum registers using $N = \log_2 N_x + \log_2 N_y + 4 + 1$ qubits with Draper QFT adders. Statevector simulation reproduces classical array streaming to machine precision ($< 1.37 \times 10^{-13}$).
2. **Preservation of Interface Conservation**:
   By retaining geometric VOF operations (pairwise anti-symmetric fluxes and neighborhood consistency) on the classical processor, the hybrid solver preserves strict mass conservation ($< 10^{-12}$) during quantum hydrodynamic stepping.
3. **Collision Interface Formulation**:
   The `QuantumCollisionOperator` abstract base class provides the formal boundary for future research into non-unitary dissipative collision realizations (linearized approximations, Carleman embedding, or block encoding).

### Next Step
Proceed to **PHASE 14 — LINEARIZED QUANTUM D2Q9 COLLISION & BLOCK ENCODING (COMPLETED)**.

---

## Phase 14: Linearized Quantum D2Q9 Collision Operator & Ancilla Block-Encoding

### Date
2026-10-27

### Objective
Formulate, implement, and validate the quantum representation of the D2Q9 hydrodynamic collision operator:
1. Low-Mach linearization around hydrostatic equilibrium ($\rho_0 = 1.0, \mathbf{u}_0 = 0$), establishing the projection matrix $\mathbf{E}$ onto conserved moments $\{\rho, j_x, j_y\}$.
2. Hermite-weighted symmetrization: $\tilde{\mathbf{C}}_{\text{lin}} = \mathbf{W}^{-1} \mathbf{C}_{\text{lin}} \mathbf{W}$ with $\|\tilde{\mathbf{C}}_{\text{lin}}\|_2 = 1.0 \le 1.0$.
3. Exact 5-qubit ($32 \times 32$) unitary block dilation ($1$ ancilla $+ 4$ velocity qubits) via spectral decomposition ($\|U U^\dagger - I\|_{\infty} < 10^{-15}$).
4. Machine-precision conservation of mass ($\Delta \rho < 10^{-15}$) and momentum ($\|\Delta \mathbf{j}\| < 10^{-15}$).
5. Analytical $O(\text{Ma}^2)$ error bound $\frac{5}{6\tau} \|\mathbf{u}\|^2$ verified empirically with slope $2.000$.
6. Full quantum dam-break simulation integration (`--mode full_quantum`) in `vof_dam_break.py` and `HybridVOFStep`.

### Files Created/Modified
- `qlbm/components/vof/collision.py`: `LinearizedQuantumCollisionOperator` implementation.
- `qlbm/components/vof/__init__.py`: exported collision operator.
- `qlbm/components/__init__.py`: updated package exports.
- `qlbm/components/vof/hybrid_stepper.py`: integrated quantum collision in `HybridVOFStep`.
- `demos/simulation/vof_dam_break.py`: added `full_quantum` Mode Q and quantum resource reporting.
- `tests/research/test_quantum_collision.py`: 11 comprehensive unit and physical benchmark tests.
- `docs/research/14_quantum_collision_linearization.md`: theoretical derivations and error analysis.

### Tests Executed
1. `pytest tests/research/test_quantum_collision.py -v`:
   - `TestLinearizedCollisionSpectra::test_projection_matrix_idempotence` PASSED ($< 10^{-15}$)
   - `TestLinearizedCollisionSpectra::test_projection_matrix_eigenvalues` PASSED (exact rank 3)
   - `TestLinearizedCollisionSpectra::test_linearized_collision_eigenvalues` PASSED (decay $1 - 1/\tau$)
   - `TestHermiteSymmetrizationAndUnitarity::test_weighted_matrix_symmetry` PASSED ($0.00$)
   - `TestHermiteSymmetrizationAndUnitarity::test_spectral_norm_subunitary` PASSED ($\le 1.0$)
   - `TestHermiteSymmetrizationAndUnitarity::test_unitary_block_dilation` PASSED ($< 10^{-14}$)
   - `TestHermiteSymmetrizationAndUnitarity::test_quantum_circuit_construction` PASSED (5 qubits)
   - `TestCollisionInvariantsAndConservation::test_mass_conservation_arbitrary_distributions` PASSED ($< 10^{-15}$)
   - `TestCollisionInvariantsAndConservation::test_momentum_conservation_arbitrary_distributions` PASSED ($< 10^{-15}$)
   - `TestMachQuadraticConvergence::test_quadratic_mach_convergence_rate` PASSED (slope $2.000$)
   - `TestEndToEndHybridDamBreakWithCollision::test_classical_vs_quantum_collision_dam_break_match` PASSED
   - Result: **11 passed in 2.39s**.
2. Combined research test suite (`pytest tests/research/ -q`):
   - **136 passed in 42.58s** (0 regressions).
3. End-to-end full quantum dam-break run (`demos/simulation/vof_dam_break.py --mode full_quantum`):
   - Mass conservation: $\Delta M / M_0 = 1.48 \times 10^{-16} < 10^{-12}$ (PASS)
   - Low-Mach stability: max $\|\mathbf{u}\| = 0.0019 < 0.25$ (PASS)

### Scientific & Architectural Findings
1. **Unitarity via Hermite Weighting:**
   The non-unitary BGK collision operator is strictly contractive and symmetric under the Hermite-scaled distribution basis $\tilde{f}_i = f_i / \sqrt{w_i}$, enabling exact unitary block dilation using a single ancilla qubit without non-physical dissipation.
2. **Quadratic Mach Convergence:**
   The deviation between linearized quantum collision and non-linear BGK scales strictly as $\frac{5}{6\tau} \|\mathbf{u}\|^2$, establishing the exact physical domain of validity ($Ma < 0.1$) for quantum linear collision in hydrodynamic free-surface flows.
3. **Complete Quantum Hydrodynamic Stepper (Mode Q):**
   With unitary streaming (QFT Draper adders), unitary solid reflection, and block-encoded collision, both kinetic substeps of LBM are realized as quantum operators coupled to classical geometric VOF tracking.

### Next Step
Proceed to **PHASE 15 — SYNTHESIS, COMPREHENSIVE VALIDATION & THESIS PATHWAY**:
1. Finalize the cross-paradigm comparison: Classical LBM vs Multiphase Shan-Chen vs Hybrid VOF-LBM vs Full-Quantum VOF.
2. Consolidate error metrics, scaling limits, and quantum circuit resource requirements into the final thesis research roadmap.

