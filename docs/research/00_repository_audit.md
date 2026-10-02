# Phase 0: Comprehensive Repository Audit

**Date:** 2026-10-02  
**Repository:** `~/Research/qlbm`  
**Current Branch:** `research/multiphase-vof` (branched cleanly from `main`)  
**Environment:** Python 3.14.4, Qiskit 2.1.0, Qiskit Aer 0.17.2, Qulacs 0.6.14, PyTket 2.18.4, VTK 9.7.1, Pytest 9.1.1  

---

## 1. Executive Summary & Context

This research project aims to investigate and extend the Quantum Lattice Boltzmann Method (QLBM) toward a technically justified framework for two-phase gas-liquid flow and dam-break simulations using a hybrid Volume-of-Fluid (VoF) + LBM methodology.

The current repository (`qlbm`, version 0.0.6) provides several quantum formulations developed primarily by researchers at TU Delft (Georgescu, Schalkers, Möller et al.). A foundational insight established at the start of this investigation is:
- **MSQLBM** (Multi-Speed QLBM) is an **amplitude-encoded, multi-speed, strictly collisionless** quantum algorithm for the transport equation. Its operational cycle is:
  $$\text{Streaming} \longrightarrow \text{Boundary Reflection} \longrightarrow \text{Ancilla Reset}$$
- The repository also contains an **ABQLBM** (Amplitude-Based QLBM) implementation designed for standard discrete velocity sets ($D_d Q_q$, specifically D2Q9), which is **likewise collisionless** at present.
- Other collision mechanics present in the repository belong to **CBSE (Computational Basis State Encoding)** and **LQLGA (Local Quantum Lattice Gas Automata)** paradigms (specifically equivalence-class permutations and unitary discrete Fourier transform mixing), which are lattice gas collision models rather than conventional BGK (Bhatnagar-Gross-Krook) relaxation models.
- **There is currently zero multiphase, free-surface, or Volume-of-Fluid (VoF) functionality in the repository.** All such physics must be developed through a grounded, scientifically validated progression starting from classical reference solvers and systematically exploring quantum compatibility.

---

## 2. Complete Repository Directory Architecture

```
qlbm/
├── pyproject.toml              # Build config, dependencies (qiskit 2.1.0, aer, qulacs, vtk, etc.)
├── Makefile                    # Target helpers (test, lint, docs)
├── LICENSE                     # MPL-2.0
├── CITATION.cff                # Citation metadata
├── README.md                   # Top-level overview and quick-start
├── qlbm/                       # Core Python library
│   ├── __init__.py             # Exposes public API: lattices, components, infra, tools
│   ├── lattice/                # Lattice abstractions, registers, geometries, encodings
│   │   ├── lattices/           # Concrete lattice classes:
│   │   │   ├── base.py         # Lattice (ABC), AmplitudeLattice (ABC)
│   │   │   ├── ms_lattice.py   # MSLattice (Multi-speed, magnitude+direction registers)
│   │   │   ├── ab_lattice.py   # ABLattice (Amplitude-based, log2(q) velocity register)
│   │   │   ├── oh_lattice.py   # OHLattice (One-hot velocity register, q qubits)
│   │   │   ├── spacetime_lattice.py # SpaceTimeLattice (CBSE encoding, lightcone truncation)
│   │   │   └── lqlga_lattice.py     # LQLGALattice (Local QLGA, N_points * q qubits)
│   │   ├── spacetime/          # Space-time discretization descriptors:
│   │   │   ├── properties_base.py   # LatticeDiscretization enum (D1Q2, D2Q4, D3Q6, D1Q3, D2Q9, CFL)
│   │   │   ├── d1q2.py, d2q4.py, d3q6.py # Specific velocity coordinate definitions
│   │   ├── eqc/                # Equivalence Class generators for CBSE/LGA collisions:
│   │   │   ├── eqc.py          # EquivalenceClass data container (mass, momentum, states)
│   │   │   └── eqc_generator.py# Generates all EQC partitions for a discretization
│   │   └── geometry/           # Solid geometry definitions and quantum mappings:
│   │       ├── shapes/         # Geometric shapes: Block (cuboid), Circle, YMonomial
│   │       └── encodings/      # Mappings from geometry to quantum comparator/ancilla logic:
│   │           ├── ms.py       # DimensionalReflectionData, ReflectionWall, ReflectionResetEdge
│   │           ├── spacetime.py# SpaceTime geometry reflection mappings
│   │           └── lqlga.py    # LQLGA geometry reflection mappings
│   ├── components/             # Quantum operators and end-to-end algorithms:
│   │   ├── base.py             # LBMPrimitive, LBMOperator, LBMAlgorithm abstract bases
│   │   ├── cqlbm.py            # CQLBM entry point (delegates to MSQLBM or ABQLBM)
│   │   ├── ms/                 # Multi-Speed Collisionless QLBM components:
│   │   │   ├── msqlbm.py       # MSQLBM full algorithm (CFL time series loop)
│   │   │   ├── streaming.py    # MSStreamingOperator, StreamingAncillaPreparation, ControlledIncrementer
│   │   │   ├── primitives.py   # MSInitialConditions, GridMeasurement, EdgeComparator
│   │   │   ├── bounceback_reflection.py # BounceBackReflectionOperator, BounceBackWallComparator
│   │   │   └── specular_reflection.py   # SpecularReflectionOperator, SpecularWallComparator
│   │   ├── ab/                 # Amplitude-Based QLBM (D2Q9/D1Q3):
│   │   │   ├── ab.py           # ABQLBM end-to-end algorithm (collisionless)
│   │   │   ├── streaming.py    # ABStreamingOperator (controlled spatial coordinate shifts)
│   │   │   ├── reflection/     # ABReflectionOperator (D2Q9 permutations), ABZoneAgnosticReflectionOperator
│   │   │   ├── averaged_collision.py # ABEAveragedCollisionOperator (WIP, truncated QFT for D1Q3)
│   │   │   ├── initial.py      # UniformStatePreparation (Shukla & Vedula 2024)
│   │   │   ├── encodings.py    # ABEncodingType enum (AB, OH)
│   │   │   └── measurement.py  # ABMeasurementOperator
│   │   ├── spacetime/          # Space-Time CBSE algorithm:
│   │   │   ├── spacetime.py    # SpaceTimeQLBM end-to-end algorithm
│   │   │   ├── streaming.py    # SpaceTimeStreamingOperator
│   │   │   ├── collision/      # GenericSpaceTimeCollisionOperator, SpaceTimeD2Q4CollisionOperator
│   │   │   ├── reflection/     # PointwiseReflectionOperator, VolumetricReflectionOperator
│   │   │   ├── initial/        # PointwiseInitialConditionOperator, VolumetricInitialConditionOperator
│   │   │   └── measurement.py  # SpaceTimeMeasurementOperator
│   │   ├── lqlga/              # Local Quantum Lattice Gas Automata:
│   │   │   ├── lqlga.py        # LQLGA end-to-end algorithm
│   │   │   ├── streaming.py    # LQLGAStreamingOperator
│   │   │   ├── collision.py    # GenericLQLGACollisionOperator (applies EQC collision across sites)
│   │   │   ├── reflection.py   # GenericLQLGAReflectionOperator
│   │   │   ├── initial.py      # LQLGAInitialConditionOperator
│   │   │   └── measurement.py  # LQLGAMeasurementOperator
│   │   └── common/             # Shared quantum primitives across algorithms:
│   │       ├── adders.py       # Quantum adders (Draper QFT adder, Beauregard adder)
│   │       ├── comparators.py  # SingleRegisterComparator, TwoRegisterComparator
│   │       ├── primitives.py   # TruncatedQFT, EmptyPrimitive
│   │       └── cbse_collision/ # CBSE Equivalence-Class collision primitives:
│   │           ├── cbse_collision.py      # EQCCollisionOperator (PRP cycle)
│   │           ├── cbse_permutation.py    # EQCPermutation
│   │           └── cbse_redistribution.py # EQCRedistribution (DFT block unitary mixing)
│   ├── infra/                  # Simulation, compilation, and execution engines:
│   │   ├── compiler.py         # CircuitCompiler (Qiskit / PyTket transpilation)
│   │   ├── runner/             # Execution runners:
│   │   │   ├── base.py         # Runner (ABC)
│   │   │   ├── qiskit_runner.py# QiskitRunner (AerBackend, statevector/sampling)
│   │   │   ├── qulacs_runner.py# QulacsRunner (high-performance statevector simulator)
│   │   │   ├── mpi_qulacs.py   # Distributed MPI Qulacs runner
│   │   │   └── simulation_config.py # SimulationConfig dataclass
│   │   ├── reinitialize/       # Inter-timestep state reinitialization:
│   │   │   ├── base.py         # Reinitializer (ABC)
│   │   │   ├── identity_reinitializer.py # Identity pass-through
│   │   │   └── spacetime_reinitializer.py# Re-initialization for space-time lattice
│   │   └── result/             # Results processing and ParaView export:
│   │       ├── base.py         # QBMResult (ABC)
│   │       ├── amplitude_result.py # AmplitudeResult (processes counts/statevectors to .vti)
│   │       ├── spacetime_result.py # SpaceTimeResult
│   │       └── lqlga_result.py     # LQLGAResult
│   └── tools/                  # Utilities, logging, exceptions:
│       ├── exceptions.py       # Custom exception classes
│       ├── logging.py          # Logging setup
│       └── utils.py            # Math helpers, bit manipulations, CFL time series generators
├── test/                       # Test suite: unit, integration, e2e, regression
├── demos/                      # Jupyter notebooks, benchmarks, visualization samples
└── docs/                       # Sphinx documentation and literature bibliography (refs.bib)
```

---

## 3. Targeted Codebase Audit & Keyword Analysis

### 3.1 Collision Implementations Search
Command: `grep -R "class .*Collision" -n qlbm`

| File | Class | Description |
| :--- | :--- | :--- |
| `qlbm/components/ab/averaged_collision.py:16` | `ABEAveragedCollisionOperator` | Averaged collision using `TruncatedQFT` on velocity qubits. Marked `"""WIP."""`, restricted to D1Q3. **Not BGK**. |
| `qlbm/components/common/cbse_collision/cbse_collision.py:25` | `EQCCollisionOperator` | Permutation-Redistribution-Permutation (PRP) unitary collision over equivalence classes sharing mass and momentum. Used for CBSE/LGA. **Not BGK**. |
| `qlbm/components/spacetime/collision/eqc_collision.py:14` | `GenericSpaceTimeCollisionOperator` | Applies `EQCCollisionOperator` to velocity registers across space-time lattice sites. |
| `qlbm/components/spacetime/collision/d2q4_old.py:16` | `SpaceTimeD2Q4CollisionOperator` | Legacy pairwise $R_y(\theta)$ rotation between opposite velocity pairs ($|0101\rangle \leftrightarrow |1010\rangle$) in D2Q4. |
| `qlbm/components/lqlga/collision.py:14` | `GenericLQLGACollisionOperator` | Applies `EQCCollisionOperator` across local velocity registers in `LQLGALattice`. |

**Crucial Finding:** None of the collision classes implement the Boltzmann collision operator with BGK relaxation or local equilibrium distribution functions ($f_i^{(eq)}$). All existing collisions are either quantum lattice gas (Q-LGA) equivalence-class mixing or experimental truncated QFT averaging.

### 3.2 Literature and Phenomenological Keyword Search
Exhaustive search across `qlbm/`, `docs/`, `demos/`, and `test/` for classical LBM and multiphase concepts:

| Keyword | Status | Details / Location |
| :--- | :--- | :--- |
| **BGK** | **ABSENT** | Zero occurrences in all source files, tests, documentation, and notebooks. |
| **relaxation** | **ABSENT** | Zero occurrences relating to single/multi-relaxation time ($\tau$). |
| **equilibrium** | **ABSENT** | Zero occurrences relating to Maxwell-Boltzmann or polynomial equilibrium distribution $f_i^{(eq)}$. |
| **D1Q2** | **IMPLEMENTED** | Present in SpaceTime and LQLGA discretizations. |
| **D2Q4** | **IMPLEMENTED** | Supported in SpaceTime, LQLGA, EQC collisions. |
| **D3Q6** | **PARTIALLY IMPLEMENTED** | Defined in `properties_base.py` and supported in `EQCCollisionOperator`, but streaming/BCs incomplete. |
| **D1Q3** | **PARTIALLY IMPLEMENTED** | Supported in `ABStreamingOperator` and `ABEAveragedCollisionOperator`. |
| **D2Q9** | **IMPLEMENTED (Collisionless)** | Fully supported in `ABLattice`, `OHLattice`, `ABStreamingOperator`, `ABReflectionOperator`. |
| **D3Q19** | **ABSENT** | Zero occurrences. Not defined in `LatticeDiscretization`. |
| **multiphase** | **ABSENT** | Zero occurrences across the entire repository. |
| **phase field** | **ABSENT** | Zero occurrences. |
| **pseudopotential** | **ABSENT** | Zero occurrences (no Shan-Chen model). |
| **free energy** | **ABSENT** | Zero occurrences. |
| **Shan-Chen** | **ABSENT** | Zero occurrences. |
| **color gradient** | **ABSENT** | Zero occurrences. |
| **entropic** | **ABSENT** | Zero occurrences. |
| **volume of fluid / VOF** | **ABSENT** | Zero occurrences. |
| **interface** | **ABSENT (Physics)** | Only matched software API interfaces (e.g., class interfaces, ParaView file headers). Zero physical interface tracking code. |
| **surface tension** | **ABSENT** | Zero occurrences. |

---

## 4. Status Breakdown of Repository Functionality

### 4.1 IMPLEMENTED
1. **Multi-Speed Collisionless QLBM (MSQLBM):**
   - Lattice representation: `MSLattice` with logarithmic position registers ($g_x, g_y$), speed magnitude registers ($v_x, v_y$), direction qubits ($vd_x, vd_y$), velocity ancillae ($a_v$), obstacle ancillae ($a_o$), and comparator ancillae ($a_c$).
   - Streaming: `MSStreamingOperator` using Draper QFT adders (`ControlledIncrementer`) conditioned on direction and CFL substep counters.
   - Geometry: 2D and 3D `Block` (cuboids) encoded into `ReflectionWall`, `ReflectionPoint`, `ReflectionResetEdge`.
   - Boundary reflection: `BounceBackReflectionOperator` and `SpecularReflectionOperator` using single-register quantum comparators and controlled phase/X logic.
   - Measurement: `GridMeasurement` extracting particle probability distributions across grid registers.

2. **Amplitude-Based D2Q9 QLBM (ABQLBM):**
   - Lattice representation: `ABLattice` with logarithmic position registers and $\lceil \log_2 9 \rceil = 4$ velocity qubits encoding discrete directions $0 \dots 8$.
   - One-Hot Alternative: `OHLattice` with 9 separate velocity qubits.
   - Streaming: `ABStreamingOperator` using controlled coordinate shifts for all 9 lattice vectors.
   - Boundary conditions: `ABReflectionOperator` (exact velocity channel permutation for bounce-back and specular walls) and `ABZoneAgnosticReflectionOperator`.

3. **Space-Time CBSE & LGA (SpaceTimeQLBM & LQLGA):**
   - Full computational basis state encoding across space and velocity channels.
   - Exact unitary equivalence-class collision (`EQCCollisionOperator`) conserving mass and momentum via permutation and DFT blocks.

4. **Infrastructure & Tooling:**
   - Runners: `QiskitRunner` (statevector, sampling) and `QulacsRunner` (CPU/GPU acceleration).
   - Compiler: `CircuitCompiler` wrapping Qiskit and PyTket.
   - Simulation pipeline: `SimulationConfig` coupling initial conditions, algorithm, postprocessing, measurement, and runner.
   - Visualization: Exporters writing ParaView `.vti` unstructured/structured grid files.

### 4.2 PARTIALLY IMPLEMENTED
1. **AB Averaged Collision (`ABEAveragedCollisionOperator`):**
   - Only defined for D1Q3 using `TruncatedQFT` on velocity qubits. Does not represent macroscopic fluid viscosity or relaxation to equilibrium.
2. **D3Q6 Space-Time Modeling:**
   - Collision operator implemented, but full 3D streaming and volumetric boundary conditions for D3Q6 are incomplete.
3. **Multi-step state carry-over (without measurement):**
   - For amplitude-encoded methods (`MSQLBM`, `ABQLBM`), evolving past step $t=1$ in a real quantum device requires either mid-circuit measurement with feed-forward, unitary non-destructive re-encoding, or repeated state preparation. On classical simulators, `Reinitializer` uses statevector snapshots.

### 4.3 ONLY DOCUMENTED
1. **Collisionless nature of MSQLBM:** Documented in docstrings citing Schalkers & Möller (2024 JCP).
2. **D2Q9 velocity vectors:** Documented in `LatticeDiscretizationProperties` and test comments, but collision operators for D2Q9 do not exist.

### 4.4 ABSENT (Must Be Developed for Thesis Scope)
1. **Classical LBM Reference Solver:** Standard classical D2Q9 BGK solver (needed as ground-truth benchmark).
2. **Quantum Collision Operator for D2Q9:** Unitary quantum representation of local relaxation toward equilibrium.
3. **Multiphase Fluid Physics:**
   - Interfacial dynamics, surface tension, density/viscosity ratios.
   - Multiphase formulations (Shan-Chen, Color-Gradient, Free-Energy, or Phase-Field).
4. **Volume-of-Fluid (VoF) Method:**
   - Phase volume fraction $\alpha(\mathbf{x}, t)$, advection, interface reconstruction (PLIC), mass conservation.
5. **Hybrid VoF + LBM Coupling Architecture:**
   - Bidirectional coupling between hydrodynamic velocity field and interface advection/reconstruction.
6. **Dam-Break Benchmark:**
   - Fluid column collapse under gravity against walls, front position tracking, interface deformation.

---

## 5. Architectural & Technical Limitations

1. **Unitary vs. Dissipative Physics:**
   - LBM collision with BGK relaxation is inherently dissipative (entropy-increasing) and non-linear (equilibrium depends quadratically on local velocity $\mathbf{u} = \frac{1}{\rho}\sum f_i \mathbf{c}_i$).
   - Quantum circuits execute strictly linear, unitary transformations ($U^\dagger U = I$).
   - This fundamental physical distinction explains why MSQLBM and ABQLBM are collisionless: streaming and boundary reflection are permutations/phase-shifts (pure unitaries), whereas standard BGK collision requires non-unitary relaxation or dilated ancilla-assisted unitary embeddings (e.g., Carleman linearization, Block Encoding, or Quantum Singular Value Transformation).

2. **Amplitude Encoding vs. Basis Encoding:**
   - In `MSLattice` and `ABLattice`, the distribution $f(\mathbf{x}, \mathbf{c})$ is encoded in quantum state amplitudes:
     $$|\Psi\rangle = \sum_{\mathbf{x}, i} \sqrt{f_i(\mathbf{x})} |\mathbf{x}\rangle |i\rangle$$
   - Calculating local density $\rho(\mathbf{x}) = \sum_i f_i(\mathbf{x})$ and velocity $\mathbf{u}(\mathbf{x}) = \frac{1}{\rho}\sum_i f_i \mathbf{c}_i$ from amplitudes at every lattice site simultaneously requires full state tomography or measurement collapse, destroying the quantum superposition.

3. **Multi-Speed Velocity Discretization:**
   - `MSLattice` uses independent Cartesian speed components $(v_x, v_y) \in \{0, \dots, N_v-1\}^2$ with directions $(\pm, \pm)$. This forms a tensor-product grid of speeds, not the isotropic discrete velocity stencils of classical LBM ($D_2Q_9$, $D_3Q_{19}$) which have specific lattice weights $w_i$ ensuring Galilean invariance and recovering the Navier-Stokes equations in the low-Mach limit.

---

## 6. Reusable Repository Components for Thesis Progression

The following existing components provide solid building blocks:
1. `MSLattice` & `MSStreamingOperator`: Verified, working multi-speed quantum streaming using Draper QFT adders.
2. `BounceBackReflectionOperator` & `SpecularReflectionOperator`: Verified comparator-driven quantum boundary reflections on cuboid geometry.
3. `ABLattice` & `ABStreamingOperator`: Validated quantum streaming on standard D2Q9 lattice topologies with compact register allocation ($\log_2 N_x + \log_2 N_y + 4$ qubits).
4. `SimulationConfig`, `QiskitRunner`, `QulacsRunner`: High-performance execution backends supporting statevector extraction, shot sampling, and ParaView `.vti` export for validation.
5. `SingleRegisterComparator`, `ControlledIncrementer`, `TruncatedQFT`: Robust quantum arithmetic subroutines in `qlbm/components/common/`.

---

## 7. Conclusions & Research Roadmap

The repository currently provides an implementation of **collisionless quantum streaming and boundary conditions** in two main flavors:
- Multi-speed Cartesian transport (`MSQLBM`)
- Discrete velocity $D_d Q_q$ transport (`ABQLBM`)

To achieve the thesis objective (Quantum Lattice Boltzmann Method for Two-Phase Gas-Liquid Flow and Dam-Break Problems):
1. **Phases 1–6:** Deeply understand and validate the existing collisionless MS components (`MSLattice`, streaming, geometry, bounce-back, specular reflection, full MSQLBM).
2. **Phase 7:** Perform a rigorous investigation into collision mechanics across the repository and quantum computing literature.
3. **Phases 8–12:** Establish a clean classical D2Q9 reference solver, formulate the quantum-classical state mapping, investigate D2Q9 quantum representation, design a small quantum collision prototype, and assemble a full quantum LBM step.
4. **Phases 13–15:** Perform literature analysis of multiphase LBM families, select the technically justified multiphase model, and implement/validate a classical multiphase reference solver.
5. **Phases 16–19:** Formulate interface tracking via Volume-of-Fluid (VoF), design the coupling architecture between LBM hydrodynamics and VoF advection, and build a classical VoF+LBM solver.
6. **Phases 20–24:** Analyze quantum compatibility across the hybrid pipeline, define the hybrid quantum-classical boundary, execute a proof-of-concept dam-break benchmark, and document the complete framework.
