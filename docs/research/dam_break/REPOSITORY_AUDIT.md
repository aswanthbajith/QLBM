# Repository Audit: Quantum-Circuit Dam-Break Simulator

**Date:** 2026-10-23  
**Project:** Hybrid Classical–Quantum Lattice Boltzmann Dam-Break Simulation  
**Repository:** `~/Research/qlbm`  
**Branch:** `research/multiphase-vof`  
**Author:** Computational Fluid Dynamics & Quantum Algorithms Research Team  

---

## 1. Existing Working Components

The repository possesses a well-structured and fully functional foundation across classical hydrodynamics, geometric Volume-of-Fluid (VOF) modeling, and quantum circuit primitives:

| Component | Location | Implementation Status | Functional Description |
| :--- | :--- | :---: | :--- |
| **D2Q9 Lattice Model** | [`qlbm/classical/d2q9.py`](file:///home/aswa/Research/qlbm/qlbm/classical/d2q9.py) | **VERIFIED** | Lattice vectors $\mathbf{c}_i$, weights $w_i$, speed of sound $c_s^2 = 1/3$, opposite directions, equilibrium $f_i^{\text{eq}}$, moments $\rho, \mathbf{u}$, Guo forcing $F_i$, BGK collision, and halfway bounce-back `stream_and_bounce`. |
| **VOF Geometry & Reconstruction** | [`qlbm/classical/vof.py`](file:///home/aswa/Research/qlbm/qlbm/classical/vof.py) | **VERIFIED** | Discrete cell states (`GAS`, `INTERFACE`, `LIQUID`, `SOLID`), 2nd-order Youngs 9-point normal vector estimation, Scardovelli–Zaleski closed-form PLIC forward area and inverse $\alpha$, `VOFInterface` container. |
| **Classical Reference Solver** | [`qlbm/classical/hybrid_vof_lbm.py`](file:///home/aswa/Research/qlbm/qlbm/classical/hybrid_vof_lbm.py) | **VERIFIED** | Full 8-stage hybrid VOF-LBM solver (`VOFHydroSolver`): decoupled ambient gas phase ($\rho_L/\rho_G = \infty$), pairwise anti-symmetric directional mass flux $\Delta M_i$, topological completeness $\mathcal{N}(\text{LIQUID}) \cap \text{GAS} = \emptyset$, Young–Laplace capillary pressure jump, dam-break initialization. |
| **Quantum Lattice Abstraction** | [`qlbm/lattice/lattices/ab_lattice.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/ab_lattice.py) | **VERIFIED** | Amplitude-based lattice encoding (`ABLattice`) allocating coordinate registers for $N_x, N_y$ (powers of 2), 4 velocity qubits for D2Q9, and 1 obstacle ancilla. |
| **Quantum Streaming Operator** | [`qlbm/components/ab/streaming.py`](file:///home/aswa/Research/qlbm/qlbm/components/ab/streaming.py) | **VERIFIED** | Draper QFT adder streaming implementation shifting coordinate registers conditioned on velocity states in Fourier space. |
| **Quantum D2Q9 Streamer** | [`qlbm/components/vof/streaming.py`](file:///home/aswa/Research/qlbm/qlbm/components/vof/streaming.py) | **VERIFIED** | High-level wrapper `QuantumD2Q9StreamingOperator` mapping 3D distribution arrays $f_i(x, y)$ into quantum statevectors, executing QFT streaming, and decoding back to classical arrays with error $< 1.37 \times 10^{-13}$. |
| **Quantum D2Q9 Bounce-Back** | [`qlbm/components/vof/boundary.py`](file:///home/aswa/Research/qlbm/qlbm/components/vof/boundary.py) | **VERIFIED** | Permutation circuit `QuantumD2Q9BounceBackOperator` executing exact unitary opposite-velocity swaps $(1 \leftrightarrow 3), (2 \leftrightarrow 4), (5 \leftrightarrow 7), (6 \leftrightarrow 8)$ on velocity registers. |
| **Hybrid Stepper Prototype** | [`qlbm/components/vof/hybrid_stepper.py`](file:///home/aswa/Research/qlbm/qlbm/components/vof/hybrid_stepper.py) | **VERIFIED** | `HybridVOFStep` orchestrating quantum streaming with classical VOF reconstruction, state transitions, and mass conservation. |

---

## 2. Existing Incomplete Components & Gaps

1. **Quantum Collision Operator**:
   - `QuantumCollisionOperator` in [`qlbm/components/vof/base.py`](file:///home/aswa/Research/qlbm/qlbm/components/vof/base.py) is currently an abstract specification.
   - The classical BGK relaxation step is dissipative and non-unitary; no unitary quantum circuit implements nonlinear BGK collision directly without ancilla dilation, block-encoding, or linearization.
2. **Integrated Spatially-Conditional Quantum Boundary**:
   - `QuantumD2Q9BounceBackOperator` operates on the velocity register as a global permutation or single-control gate. It is not currently coupled to a spatial geometry comparator circuit inside a single monolithic unitary evolution. Wall reflection in the hybrid solver is handled via classical post-streaming reflection on boundary faces.
3. **Multi-Mode Dam-Break Runner**:
   - [`demos/simulation/vof_dam_break.py`](file:///home/aswa/Research/qlbm/demos/simulation/vof_dam_break.py) currently executes the classical solver (`VOFHydroSolver`). It lacks CLI switches for Mode A (Classical reference), Mode B (Quantum streaming verification), and Mode C (Full Hybrid stepper dam break) with unified reporting.
4. **Independent Physical Calibration**:
   - Physical dimensionless number conversion (Froude, Reynolds, capillary numbers) and systematic benchmarking against experimental data (Martin & Moyce 1952) require structured export and documentation.

---

## 3. Mathematical Assumptions

- **Incompressible Hydrodynamic Regime**: The Mach number is constrained to $Ma = |\mathbf{u}| / c_s < 0.15$ where $c_s = 1/\sqrt{3}$ in lattice units ($\Delta x = 1, \Delta t = 1, c_s^2 = 1/3$).
- **BGK Viscosity Relation**: Kinematic shear viscosity $\nu = c_s^2 (\tau - 0.5) \Delta t = \frac{2\tau - 1}{6}$.
- **Guo Body Forcing**: Gravitational acceleration $\mathbf{g} = (0, -g)$ generates force $\mathbf{F} = \rho \mathbf{g}$. Momentum is corrected via $\rho \mathbf{u} = \sum f_i \mathbf{c}_i + \frac{\Delta t}{2}\mathbf{F}$.
- **Decoupled Ambient Gas**: The gas phase is treated as non-participating ($\rho_G = 0$ dynamically), corresponding to an infinite density ratio $\rho_L / \rho_G = \infty$.
- **Capillary Interface Condition**: Interface pressure incorporates the Young–Laplace capillary pressure jump $p_{\text{iface}} = p_{\text{gas}} + \sigma \kappa$, where curvature $\kappa = \nabla \cdot \hat{\mathbf{n}}$ is evaluated from 2nd-order Youngs normal vectors.
- **Exact Geometric Mass Conservation**: Net mass flux across fluid links is strictly pairwise anti-symmetric: $\Delta M_i(\mathbf{x}) = -\Delta M_{\bar{i}}(\mathbf{x}+\mathbf{c}_i)$, with contact area $A_i(\mathbf{x}) = \frac{1}{2}[C(\mathbf{x}) + C(\mathbf{x}+\mathbf{c}_i)]$.

---

## 4. Boundary-Condition Implementation

- **Solid Walls**:
  - Configured via boolean array `solid_mask` of shape $(N_y, N_x)$.
  - Standard halfway bounce-back reflects outgoing post-collision populations $f_i^*(\mathbf{x}_f)$ back into opposite fluid directions $f_{\bar{i}}(\mathbf{x}_f, t+1)$.
  - In the hybrid solver, walls along $y=0$, $x=0$, and $x=N_x-1$ intercept fluid links before coordinate wraparound can occur.
- **Free Surface / Gas Reconstruction**:
  - Körner et al. (2005) reconstruction on interface cells having gas neighbors in direction $\mathbf{c}_i$:
    $$f_{\bar{i}}(\mathbf{x}_{\text{iface}}, t+1) = f_i^{\text{eq}}(\rho_{\text{iface}}, \mathbf{u}) + f_{\bar{i}}^{\text{eq}}(\rho_{\text{iface}}, \mathbf{u}) - f_i^*(\mathbf{x}_{\text{iface}}, t)$$
- **Topological Rule**:
  - The solver enforces $\mathcal{N}(\text{LIQUID}) \cap \text{GAS} = \emptyset$. Any liquid cell in 8-neighborhood contact with a gas cell is converted into an interface cell.

---

## 5. Quantum-Circuit Register Layout

For an $N_x \times N_y$ lattice with power-of-two dimensions:
- Coordinate qubits: $n_x = \log_2 N_x$, $n_y = \log_2 N_y$.
- Velocity qubits: $n_v = \lceil \log_2 9 \rceil = 4$.
- Obstacle ancilla: $n_{\text{anc}} = 1$.
- Total qubits: $N_{\text{qubits}} = n_x + n_y + 4 + 1$.

### Qubit Allocation Table

| Register | Name | Size | Bit Range | Description |
| :--- | :--- | :---: | :---: | :--- |
| `grid_x` | $x$ coordinate | $n_x$ | $0 \dots n_x-1$ | Binary representation of $x \in \{0, \dots, N_x-1\}$ |
| `grid_y` | $y$ coordinate | $n_y$ | $n_x \dots n_x+n_y-1$ | Binary representation of $y \in \{0, \dots, N_y-1\}$ |
| `velocity` | $\mathbf{c}_i$ index | 4 | $n_x+n_y \dots n_x+n_y+3$ | Discrete velocity index $i \in \{0, \dots, 8\}$ |
| `obstacle` | $anc$ | 1 | $n_x+n_y+4$ | Obstacle / reflection marker ancilla |

**Basis-State Indexing:**
$$\text{Index} = x + (y \cdot N_x) + (i \cdot N_x N_y)$$
The velocity states $i \in \{0, \dots, 8\}$ are mapped directly to binary integers $0000_2 \dots 1000_2$. Integers $9 \dots 15$ are unused auxiliary states.

---

## 6. Classical–Quantum Data Flow

```text
[Classical Kinetic State f_i(x,y)]
              |
              v
[Classical Guo Forcing & BGK Collision] -> f_post[i, y, x]
              |
              v
[State Preparation: Amplitude Encoding]
       |psi> = sum sqrt(f_post / F) |x, y, i>
              |
              v
[Quantum Circuit: Draper QFT Streaming]
       U_stream |x, y, i> = |(x + c_ix) mod Nx, (y + c_iy) mod Ny, i>
              |
              v
[State Readout & Probability Extraction]
       P(x, y, i) = |<x, y, i | psi'>|^2
       f_streamed = P(x, y, i) * F
              |
              v
[Classical Boundary Enforcement: Wall Bounce-Back]
              |
              v
[Classical Free-Surface Gas Boundary Reconstruction]
              |
              v
[Classical Pairwise Anti-Symmetric Mass Advection Delta M_i]
              |
              v
[Classical Cell State Transitions & PLIC Geometry Reconstruction]
```

---

## 7. Existing Tests and Execution Baseline

All existing research tests run cleanly on Python 3.14.4 with Qiskit 2.1.0:

1. `tests/research/test_vof_lbm_hybrid.py`: **8/8 PASSED** in 6.07s
   - Local mass flux anti-symmetry ($< 10^{-14}$)
   - Global mass conservation over 50 steps ($< 10^{-12}$)
   - Droplet advection under shear and translation
   - Static droplet Young–Laplace pressure scaling across radii
   - Dam-break column collapse dynamics
2. `tests/research/test_quantum_vof.py`: **8/8 PASSED** in 7.02s
   - Quantum collision operator interface enforcement
   - D2Q9 bounce-back circuit construction and opposite-velocity permutation
   - D2Q9 streaming register sizing and circuit compilation
   - All 9 velocity directions verified against modulo coordinate arithmetic
   - Statevector distribution array streaming equivalence ($\max |f^q - f^c| < 1.37 \times 10^{-13}$)
   - Hybrid stepping mass conservation and state matching

---

## 8. Known Defects, Bottlenecks, and Numerical Risks

1. **Statevector Memory & Simulation Scaling**:
   - Storing $2^{N_{\text{qubits}}}$ complex double amplitudes requires $16 \times 2^{N_{\text{qubits}}}$ bytes.
   - For $N_x \times N_y = 64 \times 32$, $N = 6 + 5 + 4 + 1 = 16$ qubits ($2^{16} = 65,536$ states $\approx 1$ MB).
   - For $128 \times 128$, $N = 7 + 7 + 4 + 1 = 19$ qubits ($2^{19} = 524,288$ states $\approx 8.4$ MB).
   - State preparation and probability extraction require full array loops, introducing a classical overhead that exceeds classical array streaming for CPU simulations.
2. **Periodic Wraparound vs Solid Walls**:
   - Modulo arithmetic in the quantum streaming circuit wraps boundary populations. If the fluid column contacts an open domain edge, mass wraps to the opposite boundary.
   - **Mitigation:** Enforce solid boundaries ($solid\_mask = True$) along the entire perimeter ($y=0$, $x=0$, $x=N_x-1$) so that outgoing populations are reflected before reaching wraparound edges.
3. **Non-Unitary BGK Collision**:
   - BGK collision dissipates kinetic energy into heat / viscous entropy and cannot be represented by a simple unitary circuit without ancilla dilation. Reporting quantum collision as implemented would be scientifically false; collision must remain explicitly classical in Option A.

---

## 9. Features That Can Be Reused Directly

1. Vectorized D2Q9 equilibrium and moment functions in `qlbm/classical/d2q9.py`.
2. Youngs normal estimation and Scardovelli–Zaleski PLIC in `qlbm/classical/vof.py`.
3. `VOFHydroSolver` in `qlbm/classical/hybrid_vof_lbm.py`.
4. `ABLattice` in `qlbm/lattice/lattices/ab_lattice.py`.
5. `ABStreamingOperator` in `qlbm/components/ab/streaming.py`.
6. `QuantumD2Q9StreamingOperator` in `qlbm/components/vof/streaming.py`.
7. `QuantumD2Q9BounceBackOperator` in `qlbm/components/vof/boundary.py`.
8. `HybridVOFStep` in `qlbm/components/vof/hybrid_stepper.py`.

---

## 10. Proposed Implementation Changes

1. **Multi-Mode Simulation Entrypoint**:
   - Enhance [`demos/simulation/vof_dam_break.py`](file:///home/aswa/Research/qlbm/demos/simulation/vof_dam_break.py) with `--mode` (`classical`, `streaming_test`, `hybrid`), unified parameter handling, timing breakdown, and CSV/JSON output generation.
2. **Comprehensive Validation Test Suite**:
   - Implement `tests/research/test_dam_break_validation.py` checking initial conditions, streaming accuracy ($E_{\max} < 10^{-12}$), collision conservation, boundary non-leakage, and end-to-end dam break dynamics.
3. **Visualization Suite**:
   - Implement `demos/visualization/visualize_dam_break.py` to generate the 9 required diagnostic figures saved in `results/dam_break/plots/`.
4. **Physical Benchmark Analysis**:
   - Map lattice parameters to dimensionless time $T = t \sqrt{g/L_0}$ and surge front $Z = x/L_0$, comparing against Martin & Moyce (1952) experimental data.
5. **Research Documentation**:
   - Author `docs/research/dam_break/IMPLEMENTATION.md`, `VALIDATION_REPORT.md`, and `RESULTS.md`.
