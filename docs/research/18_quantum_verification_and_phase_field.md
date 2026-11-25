# Research Note 18: Quantum Fluid Verification, Repeat-Until-Success Scaling & Phase-Field Interface Model

## 1. Executive Summary

This research note documents the resolution of the five core scientific review issues raised on our 14-qubit multi-step quantum D3Q19 fluid-evolution prototype (commit `9613203` in `aswanthbajith/QLBM`), establishes the first complete physical basis gate transpilation audit, benchmarks Repeat-Until-Success (RUS) multi-step execution up to 16 time steps, and presents the mathematical formulation and working prototype of a two-distribution **Quantum Phase-Field (Cahn-Hilliard)** interface model.

### Key Milestones Achieved:
1. **Collision Discrepancy Resolution**: Identified and resolved the root cause of the apparent ~0.396 collision error in `collision_benchmarks.py`. The discrepancy was caused by redundant tensor dimension expansion (`u_in[:, None, None, None]`), which corrupted NumPy broadcasting. Upon passing native 4D arrays, the true error between Linearized Quantum BGK and Non-linear BGK is **$0.0090625$**, strictly adhering to the analytical low-Mach quadratic bound $\frac{5}{6\tau} \|\mathbf{u}\|^2 = 0.015104$.
2. **Independent Classical Reference**: Developed `independent_classical_d3q19.py`, an independent classical D3Q19 solver completely decoupled from quantum classes. The quantum statevector stepper matches the classical solver trajectory to machine precision (**$3.23 \times 10^{-12}$ after 16 continuous time steps**).
3. **Physical Zero-Penetration Wall Flux Verification**: Built `wall_flux_verification.py` to evaluate face-normal streaming and bounce-back across stationary basin floors, elevated shelves, vertical cliffs, and closed barrier gates. Verified **$0.00 \times 10^0$ net fluid penetration** into solid boundaries across all geometric features.
4. **Hardware Gate Transpilation Audit**: Transpiled 11-qubit and 14-qubit fluid step circuits into physical basis gates (`['cx', 'u']`). Identified gate count scaling: an 11-qubit step requires 5,238 CNOTs (depth 8,853), and a 14-qubit step requires 7,786 CNOTs (depth 12,753, estimated runtime ~1.75 ms on superconducting architectures).
5. **Multi-Step Scaling & Repeat-Until-Success (RUS) Analysis**: Benchmarked 1, 2, 4, 8, and 16 consecutive quantum time steps with circuit-executable ancilla measurements. The cumulative survival probability after 16 steps is $79.5\%$, requiring an expected Repeat-Until-Success shot multiplier of only **$1.26\times$**. Mass conservation error remained bounded at **$1.07 \times 10^{-9}$**.
6. **Quantum Phase-Field (Cahn-Hilliard) Prototype**: Designed and implemented a two-distribution model (D3Q19 for velocity, D3Q7 for order parameter $\phi \in [-1, 1]$). Formulated exact 4-qubit unitary block-encoding for non-unitary dissipative Cahn-Hilliard collision, addressed non-linear chemical potential $\mu = A(\phi^3 - \phi) - \kappa \nabla^2 \phi$ via measurement-assisted forcing, and verified the analytical $\tanh$ planar equilibrium profile with order parameter drift $< 10^{-15}$.

---

## 2. Scientific Review: Root-Cause Analysis and Resolutions

### 2.1 Resolution of the ~0.396 Collision Discrepancy (Issue 5)
In commit `9613203`, the collision benchmark reported an unexpectedly large discrepancy of $0.39589$ between the linearized quantum collision operator and classical non-linear BGK.

**Investigation**:
In `research/quantum_fluid_evolution/collision/collision_benchmarks.py`, macroscopic velocities were extracted via `compute_macroscopic(f_in[:, None, None, None])`, returning `u_field` with shape `(3, 1, 1, 1)`. In the subsequent call:
```python
# BUGGY CODE:
feq_actual = compute_equilibrium(rho_field, u_in[:, None, None, None])
```
Adding three additional `None` axes transformed a 4D array into a 7D array `(3, 1, 1, 1, 1, 1, 1)`. This broke tensor broadcasting in `compute_equilibrium`, causing erroneous dot products $\mathbf{c}_i \cdot \mathbf{u}$.

**Resolution**:
Passing `u_field` directly:
```python
# CORRECTED CODE:
feq_actual = compute_equilibrium(rho_field, u_field)
f_bgk = bgk_collision(f_in[:, None, None, None], feq_actual, tau=tau)[:, 0, 0, 0]
```
**Result**:
- Apparent Buggy Discrepancy: $0.39589$
- True Discrepancy: **$0.0090625$**
- Analytical Theoretical Bound $\frac{5}{6\tau} \|\mathbf{u}\|^2$: **$0.015104$**
The true error is $43\times$ smaller than previously reported and strictly satisfies the low-Mach asymptotic bound.

---

### 2.2 Independent Classical D3Q19 Reference Solver (Issue 2)
To provide rigorous ground truth, `IndependentClassicalD3Q19Solver` was implemented from scratch with zero dependencies on any quantum modules or quantum helper functions.

**Algorithmic Formulation**:
1. Spatial streaming via cyclic boundary shifts:
   $$f_i^*(\mathbf{x} + \mathbf{c}_i \Delta t) = f_i(\mathbf{x}, t)$$
2. Halfway bounce-back on solid obstacle nodes:
   $$f_i(\mathbf{x}_{\text{fluid}}, t + \Delta t) = f_{\bar{i}}^*(\mathbf{x}_{\text{fluid}}, t) \quad \text{for } \mathbf{x}_{\text{fluid}} + \mathbf{c}_i \in \text{Solid}$$
3. Dual collision engines:
   - **Linearized BGK**: Evaluates $\mathbf{E} \mathbf{f}$ using exact moment projection and applies $(1 - \omega)\mathbf{f} + \omega \mathbf{E}\mathbf{f} + \mathbf{G}\mathbf{f}$.
   - **Non-Linear BGK**: Computes $\rho = \sum f_i$, $\mathbf{u} = \frac{1}{\rho}\sum f_i \mathbf{c}_i + \frac{\mathbf{g}\Delta t}{2}$, evaluates full quadratic $f_i^{\text{eq}}(\rho, \mathbf{u})$, and applies Guo forcing.

**Validation**:
Comparison against `QuantumFluid3DStepper` across 4 time steps yields a maximum point-wise difference of **$9.37 \times 10^{-13}$**, confirming exact equivalence.

---

### 2.3 Physical Wall Boundary Face-Flux Verification (Issue 3)
A fluid solver must strictly prevent fluid leakage across physical boundaries. `WallFaceFluxVerifier` was constructed to measure discrete face fluxes:
$$J_{\text{face}} = \sum_{\mathbf{x} \in \text{Fluid}} \sum_{\substack{i=1 \\ \mathbf{x} + \mathbf{c}_i \in \text{Solid}}}^{18} f_i(\mathbf{x}) \cdot (\mathbf{c}_i \cdot \mathbf{n}_{\text{face}})$$

**Scorecard**:
| Boundary Feature | Solid Nodes | Net Penetration Flux | Max Face Error | Status |
|------------------|-------------|----------------------|----------------|--------|
| **Stationary Basin Floor ($z=0$)** | 32 | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASSED** |
| **Elevated Shelf & Cliff Drop** | 64 | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASSED** |
| **Closed Barrier Gate ($x=x_{\text{gate}}$)** | 32 | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASSED** |

Fluid mass penetrating solid obstacle domains is identically zero to machine precision.

---

### 2.4 Hardware Gate Transpilation & CNOT Budget (Issue 4)
High-level quantum circuits with multi-qubit unitaries and multi-controlled gates (`mcx`, `mcphase`) hide massive gate counts. Transpiling to physical basis gates (`['cx', 'u']`) reveals the true compilation footprint:

| Parameter | 11-Qubit Micro-Lattice ($2 \times 2 \times 4$) | 14-Qubit Grid ($4 \times 4 \times 8$) |
|-----------|------------------------------------------------|---------------------------------------|
| **Spatial Qubits** | 4 ($1 + 1 + 2$) | 7 ($2 + 2 + 3$) |
| **Velocity Qubits** | 5 | 5 |
| **Ancilla Qubits** | 2 | 2 |
| **Total Qubits** | **11** | **14** |
| **Undecomposed Circuit Depth** | 102 | 136 |
| **Undecomposed Gate Count** | 129 | 165 |
| **Transpiled CNOT (CX) Count** | **5,238** | **7,786** |
| **Transpiled Single-Qubit (U) Count** | 6,708 | 9,588 |
| **Total Physical Basis Gates** | 11,946 | 17,374 |
| **Transpiled Circuit Depth** | **8,853** | **12,753** |
| **Estimated Execution Time** ($\tau_{\text{2Q}}=200\text{ns}$) | **1.18 ms** | **1.75 ms** |

**Hardware Takeaway**:
Executing even a single multi-step step circuit on current NISQ processors exceeds typical coherence times ($T_2 \sim 100\text{--}300\,\mu\text{s}$). Error-mitigated or early fault-tolerant quantum hardware with logical CNOTs will be necessary for multi-step runs.

---

## 3. Multi-Step Repeat-Until-Success (RUS) Scaling Suite

We modeled circuit-executable measurement, ancilla reset, and Repeat-Until-Success (RUS) statistics across $K \in \{1, 2, 4, 8, 16\}$ consecutive time steps on the 14-qubit lattice ($4 \times 4 \times 8$, 16,384 amplitudes):

$$E[N_{\text{trials}}] = \frac{1}{P_{\text{cumul}}} = \prod_{s=1}^K \frac{1}{P_{\text{succ}}^{(s)}}$$

### Scaling Benchmark Results:
| Steps $K$ | Single-Step $P_{\text{succ}}$ | Cumulative $P_{\text{cumul}}$ | RUS Cost Multiplier | Momentum $J_z$ | Mass Error | vs Linear D3Q19 | vs Non-linear D3Q19 |
|-----------|-------------------------------|-------------------------------|---------------------|----------------|------------|-----------------|---------------------|
| **1** | $0.993568$ | $0.993568$ | $1.01\times$ | $-0.300000$ | $6.66 \times 10^{-11}$ | $3.69 \times 10^{-13}$ | $1.0531 \times 10^{-3}$ |
| **2** | $0.949279$ | $0.943173$ | $1.06\times$ | $-0.550000$ | $1.33 \times 10^{-10}$ | $5.34 \times 10^{-13}$ | $2.4114 \times 10^{-2}$ |
| **4** | $0.982524$ | $0.884778$ | $1.13\times$ | $-0.867222$ | $2.67 \times 10^{-10}$ | $9.37 \times 10^{-13}$ | $2.8901 \times 10^{-2}$ |
| **8** | $0.993317$ | $0.846661$ | $1.18\times$ | $-0.625098$ | $5.34 \times 10^{-10}$ | $1.91 \times 10^{-12}$ | $2.4026 \times 10^{-2}$ |
| **16** | $0.993256$ | $0.795036$ | **$1.26\times$** | $+0.901462$ | **$1.07 \times 10^{-9}$** | **$3.23 \times 10^{-12}$** | **$3.5799 \times 10^{-2}$** |

### Key Findings:
1. **Precision**: The quantum statevector trajectory matches the independent classical linearized solver down to $3.23 \times 10^{-12}$ after 16 continuous steps.
2. **Shot Overhead**: Because the collision and boundary block-encodings are well-conditioned, single-step herald probabilities average $\approx 99.3\%$. The cumulative survival probability after 16 steps is $79.5\%$, meaning that on average only **1.26 circuit executions** are required to complete the full 16-step trajectory.
3. **Low-Mach Discrepancy**: The difference between the linearized quantum evolution and the non-linear classical solver remains bounded at $\sim 0.0358$ across all 16 steps.

---

## 4. Quantum Phase-Field (Cahn-Hilliard) Interface Prototype

### 4.1 Why Phase-Field over Volume-of-Fluid (VOF)?
The geometric Volume-of-Fluid (VOF) method relies on sharp Heaviside step functions, discontinuous cell-state topological transitions (empty $\leftrightarrow$ interface $\leftrightarrow$ full), and geometric normal reconstruction (Youngs / PLIC). These operations are non-unitary, non-differentiable, and fundamentally incompatible with continuous unitary quantum gates.

In contrast, **Phase-Field (Cahn-Hilliard) models** track the interface via a continuous order parameter $\phi \in [-1, 1]$ governed by the Cahn-Hilliard equation:
$$\frac{\partial \phi}{\partial t} + \nabla \cdot (\phi \mathbf{u}) = \nabla \cdot (M \nabla \mu)$$
where $\mu$ is the chemical potential and $M$ is the mobility. In the Lattice Boltzmann framework, this is evolved by a second distribution set $g_i(\mathbf{x}, t)$, making it naturally compatible with quantum streaming and collision.

### 4.2 Two-Distribution Lattice Architecture
- **Hydrodynamic Field ($f_i$)**: D3Q19 lattice (19 velocities, 5 velocity qubits).
- **Order Parameter Field ($g_i$)**: D3Q7 lattice (7 velocities: $\mathbf{c}_0 = \mathbf{0}$, $\mathbf{c}_{1..6} = (\pm 1, 0, 0), (0, \pm 1, 0), (0, 0, \pm 1)$).
  - Weights: $w_0 = 1/4$, $w_{1..6} = 1/8$.
  - Sound speed squared: $c_{s,g}^2 = 1/4$.
  - Register encoding: $\lceil \log_2(7) \rceil = 3$ velocity qubits.
  - Order parameter: $\phi(\mathbf{x}) = \sum_{i=0}^6 g_i(\mathbf{x})$.

### 4.3 Addressing the Two Quantum Obstacles

#### 1. Non-Unitary Dissipation
The discrete Cahn-Hilliard BGK collision on D3Q7 is strictly dissipative:
$$\mathbf{M}_{\text{diss}} = \left(1 - \frac{1}{\tau_\phi}\right) \mathbf{I}_7 + \frac{1}{\tau_\phi} \mathbf{w} \mathbf{1}^T$$
With $\tau_\phi = 0.8$, the non-conserved eigenvalues are $1 - 1/\tau = -0.25 < 1$.
We embed $\mathbf{M}_{\text{diss}} / \alpha$ (padded to $8 \times 8$) into an exact 4-qubit unitary dilation $\mathbf{U}_{\text{block}} \in \mathbb{C}^{16 \times 16}$:
$$\mathbf{U}_{\text{block}} = \begin{bmatrix} \mathbf{A} & \sqrt{\mathbf{I} - \mathbf{A}\mathbf{A}^\dagger} \\ \sqrt{\mathbf{I} - \mathbf{A}^\dagger\mathbf{A}} & -\mathbf{A}^\dagger \end{bmatrix}, \quad \mathbf{A} = \frac{\mathbf{M}_8}{\alpha}$$
Verifying $\|\mathbf{U}_{\text{block}} \mathbf{U}_{\text{block}}^\dagger - \mathbf{I}\| < 10^{-15}$. Projecting the ancilla onto $|0\rangle$ yields exact dissipative evolution with average success probability $P_{\text{succ}} \approx 78.5\%$.

#### 2. Non-Linear Chemical Potential $\mu(\phi)$
The chemical potential derived from the Ginzburg-Landau double-well free energy $\Psi(\phi) = \frac{1}{4}A(\phi^2 - 1)^2$ is:
$$\mu = A (\phi^3 - \phi) - \kappa \nabla^2 \phi$$
The cubic term $\phi^3$ is non-linear. By the No-Cloning theorem and linearity of quantum mechanics, a linear unitary operator cannot map $\sum c_i |i\rangle \to \sum c_i^3 |i\rangle$ without ancillary resources.
We formulate a **measurement-assisted non-linear forcing oracle**:
1. Heralded projective readout samples the local order parameter $\phi(\mathbf{x})$.
2. The non-linear potential $\mu(\mathbf{x})$ and Korteweg interfacial force $\mathbf{F}_s = \mu \nabla \phi$ are computed.
3. The surface force $\mathbf{F}_s$ is encoded into the D3Q19 quantum forcing block as a linear source operator.

### 4.4 Analytical Verification of the Phase-Field Prototype
The prototype was verified against the analytical 1D planar hyperbolic-tangent equilibrium profile:
$$\phi(z) = \tanh\left(\frac{z - z_0}{\xi}\right), \quad \xi = \sqrt{\frac{2\kappa}{A}}$$
- **Interface Width**: $\xi = 1.0000$ lattice units
- **Surface Tension (Analytical)**: $\sigma_{\text{anal}} = \frac{2\sqrt{2}}{3}\sqrt{\kappa A} = 0.026667$
- **Surface Tension (Numerical)**: $\sigma_{\text{num}} = \sum \kappa (\partial_z \phi)^2 = 0.021533$ (discrete stencil error $\approx 19.2\%$)
- **Max Chemical Potential $|\mu|$**: $4.00 \times 10^{-2}$ (mean $5.75 \times 10^{-3}$)
- **Equilibrium Invariance**: Applying the quantum dissipative collision operator to the analytical equilibrium state yields an order parameter drift of **$2.22 \times 10^{-16}$** (exact machine precision).

---

## 5. Summary & Verification Scorecard

| Feedback Item | Target | Delivered Solution | Status |
|---------------|--------|--------------------|--------|
| **1. Shot Overhead & RUS** | Circuit-executable measurement & shot modeling | 1-16 step Repeat-Until-Success runner; $1.26\times$ cost at 16 steps | **COMPLETED** |
| **2. Independent Classical Solver** | Pure classical reference with zero quantum dependencies | `IndependentClassicalD3Q19Solver` matching quantum stepper to $10^{-12}$ | **COMPLETED** |
| **3. Wall Face Flux Verification** | Verify zero fluid penetration on floor, cliff, gate | `WallFaceFluxVerifier` demonstrating $0.00 \times 10^0$ net leakage | **COMPLETED** |
| **4. Hardware Gate Transpilation** | Decompose into native CX + U basis gates | Transpiled circuits: 5,238 CX (11Q) and 7,786 CX (14Q) | **COMPLETED** |
| **5. Collision Discrepancy** | Resolve ~0.396 error vs non-linear BGK | Fixed 7D broadcasting bug; true error is $0.00906 \le 0.0151$ | **COMPLETED** |
| **Phase-Field Prototype** | Minimal Cahn-Hilliard model addressing dissipation & $\mu$ | Two-distribution D3Q19+D3Q7 prototype with block-encoding | **COMPLETED** |
