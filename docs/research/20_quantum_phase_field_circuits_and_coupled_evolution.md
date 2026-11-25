# Research Note 20: Quantum Conservative Allen-Cahn (QCAC D3Q7) Circuits, Coupled Multiphase Stepper & QLKS Viscosity Decoupling

## 1. Executive Summary

This research note documents the transition of two-phase interface dynamics from classical NumPy array operations into genuine, hardware-transpiled quantum circuits and statevectors. By formulating the **Quantum Conservative Allen-Cahn (QCAC D3Q7)** architecture and establishing the **Coupled Quantum Multiphase 3D Stepper**, we eliminate the last classical kinetic shortcut in our 3D waterfall solver.

### Key Breakthroughs:
1. **Quantum D3Q7 Phase-Field Circuit Engine**:
   - Encodes the 7 discrete velocity directions of the Conservative Allen-Cahn (CAC) lattice into exactly **3 velocity qubits** ($\lceil \log_2(7) \rceil = 3$, states $|000\rangle \dots |110\rangle$), leaving 1 unpopulated zero-padding state.
   - Formulates **Decoupled Cardinal 3D QFT Streaming**: Because D3Q7 velocities lie strictly along cardinal axes ($\pm \mathbf{e}_x, \pm \mathbf{e}_y, \pm \mathbf{e}_z$ with zero diagonals), each spatial coordinate register ($q_x, q_y, q_z$) requires only a single positive and single negative controlled Draper PhaseShift gate.
   - **$20\times$ Hardware Basis Gate Reduction**: On a $2 \times 2 \times 4$ lattice, D3Q7 streaming and boundary execution requires only **265 CNOT (CX) gates** (estimated hardware runtime **$59.94 \, \mu\text{s}$**), compared to **5,238 CNOT gates** ($1.18 \, \text{ms}$) for D3Q19.
2. **Quantum Spatial Obstacle Boundary Operator for D3Q7**:
   - Constructs a multi-controlled spatial oracle $O_{\text{wall}}$ with uncomputed ancilla, driving a controlled 3-qubit conjugate velocity swap unitary $U_{\text{bounce}}$:
     $$|1\rangle \leftrightarrow |2\rangle \, (+x \leftrightarrow -x), \quad |3\rangle \leftrightarrow |4\rangle \, (+y \leftrightarrow -y), \quad |5\rangle \leftrightarrow |6\rangle \, (+z \leftrightarrow -z)$$
     leaving the rest state $|0\rangle$ invariant.
   - Verified exact involution $U_{\text{wall}}^2 = I$ to machine precision ($< 10^{-13}$) with zero residual ancilla entanglement.
3. **Dual-Register Coupled Quantum Multiphase Stepper**:
   - Unifies the 14-qubit Quantum Hydrodynamic Stepper (D3Q19, statevector $|\Psi_f\rangle$) and the 10-qubit Quantum Phase-Field Stepper (D3Q7, statevector $|\Psi_g\rangle$) into a 24-qubit coupled quantum system.
   - The phase-field statevector computes the thermodynamic chemical potential $\mu = \frac{3\sigma}{2\xi}(\phi^3 - \phi) - \frac{3\sigma\xi}{4}\nabla^2\phi$ and injects the non-singular surface tension force $\mathbf{F}_s = \mu \nabla \phi$ into the hydrodynamic forcing block-encoding.
   - The hydrodynamic statevector provides fluid velocity $\mathbf{u}$, which is advected directly in the phase-field equilibrium $g_i^{\text{eq}}(\phi, \mathbf{u})$.
4. **Viscosity & Reynolds Number Decoupling via QLKS (Xiao et al. 2025)**:
   - Implements the Quantum Lattice Kinetic Scheme parameter $A \in (0, 1]$ (arXiv:2505.10883) to scale physical kinematic viscosity $\nu = \frac{1}{3} A (\tau - 0.5)$.
   - Allows setting quantum relaxation to the maximally stable fixed point $\tau = 1.0$ (where $\lambda = 0.0$ is an exact projection without non-unitary dissipation error) while simulating high-Reynolds turbulent waterfall flows.

---

## 2. Mathematical Formulation: Quantum Conservative Allen-Cahn (QCAC D3Q7)

### 2.1 Velocity Lattice & Quantum Register Mapping
The D3Q7 lattice represents the minimal 3D isotropic discrete velocity set capable of recovering the second-order diffusion operator $\nabla^2 \phi$:
$$\mathbf{c}_0 = (0, 0, 0), \quad \mathbf{c}_{1,2} = (\pm 1, 0, 0), \quad \mathbf{c}_{3,4} = (0, \pm 1, 0), \quad \mathbf{c}_{5,6} = (0, 0, \pm 1)$$
Lattice weights:
$$w_0 = \frac{1}{4}, \quad w_{1..6} = \frac{1}{8}, \quad c_s^2 = \frac{1}{4}$$

We map the 7 discrete velocities to a 3-qubit computational basis $|v_2 v_1 v_0\rangle$:
| Velocity $\mathbf{c}_i$ | Direction | Basis State $|v\rangle$ | Binary $|v_2 v_1 v_0\rangle$ |
|---|---|:---:|:---:|
| $\mathbf{c}_0 = (0, 0, 0)$ | Rest | $|0\rangle$ | $|000\rangle$ |
| $\mathbf{c}_1 = (+1, 0, 0)$ | $+x$ | $|1\rangle$ | $|001\rangle$ |
| $\mathbf{c}_2 = (-1, 0, 0)$ | $-x$ | $|2\rangle$ | $|010\rangle$ |
| $\mathbf{c}_3 = (0, +1, 0)$ | $+y$ | $|3\rangle$ | $|011\rangle$ |
| $\mathbf{c}_4 = (0, -1, 0)$ | $-y$ | $|4\rangle$ | $|100\rangle$ |
| $\mathbf{c}_5 = (0, 0, +1)$ | $+z$ | $|5\rangle$ | $|101\rangle$ |
| $\mathbf{c}_6 = (0, 0, -1)$ | $-z$ | $|6\rangle$ | $|110\rangle$ |
| *Padding State* | *Unpopulated* | $|7\rangle$ | $|111\rangle$ |

### 2.2 Decoupled 3D QFT Coordinate Streaming Circuit
Spatial coordinate registers $|x\rangle, |y\rangle, |z\rangle$ are translated along $\mathbf{c}_i$ via Draper QFT adders:
$$U_{\text{stream}} |x, y, z, c_i\rangle = |(x + c_{ix}) \bmod N_x, \, (y + c_{iy}) \bmod N_y, \, (z + c_{iz}) \bmod N_z, \, c_i\rangle$$

In D3Q19, 12 velocities have diagonal couplings ($c_{ix} \ne 0$ and $c_{iy} \ne 0$), requiring 30 distinct controlled PhaseShift gates.
In D3Q7, each spatial dimension has **strictly one positive and one negative shifting velocity**:
- **Dimension $x$**: $+1$ shift conditioned on $|001\rangle$; $-1$ shift conditioned on $|010\rangle$.
- **Dimension $y$**: $+1$ shift conditioned on $|011\rangle$; $-1$ shift conditioned on $|100\rangle$.
- **Dimension $z$**: $+1$ shift conditioned on $|101\rangle$; $-1$ shift conditioned on $|110\rangle$.

Total PhaseShift applications for D3Q7: **exactly 6** (a $5\times$ reduction over D3Q19).

### 2.3 Quantum Spatial Obstacle Boundary Circuit
On solid obstacle nodes $\mathbf{x} \in \Omega_{\text{solid}}$, incoming phase-field populations bounce back:
$$g_i^*(\mathbf{x}_{\text{fluid}}, t + \Delta t) = g_{\bar{i}}(\mathbf{x}_{\text{fluid}}, t)$$

We implement bounce-back via an 8-dimensional orthogonal permutation matrix $U_{\text{bounce}}$ on the 3 velocity qubits:
$$U_{\text{bounce}} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}$$
Notice: $U_{\text{bounce}}^T = U_{\text{bounce}}$ and $U_{\text{bounce}}^2 = I_8$.
The full boundary operator is:
$$U_{\text{wall}} = O_{\text{wall}}^\dagger \cdot \left[ |0\rangle\langle 0|_{\text{wall}} \otimes I_8 + |1\rangle\langle 1|_{\text{wall}} \otimes U_{\text{bounce}} \right] \cdot O_{\text{wall}}$$
This leaves the obstacle ancilla strictly uncomputed in $|0\rangle$, preventing phase decoherence.

---

## 3. Hardware Basis Gate Transpilation Audit

Transpilation into physical hardware basis gates (`['cx', 'u']`) demonstrates the efficiency of the D3Q7 architecture:

| Grid Resolution | Model | Spatial Qubits | Total Qubits | Undecomposed Depth | Basis Depth | CNOT (CX) Gates | 1Q (U) Gates | Est. Hardware Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$2 \times 2 \times 2$** | **QCAC D3Q7** | 3 | **7** | 16 | 310 | **197** | 267 | **$44.74 \, \mu\text{s}$** |
| **$2 \times 2 \times 4$** | **QCAC D3Q7** | 4 | **8** | 21 | 436 | **265** | 347 | **$59.94 \, \mu\text{s}$** |
| **$2 \times 2 \times 4$** | *D3Q19 Hydro* | 4 | *11* | 102 | 8,853 | *5,238* | 6,708 | *$1.18 \, \text{ms}$* |
| **$4 \times 4 \times 4$** | **QCAC D3Q7** | 6 | **10** | 28 | 630 | **405** | 511 | **$91.22 \, \mu\text{s}$** |
| **$4 \times 4 \times 8$** | *D3Q19 Hydro* | 7 | *14* | 136 | 12,753 | *7,786* | 9,588 | *$1.75 \, \text{ms}$* |

**Key Finding**:
The D3Q7 Phase-Field circuit requires **$19.8\times$ fewer CNOT gates** and executes **$19.7\times$ faster** than the hydrodynamic D3Q19 circuit on the same spatial lattice.

---

## 4. Coupled Quantum Multiphase Stepper Architecture

`CoupledQuantumMultiphase3DStepper` orchestrates dual quantum registers:
```
      +--------------------------------------------------------+
      |  Phase-Field Register |Psi_g> (D3Q7, 10-12 Qubits)     |
      +--------------------------------------------------------+
             |                                    ^
             | Order Parameter phi(x)             | Fluid Velocity u(x)
             v                                    |
      Chemical Potential mu(phi)                  | Advection in g_eq
      Interfacial Force F_s = mu grad(phi)        |
             |                                    |
             v                                    |
      +--------------------------------------------------------+
      |  Hydrodynamic Register |Psi_f> (D3Q19, 13-15 Qubits)   |
      +--------------------------------------------------------+
```

### Verification Highlights:
- **Streaming Equivalence**: Statevector evolution matches classical 3D periodic roll with maximum discrepancy **$5.55 \times 10^{-14}$** (machine zero).
- **Boundary Involution**: Obstacle operator involution error $\|U_{\text{wall}}^2 - I\| < 10^{-13}$.
- **Mass Conservation**: Phase-field order parameter conserved across 5 coupled quantum steps to within **$6.28 \times 10^{-14}$**.
- **Solid Wall Non-Penetration**: Strict **$0.0$ fluid penetration** across obstacle nodes during waterfall descent.

---

## 5. QLKS Viscosity Decoupling (Xiao et al. 2025)

In classical BGK collision, kinematic viscosity is fixed by the relaxation time $\tau$:
$$\nu = c_s^2 \left( \tau - \frac{1}{2} \Delta t \right)$$
As the Reynolds number $Re = \frac{U L}{\nu}$ increases, $\tau \to 0.5$, causing numerical instability in quantum block-encoding and CPTP contraction channels.

Under the QLKS formulation (Xiao et al. 2025, arXiv:2505.10883), an equilibrium scale parameter $A \in (0, 1]$ is introduced:
$$f_i^{\text{eq, QLKS}} = w_i \rho \left[ A + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2} + \dots \right]$$
The resulting effective kinematic viscosity becomes:
$$\nu_{\text{eff}} = \frac{1}{3} A \left( \tau - \frac{1}{2} \right)$$
This decouples physical viscosity from quantum operator relaxation:
- We set $\tau = 1.0$ (where $\lambda = 1 - 1/\tau = 0.0$, making the collision operator an exact projection on conserved moments without non-unitary decay!).
- We tune $A \in (0, 1]$ to access arbitrarily high Reynolds numbers without altering quantum gate fidelity or numerical stability.

---

## 6. Production Usage Reference

To run the coupled quantum waterfall engine from the CLI:
```bash
# Run 3D waterfall with genuine coupled quantum circuit stepper
.venv/bin/python ASWANTH/executables/run_quantum_phase_field_waterfall.py \
  --nx 8 --ny 4 --nz 8 \
  --steps 20 \
  --engine quantum \
  --qlks-a 0.8 \
  --gz -0.010
```
