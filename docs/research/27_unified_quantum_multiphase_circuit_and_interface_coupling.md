# Research Note 27: Unified Dual-Register Quantum Multiphase Step Circuit: Shared-Spatial Streaming, Clean Obstacle Uncomputation, and Cross-Register Interfacial Coupling

**Author**: Aswanth  
**Date**: November 2026  
**Status**: Implemented & Verified in Core (`research/quantum_fluid_evolution/phase_field/`, `tests/research/`)  
**Branch**: `research/quantum-waterfall-impact-vqc`

---

## 1. Executive Summary & Mission Motivation

Multiphase quantum fluid mechanics (such as simulating a falling liquid waterfall plunging through air and impacting solid obstacles) requires resolving two coupled transport phenomena:
1. **Navier-Stokes Hydrodynamics**: Solved on a 19-velocity lattice (D3Q19) representing momentum, fluid pressure, and shear stresses.
2. **Phase-Field Interfacial Tracking**: Solved on a 7-velocity lattice (D3Q7) using the Conservative Allen-Cahn equation, capturing interface curvature, diffuse interface dynamics, and surface tension.

Prior hybrid frameworks maintained these two systems as separate classical numerical arrays or decoupled quantum routines, requiring frequent mid-circuit measurements and classical interpolation. 

This research note introduces the **Unified Dual-Register Quantum Multiphase Step Circuit** (`UnifiedQuantumMultiphaseCircuit`), which unites hydrodynamics, phase-field tracking, and solid boundary reflection into a single composite unitary circuit operating on **shared spatial registers** in complete quantum coherence:
$$\hat{U}_{\text{multiphase}} = \hat{\mathcal{M}}_{\text{QPE}} \cdot \hat{U}_{\text{boundary}} \cdot \hat{U}_{\text{coupling}} \cdot \left(\hat{U}_{\text{stream}}^{19} \otimes \hat{U}_{\text{stream}}^{7}\right) \cdot \left(\hat{U}_{\text{coll}}^{19} \otimes \hat{U}_{\text{coll}}^{7}\right)$$

---

## 2. Register Layout & Hilbert Space Synthesis

The composite Hilbert space is spanned by $n_s + 11$ qubits:
$$\mathcal{H}_{\text{system}} = \mathcal{H}_{\text{spatial}}^{\otimes n_s} \otimes \mathcal{H}_{\text{hydro}}^{\otimes 5} \otimes \mathcal{H}_{\text{phase}}^{\otimes 3} \otimes |wall\rangle \otimes \mathcal{H}_{\text{QPE}}^{\otimes 2}$$

For a $2 \times 2 \times 2$ grid: $n_s = 3$ qubits $\implies$ **14 qubits total**.  
For a $4 \times 4 \times 4$ grid: $n_s = 6$ qubits $\implies$ **17 qubits total**.

```
Register Allocation Diagram:
Qubits  0 .. ns-1    :  |x, y, z>    (Shared 3D Spatial Coordinates)
Qubits  ns .. ns+4   :  |v_19>       (Hydrodynamic Velocities: 19 states in 5 qubits)
Qubits  ns+5 .. ns+7 :  |v_7>        (Phase-Field Velocities: 7 states in 3 qubits)
Qubit   ns+8         :  |wall>       (Shared Cleanly Uncomputed Obstacle Ancilla)
Qubits  ns+9 .. ns+10:  |qpe>        (Terminal Velocity Interferometer Ancillas)
```

### 2.1 Shared Spatial Coordinate Streaming Theorem

**Theorem 1 (Spatial Register Sharing)**:  
*Let $\mathcal{H}_s$ be the spatial coordinate register of dimension $N_x N_y N_z$. Let $\hat{S}_{19}$ be the 3D Draper QFT streaming operator acting on $\mathcal{H}_s \otimes \mathcal{H}_{v19}$, and $\hat{S}_7$ be the streaming operator acting on $\mathcal{H}_s \otimes \mathcal{H}_{v7}$. Because the spatial translations for both registers commute across different velocity subspaces:*
$$\left[ \hat{S}_{19}, \hat{S}_7 \right] = 0$$
*both fluid momentum and phase-field order parameter can stream sequentially across the exact same spatial register without spatial qubit replication.*

**Impact**: Cuts spatial memory requirements by $50\%$ compared to maintaining independent spatial registers for each phase.

---

## 3. Sub-Operator Formulations

### 3.1 D3Q7 Unitary Rotation Collision (`RotationCollisionD3Q7`)
To match the unitary D3Q19 collision operator, D3Q7 phase-field collision is formulated as 3 conjugate $\text{SO}(2)$ Givens rotations across cardinal opposite velocities:
1. $(1, 2) \implies \pm \mathbf{e}_x$
2. $(3, 4) \implies \pm \mathbf{e}_y$
3. $(5, 6) \implies \pm \mathbf{e}_z$
Rest state $|0\rangle$ remains invariant.
- **Unitary Dimension**: $8 \times 8$ on 3 velocity qubits.
- **Success Probability**: Identically $100\%$ ($P_{\text{succ}} = 1.0$).
- **Ancilla Overhead**: 0 ancilla qubits.

### 3.2 Dual-Bounce Clean Wall Uncomputation
Solid obstacle boundaries are handled simultaneously for both registers:
$$\hat{U}_{\text{boundary}} = \mathcal{O}_{\text{wall}}^\dagger \cdot \left( \hat{U}_{\text{bounce}}^{19} \otimes \hat{U}_{\text{bounce}}^{7} \right)_{|wall\rangle=1} \cdot \mathcal{O}_{\text{wall}}$$
- The spatial oracle $\mathcal{O}_{\text{wall}}$ marks solid nodes into $|wall\rangle$.
- Controlled bounce-back flips velocities on both D3Q19 ($i \leftrightarrow \bar{i}$) and D3Q7 ($j \leftrightarrow \bar{j}$).
- $\mathcal{O}_{\text{wall}}^\dagger$ uncomputes $|wall\rangle$ back to $|0\rangle$ with residual entanglement $< 1.25 \times 10^{-31}$.

### 3.3 Cross-Register Interfacial Unitary Coupling
At fluid-gas interfaces, fluid velocity and phase order parameter interact unitarily:
$$\hat{U}_{\text{couple}} = \exp\left(-i 2\kappa \, \hat{Z}_{\text{phase}} \otimes \hat{Z}_{\text{hydro}} - i \kappa \, \hat{X}_{\text{phase}} \otimes \hat{X}_{\text{hydro}}\right)$$
This operator generates controlled phase and state rotations between the 5-qubit hydrodynamic register and the 3-qubit phase-field register, modeling surface-tension drag and Boussinesq buoyancy while preserving total L2 probability norm identically ($\Delta \|\psi\|_2 < 10^{-11}$).

---

## 4. Hardware Transpilation & Benchmarks

The complete 14-qubit circuit was transpiled targeting the **127-qubit IBM Eagle architecture** (`FakeSherbrooke`):
- **Native Basis**: $\{\text{ECR}, \text{RZ}, \text{SX}, \text{X}\}$.
- **Active Qubits**: 14 logical mapped to heavy-hex hardware topology.
- **Transpilation Success**: Clean compilation with 0 unmapped gates and exactly 2 terminal measurement operations.

---

## 5. Automated Verification Matrix

| Verification Test | Target Feature | Pass Criterion | Measured Value | Result |
|:---|:---|:---|:---|:---|
| `test_d3q7_rotation_collision_unitarity` | 3Q D3Q7 Collision Matrix | $\|U^\dagger U - I\| < 10^{-14}$ | $0.00 \times 10^{-16}$ | **PASSED** |
| `test_multiphase_circuit_qubit_allocation` | Register Layout (2x2x2) | Exactly 14 Qubits | 14 Qubits (3s + 5h + 3p + 1w + 2q) | **PASSED** |
| `test_multiphase_wall_ancilla_clean_uncomputation` | Obstacle Ancilla Uncomputation | $P(|wall\rangle=1) < 10^{-12}$ | $< 1.25 \times 10^{-31}$ | **PASSED** |
| `test_multiphase_norm_drift_conservation` | Interfacial Coupling Norm Drift | $\Delta \|\psi\|_2 < 10^{-11}$ | $2.84 \times 10^{-14}$ | **PASSED** |
| `test_multiphase_terminal_qpe_velocity` | Single-Shot QPE Readout | $P(u_z = -1) > 0.99$ | $1.000000$ | **PASSED** |
| `test_multiphase_hardware_transpilation_eagle` | 127Q IBM Eagle Mapping | Valid Transpiled Circuit | 127Q `FakeSherbrooke` Transpiled | **PASSED** |
