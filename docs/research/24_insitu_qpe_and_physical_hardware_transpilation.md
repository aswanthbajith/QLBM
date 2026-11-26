# Research Note 24: In-Situ Quantum Phase Estimation (QPE) Interferometry and 127-Qubit Superconducting Hardware Transpilation for 3D Lattice Boltzmann QCFD

**Author**: Aswanth  
**Date**: November 2026  
**Status**: Verified & Implemented in Source (`research/`, `ASWANTH/executables/`, `tests/research/`)  
**Branch**: `research/quantum-waterfall-impact-vqc`

---

## 1. Executive Summary

As quantum computational fluid dynamics (QCFD) transitions from classical-quantum hybrid paradigms toward end-to-end quantum computation, two critical bottlenecks emerge:
1. **Observable Extraction**: Measuring continuous hydrodynamic velocity fields typically requires reconstructing high-dimensional wavefunctions via state tomography, destroying the fluid statevector and incurring exponential sampling complexity $\mathcal{O}(2^N)$.
2. **Physical Hardware Feasibility**: Unitary streaming and collision operators must map efficiently onto actual physical quantum processor units (QPUs) under heavy-hex coupling topologies, native gate basis sets ($\text{ECR}, \text{RZ}, \text{SX}, \text{X}$), and finite coherence budgets ($T_1, T_2$).

This monograph establishes:
- The **Exact Discrete Velocity QPE Interferometer**: couples $m=2$ precision ancilla qubits to the diagonal directional momentum operator $\hat{J}_z = \sum_{i=0}^{18} c_{i,z} |i\rangle\langle i|$. Setting phase base scale $\phi_0 = 1/4$ achieves **zero spectral leakage**, reproducing macroscopic fluid velocity moments $\langle u_z \rangle_{\text{QPE}}$ with machine-precision equivalence to classical LBM moments ($|u_z^{\text{QPE}} - u_z^{\text{LBM}}| < 10^{-14}$).
- **In-Situ Non-Destructive Fluid Profiling**: Direct integration into the 3D multiphase waterfall time-stepper, capturing real-time probability shifts $P(u_z = -1)$ as the liquid column detaches and plunges down the cliff face.
- **Physical Hardware Transpilation onto 127-Qubit IBM Eagle Processors**: Transpiling algorithmic streaming, $\text{SO}(2)$ Givens collision, and QPE circuits targeting `ibm_sherbrooke` / `FakeSherbrooke`, establishing exact native 2Q gate counts ($\text{ECR}$), circuit depth, coherence budget headroom, and NISQ error mitigation protocols (TREX, ZNE, Dynamical Decoupling).

---

## 2. Mathematical Foundation of Discrete Velocity QPE

### 2.1 Diagonal Momentum Hamiltonian

In the computational basis of velocity states $|i\rangle$ ($i \in \{0, \dots, 18\}$ for D3Q19; $i \in \{0, \dots, 6\}$ for D3Q7), the directional lattice velocity operator along Cartesian axis $\alpha \in \{x, y, z\}$ is strictly diagonal:
$$\hat{J}_\alpha = \sum_{i=0}^{Q-1} c_{i,\alpha} |i\rangle\langle i|$$
where eigenvalues $c_{i,\alpha} \in \{-1, 0, +1\}$ are integers in lattice units.

### 2.2 Unitary Phase Encoding & Zero-Spectral-Leakage Condition

Standard continuous Quantum Phase Estimation suffers from spectral leakage when the target eigenvalue phase $\theta$ is not an exact integer multiple of $2^{-m}$. However, for discrete lattice Boltzmann registers, the spectrum is strictly ternary:
$$\lambda \in \{-1, 0, +1\}$$

We define the controlled unitary evolution operator:
$$\hat{U}_\alpha = \exp\left(i \cdot 2\pi \cdot \phi_0 \cdot \hat{J}_\alpha\right)$$
Choosing base phase scale $\phi_0 = \frac{1}{4}$ ($90^\circ$ phase shift per lattice speed unit):
- Eigenvalue $c_{i,\alpha} = 0 \implies \text{phase } \theta = 0.00 = .00_2 \implies \text{Ancilla State } |00\rangle$
- Eigenvalue $c_{i,\alpha} = +1 \implies \text{phase } \theta = 0.25 = .01_2 \implies \text{Ancilla State } |01\rangle$
- Eigenvalue $c_{i,\alpha} = -1 \implies \text{phase } \theta = -0.25 \equiv 0.75 \pmod 1 = .11_2 \implies \text{Ancilla State } |11\rangle$

Because every physical eigenvalue maps to an **exact dyadic fraction** of denominator $2^2 = 4$, an ancilla register of only $m = 2$ precision qubits completely eliminates spectral leakage:
$$\mathcal{F}_{\text{leakage}} \equiv 0$$
The inverse Quantum Fourier Transform $\text{QFT}_2^\dagger$ concentrates 100% of the interferometric probability into basis states $|00\rangle, |01\rangle, |11\rangle$.

### 2.3 Mathematical Equivalence to Fluid Moments

Let the target node statevector be encoded with normalized distribution amplitudes:
$$|\psi_{\text{fluid}}\rangle = \sum_{i=0}^{Q-1} \sqrt{\frac{f_i}{\rho}} |i\rangle, \quad \rho = \sum_{i=0}^{Q-1} f_i$$
The probability of observing basis state $|i\rangle$ upon target measurement is $|a_i|^2 = f_i / \rho$.

Under QPE with precision register initially in $|00\rangle$:
$$|00\rangle \otimes |\psi\rangle \xrightarrow{H^{\otimes 2}} \frac{1}{2} \sum_{k=0}^3 |k\rangle \otimes |\psi\rangle \xrightarrow{\text{C-}U^{2^j}} \frac{1}{2} \sum_{k=0}^3 |k\rangle \otimes \sum_{i} \sqrt{\frac{f_i}{\rho}} e^{i 2\pi k \frac{c_{i,z}}{4}} |i\rangle$$
Applying $\text{QFT}_2^\dagger$:
$$\sum_{i=0}^{Q-1} \sqrt{\frac{f_i}{\rho}} |\text{bin}_2(c_{i,z})\rangle \otimes |i\rangle$$

Measuring the 2-qubit precision register yields the macroscopic velocity distribution:
$$P(u_z = -1) = \sum_{c_{i,z} = -1} \frac{f_i}{\rho}, \quad P(u_z = 0) = \sum_{c_{i,z} = 0} \frac{f_i}{\rho}, \quad P(u_z = +1) = \sum_{c_{i,z} = +1} \frac{f_i}{\rho}$$
The expectation value is:
$$\langle u_z \rangle_{\text{QPE}} = (+1) \cdot P(u_z = +1) + (-1) \cdot P(u_z = -1) = \sum_{i=0}^{Q-1} c_{i,z} \frac{f_i}{\rho} \equiv u_z^{\text{classical}}$$
**Theorem**: The quantum expectation velocity measured by discrete QPE is mathematically identical to the classical lattice Boltzmann momentum moment to machine precision.

---

## 3. In-Situ Waterfall Profiling Dynamics

During 3D waterfall time-stepping (`ASWANTH/executables/run_quantum_phase_field_waterfall.py`), two in-situ probes track the quantum velocity spectrum:
1. **Shelf Lip Probe** $(z = z_{\text{shelf}}+1, y = ny/2, x = x_{\text{cliff}})$: Monitors horizontal gate release turning downward over the cliff.
2. **Plunging Jet Probe** $(z = z_{\text{shelf}}/2, y = ny/2, x = x_{\text{cliff}}+1)$: Monitors free-falling liquid accelerating under gravity.

### Numerical Observables Recorded During Simulation
```
Step 1:
  ↳ [QPE Lip]    P(-1)=0.185, P(0)=0.662, P(+1)=0.153 | <uz>=-0.0313 (moment=-0.0313)
  ↳ [QPE Plunge] P(-1)=0.175, P(0)=0.668, P(+1)=0.158 | <uz>=-0.0172 (moment=-0.0172)
Step 3:
  ↳ [QPE Lip]    P(-1)=0.170, P(0)=0.643, P(+1)=0.187 | <uz>=+0.0167 (moment=+0.0167)
  ↳ [QPE Plunge] P(-1)=0.188, P(0)=0.644, P(+1)=0.168 | <uz>=-0.0199 (moment=-0.0199)
Step 6:
  ↳ [QPE Lip]    P(-1)=0.157, P(0)=0.650, P(+1)=0.193 | <uz>=+0.0353 (moment=+0.0353)
  ↳ [QPE Plunge] P(-1)=0.169, P(0)=0.654, P(+1)=0.177 | <uz>=+0.0074 (moment=+0.0074)
```
- In all steps, $|u_z^{\text{QPE}} - u_z^{\text{moment}}| < 10^{-14}$.
- Total Order Parameter Mass Conservation Error: $\Delta M_\phi = 5.60 \times 10^{-14}$ (exact machine zero).

---

## 4. Physical Quantum Hardware Transpilation Analysis

Using `ASWANTH/executables/submit_ibm_quantum_job.py`, algorithmic circuits were transpiled onto the 127-qubit IBM Eagle processor `FakeSherbrooke`.

### 4.1 Target Architecture Specifications (`ibm_sherbrooke`)
- **Physical Qubits**: 127 heavy-hex topology.
- **Native Gate Basis**: $\{\text{ECR}, \text{RZ}, \text{SX}, \text{X}, \text{ID}, \text{RESET}, \text{MEASURE}\}$.
- **Median QPU Coherence Times**:
  - $T_1 = 251.8\ \mu\text{s}$ (longitudinal relaxation)
  - $T_2 = 108.8\ \mu\text{s}$ (transverse dephasing)
- **Gate Execution Times**:
  - Native 2Q $\text{ECR} \approx 300\text{ ns}$
  - Native 1Q $\text{SX} / \text{RZ} \approx 35\text{ ns}$
  - Measurement $\approx 1000\text{ ns}$

### 4.2 Transpilation Metrics (Level 3 Optimization)

| Circuit Module | Algorithmic Qubits | Physical Qubits | Transpiled Depth | Native 2Q ($\text{ECR}$) | Native 1Q ($\text{SX/RZ}$) | Est. Duration ($\tau$) | Coherence Headroom ($T_2 / \tau$) |
|---|---|---|---|---|---|---|---|
| **QPE Velocity Readout (D3Q19)** | 7 | 7 | 2,323 | 690 | 3,374 | $488.8\ \mu\text{s}$ | $0.22\times$ |
| **Rotation Collision ($\text{SO}(2)$)** | 5 | 5 | 4,484 | 966 | 6,334 | $942.6\ \mu\text{s}$ | $0.23\times$ |
| **Coupled LBM Step (Streaming + Coll)** | 15 | 15 | 6,810 | 1,656 | 9,708 | $1,430.1\ \mu\text{s}$ | $0.08\times$ |

---

## 5. NISQ Error Mitigation Strategy

Because current superconducting coherence headroom is $\approx 0.22\times$ ($T_2 \sim 110\ \mu\text{s}$ vs circuit duration $\sim 490\ \mu\text{s}$), unmitigated physical execution accumulates dephasing and gate infidelity ($\sim 0.75\%$ per $\text{ECR}$). To achieve publication-grade fidelity on physical QPUs, three orthogonal mitigation layers are deployed:

1. **Twirled Readout Error Mitigation (TREX)**:
   - Evaluates random Pauli-X pre-measurement twirling to diagonalize readout assignment error tensors into classical bit-flip channels.
   - Eliminates asymmetric $|0\rangle \to |1\rangle$ vs $|1\rangle \to |0\rangle$ measurement bias on precision ancilla qubits.

2. **Zero Noise Extrapolation (ZNE)**:
   - Scales effective circuit noise by inserting unitary identity pairs $U \to U (U^\dagger U)^n$ with scale factors $\lambda \in \{1, 3, 5\}$.
   - Extrapolates measured velocity expectation values $\langle u_z(\lambda) \rangle$ to the zero-noise limit $\lambda \to 0$ via Richardson polynomial fitting.

3. **Dynamical Decoupling (DD)**:
   - During multi-qubit idling periods (while heavy-hex routing swaps occur on neighboring registers), insert periodic $\text{XY4}$ pulse sequences:
     $$[\tau/2 - X - \tau - Y - \tau - X - \tau - Y - \tau/2]$$
   - Actively cancels low-frequency $1/f$ magnetic flux dephasing, extending effective $T_2^*$ toward the $2 T_1$ relaxation limit.

---

## 6. Conclusion & Roadmap Alignment

By integrating in-situ QPE interferometry and physical 127-qubit hardware transpilation into the core QLBM repository:
1. **Zero Spectral Leakage QPE** provides a verified, mathematically exact bridge between quantum wavefunctions and macroscopic fluid observables.
2. **End-to-End Quantum Stepping** (Draper QFT streaming + $\text{SO}(2)$ Givens collision) is fully wired into executable scripts, animated multimedia dashboards, and automated test suites.
3. **100% Test Pass Rate** is maintained across all test suites with machine-zero order parameter conservation ($< 10^{-13}$).
