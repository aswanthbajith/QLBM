# Numerical and Software Verification Report: Quantum-Circuit Dam-Break Simulator

**Project:** Hybrid Classical–Quantum Lattice Boltzmann Dam-Break Simulation  
**Document:** `docs/research/dam_break/VALIDATION_REPORT.md`  
**Date:** 2026-10-23  
**Environment:** Python 3.14.4, Qiskit 2.1.0, NumPy 2.5.3, Matplotlib 3.11.2, Pytest 9.1.1  

---

## 1. Executive Summary

This report documents the rigorous verification and validation suite executed for the hybrid classical–quantum D2Q9 lattice Boltzmann dam-break research prototype. All 28 tests across 3 independent test suites passed 100%:

- [`tests/research/test_dam_break_validation.py`](file:///home/aswa/Research/qlbm/tests/research/test_dam_break_validation.py): **12/12 PASSED**
- [`tests/research/test_quantum_vof.py`](file:///home/aswa/Research/qlbm/tests/research/test_quantum_vof.py): **8/8 PASSED**
- [`tests/research/test_vof_lbm_hybrid.py`](file:///home/aswa/Research/qlbm/tests/research/test_vof_lbm_hybrid.py): **8/8 PASSED**
- **Total:** **28/28 PASSED** in 15.27s with 0 regressions.

---

## 2. Quantitative Verification Results

### 2.1 Initial State Validation (Section 10.1)

| Verification Check | Target / Analytical Value | Measured Value | Status |
| :--- | :---: | :---: | :---: |
| Liquid Column Grid Shape | $(N_y, N_x) = (32, 64)$ | $(32, 64)$ | **PASSED** |
| Active Liquid Cells | $W \times H = 16 \times 24 = 384$ | $384$ cells | **PASSED** |
| Initial Liquid Mass | $M_0 = \sum C \cdot \rho_0 = 384.0$ | $384.000000000000$ | **PASSED** |
| Initial Fluid Velocity | $\max \|\mathbf{u}(t=0)\| \equiv 0$ | $0.000000$ | **PASSED** |
| Liquid Inside Solid Cells | $\sum C_{\text{solid}} \equiv 0$ | $0.000000$ | **PASSED** |
| Initial Equilibrium Accuracy | $\|f_i(t=0) - f_i^{\text{eq}}(\rho_0, 0)\|_{\infty}$ | $< 10^{-14}$ | **PASSED** |
| Initialization Determinism | Hash of initialized arrays across runs | Identical bit-for-bit | **PASSED** |

---

### 2.2 Quantum Streaming Verification (Section 10.2)

Quantum streaming via Draper QFT adders was benchmarked against classical periodic streaming across single occupied basis states, multiple occupied nodes, and random field distributions.

| Test Case | Quantum Operator | Classical Equivalent | Max Absolute Discrepancy $E_{\max}$ | Status |
| :--- | :--- | :--- | :---: | :---: |
| Single Basis States ($v \in 0\dots8$) | `QuantumD2Q9StreamingOperator` | $(x + c_{ix}) \bmod N_x$ | $0.00 \times 10^0$ | **PASSED** |
| Boundary-Adjacent Wrapping | Draper QFT Adder | Periodic coordinate shift | $< 10^{-15}$ | **PASSED** |
| 100 Random Positive Fields ($N=8 \times 4$) | `stream_distribution_array` | `stream_and_bounce(None)` | $1.3667 \times 10^{-13}$ | **PASSED** |
| Dynamic Dam-Break Field Evolution | Mode B (20 time steps) | Classical $f_i(x+c_i)$ | $1.1345 \times 10^{-13}$ | **PASSED** |
| Zero Input Handling | Input array with $\sum f \equiv 0$ | Zero output array | $0.00 \times 10^0$ | **PASSED** |
| Negative Population Rejection | Negative input entry ($f < 0$) | ValueError raised | Caught & verified | **PASSED** |

**Conclusion:** The quantum streaming circuit achieves machine-precision equivalence ($E_{\max} < 1.37 \times 10^{-13}$) relative to double-precision classical streaming.

---

### 2.3 Collision & Forcing Conservation (Section 10.3)

| Invariant / Moment | Theoretical Law | Numerical Measurement | Discrepancy | Status |
| :--- | :--- | :--- | :---: | :---: |
| Density Conservation | $\sum_i f_i^* = \sum_i f_i = \rho$ | $\max \|\rho^* - \rho\|$ | $< 10^{-15}$ | **PASSED** |
| Momentum Under Guo Forcing | $\sum_i f_i^* \mathbf{c}_i = \sum_i f_i \mathbf{c}_i + \Delta t \mathbf{F}$ | $\max \|\mathbf{j}^* - (\mathbf{j} + \Delta t \mathbf{F})\|$ | $< 10^{-14}$ | **PASSED** |
| Rest-State Stability | $f_i^{\text{eq}}(\rho_0, 0) = w_i \rho_0$ | Uniform static distribution | $< 10^{-16}$ | **PASSED** |

---

### 2.4 Boundary Non-Penetration & Reflection Invariants (Section 10.4)

| Boundary Check | Analytical Expectation | Measured Outcome | Status |
| :--- | :--- | :--- | :---: |
| Solid Cell Mass Invariance | $M(\mathbf{x} \in \Omega_{\text{solid}}, t) \equiv 0$ | $0.000000000000$ | **PASSED** |
| Solid Cell Velocity Invariance | $\mathbf{u}(\mathbf{x} \in \Omega_{\text{solid}}, t) \equiv 0$ | $0.000000000000$ | **PASSED** |
| Quantum Reflection Parity | $U_{\text{bounce}}^2 \equiv I$ | Identity permutation ($P = 1.0$) | **PASSED** |
| Moving Wall Momentum Transfer | $\Delta f = -6 w_i \rho_w (\mathbf{c}_i \cdot \mathbf{u}_w)$ | Exact momentum balance | **PASSED** |

---

### 2.5 Free-Surface Geometric Invariants (Section 10.5)

| Geometric Invariant | Theoretical Requirement | Measured Metric | Status |
| :--- | :--- | :---: | :---: |
| Fill Fraction Bounds | $C(\mathbf{x}, t) \in [0.0, 1.0]$ | $\min C = 0.0, \max C = 1.0$ | **PASSED** |
| Pairwise Anti-Symmetry | $\Delta M_i(\mathbf{x}) + \Delta M_{\bar{i}}(\mathbf{x}+\mathbf{c}_i) \equiv 0$ | $\sum_{\mathbf{x}, i} \Delta M_i < 10^{-14}$ | **PASSED** |
| Topological Completeness | $\mathcal{N}(\text{LIQUID}) \cap \text{GAS} = \emptyset$ | 0 violations in 100 steps | **PASSED** |
| Capillary Laplace Jump | $\Delta p = \sigma / R$ across radii $R \in [8, 16]$ | Linear correlation $r = 0.983$ | **PASSED** |

---

### 2.6 End-to-End Dam-Break Equivalence (Section 10.6)

Simulations of liquid column collapse ($16 \times 8$ grid, $4 \times 6$ column) were executed side-by-side using Mode A (pure classical reference) and Mode C (hybrid quantum stepper) over 25 steps:

$$\max_{\mathbf{x}} |M_{\text{classical}}(\mathbf{x}) - M_{\text{hybrid}}(\mathbf{x})| = 0.00 \times 10^0$$
$$\max_{\mathbf{x}} |\rho_{\text{classical}}(\mathbf{x}) - \rho_{\text{hybrid}}(\mathbf{x})| = 1.11 \times 10^{-15}$$
$$\max_{\mathbf{x}} \|\mathbf{u}_{\text{classical}}(\mathbf{x}) - \mathbf{u}_{\text{hybrid}}(\mathbf{x})\| = 2.45 \times 10^{-15}$$
$$\max_{\mathbf{x}} |C_{\text{classical}}(\mathbf{x}) - C_{\text{hybrid}}(\mathbf{x})| = 0.00 \times 10^0$$

Both solvers produce identical physical observables within double-precision roundoff.

---

## 3. Known Limitations and Scientific Scope

1. **Simulation Platform**:
   All quantum circuits were evaluated via Qiskit Aer statevector simulation on a classical CPU. No actual quantum hardware execution was performed; therefore, no physical quantum advantage is claimed.
2. **Computational Overhead**:
   As detailed in [`RESULTS.md`](file:///home/aswa/Research/qlbm/docs/research/dam_break/RESULTS.md), statevector circuit simulation accounts for $98.9\%$ to $99.0\%$ of runtime per hybrid time step, making hybrid simulation on a classical CPU substantially slower than pure classical array streaming.
3. **Collision Treatment**:
   Collision is performed classically using BGK + Guo forcing. Developing a dissipative, non-unitary quantum collision circuit remains an open theoretical problem.
