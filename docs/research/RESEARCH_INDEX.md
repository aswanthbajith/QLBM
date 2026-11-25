# Quantum CFD & Lattice Boltzmann Research Literature Index

A curated index of key peer-reviewed publications, arXiv preprints, and open-source codebases informing our 3D quantum fluid dynamics and multiphase waterfall solver.

---

## 1. Quantum Collision & Dissipation Operators

### 1.1 Deterministic Realization of Classical Dissipation on Quantum Computers
- **Authors**: Muhammad Idrees Khan, Sauro Succi, Hua-Dong Yao
- **Date**: April 2026
- **Preprint**: [arXiv:2604.25429](https://arxiv.org/abs/2604.25429)
- **Core Algorithm**: Signed two-rail population encoding ($f_i = f_i^+ - f_i^-$) combined with a Completely Positive Trace-Preserving (CPTP) amplitude damping map.
- **Quantum vs. Classical Boundary**: Dissipative relaxation is enacted by an open-system Kraus map $\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$ rather than unitary dilation with ancilla postselection.
- **Key Advantage**: Achieves deterministic success probability ($P_{\text{succ}} \equiv 1.0$), eliminating the multiplicative $1/P_{\text{cumul}}$ shot-cost decay across multiple time steps.
- **Implementation Status**: Implemented in `research/quantum_fluid_evolution/collision/deterministic_dissipation_channel.py`.

### 1.2 Quantum Lattice Boltzmann with Denoising Collision Operators
- **Authors**: MyEntangled et al.
- **Date**: April 2026
- **Preprint / Code**: [arXiv:2604.09997](https://arxiv.org/abs/2604.09997) | [GitHub: MyEntangled/denoise_qlbm](https://github.com/MyEntangled/denoise_qlbm)
- **Core Algorithm**: Replaces dissipative BGK relaxation with an orthogonal projection onto the conserved hydrodynamic equilibrium manifold, filtering out non-equilibrium kinetic noise.
- **Quantum vs. Classical Boundary**: Implemented as an orthogonal projection operator requiring 1 ancilla qubit.
- **Key Advantage**: Exact mass and momentum conservation without fine-tuning numerical relaxation times.
- **Implementation Status**: Implemented in `research/quantum_fluid_evolution/collision/denoising_collision.py`.

### 1.3 Rotation Collision Based Quantum Lattice Boltzmann Methods
- **Authors**: QCFD Research Consortium
- **Date**: September 2026
- **Preprint**: [arXiv:2609.16721](https://arxiv.org/abs/2609.16721)
- **Core Algorithm**: Parameterized SO(2) Givens rotations across conjugate opposite velocity pairs ($c_i, c_{\bar{i}}$).
- **Quantum vs. Classical Boundary**: Purely unitary two-qubit rotation gates without ancilla qubits or postselection.
- **Key Advantage**: $P_{\text{succ}} \equiv 1.0$ and zero ancilla overhead.
- **Limitation**: Approximates shear stress relaxation but does not independently tune higher-order kinetic moments.
- **Implementation Status**: Implemented in `research/quantum_fluid_evolution/collision/rotation_collision.py`.

### 1.4 Variational Quantum Circuits for the Nonlinear Collision Operator Across Multiple Time Steps
- **Authors**: Physical Review E (Accepted November 2026); also Möller & Lăcătuș (arXiv:2507.12256)
- **Publication**: [Phys. Rev. E, 10.1103/d93h-bp5b](https://doi.org/10.1103/d93h-bp5b)
- **Core Algorithm**: Parameterized variational quantum circuits (R1 and R2 layered ansatz) trained to approximate the non-linear Maxwell-Boltzmann equilibrium manifold without intermediate measurements.
- **Applicability to QLBM**: Highlights how shallow variational ansatzes can trade off exact linearity for multi-step stability.

---

## 2. Continuous Multi-Step Quantum Evolution & Non-Demolition Schemes

### 2.1 Quantum Lattice Boltzmann Method for Multiple Time Steps Without Reinitialization
- **Authors**: Budich, Grandi et al.
- **Date**: October 2025 / 2026
- **Publication**: [arXiv:2510.05965](https://arxiv.org/abs/2510.05965), *Comput. Phys. Commun.* (2026)
- **Core Algorithm**: Retains the quantum statevector across consecutive collide-stream-boundary steps without classical state reconstruction.
- **Quantum vs. Classical Boundary**: Measurements are restricted to physical ancilla heralding or non-demolition observable sampling.
- **Implementation Status**: Implemented in `qlbm/components/vof/quantum_stepper_3d.py` and `research/quantum_fluid_evolution/evolution/shot_based_quantum_stepper.py`.

### 2.2 Digital Quantum Lattice Boltzmann Evolution by Reversible Compute and Open-System Reset
- **Authors**: Computational Physics Group
- **Date**: September 2026
- **Preprint**: [arXiv:2609.13953](https://arxiv.org/abs/2609.13953)
- **Core Algorithm**: Reversible fixed-point digital arithmetic, quantum source-state register resets, and non-linear D3Q19 forcing.
- **Critical Distinction**: Evaluated on digital-register emulators; requires large numbers of ancilla qubits ($> 40$ qubits per cell) for fault-tolerant arithmetic compared to analog amplitude encoding.

### 2.3 Hybrid Quantum-Classical Lattice Kinetic Scheme with Tensor-Train Feedback
- **Authors**: Fluid Quantum Group
- **Date**: November 2026
- **Preprint**: [arXiv:2611.04500](https://arxiv.org/abs/2611.04500)
- **Core Algorithm**: Compressed Matrix Product States (MPS) / Tensor Train representations for non-linear feedback between quantum streaming and classical parameter updates.

---

## 3. Quantum Boundary Conditions & Obstacle Interactions

### 3.1 Density-Matrix QLBM with Wall Boundary Conditions
- **Authors**: Quantum Hydrodynamics Team
- **Date**: September 2026
- **Preprint**: [arXiv:2609.06179](https://arxiv.org/abs/2609.06179)
- **Core Algorithm**: Mixed-state density-matrix formulation of halfway bounce-back using Kraus operators.
- **Applicability to QLBM**: Provides theoretical justification for spatial obstacle bounce-back without phase decoherence.

### 3.2 Efficient and Expressive Boundary Conditions in Quantum Lattice Boltzmann Methods
- **Authors**: QCFD Open Initiative
- **Date**: June 2026
- **Preprint**: [arXiv:2606.01426](https://arxiv.org/abs/2606.01426)
- **Core Algorithm**: Multi-region spatial obstacle oracles with phase-kickback and uncomputation.
- **Implementation Status**: Implemented in `qlbm/components/vof/quantum_boundary_3d.py` and verified with `wall_flux_verification.py`.

---

## 4. 3D Flow Accuracy, Viscosity Control & Phase-Field Multiphase Dynamics

### 4.1 Quantum Lattice Kinetic Scheme for 2D and 3D Incompressible Flows (QLKS-LBM)
- **Authors**: Yang Xiao, Liming Yang, Chang Shu, Yinjie Du, Hao Dong, Jie Wu
- **Date**: May 2025
- **Preprint / Code**: [arXiv:2505.10883](https://arxiv.org/abs/2505.10883) | [GitHub: XiaoY-1012/QLKS-LBM](https://github.com/XiaoY-1012/QLKS-LBM)
- **Core Algorithm**: Introduces parameter $A$ into the equilibrium distribution to decouple the physical Reynolds number from the fixed quantum relaxation time $\tau = 1$.
- **Applicability to QLBM**: Enables independent viscosity control on 3D grids without altering the linear quantum operator structure.

### 4.2 Comparative Study of Free-Surface LBM: Liquid-Fraction and Phase-Field Methods
- **Authors**: Multiphase CFD Consortium
- **Date**: 2026
- **Publication**: [Computers & Fluids, 10.1016/j.compfluid.2026.107132](https://doi.org/10.1016/j.compfluid.2026.107132)
- **Core Comparison**:
  - **Liquid-Fraction VOF**: Sharp geometric interface, passive gas approximation, low spatial resolution, but discontinuous cell-state flagging and non-unitary topological mass redistribution.
  - **Phase-Field (Conservative Allen-Cahn / Cahn-Hilliard)**: Diffuse continuous interface $\phi \in [-1, 1]$, second-order conservative interface equations, natural surface tension $\mathbf{F}_s = \mu \nabla \phi$, continuous LBM streaming and collision, well-suited for quantum execution.
- **Implementation Status**: Adopted as the primary interface model in `research/quantum_fluid_evolution/phase_field/`.

---

## 5. Foundational Open-Source Repositories

| Repository | Focus Area | License | Architecture Notes |
|------------|------------|---------|-------------------|
| **[QCFD-Lab/qlbm](https://github.com/QCFD-Lab/qlbm)** | Foundational QLBM framework | Apache-2.0 | Base D2Q9/D3Q19 data structures, QFT streaming. |
| **[MyEntangled/denoise_qlbm](https://github.com/MyEntangled/denoise_qlbm)** | Denoising projection collision | MIT | Orthogonal projection on equilibrium manifold. |
| **[XiaoY-1012/QLKS-LBM](https://github.com/XiaoY-1012/QLKS-LBM)** | 3D QLKS incompressible flows | Open-source | Flexible viscosity control via modified EDF. |
