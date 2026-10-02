# Shan-Chen Pseudopotential Multiphase Model in D2Q9

## 1. Executive Summary

In the roadmap toward two-phase gas-liquid dam-break simulations, **Phase 9** introduces the first multiphase fluid mechanism into our validated D2Q9 Lattice Boltzmann solver: the **Shan-Chen pseudopotential formulation** (Shan & Chen 1993, 1994).

While our architectural survey in Phase 8 concluded that high-density-ratio dam-break flows ($1000:1$ water-to-air) ultimately require a **Volume-of-Fluid (VOF) / Free-Surface LBM** framework, the Shan-Chen model provides an indispensable kinetic foundation:
1. It validates non-ideal equations of state and phase coexistence directly from microscopic particle interactions.
2. It establishes spontaneous spinodal decomposition (liquid-vapor phase separation) without explicit front tracking.
3. It benchmarks diffuse interface dynamics, surface tension, and pressure jumps against analytical thermodynamic relations.
4. It verifies the coupling of interactive inter-particle body forces with the Guo discrete forcing operator.

All implementations reside in [`qlbm/classical/multiphase.py`](file:///home/aswa/Research/qlbm/qlbm/classical/multiphase.py) and are validated in [`tests/research/test_shan_chen.py`](file:///home/aswa/Research/qlbm/tests/research/test_shan_chen.py).

---

## 2. Mathematical Formulation

### 2.1 Microscopic Inter-Particle Interaction Force

In the single-component multiphase Shan-Chen model, fluid particles experience a non-local attractive/repulsive interaction force modeled via a scalar pseudopotential $\psi(\mathbf{x}) = \psi(\rho(\mathbf{x}))$.

On a discrete D2Q9 lattice, the interaction force $\mathbf{F}_{\text{SC}}(\mathbf{x})$ exerted on the fluid at node $\mathbf{x}$ by fluid at neighboring nodes $\mathbf{x} + \mathbf{c}_i$ is defined as:

$$\mathbf{F}_{\text{SC}}(\mathbf{x}) = -G \psi(\mathbf{x}) \sum_{i=0}^{8} w_i \psi(\mathbf{x} + \mathbf{c}_i) \mathbf{c}_i$$

where:
- $G$ is the interaction strength parameter ($G < 0$ creates cohesive, attractive forces necessary for phase separation).
- $w_i$ are the standard D2Q9 lattice weights ($w_0 = 4/9$, $w_{1..4} = 1/9$, $w_{5..8} = 1/36$).
- $\mathbf{c}_i$ are the discrete D2Q9 velocity vectors.

### 2.2 Pseudopotential Functional Forms

Two standard functional forms are implemented:

#### 1. Exponential (Shan-Chen standard):
$$\psi(\rho) = \rho_0 \left[ 1 - \exp\left( -\frac{\rho}{\rho_0} \right) \right]$$

This potential satisfies:
- $\psi(0) = 0$ (no interaction in pure vacuum).
- $\psi'(\rho) = \exp(-\rho/\rho_0) > 0$ (monotonic increase).
- $\lim_{\rho \to \infty} \psi(\rho) = \rho_0$ (saturation at high density prevents unphysical particle collapse).

#### 2. Carnahan-Starling Non-Ideal EOS Form:
To model realistic van der Waals / Carnahan-Starling fluids with higher density ratios:
$$\psi_{\text{CS}}(\rho) = \sqrt{\frac{2 \left( p_{\text{CS}}(\rho) - c_s^2 \rho \right)}{c_s^2 |G|}}$$

### 2.3 Continuum Limit & Pressure Tensor

Expanding $\psi(\mathbf{x} + \mathbf{c}_i)$ in a Taylor series about node $\mathbf{x}$:

$$\psi(\mathbf{x} + \mathbf{c}_i) = \psi(\mathbf{x}) + c_{i\alpha} \partial_\alpha \psi + \frac{1}{2} c_{i\alpha} c_{i\beta} \partial_\alpha \partial_\beta \psi + \dots$$

Using the isotropic lattice tensor moments $\sum_i w_i c_{i\alpha} c_{i\beta} = c_s^2 \delta_{\alpha\beta}$:

$$\mathbf{F}_{\text{SC}} = -G c_s^2 \psi \nabla \psi - \frac{G c_s^4}{2} \psi \nabla (\nabla^2 \psi) + O(\nabla^4)$$

Rewriting the leading-order term as the gradient of a scalar potential:
$$-G c_s^2 \psi \nabla \psi = -\nabla \left( \frac{c_s^2 G}{2} \psi^2 \right)$$

This extra term alters the macroscopic momentum balance:
$$\partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p + \mathbf{F}_{\text{SC}} = -\nabla \left( c_s^2 \rho + \frac{c_s^2 G}{2} \psi^2 \right) + \mathbf{F}_{\text{dispersion}}$$

Thus, the effective equation of state (EOS) governing the Shan-Chen fluid is non-ideal:

$$p(\rho) = c_s^2 \rho + \frac{c_s^2 G}{2} \psi^2(\rho)$$

### 2.4 Spinodal Decomposition & Critical Point

Phase separation requires the fluid to enter the spinodal regime where compressibility is negative, i.e., $\frac{\partial p}{\partial \rho} < 0$.

Differentiating the Shan-Chen EOS with the exponential potential ($\rho_0 = 1$):
$$\frac{\partial p}{\partial \rho} = c_s^2 \left[ 1 + G \psi(\rho) \psi'(\rho) \right] = c_s^2 \left[ 1 + G (1 - e^{-\rho}) e^{-\rho} \right]$$

The function $g(\rho) = (1 - e^{-\rho}) e^{-\rho}$ has a single maximum at $\rho_c = \ln(2) \approx 0.69315$, where:
$$g(\rho_c) = (1 - 0.5)(0.5) = \frac{1}{4}$$

Therefore, the critical interaction parameter $G_c$ is:
$$1 + G_c \cdot \frac{1}{4} = 0 \implies G_c = -4.0$$

- For $G > -4.0$ (subcritical / high temperature): $\frac{\partial p}{\partial \rho} > 0$ for all $\rho$. The fluid remains single-phase; density fluctuations are exponentially damped by acoustic dissipation.
- For $G < -4.0$ (supercritical / sub-critical temperature): $\frac{\partial p}{\partial \rho} < 0$ in the interval between the spinodal densities $\rho_{s1} < \rho < \rho_{s2}$. Any infinitesimal density fluctuation triggers spontaneous phase separation into a dense liquid phase $\rho_l$ and a dilute vapor phase $\rho_v$.

---

## 3. Numerical Architecture

The implementation in [`qlbm/classical/multiphase.py`](file:///home/aswa/Research/qlbm/qlbm/classical/multiphase.py) defines:
- `compute_pseudopotential(rho, form, rho_0, G)`: Vectorized calculation supporting exponential and Carnahan-Starling forms.
- `compute_shan_chen_force(psi, G)`: Computes the 2D interaction force via periodic lattice stencil shifts.
- `shan_chen_pressure(rho, G, rho_0)`: Calculates the thermodynamic non-ideal pressure $p(\rho)$.
- `ShanChenMultiphaseSolver`: Subclasses `D2Q9BGKSolver`, integrating:
  1. Microscopic interaction force $\mathbf{F}_{\text{SC}}$.
  2. Optional external gravity $\mathbf{F}_{\text{grav}} = \rho \mathbf{g}$.
  3. Half-step discrete velocity correction $\mathbf{u} = \frac{1}{\rho}\sum f_i \mathbf{c}_i + \frac{\Delta t}{2\rho}(\mathbf{F}_{\text{SC}} + \mathbf{F}_{\text{grav}})$.
  4. 2nd-order Guo discrete forcing source term $F_i$.

### 3.1 Exact Mass Conservation

Because the Guo forcing source satisfies $\sum_{i=0}^8 F_i = 0$ identically, and standard bounce-back preserves local mass on solid walls, total domain mass is conserved to machine precision:
$$\sum_{\mathbf{x}} \rho(\mathbf{x}, t) = \sum_{\mathbf{x}} \sum_{i=0}^8 f_i(\mathbf{x}, t) = \text{constant}$$

---

## 4. Verification & Analytical Benchmarks

All 8 tests in [`tests/research/test_shan_chen.py`](file:///home/aswa/Research/qlbm/tests/research/test_shan_chen.py) validate the theoretical predictions:

| Benchmark Test | Theoretical Expectation | Simulation Result | Status |
| :--- | :--- | :--- | :---: |
| **Pseudopotential Properties** | $\psi(0)=0$, strictly monotonic, $\psi \to 1.0$ | $\psi(0)=0$, diff $> 0$, $|\psi(20)-1| < 10^{-6}$ | **PASSED** |
| **Uniform Force Isotropy** | $\mathbf{F}_{\text{SC}}(\rho \equiv \text{const}) \equiv \mathbf{0}$ | $\max \|\mathbf{F}_{\text{SC}}\| < 10^{-15}$ | **PASSED** |
| **Cohesive Interface Direction** | Both slab edges pulled inward into dense core | $F_x(x=0) = +0.248 > 0$, $F_x(x=7) = -0.248 < 0$ | **PASSED** |
| **Subcritical Stability ($G=-3.0$)** | Density perturbation damps out ($G > G_c = -4$) | Spread decays from $0.04$ to $< 0.005$ | **PASSED** |
| **Supercritical Separation ($G=-5.0$)** | Spontaneous spinodal decomposition into liquid & gas | $\rho_l \approx 1.894$, $\rho_v \approx 0.118$, ratio $> 16$ | **PASSED** |
| **Spinodal Mass Conservation** | Discrete mass invariant throughout phase separation | $|\Delta M| < 10^{-11}$ across 500 steps | **PASSED** |
| **Equation of State Spinodal Loop** | Van der Waals-like loop with $\partial p / \partial \rho < 0$ | Spinodal region confirmed with negative slope | **PASSED** |
| **Gravity-Driven Stratification** | Liquid settles to lower domain, vapor rises to top | Lower core $\rho = 1.89 > 1.7$, Upper $\rho = 0.118 < 0.2$ | **PASSED** |

---

## 5. Architectural Implications for the Two-Phase Dam-Break Goal

While the Shan-Chen model successfully reproduces multiphase thermodynamics, phase separation, and gravity stratification, it exhibits well-documented limitations for large-scale free-surface hydrodynamics:
1. **Density Ratio Limitation:** The exponential potential achieves density ratios $\rho_l / \rho_v \sim 10 - 20$. Attempting higher ratios ($> 50$) causes parasitic spurious currents at the diffuse interface, leading to numerical divergence.
2. **Diffuse Interface Width:** The interface spans $4 - 6$ lattice cells. For macroscopic dam-break waves and thin splash sheets, this diffuse layer introduces unphysical numerical diffusion.
3. **Air-Water Contrast:** Physical air-water systems have a density ratio of approximately $800:1$ to $1000:1$ and dynamic viscosity ratio $50:1$.

**Conclusion for Thesis Architecture:**
The Shan-Chen solver acts as our **thermodynamic and diffuse-interface kinetic reference**.
For the two-phase dam-break simulation, we will proceed to:
- **Phase 10:** Quantify surface tension via Laplace's Law $\Delta p = \sigma / R$ using the Shan-Chen droplet test.
- **Phases 11–13:** Implement the sharp-interface **Volume-of-Fluid (VOF) / Free-Surface LBM** (Körner / conservative PLIC) capable of high density ratios and sharp liquid-gas boundary tracking.
