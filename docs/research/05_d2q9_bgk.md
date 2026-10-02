# D2Q9 BGK Operator Deep Verification & Collision Dynamics

## 1. Executive Summary

In Lattice Boltzmann hydrodynamics, the collision operator $\Omega(\mathbf{f})$ is the dissipative engine that drives microscopic velocity distributions toward thermodynamic equilibrium, dissipates kinetic energy into heat, and generates physical viscosity according to the Navier-Stokes equations.

**Phase 5** performs a rigorous mathematical and numerical investigation of the Bhatnagar-Gross-Krook (**BGK**) Single-Relaxation-Time collision operator on the D2Q9 lattice stencil. Specifically, we:
1. Construct the Gram-Schmidt orthogonal moment basis matrix $M \in \mathbb{R}^{9 \times 9}$ (D'Humières et al., Lallemand & Luo) to decompose kinetic states into conserved hydrodynamic modes and relaxing non-hydrodynamic modes.
2. Prove and numerically verify the spectral decomposition of the collision operator: exactly 3 zero eigenvalues (mass and momentum conservation) and 6 degenerate relaxation eigenvalues $\lambda = -\omega = -1/\tau$.
3. Mathematically define the linear numerical stability domain $\tau > 0.5$ ($\omega \in (0, 2)$) and observe the bifurcation to exponential numerical instability when $\tau < 0.5$.
4. Reconcile kinetic non-equilibrium momentum flux $\Pi_{\alpha \beta}^{(neq)}$ with macroscopic Navier-Stokes shear stress $\sigma_{\alpha \beta}$ via the discrete lattice time step factor $(1 - \frac{1}{2\tau})$.
5. Clarify the thermodynamic status of polynomial LBM: proving that polynomial BGK strictly minimizes a quadratic Lyapunov functional $\Phi(f) = \sum \frac{(f_i - f_i^{(eq)})^2}{w_i}$ at rate $(1 - \omega)^2$, whereas the logarithmic $H$-theorem is only approximately satisfied up to $O(\text{Ma}^3)$ due to polynomial truncation of the Maxwellian.
6. Establish the fundamental quantum-classical collision gap that dictates how quantum dissipative collisions must be approached.

---

## 2. Orthogonal Moment Space Decomposition (D'Humières / Lallemand & Luo)

The microscopic state $\mathbf{f} \in \mathbb{R}^9$ can be mapped into a macroscopic/kinetic moment basis $\mathbf{m} \in \mathbb{R}^9$ via an orthogonal linear transformation:

$$\mathbf{m} = M \mathbf{f}, \quad \mathbf{f} = M^{-1} \mathbf{m}$$

### 2.1 The Moment Transformation Matrix $M$

On the D2Q9 lattice with discrete velocities $\mathbf{c}_i = (c_{ix}, c_{iy})$, the 9 orthogonal basis polynomials are:

$$\begin{aligned}
m_0 &= \sum_i f_i = \rho && \text{(fluid density, conserved)} \\
m_1 &= \sum_i f_i \left[ -4 + 3(c_{ix}^2 + c_{iy}^2) \right] = e && \text{(kinetic energy mode)} \\
m_2 &= \sum_i f_i \left[ 4 - \frac{21}{2}(c_{ix}^2 + c_{iy}^2) + \frac{9}{2}(c_{ix}^2 + c_{iy}^2)^2 \right] = \epsilon && \text{(energy squared mode)} \\
m_3 &= \sum_i f_i c_{ix} = j_x && \text{(x-momentum, conserved)} \\
m_4 &= \sum_i f_i \left[ -3 + 2(c_{ix}^2 + c_{iy}^2) \right] c_{ix} = q_x && \text{(heat flux mode in x)} \\
m_5 &= \sum_i f_i c_{iy} = j_y && \text{(y-momentum, conserved)} \\
m_6 &= \sum_i f_i \left[ -3 + 2(c_{ix}^2 + c_{iy}^2) \right] c_{iy} = q_y && \text{(heat flux mode in y)} \\
m_7 &= \sum_i f_i (c_{ix}^2 - c_{iy}^2) = p_{xx} && \text{(normal stress difference)} \\
m_8 &= \sum_i f_i c_{ix} c_{iy} = p_{xy} && \text{(off-diagonal shear stress)}
\end{aligned}$$

Arranged into matrix form:

$$M = \begin{pmatrix}
 1 &  1 &  1 &  1 &  1 &  1 &  1 &  1 &  1 \\
-4 & -1 & -1 & -1 & -1 &  2 &  2 &  2 &  2 \\
 4 & -2 & -2 & -2 & -2 &  1 &  1 &  1 &  1 \\
 0 &  1 &  0 & -1 &  0 &  1 & -1 & -1 &  1 \\
 0 & -2 &  0 &  2 &  0 &  1 & -1 & -1 &  1 \\
 0 &  0 &  1 &  0 & -1 &  1 &  1 & -1 & -1 \\
 0 &  0 & -2 &  0 &  2 &  1 &  1 & -1 & -1 \\
 0 &  1 & -1 &  1 & -1 &  0 &  0 &  0 &  0 \\
 0 &  0 &  0 &  0 &  0 &  1 & -1 &  1 & -1
\end{pmatrix}$$

### 2.2 Gram-Schmidt Orthogonality & Analytical Inversion

The row vectors $\mathbf{m}_k$ are mutually orthogonal in Euclidean $\mathbb{R}^9$:

$$M M^T = \text{diag}(9, 36, 36, 6, 12, 6, 12, 4, 4)$$

Consequently, the inverse transformation $M^{-1}$ requires no matrix inversion:

$$M^{-1} = M^T \left[ \text{diag}(M M^T) \right]^{-1}$$

---

## 3. BGK Collision Operator & Relaxation Spectrum

### 3.1 Operator Definition in Distribution Space

The Bhatnagar-Gross-Krook collision step updates distributions toward local polynomial equilibrium:

$$\mathbf{f}^* = \mathbf{f} - \frac{1}{\tau} (\mathbf{f} - \mathbf{f}^{(eq)}) = (1 - \omega) \mathbf{f} + \omega \mathbf{f}^{(eq)}$$

where $\omega = \Delta t / \tau$ is the collision relaxation frequency.

Linearizing around a homogeneous rest state $(\rho_0, \mathbf{u}_0 = \mathbf{0})$:
$$f_i^{(eq)} \approx w_i \left[ \sum_j f_j + 3 \mathbf{c}_i \cdot \sum_j f_j \mathbf{c}_j \right]$$

The equilibrium projection matrix $P_{eq} \in \mathbb{R}^{9 \times 9}$ has entries:
$$(P_{eq})_{ij} = w_i \left[ 1 + 3 (c_{ix} c_{jx} + c_{iy} c_{jy}) \right]$$

The post-collision evolution operator is:
$$C = I + \Omega = (1 - \omega) I + \omega P_{eq}$$

### 3.2 Spectral Analysis & Eigenvalues

Because $P_{eq}$ is an orthogonal projection operator with rank 3 (projecting onto the 3 conserved moments $\rho, j_x, j_y$):
1. **Conserved Subspace ($\text{dim} = 3$):**
   For any perturbation with non-zero mass or momentum, $P_{eq} \mathbf{v} = \mathbf{v}$.
   $$C \mathbf{v} = (1 - \omega) \mathbf{v} + \omega \mathbf{v} = 1.0 \cdot \mathbf{v}$$
   $$\implies \lambda_0 = \lambda_1 = \lambda_2 = 1.0 \quad (\Omega \text{ eigenvalue } 0.0)$$

2. **Relaxing Kinetic Subspace ($\text{dim} = 6$):**
   For any non-equilibrium perturbation orthogonal to mass and momentum, $P_{eq} \mathbf{v} = \mathbf{0}$.
   $$C \mathbf{v} = (1 - \omega) \mathbf{v}$$
   $$\implies \lambda_3 = \dots = \lambda_8 = 1 - \omega = 1 - \frac{1}{\tau} \quad (\Omega \text{ eigenvalue } -\omega)$$

In `tests/research/test_d2q9_bgk_relaxation.py`, this analytical spectrum was evaluated numerically across $\tau \in \{0.6, 0.8, 1.0, 1.5, 2.0\}$ and verified to match within $10^{-13}$.

---

## 4. Linear Numerical Stability Bounds

The post-collision amplification factor for non-equilibrium fluctuations is:

$$\sigma = 1 - \omega = 1 - \frac{1}{\tau}$$

For a non-equilibrium state $\mathbf{f} = \mathbf{f}^{(eq)} + \delta \mathbf{f}$, the perturbation after $k$ collision steps satisfies:

$$\delta \mathbf{f}^{(k)} = (1 - \omega)^k \delta \mathbf{f}^{(0)}$$

### Stability Regime Classification:

```
        Unstable               Over-relaxation             Under-relaxation
  (exponential explosion)     (damped oscillation)        (monotonic decay)
<-------------------------|--------------------------|------------------------->
0                       0.5                         1.0                       tau
(omega -> inf)        (omega=2)                  (omega=1)                 (omega=0)
                      Critical                   Projective
                      Stability                 Equilibrium
```

1. **Under-relaxation ($\tau > 1.0 \implies 0 < \omega < 1$):**
   $0 < 1 - \omega < 1$. Perturbations decay monotonically without overshoot.
2. **Projective relaxation ($\tau = 1.0 \implies \omega = 1$):**
   $1 - \omega = 0$. Perturbations collapse to exact equilibrium in a single step ($\delta \mathbf{f}^{(1)} = \mathbf{0}$).
3. **Over-relaxation ($0.5 < \tau < 1.0 \implies 1 < \omega < 2$):**
   $-1 < 1 - \omega < 0$. Perturbations alternate sign (overshoot equilibrium) while exponentially damping in amplitude.
4. **Instability Threshold ($\tau = 0.5 \implies \omega = 2$):**
   $1 - \omega = -1$. Perturbations oscillate with constant undamped amplitude (zero physical viscosity $\nu = 0$).
5. **Exponential Explosion ($\tau < 0.5 \implies \omega > 2$):**
   $|1 - \omega| > 1$. Perturbations grow exponentially without bound: $||\delta \mathbf{f}^{(k)}|| \sim |1 - \omega|^k \to \infty$.

---

## 5. Non-Equilibrium Stress Tensor & Chapman-Enskog Relation

The non-equilibrium momentum flux tensor $\Pi_{\alpha \beta}^{(neq)}$ is defined as:

$$\Pi_{\alpha \beta}^{(neq)} = \sum_{i=0}^8 (f_i - f_i^{(eq)}) c_{i\alpha} c_{i\beta}$$

From the multi-scale Chapman-Enskog expansion, to first order in Knudsen number $\text{Kn}$:

$$f_i^{(1)} = -\tau \Delta t \left[ \partial_t^{(1)} + c_{i\gamma} \partial_\gamma^{(1)} \right] f_i^{(eq)}$$

Contracting with discrete velocity dyadics $c_{i\alpha} c_{i\beta}$:

$$\Pi_{\alpha \beta}^{(neq)} = -\tau \Delta t \rho c_s^2 \left( \frac{\partial u_\alpha}{\partial x_\beta} + \frac{\partial u_\beta}{\partial x_\alpha} \right) + O(\text{Kn}^2)$$

For plane Couette flow with shear rate $\dot{\gamma} = \partial u_x / \partial y = U_0 / H$:

$$\Pi_{xy}^{(neq)} = -\frac{\tau}{3} \rho \dot{\gamma}$$

### The Discrete Lattice Factor

In continuous kinetic theory, shear stress equals $-\Pi_{xy}^{(neq)}$. However, in discrete Lattice Boltzmann methods with finite time step $\Delta t$, the physical Navier-Stokes viscous stress $\sigma_{xy}$ is related to the kinetic non-equilibrium tensor by the factor:

$$\sigma_{xy} = -\left( 1 - \frac{\Delta t}{2\tau} \right) \Pi_{xy}^{(neq)}$$

Substituting $\Pi_{xy}^{(neq)}$:

$$\sigma_{xy} = \left( 1 - \frac{1}{2\tau} \right) \frac{\tau}{3} \rho \dot{\gamma} = \frac{2\tau - 1}{6} \rho \dot{\gamma} = \rho \nu \dot{\gamma}$$

This recovers Newton's law of viscosity with exact kinematic shear viscosity $\nu = \frac{2\tau - 1}{6}$.

In `tests/research/test_d2q9_bgk_relaxation.py`, both $\Pi_{xy}^{(neq)}$ and the recovered $\sigma_{xy}$ were evaluated in steady Couette flow and matched the analytical formulas to within **$1.5\%$**.

---

## 6. Thermodynamic Status: Discrete Lyapunov Theorem vs $H$-Theorem

A foundational question in kinetic theory is whether the Lattice Boltzmann BGK operator satisfies Boltzmann's $H$-theorem:

$$H(t) = \sum_{\mathbf{x}} \sum_{i=0}^8 f_i \ln\left( \frac{f_i}{w_i} \right), \quad \frac{dH}{dt} \le 0$$

### 6.1 Why Standard Polynomial LBM Does Not Satisfy the Logarithmic $H$-Theorem

In continuous gas dynamics, the Maxwell-Boltzmann distribution $f_M \propto \exp(-v^2/2T)$ is the exact unconstrained minimizer of $H = \int f \ln f \, d\mathbf{v}$.

In standard LBM, however, the equilibrium distribution $f_i^{(eq)}$ is a **2nd-order Taylor series polynomial**:
$$f_i^{(eq)} = w_i \rho \left[ 1 + 3 \mathbf{c}_i \cdot \mathbf{u} + \frac{9}{2} (\mathbf{c}_i \cdot \mathbf{u})^2 - \frac{3}{2} |\mathbf{u}|^2 \right]$$

Because $f_i^{(eq)}$ is a polynomial truncation rather than an exponential, it is **not the exact minimum** of $\sum f_i \ln(f_i / w_i)$. Consequently, under oscillatory over-relaxation ($\tau < 1.0$), $H(t)$ can exhibit tiny fluctuations of order $O(\text{Ma}^3)$ instead of monotonic decrease.

### 6.2 The Quadratic Lyapunov Functional

Instead of the logarithmic entropy, the standard polynomial BGK collision operator strictly minimizes the **quadratic distance functional**:

$$\Phi(\mathbf{f}) = \sum_{i=0}^8 \frac{(f_i - f_i^{(eq)})^2}{w_i}$$

At each collision step:
$$f_i^* - f_i^{(eq)} = (1 - \omega) (f_i - f_i^{(eq)})$$

$$\Phi(\mathbf{f}^*) = \sum_{i=0}^8 \frac{(1 - \omega)^2 (f_i - f_i^{(eq)})^2}{w_i} = (1 - \omega)^2 \Phi(\mathbf{f})$$

Because $|1 - \omega| < 1$ for all $\tau > 0.5$:

$$\Phi(\mathbf{f}^*) < \Phi(\mathbf{f}) \quad \forall \, \mathbf{f} \ne \mathbf{f}^{(eq)}$$

This proves rigorously that **polynomial BGK possesses an exact, monotonic quadratic Lyapunov functional** that guarantees asymptotic convergence to equilibrium for all stable relaxation times.

---

## 7. The Quantum Collision Challenge (Scientific Gap)

Analyzing the BGK collision operator in moment and distribution space illuminates why implementing LBM collisions on quantum hardware is fundamentally non-trivial:

| Classical BGK Operator | Quantum Circuit Requirements |
| :--- | :--- |
| **Non-unitary contraction:** $||\mathbf{f}^* - \mathbf{f}^{(eq)}|| = |1 - \omega| ||\mathbf{f} - \mathbf{f}^{(eq)}|| < ||\mathbf{f} - \mathbf{f}^{(eq)}||$ | Quantum gates must be strictly unitary ($U^\dagger U = I$), preserving norm $||\psi||=1$. |
| **Non-linear equilibrium:** $f_i^{(eq)} \sim \rho \mathbf{u}^2 = \frac{(\sum f_j \mathbf{c}_j)^2}{\sum f_j}$ involves multiplication and division of state variables | Quantum mechanics is strictly linear ($\hat{U} (\alpha |a\rangle + \beta |b\rangle) = \alpha \hat{U}|a\rangle + \beta \hat{U}|b\rangle$). |
| **Dissipative entropy production:** irreversible loss of kinetic information into viscous heat | Closed quantum systems are time-reversible ($U^\dagger = U^{-1}$) with constant von Neumann entropy. |

### Scientifically Defensible Quantum Paths Forward

To implement collision dynamics in a quantum framework without violating quantum mechanics, three legitimate pathways exist in the literature:

1. **Unitary Equivalence-Class Permutations (LQLGA / Georgescu et al. 2026):**
   Already partially present in the repository (`EQCCollisionOperator`). Restricts collisions to unitary permutations within degenerate energy/momentum manifolds. Does not recover continuous BGK relaxation.
2. **Linear Embedding via Carleman Linearization / Quantum Block Encoding:**
   Maps the non-linear polynomial equilibrium and dissipative relaxation into a higher-dimensional linear system embeddable in a block-encoded unitary $U = \begin{pmatrix} A & * \\ * & * \end{pmatrix}$. Requires ancilla qubits and success probability scaling.
3. **Hybrid Classical-Quantum Architecture:**
   Uses quantum circuits for the multi-scale, high-dimensional advection and boundary reflection steps (where quantum streaming achieves exponential state packing), while performing non-linear multiphase collision and interface tracking classically or via hybrid coprocessor calls.

This fundamental insight guides our transition through multiphase physics and Volume-of-Fluid coupling.

---

## 8. Summary of Phase 5 Verification Suite

The verification suite `tests/research/test_d2q9_bgk_relaxation.py` contains 18 comprehensive tests:

1. `test_moment_matrix_orthogonality`: $M M^T$ is strictly diagonal. **PASSED**.
2. `test_analytical_inverse`: $M^{-1}$ matches $M^T [\text{diag}(M M^T)]^{-1}$. **PASSED**.
3. `test_moment_roundtrip_transformation`: $\mathbf{f} \to \mathbf{m} \to \mathbf{f}$ machine precision inversion ($< 10^{-14}$). **PASSED**.
4. `test_conserved_moment_mapping`: $m_0, m_3, m_5$ reproduce $\rho, j_x, j_y$. **PASSED**.
5. `test_collision_eigenvalues` (5 relaxation rates $\tau \in \{0.6, 0.8, 1.0, 1.5, 2.0\}$): strictly 3 unitary eigenvalues and 6 degenerate decaying eigenvalues $1 - 1/\tau$. **5 PASSED**.
6. `test_asymptotic_stability_tau_greater_than_half`: exponential damping of fluctuations at rate $|1 - 1/\tau|$. **PASSED**.
7. `test_instability_tau_less_than_half`: exponential divergence of fluctuations for $\tau = 0.4$. **PASSED**.
8. `test_nonequilibrium_stress_zero_at_equilibrium`: $\Pi_{\alpha \beta}^{(neq)} \equiv 0$ at equilibrium. **PASSED**.
9. `test_couette_shear_stress_relation`: steady shear stress recovery $\Pi_{xy}^{(neq)} = -\frac{\tau}{3} \rho \dot{\gamma}$ and $\sigma_{xy} = \rho \nu \dot{\gamma}$ ($< 1.5\%$ error). **PASSED**.
10. `test_quadratic_lyapunov_monotonic_decrease` (4 relaxation rates): strictly monotonic decrease of $\Phi(f)$ at exact rate $(1 - \omega)^2$. **4 PASSED**.
11. `test_logarithmic_h_convergence_to_equilibrium`: convergence of Boltzmann's $H$ to $H_{eq}$. **PASSED**.
