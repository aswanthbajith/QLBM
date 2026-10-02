# Volume-of-Fluid (VOF) Geometric Interface Reconstruction

## 1. Executive Summary

Having established that diffuse-interface methods (such as the Shan-Chen model in Phases 9–10) are restricted by parasitic spurious currents to low density ratios ($\sim 10 - 30$), **Phase 11** initiates the core architecture for the thesis's primary physics solver: the sharp-interface **Volume-of-Fluid (VOF) Free-Surface Lattice Boltzmann framework** (Körner et al. 2005, Bogner et al. 2015).

Phase 11 develops, implements, and benchmarks the complete geometric foundation:
1. **Discrete Cell State Classification:** Categorizes lattice nodes into `GAS` ($C \le \epsilon$), `INTERFACE` ($\epsilon < C < 1 - \epsilon$), `LIQUID` ($C \ge 1 - \epsilon$), and `SOLID` obstacle nodes.
2. **2nd-Order Youngs Gradient Operator:** Estimates outward interface unit normals $\mathbf{n} = (n_x, n_y)$ using a 9-point weighted central-difference stencil.
3. **Exact Analytical PLIC (Piecewise Linear Interface Calculation):** Implements closed-form forward area and inverse line-constant ($\alpha$) calculation across all four quadrants ($2\pi$ space) following the classical geometry of Scardovelli & Zaleski (2000).
4. **Machine-Precision Consistency:** Proves roundtrip consistency $|C_{\text{calculated}}(\alpha(C)) - C| < 10^{-14}$ across 1000 random orientations and fill levels.
5. **Dam-Break Free-Surface Topology:** Validates the geometric initialization of rectangular liquid columns characteristic of two-phase dam-break problems.

All implementations reside in [`qlbm/classical/vof.py`](file:///home/aswa/Research/qlbm/qlbm/classical/vof.py) and are validated in [`tests/research/test_vof_reconstruction.py`](file:///home/aswa/Research/qlbm/tests/research/test_vof_reconstruction.py).

---

## 2. Mathematical Formulation

### 2.1 Volume Fraction & Discrete Cell Classification

Let $\Omega_i$ be the Cartesian control volume of lattice cell $\mathbf{x} = (x_i, y_j)$ with volume $\Delta V = \Delta x \Delta y$. The liquid volume fraction (fill level) $C(\mathbf{x}) \in [0, 1]$ is:

$$C(\mathbf{x}, t) = \frac{1}{\Delta V} \int_{\Omega(\mathbf{x})} \chi_{\text{liquid}}(\mathbf{r}, t) \, d\mathbf{r}$$

where $\chi_{\text{liquid}}$ is the characteristic indicator function of the liquid phase.

To eliminate the computational burden of simulating ambient air at high density ratios, the domain is partitioned into four distinct topological states:

$$\text{State}(\mathbf{x}) = \begin{cases}
\text{SOLID} & \text{if } \mathbf{x} \in \Omega_{\text{solid}} \\
\text{GAS} & \text{if } C(\mathbf{x}) \le \epsilon \\
\text{LIQUID} & \text{if } C(\mathbf{x}) \ge 1 - \epsilon \\
\text{INTERFACE} & \text{if } \epsilon < C(\mathbf{x}) < 1 - \epsilon
\end{cases}$$

with classification tolerance $\epsilon = 10^{-5}$.
- In **LIQUID** cells, standard D2Q9 Navier-Stokes collision and streaming occur.
- In **GAS** cells, no distribution functions are computed (effective $\rho_L / \rho_G = \infty$).
- In **INTERFACE** cells, the free-surface boundary condition is applied and mass exchange is tracked geometrically.

---

### 2.2 Interface Normal Vector Estimation (Youngs Method)

Inside an interface cell, the boundary is approximated by a planar segment:
$$\mathbf{n} \cdot \mathbf{x} = \alpha$$
where $\mathbf{n} = (n_x, n_y)$ is the unit normal vector pointing from liquid into gas.

The normal vector is determined from the gradient of the volume fraction field $\nabla C$:
$$\mathbf{m} = -\nabla C, \quad \hat{\mathbf{n}} = \frac{\mathbf{m}}{\|\mathbf{m}\|_2}$$

On a Cartesian grid, Youngs (1982) 2nd-order 9-point weighted gradient stencil is applied:

$$m_x = -\frac{1}{8\Delta x} \left[ (C_{i+1, j+1} + 2C_{i+1, j} + C_{i+1, j-1}) - (C_{i-1, j+1} + 2C_{i-1, j} + C_{i-1, j-1}) \right]$$

$$m_y = -\frac{1}{8\Delta y} \left[ (C_{i+1, j+1} + 2C_{i, j+1} + C_{i-1, j+1}) - (C_{i+1, j-1} + 2C_{i, j-1} + C_{i-1, j-1}) \right]$$

This diagonal corner weighting suppresses grid-alignment bias and reduces orientation errors by over an order of magnitude compared to standard 2-point central differences.

---

### 2.3 Piecewise Linear Interface Calculation (PLIC)

In the local unit square $(x, y) \in [0, 1] \times [0, 1]$, the interface line is:
$$n_x x + n_y y = \alpha$$

Given the unit normal $\mathbf{n}$ and target volume fraction $C$, we require the exact line constant $\alpha$ such that the area beneath the line equals $C$.

#### 1. Quadrant Reduction:
Let $m_1 = \min(|n_x|, |n_y|)$ and $m_2 = \max(|n_x|, |n_y|)$.  
Reflecting negative coordinate directions shifts the line constant:
$$\alpha' = \alpha + \max(0, -n_x) + \max(0, -n_y)$$

#### 2. Forward Area Formula:
In the normalized first quadrant $0 \le m_1 \le m_2$:
$$\text{Area}(\alpha') = \begin{cases}
0 & \text{if } \alpha' \le 0 \\
\frac{(\alpha')^2}{2 m_1 m_2} & \text{if } 0 < \alpha' \le m_1 \quad \text{(Triangle)} \\
\frac{2\alpha' m_1 - m_1^2}{2 m_1 m_2} = \frac{\alpha' - m_1/2}{m_2} & \text{if } m_1 < \alpha' \le m_2 \quad \text{(Trapezoid)} \\
1 - \frac{(m_1 + m_2 - \alpha')^2}{2 m_1 m_2} & \text{if } m_2 < \alpha' \le m_1 + m_2 \quad \text{(Pentagon)} \\
1 & \text{if } \alpha' > m_1 + m_2
\end{cases}$$

#### 3. Closed-Form Inverse Formula (Scardovelli & Zaleski 2000):
Critical volume thresholds: $C_1 = \frac{m_1}{2 m_2}$, $C_2 = 1 - C_1$.
$$\alpha' = \begin{cases}
\sqrt{2 m_1 m_2 C} & \text{if } C \le C_1 \\
\frac{m_1}{2} + m_2 C & \text{if } C_1 < C \le C_2 \\
m_1 + m_2 - \sqrt{2 m_1 m_2 (1 - C)} & \text{if } C > C_2
\end{cases}$$

Finally, transforming back to global cell coordinates:
$$\alpha = \alpha' - \max(0, -n_x) - \max(0, -n_y)$$

This analytical inversion requires **zero numerical iterations** (no Newton-Raphson) and is exact to machine precision ($< 10^{-15}$).

---

## 4. Verification & Analytical Benchmarks

All 10 tests in [`tests/research/test_vof_reconstruction.py`](file:///home/aswa/Research/qlbm/tests/research/test_vof_reconstruction.py) validate the mathematical formulation:

| Benchmark Test | Criterion | Result | Status |
| :--- | :--- | :--- | :---: |
| **Cell State Classification** | GAS, LIQUID, INTERFACE, SOLID | Exact state mapping matching tolerance $\epsilon=10^{-5}$ | **PASSED** |
| **Horizontal Interface Normal** | Liquid below $\implies \mathbf{n} = (0, 1)$ | $\|\mathbf{n} - (0, 1)\| < 10^{-12}$ | **PASSED** |
| **Vertical Interface Normal** | Liquid left $\implies \mathbf{n} = (1, 0)$ | $\|\mathbf{n} - (1, 0)\| < 10^{-12}$ | **PASSED** |
| **Diagonal $45^\circ$ Interface** | Normal $\mathbf{n} = (1/\sqrt{2}, 1/\sqrt{2})$ | Angular error $< 10^{-10}$ degrees | **PASSED** |
| **Oblique Interfaces ($30^\circ, 60^\circ, 120^\circ, 150^\circ$)** | Youngs stencil accuracy on linear interfaces | Max angular error $2.089^\circ < 2.5^\circ$ | **PASSED** |
| **Circular Droplet Radial Normals** | Outward pointing normals everywhere | Mean error $0.64^\circ$, all $\mathbf{n} \cdot \mathbf{n}_{\text{exact}} > 0.95$ | **PASSED** |
| **PLIC Limiting Cases** | $C=0 \implies \text{Area}=0$, $C=1 \implies \text{Area}=1$ | Exact analytical limits satisfied | **PASSED** |
| **PLIC 4-Quadrant Roundtrip** | 1000 random orientations $\theta \in [0, 2\pi]$ | $\max \|C_{\text{calc}} - C_{\text{true}}\| < 10^{-14}$ | **PASSED** |
| **Axis-Aligned PLIC Inversion** | $n_x = 1, n_y = 0 \implies \alpha = C$ | Difference $< 10^{-14}$ | **PASSED** |
| **Dam-Break Column Initialization** | Rectangular column geometry ($H \times L$) | Clean interface identification and volume match | **PASSED** |

---

## 5. Architectural Bridge to Full Free-Surface Solver

Phase 11 provides the geometric core. The remaining components of the hybrid VOF-LBM solver are mapped across the subsequent phases:
- **Phase 12:** Mass flux advection across cell faces $\Delta M_{i}(\mathbf{x})$ and topological cell state transitions (interface to gas / liquid).
- **Phase 13:** Free-surface boundary condition: reconstructing distribution functions $f_i$ entering from the empty gas phase into interface cells.
- **Phase 14–15:** Two-phase liquid column collapse (dam-break) simulation and benchmark comparison against classical experimental data.
