# Multiphase LBM Model Survey & Architectural Selection

## 1. Executive Summary & Objective

The overarching objective of this research is to establish a scientifically defensible pathway toward simulating a **two-phase gas-liquid dam-break problem** using the Lattice Boltzmann Method (LBM) coupled with Volume-of-Fluid (VOF), with an architectural eye toward quantum compatibility.

A dam-break problem involves:
1. High density ratio ($\rho_L / \rho_G \approx 1000$ for water and air; at minimum $\rho_L / \rho_G \ge 100$ for benchmark models).
2. Large Reynolds numbers ($\text{Re} = \frac{\rho U H}{\mu} \sim 10^3 - 10^5$) and moderate Weber numbers ($\text{We} = \frac{\rho U^2 H}{\sigma} \sim 10^2 - 10^4$).
3. Violent topological changes: column collapse, surge wave propagation, impact against solid walls, splashing, droplet ejection, and air entrapment.
4. Strict mass/volume conservation of the liquid column.

**Phase 8** conducts a comprehensive, rigorous comparative survey of the four primary multiphase LBM paradigms and Volume-of-Fluid methodologies, evaluates their strengths, failure modes, and quantum suitability, and formalizes our architectural selection.

---

## 2. Comparative Analysis of Multiphase LBM Paradigms

```
+---------------------------------------------------------------------------------------------------+
| Paradigm           | Mathematical Basis             | Max Density Ratio | Spurious Currents       |
+--------------------+--------------------------------+-------------------+-------------------------+
| Shan-Chen          | Inter-particle pseudo-force    | 10 - 50 (std)     | High (O(10^-2))         |
| (Pseudopotential)  | F = -G psi sum w_i psi(x+c_i)  | ~1000 with PR/CS  | (thermodynamic artifact)|
+--------------------+--------------------------------+-------------------+-------------------------+
| Free Energy        | Ginzburg-Landau free energy    | 10 - 50           | Moderate (O(10^-3))     |
| (Swift / Yeomans)  | Pressure tensor P_ab           |                   |                         |
+--------------------+--------------------------------+-------------------+-------------------------+
| Color-Gradient     | Two-color populations +        | 100 - 1000        | Low (O(10^-4))          |
| (Gunstensen / RK)  | recoloring step                |                   |                         |
+--------------------+--------------------------------+-------------------+-------------------------+
| Phase-Field        | Conservative Allen-Cahn /      | ~ 1000 (air-water)| Very low (O(10^-4))     |
| (He-Chen-Zhang)    | Cahn-Hilliard + Hydrodynamic   |                   |                         |
+--------------------+--------------------------------+-------------------+-------------------------+
| VOF-LBM Hybrid     | Interface reconstruction       | Arbitrary         | Zero at interface       |
| (Körner / Hirt)    | (PLIC/Donut) + LBM hydro       | (1000 - inf)      | (sharp kinetic balance) |
+--------------------+--------------------------------+-------------------+-------------------------+
```

---

### 2.1 The Pseudopotential (Shan-Chen) Model

Introduced by Shan and Chen (1993, 1994), this model represents inter-particle molecular attractions through an interactive non-local pseudo-force:

$$\mathbf{F}_{\text{SC}}(\mathbf{x}) = -G \psi(\mathbf{x}) \sum_{i=0}^8 w_i \psi(\mathbf{x} + \mathbf{c}_i \Delta t) \mathbf{c}_i$$

where $\psi(\rho)$ is an effective density (pseudopotential).

#### Thermodynamic Equation of State
The interaction force shifts the equation of state from isothermal ideal gas to non-ideal gas:

$$p = c_s^2 \rho + \frac{c_s^2 G}{2} \psi^2(\rho)$$

Using real-gas equations of state (Carnahan-Starling or Peng-Robinson), $\psi(\rho)$ is engineered such that $\frac{c_s^2 G}{2} \psi^2(\rho) = p_{\text{EOS}}(\rho) - c_s^2 \rho$.

#### Strengths:
1. **Algorithmic Elegance:** Surface tension, phase separation, and droplet coalescence emerge spontaneously from local kinetic rules without explicit interface tracking.
2. **Computational Speed:** Requires only nearest-neighbor communication and a single set of distribution functions.

#### Critical Bottlenecks for Dam-Break:
1. **Severe Spurious Currents:** Due to lack of Galilean invariance and discrete gradient errors, large unphysical circular velocity currents ($u \sim 0.05 c_s$) circulate continuously around high-curvature interfaces.
2. **Density Ratio Limitation:** Standard Shan-Chen model is unstable for $\rho_L / \rho_G > 20$. While advanced EOS (Yuan & Schaefer 2006) allow $\rho_L / \rho_G \sim 1000$, the interface becomes extremely stiff, requiring tiny time steps and artificial numerical diffusion.
3. **Mass Loss / Diffuse Dissolution:** In high-velocity flows, small liquid droplets artificially dissolve into the vapor phase due to finite interface thickness (typically 4–6 grid cells).

---

### 2.2 The Free Energy Model

Pioneered by Swift et al. (1995, 1996), this approach grounds multiphase flow in equilibrium thermodynamics by specifying a Ginzburg-Landau free energy functional:

$$\mathcal{F}[\rho, \nabla \rho] = \int \left[ \psi_0(\rho) + \frac{\kappa}{2} |\nabla \rho|^2 \right] dV$$

Surface tension is defined thermodynamically:

$$\sigma = \kappa \int_{-\infty}^{\infty} \left( \frac{\partial \rho}{\partial z} \right)^2 dz$$

The non-equilibrium pressure tensor is directly embedded into the second moment of the equilibrium distribution:

$$P_{\alpha \beta} = \left( p_0 - \kappa \rho \nabla^2 \rho - \frac{\kappa}{2} |\nabla \rho|^2 \right) \delta_{\alpha \beta} + \kappa \partial_\alpha \rho \partial_\beta \rho$$

#### Strengths:
- Fully thermodynamically consistent: Maxwell equal-area construction is strictly satisfied.
- Surface tension and interface width are analytically decoupled from the grid spacing.

#### Critical Bottlenecks for Dam-Break:
- Evaluating $\nabla^2 \rho$ and gradient terms requires wide stencils that complicate solid wall boundary conditions.
- Highly vulnerable to numerical instability at density ratios $\rho_L / \rho_G > 10$.
- Lacks Galilean invariance unless higher-order non-linear terms are added to the equilibrium.

---

### 2.3 The Color-Gradient Model

Developed by Gunstensen et al. (1991) and refined by Halliday et al. (2007) and Reis & Phillips (2007), the color-gradient model tracks two distinct immiscible components ("red" and "blue") with distribution functions $f_i^R$ and $f_i^B$:

$$\rho_R = \sum_i f_i^R, \quad \rho_B = \sum_i f_i^B, \quad \rho = \rho_R + \rho_B$$

The phase interface is indicated by the normalized color gradient:

$$\mathbf{n} = -\frac{\nabla \rho_N}{|\nabla \rho_N|}, \quad \text{where } \rho_N = \frac{\rho_R - \rho_B}{\rho_R + \rho_B} \in [-1, 1]$$

#### The Three-Step Algorithm:
1. **Collision:** Standard BGK collision on total distribution $f_i = f_i^R + f_i^B$.
2. **Surface Tension Perturbation:** A discrete force perturbation $A_i(\mathbf{n}) = \frac{A}{2} |\mathbf{n}| [w_i \frac{(\mathbf{c}_i \cdot \mathbf{n})^2}{\mathbf{n} \cdot \mathbf{n}} - B_i]$ is added to create anisotropic interfacial stress.
3. **Recoloring (Segregation):** A geometric segregation step redistributes post-collision populations into $f_i^{*R}$ and $f_i^{*B}$ to maximize the flux of red particles along $\mathbf{n}$ and blue particles along $-\mathbf{n}$.

#### Strengths:
- Very low spurious currents ($O(10^{-4})$).
- Sharp interfaces across only 2–3 lattice nodes.
- Capable of handling density ratios up to $1000$ with MRT (Multiple Relaxation Time).

#### Critical Bottlenecks for Dam-Break:
- The recoloring step is purely heuristic and destroys Galilean invariance unless complex anisotropic corrections are applied.
- Recoloring can induce artificial lattice-orientation pinning for moving interfaces.

---

### 2.4 The Phase-Field (Cahn-Hilliard / Conservative Allen-Cahn) Model

Pioneered by He, Chen, and Zhang (1999) and matured by Lee & Lin (2005) and Fakhari et al. (2017), this approach decouples interface capturing from fluid hydrodynamics using two distribution sets:
1. **Order Parameter Distribution $g_i(\mathbf{x}, t)$:** Solves the conservative Allen-Cahn or Cahn-Hilliard convective-diffusion equation for the volume fraction / phase indicator $\phi \in [0, 1]$:
   $$\partial_t \phi + \nabla \cdot (\phi \mathbf{u}) = \nabla \cdot \left[ M \left( \nabla \phi - \frac{\mathbf{n}}{W} (1 - 4(\phi - 0.5)^2) \right) \right]$$
2. **Hydrodynamic Distribution $f_i(\mathbf{x}, t)$:** Solves the Navier-Stokes equations with variable density $\rho(\phi) = \phi \rho_L + (1 - \phi) \rho_G$ and surface tension force:
   $$\mathbf{F}_{\sigma} = \mu_{\phi} \nabla \phi = 6 \sigma \left[ -\nabla^2 \phi + \frac{1}{W^2} \phi(1-\phi)(1-2\phi) \right] \nabla \phi$$

#### Strengths:
- **True high density ratios:** Regularly simulates $\rho_L / \rho_G = 1000$ (water-air) with exceptional numerical stability.
- **Very low spurious currents:** Spurious velocities are an order of magnitude smaller than Shan-Chen.
- **Galilean Invariance:** Preserves physical convective momentum transport without lattice pinning.

#### Critical Bottlenecks:
- Requires dual distribution storage ($2 \times 9 = 18$ distributions per node in 2D).
- Computing the chemical potential $\mu_\phi$ requires numerical Laplace stencils ($\nabla^2 \phi$).

---

### 2.5 The Volume-of-Fluid (VOF) Method & Free-Surface Coupling

The Volume-of-Fluid method (Hirt & Nichols 1981) tracks the liquid volume fraction $\alpha(\mathbf{x}, t) \in [0, 1]$ in each computational cell:

$$\alpha(\mathbf{x}, t) = \frac{V_{\text{liquid}}(\mathbf{x}, t)}{V_{\text{cell}}}$$

$$\partial_t \alpha + \nabla \cdot (\alpha \mathbf{u}) = 0$$

#### The Free-Surface LBM Formulation (Körner, Thies, Franz, Zeiser 2005)
In the pioneering work of Körner et al. (2005) at FAU Erlangen, cells are dynamically partitioned into three active categories:
1. **Liquid Cells ($\alpha = 1$):** Standard LBM fluid nodes governed by BGK collision and streaming.
2. **Gas Cells ($\alpha = 0$):** Inactive nodes; no distributions are stored or updated.
3. **Interface Cells ($0 < \alpha < 1$):** Form a closed, contiguous monolayer isolating liquid from gas.
   - Interface cells stream and collide normally with adjacent liquid and interface nodes.
   - For links pointing toward gas cells, boundary distribution functions are reconstructed directly from the ambient gas pressure $p_{\text{gas}}$ and interface normal $\mathbf{n}$.
   - Mass flux across cell boundaries is computed from post-collision distributions:
     $$\Delta m_i(\mathbf{x}, t) = \left[ f_{\bar{i}}(\mathbf{x} + \mathbf{c}_i \Delta t, t) - f_i(\mathbf{x}, t) \right] \Delta t$$
     $$\alpha(\mathbf{x}, t + \Delta t) = \alpha(\mathbf{x}, t) + \frac{\sum_i \Delta m_i}{\rho_{\text{liquid}} V_{\text{cell}}}$$
   - When $\alpha \ge 1$, the cell converts to Liquid; when $\alpha \le 0$, the cell converts to Gas.

#### Strengths for Dam-Break Simulation:
1. **Infinite Density Ratio:** Because the gas phase is treated as a pressure boundary condition, the effective density ratio is $\rho_L / \rho_G = \infty$! No numerical instability can ever arise from the gas phase.
2. **Exact Volume & Mass Conservation:** Liquid mass is governed by discrete flux accounting ($\Delta m_i$), preventing droplet evaporation or mass drift.
3. **Memory & Computational Efficiency:** $50\% - 80\%$ of typical dam-break domain nodes are gas cells and do not need to be computed!
4. **Sharp, Crisp Interfaces:** No diffuse interface layers; the water-air interface is represented as a crisp geometric boundary.

---

## 3. Comprehensive Decision Matrix for Two-Phase Dam-Break

| Criterion | Shan-Chen | Free Energy | Color-Gradient | Phase-Field | Free-Surface / VOF-LBM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Air-Water Density Ratio ($\ge 1000$)** | Poor (stiff) | Unstable ($< 20$) | Good ($100-1000$) | Excellent ($1000$) | **Infinite / Exact ($\infty$)** |
| **Liquid Mass Conservation** | Poor (droplet loss) | Moderate | Good | Moderate | **Exact (machine precision)** |
| **Spurious Interfacial Currents** | High ($10^{-2}$) | Moderate ($10^{-3}$) | Low ($10^{-4}$) | Very Low ($10^{-4}$) | **Zero (no gas currents)** |
| **Interface Sharpness** | 4–6 nodes (diffuse) | 4–6 nodes | 2–3 nodes | 3–5 nodes | **1 node (crisp boundary)** |
| **Violent Wave Impact / Splashing** | Weak (evaporates) | Unstable | Moderate | High | **Exceptional** |
| **Memory Footprint** | $1 \times$ | $1 \times$ | $2 \times$ | $2 \times$ | **$\le 0.5 \times$ (gas ignored)** |
| **Quantum Compatibility** | Low | Low | Moderate | Moderate | **High (Hybrid coupling)** |

---

## 4. Architectural Selection: The Hybrid LBM-VOF Architecture

Based on rigorous scientific, physical, and numerical criteria, we establish:

### Primary Production Architecture: Hybrid LBM-VOF (Körner Formulation)
For the target two-phase gas-liquid dam-break problem, the **Free-Surface Volume-of-Fluid + Lattice Boltzmann Method (FS-VOF-LBM)** is selected as the primary physics engine.
- **Physical Justification:** It eliminates density ratio instability completely, conserves liquid volume to machine precision, and operates cleanly on violent surface splashes.
- **Algorithmic Coupling:** LBM solves Navier-Stokes hydrodynamics within the liquid domain, while the VOF advection and cell-state conversion engine tracks interface deformation and applies ambient pressure and surface tension boundary conditions.

### Supporting Kinetic Architecture: Conservative Multiphase LBM (Phase-Field / Shan-Chen)
To maintain complete generality and support benchmark comparisons:
- In **Phase 9**, we implement the Shan-Chen pseudopotential operator in `qlbm/classical/` as an internal kinetic benchmark.
- In **Phase 10**, we formulate and test Laplace's law of surface tension: $\Delta p = \frac{\sigma}{R}$.
- In **Phases 11–13**, we develop the dedicated Volume-of-Fluid engine (PLIC reconstruction, flux advection, interface cell state transitions) and couple it directly to the D2Q9 hydrodynamic solver.
- In **Phases 14–15**, we execute the classical two-phase dam-break simulation.
- In **Phases 16–18**, we analyze how the quantum transport engine (`MSQLBM`) can accelerate the advection/streaming step within this hybrid architecture.

---

## 5. Summary of Completed Survey

This survey formally anchors the transition from single-phase hydrodynamics to multiphase flow. The selection of the **Hybrid VOF-LBM methodology** provides an unambiguous, scientifically unassailable foundation for simulating gas-liquid dam breaks without resorting to unphysical density ratios or risking numerical divergence.
