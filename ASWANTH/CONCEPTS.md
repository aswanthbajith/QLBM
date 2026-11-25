# Theoretical Concepts and Mathematical Foundations

This document provides a rigorous explanation of each core scientific and algorithmic concept implemented in the **`ASWANTH`** suite.

---

## 1. D3Q19 Discrete Lattice Hydrodynamics

### Lattice Topology
In three dimensions, the standard lattice Boltzmann discretization is **D3Q19**, which defines 19 discrete velocity vectors $\mathbf{c}_i \in \mathbb{R}^3$:
- 1 rest particle: $\mathbf{c}_0 = (0, 0, 0)$
- 6 face-connected neighbor directions: $(\pm 1, 0, 0), (0, \pm 1, 0), (0, 0, \pm 1)$
- 12 edge-connected diagonal directions: $(\pm 1, \pm 1, 0), (\pm 1, 0, \pm 1), (0, \pm 1, \pm 1)$

The lattice speed of sound is:
$$c_s^2 = \frac{1}{3}$$

The lattice weights $w_i$ satisfy exact moment isotropy:
$$w_0 = \frac{1}{3}, \quad w_{1 \dots 6} = \frac{1}{18}, \quad w_{7 \dots 18} = \frac{1}{36}, \quad \sum_{i=0}^{18} w_i = 1$$

### Equilibrium Distribution
The Maxwell-Boltzmann equilibrium distribution expanded to second order in Mach number is:
$$f_i^{\text{eq}}(\rho, \mathbf{u}) = w_i \rho \left[ 1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\|\mathbf{u}\|^2}{2 c_s^2} \right]$$

### Guo Second-Order 3D Body Forcing
External gravity $\mathbf{g} = (0, 0, -g_z)$ enters through Guo's force term:
$$F_i = w_i \left( 1 - \frac{1}{2\tau} \right) \left[ \frac{\mathbf{c}_i - \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i \cdot \mathbf{u})}{c_s^4} \mathbf{c}_i \right] \cdot \mathbf{F}$$

Macroscopic moments are decoded as:
$$\rho = \sum_{i=0}^{18} f_i, \quad \rho \mathbf{u} = \sum_{i=0}^{18} f_i \mathbf{c}_i + \frac{\Delta t}{2} \mathbf{F}$$

---

## 2. 3D Draper QFT Spatial Streaming on Qubit Registers

### Quantum Register Mapping
The hydrodynamic distribution field $f_i(x, y, z)$ on an $N_x \times N_y \times N_z$ grid is mapped to a quantum state:
$$|\psi\rangle = \frac{1}{\sqrt{F_{\text{total}}}} \sum_{x, y, z, i} \sqrt{f_i(x, y, z)} \, |x\rangle |y\rangle |z\rangle |i\rangle$$

The qubit register layout is:
- $n_x = \log_2 N_x$ qubits for coordinate $x$
- $n_y = \log_2 N_y$ qubits for coordinate $y$
- $n_z = \log_2 N_z$ qubits for coordinate $z$
- $n_v = \lceil \log_2 19 \rceil = 5$ qubits for discrete velocity $i$

For an $8 \times 4 \times 8$ grid:
$$n_{\text{total}} = 3 + 2 + 3 + 5 = 13 \text{ qubits} \quad (\text{statevector dimension } 2^{13} = 8192)$$

### Draper QFT Translation Operator
Streaming translates populations by $\mathbf{c}_i$:
$$|x, y, z, i\rangle \to |(x + c_{ix}) \bmod N_x, \, (y + c_{iy}) \bmod N_y, \, (z + c_{iz}) \bmod N_z, \, i\rangle$$

In each spatial dimension $d \in \{x, y, z\}$:
1. Apply the Quantum Fourier Transform: $\text{QFT}(n_d)$ maps the spatial register to the Fourier momentum basis:
   $$|x\rangle \xrightarrow{\text{QFT}} \frac{1}{\sqrt{N_d}} \sum_{k=0}^{N_d-1} \omega^{x k} |k\rangle, \quad \omega = e^{2\pi i / N_d}$$
2. In Fourier space, coordinate translation $\Delta x = \pm 1$ becomes a diagonal phase rotation:
   $$\text{PhaseShift}_j = \exp\left( \pm \frac{i \pi}{2^{n_d - 1 - j}} \right)$$
   This phase shift is conditioned on the velocity register state $|i\rangle$.
3. Apply the inverse QFT: $\text{QFT}^\dagger(n_d)$ returns the register to the computational basis.

---

## 3. 3D Volume-of-Fluid (VOF) Interface Tracking

### Cell Classification
Each lattice cell $\mathbf{x}$ is classified based on its liquid fill fraction $C(\mathbf{x}) \in [0, 1]$:
- **GAS ($C = 0$)**: Empty cell without fluid populations.
- **INTERFACE ($0 < C < 1$)**: Surface cell containing the liquid-gas boundary.
- **LIQUID ($C = 1$)**: Fully filled fluid cell governed by Navier-Stokes hydrodynamics.
- **SOLID**: Immobile obstacle (wall, shelf, gate).

### 3D Isotropic Interface Normals
The unit normal vector $\mathbf{n} = -\nabla C / \|\nabla C\|$ pointing from liquid into gas is computed via the 19-point isotropic D3Q19 finite difference gradient:
$$\nabla C(\mathbf{x}) = \frac{1}{c_s^2} \sum_{i=1}^{18} w_i \mathbf{c}_i C(\mathbf{x} + \mathbf{c}_i)$$

---

## 4. Anti-Symmetric Pairwise Mass Fluxes & Strict Conservation

To prevent artificial mass loss or gain during interface motion, mass exchange $\Delta M_i(\mathbf{x})$ between neighboring cells along direction $\mathbf{c}_i$ is formulated symmetrically:
$$\Delta M_i(\mathbf{x}) = A_i(\mathbf{x}) \cdot [f_{\bar{i}}^*(\mathbf{x} + \mathbf{c}_i) - f_i^*(\mathbf{x})] \Delta t$$
where $A_i(\mathbf{x}) = \frac{1}{2}[C(\mathbf{x}) + C(\mathbf{x} + \mathbf{c}_i)]$ is the symmetric interface contact area.

Because:
$$A_{\bar{i}}(\mathbf{x} + \mathbf{c}_i) = A_i(\mathbf{x}) \quad \text{and} \quad [f_i^*(\mathbf{x}) - f_{\bar{i}}^*(\mathbf{x} + \mathbf{c}_i)] = -[f_{\bar{i}}^*(\mathbf{x} + \mathbf{c}_i) - f_i^*(\mathbf{x})]$$
we have the strict anti-symmetry identity:
$$\Delta M_i(\mathbf{x}) \equiv -\Delta M_{\bar{i}}(\mathbf{x} + \mathbf{c}_i)$$

Summing across all cells in the domain yields:
$$\sum_{\mathbf{x}} \sum_{i=1}^{18} \Delta M_i(\mathbf{x}) \equiv 0$$
guaranteeing machine-precision mass conservation throughout gate opening, vertical free fall, and basin impact.

---

## 5. Free-Surface Gas Boundary Reconstruction

Gas cells do not track fluid distribution functions. When streaming occurs, populations $f_{\bar{i}}(\mathbf{x}, t + \Delta t)$ entering interface cells from gas neighbors must be reconstructed to enforce atmospheric gas pressure $p_{\text{gas}}$:
$$f_{\bar{i}}(\mathbf{x}, t + \Delta t) = f_i^{\text{eq}}(\rho_{\text{gas}}, \mathbf{u}) + f_{\bar{i}}^{\text{eq}}(\rho_{\text{gas}}, \mathbf{u}) - f_i^*(\mathbf{x}, t)$$

This boundary condition (Körner et al. 2005) ensures exact dynamic pressure balance across the free boundary without spurious interface oscillations.

---

## 6. Unitary Velocity Reflection & Quantum Bounce-Back

Solid wall no-slip boundary conditions reflect particles back along their incoming path:
$$\mathbf{c}_i \to \mathbf{c}_{\bar{i}} = -\mathbf{c}_i$$

For D3Q19, this requires swapping the 9 opposite velocity pairs:
$$(1 \leftrightarrow 2), (3 \leftrightarrow 4), (5 \leftrightarrow 6), (7 \leftrightarrow 8), (9 \leftrightarrow 10), (11 \leftrightarrow 12), (13 \leftrightarrow 14), (15 \leftrightarrow 16), (17 \leftrightarrow 18)$$
while leaving the rest particle $0$ invariant.

In quantum amplitude encoding on 5 qubits ($2^5 = 32$ states), this permutation is synthesized as a 5-qubit unitary permutation operator:
$$U_{\text{bb}} = \sum_{i=0}^{18} |\bar{i}\rangle \langle i| + \sum_{k=19}^{31} |k\rangle \langle k|$$
which satisfies $U_{\text{bb}}^\dagger U_{\text{bb}} = I$ with zero phase distortion.
