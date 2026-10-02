# Phase 2: MSQLBM Streaming Mathematics & Quantum Fourier Transform Advection

**Date:** 2026-10-02  
**Target Module:** [`qlbm/components/ms/streaming.py`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py)  
**Primary Classes:** [`MSStreamingOperator`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py#L241), [`ControlledIncrementer`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py#L114), [`StreamingAncillaPreparation`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py#L18)  
**Adders Module:** [`qlbm/components/common/adders.py`](file:///home/aswa/Research/qlbm/qlbm/components/common/adders.py) (`PhaseShift`)  
**Theoretical Foundation:** Draper (2000) :cite:`draper`, Beauregard (2002) :cite:`qftadder`, Schalkers & Möller (2024) :cite:`collisionless`  

---

## 1. Physical & Mathematical Problem Formulation

In the continuous Boltzmann transport equation without collisions, particles stream along characteristic trajectories:
$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f = 0 \implies f(\mathbf{x}, \mathbf{v}, t + \Delta t) = f(\mathbf{x} - \mathbf{v} \Delta t, \mathbf{v}, t)$$

In a discrete Cartesian grid with spatial coordinates $\mathbf{x} = (x, y, z) \in \prod_{k=0}^{d-1} [0, N_k - 1]$ and periodic boundary conditions, advection translates the particle coordinates by:
$$x'_k = (x_k \pm v_k) \bmod N_k, \quad \forall k \in \{0, \dots, d-1\}$$

In the Multi-Speed Quantum Lattice Boltzmann Method (MSQLBM), this advection is realized **coherently across all basis states** using a combination of:
1. **Multi-Speed CFL Substepping:** Decomposing heterogeneous speeds into a sequence of unit-increment substeps.
2. **Quantum Ancilla Preparation:** Dynamically tagging the active velocity components with an ancilla qubit $a_v$.
3. **QFT-Based Draper Increment/Decrement:** Diagonalizing spatial translation into phase rotations in Fourier space.

---

## 2. Quantum State Encoding of Coordinates and Velocities

Recall from Phase 1 that a multi-speed basis state on an $N_x \times N_y$ lattice is represented as:
$$|\psi\rangle = |a_v\rangle |a_o\rangle |a_c\rangle |g_x\rangle |g_y\rangle |v_x\rangle |v_y\rangle |vd_x\rangle |vd_y\rangle$$

### 2.1 Spatial Register
Let $n_g = \log_2 N$. Position $x \in \{0, \dots, N-1\}$ is encoded in binary in the computational basis:
$$|x\rangle = |x_{n_g - 1} \dots x_1 x_0\rangle, \quad x = \sum_{j=0}^{n_g - 1} x_j 2^j$$
Where qubit $j=0$ is the least significant bit (LSB).

### 2.2 Velocity Magnitude & Direction Registers
The velocity along Cartesian axis $k$ is decoupled into:
- **Magnitude tier:** $v_k \in \{0, \dots, N_v/2 - 1\}$. For $N_v = 4$:
  - $v_k = 0 \implies \text{slow speed tier}$ ($c = 1$).
  - $v_k = 1 \implies \text{fast speed tier}$ ($c = 3$).
- **Direction sign:** $vd_k \in \{0, 1\}$:
  - $vd_k = 1 \implies \text{positive translation } (+)$.
  - $vd_k = 0 \implies \text{negative translation } (-)$.

---

## 3. Mathematical Derivation of Draper QFT Advection

### 3.1 The Fourier Representation of Position
The Quantum Fourier Transform ($\text{QFT}$) maps the computational coordinate state $|x\rangle$ into a product state in the Fourier basis:
$$\text{QFT}_n |x\rangle = \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n - 1} \exp\left( \frac{2\pi i \, x \, y}{2^n} \right) |y\rangle$$
Expanding $y = \sum_{j=0}^{n-1} y_j 2^j$ reveals the separable product form:
$$\text{QFT}_n |x\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{j=0}^{n-1} \left( |0\rangle + \exp\left( 2\pi i \, x \cdot 2^{-(n-j)} \right) |1\rangle \right)$$

### 3.2 Diagonal Phase Rotation in Fourier Space
In Fourier space, spatial translation by $\Delta x = \pm 1$ becomes a diagonal phase operator $\hat{P}(\pm 1)$:
$$\hat{P}(\pm 1) |y\rangle = \exp\left( \pm \frac{2\pi i \, y}{2^n} \right) |y\rangle$$
Applying $\hat{P}(\pm 1)$ to the state $\text{QFT}_n |x\rangle$:
$$\hat{P}(\pm 1) \text{QFT}_n |x\rangle = \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n - 1} \exp\left( \frac{2\pi i \, x \, y}{2^n} \right) \exp\left( \pm \frac{2\pi i \, y}{2^n} \right) |y\rangle$$
$$= \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n - 1} \exp\left( \frac{2\pi i \, (x \pm 1) \, y}{2^n} \right) |y\rangle = \text{QFT}_n |(x \pm 1) \bmod 2^n\rangle$$

Applying the inverse Quantum Fourier Transform ($\text{QFT}_n^\dagger$) yields the translated computational state:
$$\text{QFT}_n^\dagger \, \hat{P}(\pm 1) \, \text{QFT}_n |x\rangle = |(x \pm 1) \bmod 2^n\rangle$$

### 3.3 Implementation in `PhaseShift`
The unitary $\hat{P}(\pm 1)$ is implemented in [`PhaseShift`](file:///home/aswa/Research/qlbm/qlbm/components/common/adders.py#L217) by applying single-qubit phase gates $P(\phi_j) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi_j} \end{pmatrix}$ to each qubit $j \in \{0, \dots, n-1\}$ of the grid register:
$$\phi_j = \pm \frac{\pi}{2^{n - 1 - j}} = \pm \frac{2\pi}{2^{n - j}}$$
Because $y = \sum_{j=0}^{n-1} y_j 2^j$, the total accumulated phase when the target is $|y\rangle$ is:
$$\sum_{j=0}^{n-1} y_j \phi_j = \pm 2\pi \sum_{j=0}^{n-1} \frac{y_j 2^j}{2^n} = \pm \frac{2\pi y}{2^n}$$
This reproduces the exact analytic phase shift.

---

## 4. Directional Control Logic (`ControlledIncrementer`)

In [`ControlledIncrementer`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py#L114), advection must increment if $vd = 1$ and decrement if $vd = 0$, but only when the CFL velocity ancilla is active ($a_v = 1$).

This is executed using a sequence of controlled phase shifts and Pauli-$X$ bit flips:

```
          ┌───────┐┌───────────────┐┌───┐┌────────────────┐┌───┐┌───────────┐
     a_v: ┤───────├┤■──────────────├┤───├┤■───────────────├┤───├┤───────────┤
          │       ││               ││   ││                ││   ││           │
      vd: ┤───────├┤■──────────────├┤─X─├┤■───────────────├┤─X─├┤───────────┤
          │ QFT_n ││  P(+1) block  ││   ││  P(-1) block   ││   ││ QFT_n^dag │
       g: ┤───────├┤───────────────├┤───├┤────────────────├┤───├┤───────────┤
          └───────┘└───────────────┘└───┘└────────────────┘└───┘└───────────┘
```

### Truth Table Analysis:

| $a_v$ | $vd_{\text{initial}}$ | Step 1: $C_2(P(+1))$ | Step 2: $X(vd)$ | Step 3: $C_2(P(-1))$ | Step 4: $X(vd)$ | Net Shift on $g$ | $vd_{\text{final}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **0** | Inactive ($a_v=0$) | $vd \to 1$ | Inactive ($a_v=0$) | $vd \to 0$ | $0$ (No shift) | $0$ |
| **0** | **1** | Inactive ($a_v=0$) | $vd \to 0$ | Inactive ($a_v=0$) | $vd \to 1$ | $0$ (No shift) | $1$ |
| **1** | **0** | Inactive ($vd=0$) | $vd \to 1$ | **Active** ($a_v=1, vd=1$) | $vd \to 0$ | **$-1 \bmod N$** | $0$ |
| **1** | **1** | **Active** ($a_v=1, vd=1$) | $vd \to 0$ | Inactive ($vd=0$) | $vd \to 1$ | **$+1 \bmod N$** | $1$ |

**Key Properties:**
1. The direction qubit $vd$ is preserved identically ($X \cdot X = I$).
2. Exactly one branch activates when $a_v = 1$.
3. When $a_v = 0$, the spatial register remains completely untouched.

---

## 5. Multi-Speed CFL Scheduling & Ancilla Preparation

When different velocity tiers travel at different physical speeds, how does a quantum circuit stream them concurrently without physical collision or race conditions?

### 5.1 The CFL Substepping Formulation
Let the physical speeds be $c_0 = 1$ (slow) and $c_1 = 3$ (fast).
In classical LBM with uniform time step $\Delta t$, speed $c_1$ covers 3 grid units while speed $c_0$ covers 1 grid unit.
In MSQLBM, each macroscopic time step $\Delta t$ is divided into sub-intervals according to a **CFL time series**:
$$\text{TimeSeries} = \big[ \mathcal{V}_1, \; \mathcal{V}_2, \; \dots, \; \mathcal{V}_M \big]$$
For $N_v = 4$, [`get_time_series(4)`](file:///home/aswa/Research/qlbm/qlbm/tools/utils.py#L236) yields $M = 3$ substeps:
$$\mathcal{V}_1 = [1], \quad \mathcal{V}_2 = [1], \quad \mathcal{V}_3 = [0, 1]$$

### 5.2 Substep Execution Cycle
During each substep $m \in \{1, 2, 3\}$:
1. **[`StreamingAncillaPreparation`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py#L18):**
   Applies an $X$-masked multi-controlled Toffoli gate to tag the active velocities:
   $$|v\rangle |a_v=0\rangle \xrightarrow{\text{MCX}} |v\rangle |a_v = [v \in \mathcal{V}_m]\rangle$$
2. **[`ControlledIncrementer`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py#L114):**
   Applies the $\pm 1$ QFT shift conditioned on $a_v$.
3. **Boundary Reflection (if geometry present):**
   Specular or bounce-back reflections are applied at each substep.
4. **Ancilla Uncomputation:**
   Applying `StreamingAncillaPreparation` again uncomputes $a_v \to 0$ because $\text{MCX}^\dagger = \text{MCX}$.

### 5.3 Cumulative Displacement per Macro-Step:
- **Slow speed tier ($v=0$):**
  Active only in substep 3 $\implies \Delta x = \pm 1$ grid unit.
- **Fast speed tier ($v=1$):**
  Active in substeps 1, 2, and 3 $\implies \Delta x = (\pm 1) + (\pm 1) + (\pm 1) = \pm 3$ grid units.

---

## 6. Deterministic Validation Matrix (2D $8 \times 8$ Lattice)

To verify the streaming operator rigorously, 9 deterministic cases spanning all 4 quadrants, both speed tiers, and periodic modular boundaries were simulated:

| Case | Initial $(x, y)$ | Speed Tier $(v_x, v_y)$ | Direction $(vd_x, vd_y)$ | Expected Displacement | Expected Final $(x', y')$ | Statevector Result | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $(2, 2)$ | $(0, 0)$ [slow] | $(1, 1)$ [$+x, +y$] | $(+1, +1)$ | $(3, 3)$ | $(3, 3)$ | **PASSED** |
| 2 | $(2, 2)$ | $(0, 0)$ [slow] | $(1, 0)$ [$+x, -y$] | $(+1, -1)$ | $(3, 1)$ | $(3, 1)$ | **PASSED** |
| 3 | $(2, 2)$ | $(0, 0)$ [slow] | $(0, 1)$ [$-x, +y$] | $(-1, +1)$ | $(1, 3)$ | $(1, 3)$ | **PASSED** |
| 4 | $(2, 2)$ | $(0, 0)$ [slow] | $(0, 0)$ [$-x, -y$] | $(-1, -1)$ | $(1, 1)$ | $(1, 1)$ | **PASSED** |
| 5 | $(2, 2)$ | $(1, 1)$ [fast] | $(1, 1)$ [$+x, +y$] | $(+3, +3)$ | $(5, 5)$ | $(5, 5)$ | **PASSED** |
| 6 | $(2, 2)$ | $(1, 1)$ [fast] | $(1, 0)$ [$+x, -y$] | $(+3, -3)$ | $(5, 7)$ [modular wrap] | $(5, 7)$ | **PASSED** |
| 7 | $(2, 2)$ | $(1, 1)$ [fast] | $(0, 1)$ [$-x, +y$] | $(-3, +3)$ | $(7, 5)$ [modular wrap] | $(7, 5)$ | **PASSED** |
| 8 | $(2, 2)$ | $(1, 1)$ [fast] | $(0, 0)$ [$-x, -y$] | $(-3, -3)$ | $(7, 7)$ [modular wrap] | $(7, 7)$ | **PASSED** |
| 9 | $(0, 0)$ | $(0, 0)$ [slow] | $(0, 0)$ [$-x, -y$] | $(-1, -1)$ | $(7, 7)$ [origin wrap] | $(7, 7)$ | **PASSED** |

---

## 7. Comparison with Classical LBM Streaming

| Feature | Classical D2Q9 LBM | MSQLBM |
| :--- | :--- | :--- |
| **Grid Representation** | Dense 2D arrays $f_i(x, y)$ | Logarithmic qubit registers $|g_x\rangle |g_y\rangle$ |
| **Streaming Mechanism** | Memory shift / pointer offset | QFT $\rightarrow$ Controlled Phase Rotations $\rightarrow \text{QFT}^\dagger$ |
| **Discrete Velocities** | 9 isotropic vectors ($c_0=0, c_{1-4}=1, c_{5-8}=\sqrt{2}$) | 16 tensor-product speeds $(c_x, c_y) \in \{\pm 1, \pm 3\}^2$ |
| **Boundary Handling** | Array slicing / bounce-back assignments | Unitary quantum comparator gates + phase reflections |
| **Collision Integration** | Local non-linear algebraic relaxation | **Collisionless** (pure unitary advection) |

This comparison highlights that while MSQLBM achieves spatial advection with exponential qubit efficiency ($\mathcal{O}(\log N)$ qubits), the resulting velocity tensor product lacks isotropic lattice weights ($w_i$), rest particles, and local non-unitary collision relaxation. These missing physical capabilities must be grounded classically before quantum hybridizations can be pursued.
