# Phase 1: Mathematical Specification & Register Encoding of MSLattice

**Date:** 2026-10-02  
**Target Class:** [`MSLattice`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/ms_lattice.py#L17) (`qlbm/lattice/lattices/ms_lattice.py`)  
**Base Classes:** [`AmplitudeLattice`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/base.py#L62) $\rightarrow$ [`Lattice`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/base.py#L31)  
**Literature Reference:** Schalkers & Möller (2024), *Journal of Computational Physics* 502, 112816 :cite:`collisionless`  

---

## 1. Overview & Conceptual Architecture

The Multi-Speed (MS) Lattice representation (`MSLattice`) is an **amplitude-encoded, multi-speed** lattice structure designed specifically for the collisionless quantum transport equation.

Unlike discrete-velocity lattices ($D_d Q_q$) such as D2Q9, which encode discrete directional velocities on a stencil, `MSLattice` discretizes velocity independently along each Cartesian dimension as a combination of:
1. **Speed magnitude** (binary integer $v_d \in \{0, \dots, N_{v_d}/2 - 1\}$).
2. **Direction sign** (Boolean qubit $vd_d \in \{0, 1\}$ where $1 \equiv +$ and $0 \equiv -$).

Physical spatial coordinates $(x, y, z)$ are encoded into logarithmic quantum position registers ($g_x, g_y, g_z$). Quantum ancilla registers handle multi-speed CFL stepping ($a_v$), obstacle boundary detection ($a_o$), and coordinate bounding box comparisons ($a_c$).

---

## 2. Register Allocation and Sizing Formulation

Let $d \in \{1, 2, 3\}$ denote the spatial dimensionality of the lattice.
For each dimension $k \in \{0, \dots, d-1\}$:
- $N_{g_k}$: Number of grid points along dimension $k$ (must satisfy $N_{g_k} = 2^{n_{g_k}}$).
- $N_{v_k}$: Number of discrete velocity levels along dimension $k$ (must satisfy $N_{v_k} = 2^{n_{v_k}}$ with $n_{v_k} \ge 2$).

### 2.1 Internal Representation Convention
In `MSLattice.__init__`, the input dictionary parameters undergo a decrement by 1:
$$\text{num\_gridpoints}[k] = N_{g_k} - 1$$
$$\text{num\_velocities}[k] = N_{v_k} - 1$$
This convention facilitates bitwise arithmetic:
$$n_{g_k} = (\text{num\_gridpoints}[k]).\text{bit\_length}() = \lceil \log_2 N_{g_k} \rceil$$
$$n_{v_k} = (\text{num\_velocities}[k]).\text{bit\_length}() = \lceil \log_2 N_{v_k} \rceil$$

### 2.2 Register Breakdown

| Register Name | Attribute | Size (Qubits) | Physical Meaning / Algorithmic Role |
| :--- | :--- | :--- | :--- |
| **`a_v`** | `ancilla_velocity_register` | $d$ | Velocity CFL ancillae. Tracks whether speed $v_k$ streams during the current CFL substep. |
| **`a_o`** | `ancilla_object_register` | $n_{\text{obs}}$ | Obstacle detection ancillae. Set to $\|1\rangle$ if a particle streams into an obstacle. |
| **`a_c`** | `ancilla_comparator_register`| $2(d-1)$ | Boundary comparator ancillae. Holds lower/upper bound comparison results along orthogonal dimensions. |
| **`g_k`** | `grid_registers[k]` | $n_{g_k}$ | Logarithmic spatial position register for dimension $k$. |
| **`v_k`** | `velocity_registers[k]` | $n_{v_k} - 1$ | Discrete speed magnitude register for dimension $k$. |
| **`v_dir_k`** | `velocity_dir_registers[k]` | $1$ | Discrete velocity direction qubit for dimension $k$ ($1 \equiv +, 0 \equiv -$). |

### 2.3 Obstacle Ancillae Sizing Rule ($n_{\text{obs}}$)
The number of obstacle ancillae $n_{\text{obs}}$ depends dynamically on the boundary condition types present in `shapes`:
$$n_{\text{obs}} = \begin{cases} 1, & \text{if all obstacles are bounce-back (or geometry is empty)} \\ d, & \text{if at least one specular reflection obstacle is present} \end{cases}$$
- **Bounce-Back:** All directional velocity components are simultaneously inverted upon collision. A single scalar flag ($n_{\text{obs}} = 1$) indicates collision with the solid domain.
- **Specular Reflection:** Only the velocity component normal to the contacted face is inverted. Determining which face or corner was struck requires $d$ separate collision flags ($n_{\text{obs}} = d$).

### 2.4 Total Qubit Count Formula
$$N_{\text{total}} = N_{\text{ancilla}} + N_{\text{grid}} + N_{\text{velocity}}$$
Where:
$$N_{\text{ancilla}} = d + n_{\text{obs}} + 2(d-1)$$
$$N_{\text{grid}} = \sum_{k=0}^{d-1} n_{g_k} = \sum_{k=0}^{d-1} \lceil \log_2 N_{g_k} \rceil$$
$$N_{\text{velocity}} = \sum_{k=0}^{d-1} \left( (n_{v_k} - 1) + 1 \right) = \sum_{k=0}^{d-1} n_{v_k} = \sum_{k=0}^{d-1} \lceil \log_2 N_{v_k} \rceil$$

---

## 3. Register Ordering & Qubit Index Mapping

The quantum registers are appended to the `QuantumCircuit` in a fixed contiguous sequence:
$$\text{Registers} = \Big( a_v, \; a_o, \; a_c, \; g_0, \dots, g_{d-1}, \; v_0, \dots, v_{d-1}, \; v\_dir_0, \dots, v\_dir_{d-1} \Big)$$

### 3.1 Accessor Methods & Exact Qubit Indices

#### 1. `ancillae_velocity_index(dim=None)`
- For specific dimension $k \in \{0, \dots, d-1\}$:
  $$\text{index} = [k]$$
- For all dimensions (`dim=None`):
  $$\text{indices} = [0, \dots, d-1]$$

#### 2. `ancillae_obstacle_index(index=None)`
- Offset ahead: $d$ qubits.
- For specific index $i \in \{0, \dots, n_{\text{obs}}-1\}$:
  $$\text{index} = [d + i]$$
- For all obstacle ancillae (`index=None`):
  $$\text{indices} = [d, \dots, d + n_{\text{obs}} - 1]$$

#### 3. `ancillae_comparator_index(index=None)`
- Offset ahead: $d + n_{\text{obs}}$ qubits.
- For 2D ($d=2$), $2(d-1) = 2$ comparator qubits: $[l, u]$ (lower bound, upper bound).
- For 3D ($d=3$), $2(d-1) = 4$ comparator qubits: $[l_0, u_0, l_1, u_1]$.
- For specific index $i \in \{0, \dots, d-2\}$:
  $$\text{indices} = [d + n_{\text{obs}} + 2i, \; d + n_{\text{obs}} + 2i + 1]$$
- For all comparator ancillae (`index=None`):
  $$\text{indices} = [d + n_{\text{obs}}, \dots, d + n_{\text{obs}} + 2(d-1) - 1]$$

#### 4. `grid_index(dim=None)`
- Total preceding ancillae: $N_{\text{ancilla}} = d + n_{\text{obs}} + 2(d-1)$.
- For specific dimension $k$:
  $$\text{start}_k = N_{\text{ancilla}} + \sum_{j=0}^{k-1} n_{g_j}$$
  $$\text{indices} = [\text{start}_k, \dots, \text{start}_k + n_{g_k} - 1]$$
- For all grid dimensions (`dim=None`):
  $$\text{indices} = [N_{\text{ancilla}}, \dots, N_{\text{ancilla}} + N_{\text{grid}} - 1]$$

#### 5. `velocity_index(dim=None)` (Speed Magnitudes)
- Preceding qubits: $N_{\text{ancilla}} + N_{\text{grid}}$.
- For specific dimension $k$:
  $$\text{start}_k = N_{\text{ancilla}} + N_{\text{grid}} + \sum_{j=0}^{k-1} (n_{v_j} - 1)$$
  $$\text{indices} = [\text{start}_k, \dots, \text{start}_k + (n_{v_k} - 1) - 1]$$
- For all speed magnitude registers (`dim=None`):
  $$\text{indices} = [N_{\text{ancilla}} + N_{\text{grid}}, \dots, N_{\text{ancilla}} + N_{\text{grid}} + \sum_{j=0}^{d-1}(n_{v_j}-1) - 1]$$

#### 6. `velocity_dir_index(dim=None)` (Velocity Directions)
- Preceding qubits: $N_{\text{ancilla}} + N_{\text{grid}} + \sum_{j=0}^{d-1} (n_{v_j} - 1)$.
- For specific dimension $k$:
  $$\text{index} = \left[ N_{\text{ancilla}} + N_{\text{grid}} + \sum_{j=0}^{d-1} (n_{v_j} - 1) + k \right]$$
- For all direction qubits (`dim=None`):
  $$\text{indices} = [N_{\text{total}} - d, \dots, N_{\text{total}} - 1]$$

---

## 4. Concrete Example: 2D $8 \times 8$ Lattice with $N_v = 4$

Configuration:
```json
{
  "lattice": {
    "dim": {"x": 8, "y": 8},
    "velocities": {"x": 4, "y": 4}
  },
  "geometry": []
}
```

### Parameter Calculation:
- Dimensions: $d = 2$
- Grid points: $N_x = 8, N_y = 8 \implies n_{g_x} = 3, n_{g_y} = 3$ (qubit sizes for $g_x, g_y$).
- Velocities: $N_{vx} = 4, N_{vy} = 4 \implies n_{vx} = 2, n_{vy} = 2$.
  - Magnitude qubits per dim: $n_{v_k} - 1 = 1$ qubit each ($v_x, v_y$).
  - Direction qubits per dim: 1 qubit each ($vd_x, vd_y$).
- Ancillae:
  - $a_v$: $d = 2$ qubits.
  - $a_o$: 1 qubit (no specular obstacle).
  - $a_c$: $2(2-1) = 2$ qubits.
  - $N_{\text{ancilla}} = 2 + 1 + 2 = 5$ qubits.
- Total qubits: $5 + (3 + 3) + (1 + 1) + (1 + 1) = \mathbf{15\text{ qubits}}$.

### Exact Qubit Map (No Obstacle / Bounce-Back):

```
Qubit Index:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
Register:    |--a_v--| a_o |--a_c--|----g_x----|----g_y---| v_x v_y vd_x vd_y
Role:         av0 av1  ao  ac0 ac1  gx0 gx1 gx2 gy0 gy1 gy2  vx  vy  vdx vdy
```

If a **specular** obstacle is added:
- $a_o$ expands from 1 to 2 qubits ($a_{o0}, a_{o1}$).
- Total qubits become $\mathbf{16\text{ qubits}}$.
- Registers shift: $a_c$ at [4, 5], $g_x$ at [6, 7, 8], $g_y$ at [9, 10, 11], $v_x$ at [12], $v_y$ at [13], $vd_x$ at [14], $vd_y$ at [15].

---

## 5. Endian Convention & Basis-State Interpretation

### 5.1 Bit Endianness Within Registers
Within every multi-qubit register, indexing is strictly **Little-Endian**:
- In coordinate register $g_x$ of size $n$, qubit $g_x[0]$ is the **least significant bit (LSB)** with numerical weight $2^0 = 1$.
- Qubit $g_x[n-1]$ is the **most significant bit (MSB)** with numerical weight $2^{n-1}$.
- A coordinate value $X \in \{0, \dots, N_x-1\}$ is represented as:
  $$X = \sum_{i=0}^{n-1} b_i 2^i \implies |X\rangle = |b_{n-1} \dots b_1 b_0\rangle_{g_x}$$
  Example for $x = 5 = 101_2$ in a 3-qubit register ($g_x = [5, 6, 7]$):
  - Qubit 5 (bit 0): $\|1\rangle$
  - Qubit 6 (bit 1): $\|0\rangle$
  - Qubit 7 (bit 2): $\|1\rangle$

### 5.2 Qiskit Statevector Basis Index Encoding
In Qiskit, quantum basis states are indexed by integers $I \in \{0, \dots, 2^{N_{\text{total}}} - 1\}$:
$$I = \sum_{q=0}^{N_{\text{total}}-1} \text{bit}_q \cdot 2^q$$
Where $\text{bit}_q$ is the state ($\{0, 1\}$) of qubit $q$ in the circuit.
- Qubit 0 ($a_{v0}$) is the **least significant bit** of the global integer $I$.
- Qubit $N_{\text{total}}-1$ ($vd_{y}$) is the **most significant bit** of the global integer $I$.
- In Dirac string notation $|b_{N-1} b_{N-2} \dots b_1 b_0\rangle$:
  $$\text{Leftmost bit} = \text{Qubit } N_{\text{total}}-1 \quad (\text{MSB})$$
  $$\text{Rightmost bit} = \text{Qubit } 0 \quad (\text{LSB})$$

### 5.3 Physical Decoding of Basis States
Given an active basis state index $I$, the physical variables are decoded by bitmasking:
$$\text{val}(\text{reg}) = \sum_{i=0}^{\text{size}-1} \left( \frac{I}{2^{\text{start} + i}} \bmod 2 \right) 2^i$$

| Decoded Field | Symbol | Physical Interpretation |
| :--- | :--- | :--- |
| **Grid Position** | $g_x, g_y, g_z$ | Discrete spatial coordinates $x \in [0, N_x-1]$, $y \in [0, N_y-1]$. |
| **Speed Magnitude** | $v_x, v_y, v_z$ | Magnitude tier $v_d \in \{0, \dots, N_{v}/2 - 1\}$. In MSQLBM, $v=0$ denotes the slow speed channel ($c=1$), while $v=1$ denotes the fast speed channel ($c=3$). |
| **Velocity Direction**| $vd_x, vd_y, vd_z$ | Direction sign: $vd = 1 \implies +$ direction; $vd = 0 \implies -$ direction. |
| **Obstacle State** | $a_o$ | Boundary marker: $0 \equiv \text{fluid}$; $1 \equiv \text{inside solid obstacle}$. |
| **CFL Ancilla** | $a_v$ | Step control: $1 \equiv \text{streaming active in current CFL substep}$; $0 \equiv \text{inactive/reset}$. |

---

## 6. Physical Implications & Scientific Limitations

1. **No Rest Population ($c = 0$):**
   In `MSLattice`, velocity magnitude $v \in \{0, 1\}$ does not include a zero-velocity rest state. Speed $v=0$ streams 1 step, and speed $v=1$ streams 3 steps under CFL scheduling. In standard LBM ($D_2Q_9$), the rest particle population $f_0$ carries significant weight ($w_0 = 4/9$) and is essential for fluid compressibility and Galilean invariance.

2. **Tensor-Product Speed Grid:**
   The velocities in 2D form a Cartesian product:
   $$(c_x, c_y) \in \{\pm 1, \pm 3\} \times \{\pm 1, \pm 3\}$$
   This yields 16 discrete velocities (4 speeds in $x$ $\times$ 4 speeds in $y$). This is fundamentally different from the 9-velocity stencil of $D_2Q_9$, which has isotropic diagonal vectors $(\pm 1, \pm 1)$ and coordinate-axis vectors $(\pm 1, 0), (0, \pm 1)$.

3. **Amplitude Encoding Nature:**
   A single-particle state $|\psi\rangle = |x, y, v_x, v_y, vd_x, vd_y, 0, 0, 0\rangle$ represents a localized deterministic packet. A general fluid distribution is encoded as a coherent quantum superposition:
   $$|\Psi\rangle = \sum_{x, y, \mathbf{v}} \sqrt{f(\mathbf{x}, \mathbf{v})} |\mathbf{x}\rangle |\mathbf{v}\rangle |0\rangle_{\text{ancilla}}$$
   Interfering or measuring this state without projective destruction is the core challenge for multi-step collision algorithms.

---

## 7. Experimental Validation Summary

The encoding and basis-state decoding formulations were experimentally validated via [`tests/research/test_ms_register_encoding.py`](file:///home/aswa/Research/qlbm/tests/research/test_ms_register_encoding.py):
- **Test suite:** 7 passed in 3.02s.
- **2D No-Obstacle (8x8):** 15 qubits verified; contiguous partitioning across $[0, \dots, 14]$ confirmed.
- **2D Specular (8x8):** 16 qubits verified; $a_o$ expanded to 2 qubits confirmed.
- **3D Specular (8x8x8):** 25 qubits verified; $a_v=3, a_o=3, a_c=4, g=9, v=3, vd=3$ confirmed.
- **Basis State Decoding:** Bitmask decoding verified on multiple test points $((0,0), (1,2), (5,3), (7,7))$ with pure statevector outputs ($|c_k| = 1.0$).
- **Little-Endian Verification:** Explicit bit inspection confirmed LSB at lower qubit indices.
- **Existing MS Tests:** All 40 existing unit/e2e tests in `test/unit/lattice/ms_lattice_test.py` and `test/e2e/test_ms_e2e.py` passed cleanly.
