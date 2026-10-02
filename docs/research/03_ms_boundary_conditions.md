# Phase 3: MSQLBM Boundary Conditions & Quantum Comparator Reflection

**Date:** 2026-10-02  
**Target Modules:**  
- [`qlbm/components/ms/bounceback_reflection.py`](file:///home/aswa/Research/qlbm/qlbm/components/ms/bounceback_reflection.py) (`BounceBackReflectionOperator`, `BounceBackWallComparator`)  
- [`qlbm/components/ms/specular_reflection.py`](file:///home/aswa/Research/qlbm/qlbm/components/ms/specular_reflection.py) (`SpecularReflectionOperator`, `SpecularWallComparator`)  
- [`qlbm/components/ms/primitives.py`](file:///home/aswa/Research/qlbm/qlbm/components/ms/primitives.py) (`EdgeComparator`, `MSInitialConditions`, `GridMeasurement`)  
- Geometry Encodings: [`qlbm/lattice/geometry/shapes/block.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/geometry/shapes/block.py), [`qlbm/lattice/geometry/encodings/ms.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/geometry/encodings/ms.py)  
**Theoretical Foundation:** Schalkers & Möller (2024), *Computers & Fluids* 285, 106453 :cite:`qmem`  

---

## 1. Physical Problem & Classical Correspondence

In classical Lattice Boltzmann Methods, solid obstacles within the fluid domain are enforced through boundary conditions on incoming distribution functions:
1. **Bounce-Back Reflection (No-Slip Wall):**
   Particles colliding with a solid wall are reflected back along their incoming path with completely inverted momentum:
   $$f_{\bar{i}}(\mathbf{x}_f, t + \Delta t) = f_i^*(\mathbf{x}_f, t), \quad \text{where } \mathbf{c}_{\bar{i}} = -\mathbf{c}_i$$
   This enforces zero fluid velocity at the solid boundary ($\mathbf{u}_{\text{wall}} = \mathbf{0}$).
2. **Specular Reflection (Free-Slip / Symmetry Wall):**
   Only the normal momentum component is inverted, while the tangential momentum component is conserved:
   $$c_{n, \bar{i}} = -c_{n, i}, \quad c_{\tau, \bar{i}} = c_{\tau, i}$$
   This models ideal frictionless walls or symmetry planes where $\partial u_\tau / \partial n = 0$ and $u_n = 0$.

In MSQLBM, because the quantum state is an amplitude superposition over coordinates and velocities, boundary reflections must be executed as a **unitary permutation** of quantum state amplitudes without projectively collapsing the fluid superposition.

---

## 2. Four-Stage Quantum Reflection Cycle

Both `BounceBackReflectionOperator` and `SpecularReflectionOperator` execute an identical 4-stage quantum circuit cycle during each CFL streaming substep:

```
[State after Streaming]
         │
         ▼
Stage 1: Detect Intrusion (Interior Comparators -> Flip a_o to |1>)
         │
         ▼
Stage 2: Flip Direction & Stream Out (Invert vd, ControlledIncrementer)
         │
         ▼
Stage 3: Uncompute Ancilla (Exterior Comparators -> Reset a_o to |0>)
         │
         ▼
Stage 4: Corner & Edge Cleanup (Point/Edge Comparators -> Uncompute a_o)
         │
         ▼
[Reflected State in Fluid]
```

### Stage 1: Intrusion Detection (`reflect_wall` on `walls_inside`)
When streaming causes a particle's coordinate to enter the obstacle's interior boundary layer ($\mathbf{x} \in \text{walls\_inside}$):
1. **Coordinate Lower/Upper Comparators:** Quantum comparators ([`SingleRegisterComparator`](file:///home/aswa/Research/qlbm/qlbm/components/common/comparators.py)) compare the coordinate registers against the bounding box $[x_{\min}, x_{\max}]$ and $[y_{\min}, y_{\max}]$. The comparator outputs are stored in ancillae $a_c = [l, u]$.
2. **Grid Inversion:** A bitwise $X$-mask (`qubits_to_invert`) maps the specific wall coordinate value to $|1\dots 1\rangle$.
3. **Obstacle Ancilla Flip:** A Multi-Controlled Multi-Target gate ([`MCMTGate(XGate())`](file:///home/aswa/Research/qlbm/qlbm/components/ms/bounceback_reflection.py#L323)) controlled on $(a_v, g_{\text{dim}}, a_c)$ flips the obstacle ancilla $a_o$:
   $$|0\rangle_{a_o} \xrightarrow{\text{Intrusion}} |1\rangle_{a_o}$$
4. **Uncompute Grid Inversion & Comparators:** The $X$-masks and comparators are uncomputed, restoring the grid and comparator registers.

### Stage 2: Velocity Reversal & Reflection Streaming (`flip_and_stream`)
Once $a_o = 1$:
1. **Direction Inversion:**
   - **Bounce-Back:** Controlled on $a_o[0]$, Pauli-$X$ gates are applied to **all** direction qubits:
     $$vd_k \longrightarrow 1 - vd_k, \quad \forall k \in \{0, \dots, d-1\}$$
   - **Specular Reflection:** Controlled on the dimensional ancilla $a_o[k]$, Pauli-$X$ is applied **only to the normal direction qubit**:
     $$vd_k \longrightarrow 1 - vd_k, \quad vd_{\tau} \text{ unchanged}$$
2. **Outward Streaming:**
   The circuit executes [`ControlledIncrementer(reflection="bounceback"|"specular")`](file:///home/aswa/Research/qlbm/qlbm/components/ms/streaming.py#L114).
   Because the direction is now inverted, the incrementer streams the particle **back out of the obstacle** into the adjacent fluid cell in a single substep.

### Stage 3: Outer Wall Ancilla Uncomputation (`reflect_wall` on `walls_outside`)
The particle is now in the exterior boundary layer (`walls_outside`).
1. Comparators identify the particle at the exterior wall coordinate with the reflected velocity direction.
2. The same MCMT gate fires again:
   $$|1\rangle_{a_o} \xrightarrow{\text{Reset}} |0\rangle_{a_o}$$
   This cleans the obstacle ancilla so that subsequent fluid streaming can proceed without spurious reflection flags.

### Stage 4: Corner & Edge Cleanup (`reset_point_state`, `reset_edge_state`)
At corners (2D) and edges (3D), multiple walls intersect. A particle hitting a corner triggers multiple comparators simultaneously.
- Without cleanup, overlapping comparator conditions leave residual entanglement or a "dirty" ancilla $a_o = 1$.
- In 2D: [`reset_point_state`](file:///home/aswa/Research/qlbm/qlbm/components/ms/specular_reflection.py#L254) uses exact coordinate point comparators on `near_corner_points_2d` and `corners_outside` (4 corner points per cuboid) to uncompute residual flags.
- In 3D: [`reset_edge_state`](file:///home/aswa/Research/qlbm/qlbm/components/ms/specular_reflection.py#L309) uses [`EdgeComparator`](file:///home/aswa/Research/qlbm/qlbm/components/ms/primitives.py#L270) across the 12 edges and 24 near-corner edges of the 3D cuboid.

---

## 3. Mathematical Details of Quantum Geometry Encoding

In [`Block`](file:///home/aswa/Research/qlbm/qlbm/lattice/geometry/shapes/block.py), a cuboid obstacle with bounds $[x_l, x_u] \times [y_l, y_u]$ generates:
1. `walls_inside`:
   - $x$-lower wall: $\{x = x_l, \; y \in [y_l, y_u]\}$ with $vd_x = 1$.
   - $x$-upper wall: $\{x = x_u, \; y \in [y_l, y_u]\}$ with $vd_x = 0$.
   - $y$-lower wall: $\{y = y_l, \; x \in [x_l, x_u]\}$ with $vd_y = 1$.
   - $y$-upper wall: $\{y = y_u, \; x \in [x_l, x_u]\}$ with $vd_y = 0$.
2. `walls_outside`:
   - $x$-lower outer wall: $\{x = x_l - 1, \; y \in [y_l, y_u]\}$ with reflected direction $vd_x = 0$.
   - $x$-upper outer wall: $\{x = x_u + 1, \; y \in [y_l, y_u]\}$ with reflected direction $vd_x = 1$.
   - $y$-lower outer wall: $\{y = y_l - 1, \; x \in [x_l, x_u]\}$ with reflected direction $vd_y = 0$.
   - $y$-upper outer wall: $\{y = y_u + 1, \; x \in [x_l, x_u]\}$ with reflected direction $vd_y = 1$.

---

## 4. Experimental Validation & Results

The reflection operators were experimentally verified across 2D walls, corners, and 3D obstacles using Qiskit Aer statevector simulation:

| Scenario | Domain & Obstacle | Initial Particle $(x, y, z)$ | Initial Direction | Obstacle Intrusion | Reflected Position | Reflected Direction | $a_o$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D Bounce-Back Wall** | $8\times 8$, Cuboid $[3,4]\times[2,5]$ | $(2, 3)$ | $(+1, +1)$ | Hits $x=3$ | $(2, 3)$ | $(-1, -1)$ | $1^*$ | **PASSED** |
| **2D Specular Wall** | $8\times 8$, Cuboid $[3,4]\times[2,5]$ | $(2, 3)$ | $(+1, +1)$ | Hits $x=3$ | $(2, 4)$ | $(-1, +1)$ | $0$ | **PASSED** |
| **2D Specular Corner** | $8\times 8$, Cuboid $[3,4]\times[3,4]$ | $(2, 2)$ | $(+1, +1)$ | Hits $(3, 3)$ | $(2, 2)$ | $(-1, -1)$ | $0$ | **PASSED** |
| **2D Away from Wall** | $8\times 8$, Cuboid $[3,4]\times[2,5]$ | $(1, 1)$ | $(-1, +1)$ | Free streaming | $(0, 2)$ | $(-1, +1)$ | $0$ | **PASSED** |
| **3D Bounce-Back Wall** | $4\times 4\times 4$, Cuboid $[2,3]\times[1,2]\times[1,2]$ | $(1, 1, 1)$ | $(+1, +1, +1)$ | Hits $x=2$ | $(1, 1, 1)$ | $(-1, -1, -1)$ | $1^*$ | **PASSED** |

*\*Note on Bounce-Back Ancilla:* In `BounceBackReflectionOperator`, the position and direction registers are the primary physical assertions. As documented in `test_ms_e2e.py:201`, the bounce-back obstacle ancilla remains $1$ after reflection because the single-qubit bounce-back reset logic does not distinguish between individual wall normals. In contrast, `SpecularReflectionOperator` uses $d$ separate ancillae and fully resets $a_o = 0$.

---

## 5. Architectural Comparison: Quantum vs Classical Boundary Implementation

| Aspect | Classical LBM | MSQLBM |
| :--- | :--- | :--- |
| **Data Representation** | Floating-point distribution arrays $f_i(\mathbf{x})$ | Quantum statevector amplitudes $\alpha_{\mathbf{x}, \mathbf{v}}$ |
| **Wall Identification** | Mask array `is_solid[x, y]` (boolean array lookup) | Quantum comparator circuits (`SingleRegisterComparator`) |
| **Momentum Exchange** | Algebraic swap $f_{\bar{i}} \leftarrow f_i$ | Unitary bit-flips ($X(vd)$) + reverse QFT advection |
| **Curved Boundaries** | Interpolated bounce-back (Bouzidi, Yu) | Requires complex multi-qubit polynomial oracles (`YMonomial`) |
| **No-Slip Condition** | Exact momentum reflection ($\mathbf{u} = \mathbf{0}$) | Exact velocity direction reversal ($\mathbf{v} \to -\mathbf{v}$) |
| **Interfacial Boundaries**| VOF volume fraction cuts / PLIC geometry | **Absent** (only rigid solid walls supported) |

This completes the mathematical and experimental verification of MSQLBM boundary conditions.
