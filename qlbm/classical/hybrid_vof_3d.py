"""3D Hybrid Volume-of-Fluid (VOF) Free-Surface Lattice Boltzmann Solver.

Simulates 3D gravity-driven multiphase flows, liquid column collapse, and vertical
falling water (waterfall into a lower collection basin) with solid walls, an
openable vertical gate, and exact free-fall kinetics under g = (0, 0, -9.81) m/s^2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from qlbm.classical.d3q19 import (
    C,
    CS_SQ,
    WEIGHTS,
    bgk_collision,
    compute_equilibrium,
    compute_guo_force,
    compute_macroscopic,
    stream_and_bounce,
)
from qlbm.classical.vof_3d import (
    CellType,
    classify_cells_3d,
    compute_directional_mass_fluxes_3d,
    compute_normals_3d,
    reconstruct_gas_boundaries_3d,
    update_cell_states_3d,
)


class VOF3DHydroSolver:
    r"""3D Free-Surface Volume-of-Fluid (VOF) Lattice Boltzmann Solver.

    Integrates standard 3D D3Q19 lattice kinetics with a conservative,
    sharp-interface geometric Volume-of-Fluid method capable of simulating
    vertical water detachment, gravity-driven waterfalls, and basin impacts.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions along x, y, and z axes.
    tau : float, optional
        BGK relaxation time :math:`\tau > 0.5`, default 0.8.
    gravity : tuple or float, optional
        Gravitational acceleration vector :math:`(g_x, g_y, g_z)`. If float,
        acts purely downwards along -z: :math:`(0, 0, -g)`. Default (0, 0, -0.005).
    solid_mask : np.ndarray, optional
        Boolean array of shape `(nz, ny, nx)` marking solid boundaries.
    p_gas : float, optional
        External atmospheric gas pressure, default 1.0.
    dt : float, optional
        Lattice time step, default 1.0.
    conversion_hysteresis : float, optional
        Cell transition buffer, default 0.02.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        tau: float = 0.8,
        gravity: Optional[Union[float, Tuple[float, float, float]]] = None,
        solid_mask: Optional[np.ndarray] = None,
        p_gas: float = 1.0,
        dt: float = 1.0,
        conversion_hysteresis: float = 0.02,
    ) -> None:
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({nx}, {ny}, {nz}).")
        if tau <= 0.5:
            raise ValueError(f"Relaxation time must satisfy tau > 0.5, got {tau}.")

        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.tau = float(tau)
        self.dt = float(dt)
        self.p_gas = float(p_gas)
        self.conversion_hysteresis = float(conversion_hysteresis)

        # Parse 3D gravity vector
        if gravity is None:
            self.gravity = np.array([0.0, 0.0, -0.005], dtype=np.float64)
        elif isinstance(gravity, (int, float)):
            self.gravity = np.array([0.0, 0.0, -float(gravity)], dtype=np.float64)
        else:
            self.gravity = np.array(gravity, dtype=np.float64)

        # Initialize solid mask
        if solid_mask is not None:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
        else:
            self.solid_mask = np.zeros((nz, ny, nx), dtype=bool)

        # Dynamic gate management
        self.gate_mask = np.zeros((nz, ny, nx), dtype=bool)
        self.is_gate_open: bool = False

        # State fields
        self.fill_level = np.zeros((nz, ny, nx), dtype=np.float64)
        self.mass = np.zeros((nz, ny, nx), dtype=np.float64)
        self.rho = np.ones((nz, ny, nx), dtype=np.float64)
        self.u = np.zeros((3, nz, ny, nx), dtype=np.float64)
        self.cell_types = np.full((nz, ny, nx), CellType.GAS, dtype=np.int8)
        self.normals = np.zeros((3, nz, ny, nx), dtype=np.float64)

        # Distribution functions: 19 populations
        self.f = np.zeros((19, nz, ny, nx), dtype=np.float64)

        self.initial_mass: float = 0.0
        self.time_step: int = 0

    @property
    def total_mass(self) -> float:
        """Total fluid mass inside the computational domain."""
        return float(np.sum(self.mass[~self.solid_mask]))

    @property
    def fluid_mask(self) -> np.ndarray:
        """Boolean mask of active fluid cells (liquid and interface)."""
        return (
            (self.cell_types == CellType.LIQUID)
            | (self.cell_types == CellType.INTERFACE)
        ) & (~self.solid_mask)

    @property
    def pressure(self) -> np.ndarray:
        """Macroscopic pressure field: P = cs^2 * rho."""
        return CS_SQ * self.rho

    @property
    def speed(self) -> np.ndarray:
        """Velocity magnitude field: |u|."""
        return np.sqrt(self.u[0]**2 + self.u[1]**2 + self.u[2]**2)

    def initialize_waterfall_dam_break(
        self,
        shelf_height: int,
        shelf_length: int,
        gate_x: int,
        water_height: int,
        water_width: Optional[int] = None,
        water_depth: Optional[int] = None,
        rho_liquid: float = 1.0,
    ) -> None:
        r"""Initialize the 3D reference geometry with elevated reservoir, gate, drop, and basin.

        Parameters
        ----------
        shelf_height : int
            Height of the elevated shelf floor :math:`z_{\text{shelf}}`.
        shelf_length : int
            Extent of the elevated shelf from :math:`x = 0` to :math:`x_{\text{shelf}}`.
        gate_x : int
            X-coordinate of the vertical openable gate.
        water_height : int
            Height of the liquid column above the shelf floor.
        water_width : int, optional
            Width of the liquid column along x (from x=1 to gate_x-1 if None).
        water_depth : int, optional
            Depth of the liquid column along y (from y=1 to ny-2 if None).
        rho_liquid : float, optional
            Initial liquid density, default 1.0.
        """
        nx, ny, nz = self.nx, self.ny, self.nz

        # 1. Enclosing solid boundaries
        self.solid_mask[:, :, :] = False
        self.solid_mask[:, :, 0] = True       # Left wall (x = 0)
        self.solid_mask[:, :, -1] = True      # Right wall (x = nx - 1)
        self.solid_mask[:, 0, :] = True       # Back wall (y = 0)
        self.solid_mask[:, -1, :] = True      # Front wall (y = ny - 1)
        self.solid_mask[0, :, :] = True       # Basin floor (z = 0)
        self.solid_mask[-1, :, :] = True      # Ceiling (z = nz - 1)

        # 2. Elevated shelf floor
        # Solid floor from x=0 to x=shelf_length at z=shelf_height
        self.solid_mask[shelf_height, :, :shelf_length] = True
        # Below the shelf (solid ground under reservoir)
        self.solid_mask[:shelf_height, :, :shelf_length] = True

        # 3. Openable gate at gate_x
        # Gate spans from z=shelf_height+1 up to z=shelf_height+water_height+1
        self.gate_mask[:, :, :] = False
        self.gate_mask[
            (shelf_height + 1):(shelf_height + water_height + 2),
            1:(ny - 1),
            gate_x,
        ] = True
        self.solid_mask |= self.gate_mask
        self.is_gate_open = False

        # 4. Fill liquid column behind gate on the shelf
        if water_width is None:
            x_end = gate_x
        else:
            x_end = min(gate_x, 1 + water_width)

        if water_depth is None:
            y_start, y_end = 1, ny - 1
        else:
            y_start = max(1, (ny - water_depth) // 2)
            y_end = min(ny - 1, y_start + water_depth)

        z_start = shelf_height + 1
        z_end = min(nz - 1, shelf_height + 1 + water_height)

        self.fill_level[:, :, :] = 0.0
        self.fill_level[z_start:z_end, y_start:y_end, 1:x_end] = 1.0

        # Classify cells
        self.cell_types = classify_cells_3d(self.fill_level, self.solid_mask)

        # Base density & velocity
        self.rho[:, :, :] = 1.0
        self.u[:, :, :, :] = 0.0

        # Mass = C * rho
        self.mass = self.fill_level * self.rho

        # Initialize distribution functions to equilibrium in active fluid cells
        self.f = compute_equilibrium(self.rho, self.u)
        inactive = (self.cell_types == CellType.GAS) | (self.cell_types == CellType.SOLID)
        self.f[:, inactive] = 0.0

        self.normals, _ = compute_normals_3d(self.fill_level, self.solid_mask)
        self.initial_mass = self.total_mass
        self.time_step = 0

    def open_gate(self) -> None:
        """Remove the solid vertical gate to release the elevated liquid column."""
        if not self.is_gate_open:
            self.solid_mask &= ~self.gate_mask
            # Gate cells convert to empty gas in the open domain
            gate_cells = self.gate_mask
            self.cell_types[gate_cells] = CellType.GAS
            self.fill_level[gate_cells] = 0.0
            self.mass[gate_cells] = 0.0
            self.f[:, gate_cells] = 0.0
            self.u[:, gate_cells] = 0.0

            # Demote any liquid cells bordering the newly exposed open air (gas) to interface
            for i in range(1, 19):
                cx, cy, cz = C[i]
                gas_nbr = (
                    np.roll(
                        np.roll(np.roll(self.cell_types, shift=-cz, axis=0), shift=-cy, axis=1),
                        shift=-cx,
                        axis=2,
                    )
                    == CellType.GAS
                )
                touch_gas = (self.cell_types == CellType.LIQUID) & gas_nbr
                self.cell_types[touch_gas] = CellType.INTERFACE

            self.is_gate_open = True

    def center_of_mass(self) -> Tuple[float, float, float]:
        """Compute the 3D center of mass of the liquid water (x_cm, y_cm, z_cm)."""
        active = self.fluid_mask & (self.mass > 1e-8)
        if not np.any(active):
            return (0.0, 0.0, 0.0)

        weights = self.mass[active]
        z_idx, y_idx, x_idx = np.where(active)

        x_cm = float(np.average(x_idx, weights=weights))
        y_cm = float(np.average(y_idx, weights=weights))
        z_cm = float(np.average(z_idx, weights=weights))

        return (x_cm, y_cm, z_cm)

    def front_position(self) -> float:
        """Compute the farthest forward advancement of the liquid surge front along x."""
        active = self.fluid_mask & (self.fill_level > 0.05)
        if not np.any(active):
            return 0.0
        return float(np.max(np.where(active)[2]))

    def step(self, force_classical: bool = True) -> None:
        """Execute one complete time step of the 3D free-surface LBM solver."""
        active = self.fluid_mask

        # 1. Classical BGK collision with Guo 3D gravity forcing
        feq = compute_equilibrium(self.rho, self.u)
        f_post_collide = np.copy(self.f)

        # Collision only in fluid cells
        if np.any(active):
            body_force = self.rho * self.gravity[:, None, None, None]
            f_post_collide[:, active] = bgk_collision(
                self.f[:, active],
                feq[:, active],
                self.tau,
                force=body_force[:, active],
                u=self.u[:, active],
                dt=self.dt,
            )

        f_post_collide[:, ~active] = 0.0
        f_post_collide = np.maximum(0.0, f_post_collide)

        # 2. 3D Spatial streaming & solid bounce-back
        f_post_stream = stream_and_bounce(f_post_collide, self.solid_mask)

        # 3. Directional mass fluxes across interface links
        delta_m = compute_directional_mass_fluxes_3d(
            f_post_collide, self.fill_level, self.cell_types, solid_mask=self.solid_mask, dt=self.dt
        )

        # 4. Free-surface gas pressure boundary reconstruction
        f_reconstructed = reconstruct_gas_boundaries_3d(
            f_post_stream,
            f_post_collide,
            self.cell_types,
            rho_gas=self.p_gas,
            u=self.u,
        )

        # 5. Cell state transitions (mass update, fill fraction, conversions)
        old_types = np.copy(self.cell_types)
        new_mass, new_fill, new_types = update_cell_states_3d(
            self.mass,
            self.rho,
            self.cell_types,
            self.solid_mask,
            delta_m,
            conversion_hysteresis=self.conversion_hysteresis,
        )

        # Initialize newly converted interface cells from gas with neighbor equilibrium
        new_iface = (old_types == CellType.GAS) & (new_types == CellType.INTERFACE)
        if np.any(new_iface):
            for z, y, x in zip(*np.where(new_iface)):
                u_sum = np.zeros(3, dtype=np.float64)
                cnt = 0
                for ci in range(1, 19):
                    cx, cy, cz = C[ci]
                    nz_c, ny_c, nx_c = z + cz, y + cy, x + cx
                    if 0 <= nz_c < self.nz and 0 <= ny_c < self.ny and 0 <= nx_c < self.nx:
                        if old_types[nz_c, ny_c, nx_c] in (CellType.LIQUID, CellType.INTERFACE):
                            u_sum += self.u[:, nz_c, ny_c, nx_c]
                            cnt += 1
                u_avg = u_sum / max(cnt, 1)
                u_mag = float(np.linalg.norm(u_avg))
                if u_mag > 0.15:
                    u_avg = u_avg * (0.15 / u_mag)
                f_reconstructed[:, z, y, x] = compute_equilibrium(1.0, u_avg)

        self.mass = new_mass
        self.fill_level = new_fill
        self.cell_types = new_types

        # 6. Update distribution functions & macroscopic moments
        self.f = f_reconstructed
        self.f[:, self.cell_types == CellType.GAS] = 0.0
        self.f[:, self.solid_mask] = 0.0

        body_force = self.rho * self.gravity[:, None, None, None]
        new_rho, new_u = compute_macroscopic(self.f, force=body_force, dt=self.dt)

        self.rho = new_rho
        self.u = new_u
        self.u[:, self.solid_mask] = 0.0
        self.u[:, self.cell_types == CellType.GAS] = 0.0

        # Update interface normals
        self.normals, _ = compute_normals_3d(self.fill_level, self.solid_mask)
        self.time_step += 1

    def step_hybrid(self, hybrid_stepper_3d: Any) -> None:
        """Execute one complete hybrid time step delegating 3D streaming to quantum circuits."""
        hybrid_stepper_3d.step()
