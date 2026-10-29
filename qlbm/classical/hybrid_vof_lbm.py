"""Hybrid Volume-of-Fluid (VOF) Free-Surface Lattice Boltzmann Solver for D2Q9.

This module couples the classical D2Q9 BGK hydrodynamic solver with the
geometric Volume-of-Fluid (VOF) free-surface formulation (Körner et al. 2005,
Bogner et al. 2015).

Key Capabilities:
1. Clear architectural separation: Hydrodynamic Kinetic State vs Geometric VOF State.
2. Free-surface LBM boundary reconstruction from ambient gas pressure and surface tension.
3. Directional, pairwise anti-symmetric mass flux advection: Delta M_i(x) = -Delta M_bar_i(x+c_i).
4. Strict mass-conservative cell state transitions with topological neighborhood completeness.
5. 2nd-order Youngs normal vector estimation and Scardovelli-Zaleski PLIC reconstruction.
6. Hydrodynamic coupling with gravity (Guo forcing) and Young-Laplace capillary pressure.
"""

from typing import Optional, Tuple, Union

import numpy as np

from qlbm.classical.d2q9 import (
    C,
    CS,
    CS_SQ,
    OPPOSITE,
    WEIGHTS,
    bgk_collision,
    compute_equilibrium,
    compute_guo_force,
    compute_macroscopic,
    stream_and_bounce,
)
from qlbm.classical.vof import (
    CellType,
    classify_cells,
    compute_normal_youngs,
    plic_forward_area,
    plic_inverse_alpha,
)


def enforce_neighborhood_consistency(
    cell_types: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Enforce the topological rule that LIQUID cells must never touch GAS cells directly.

    Any LIQUID cell adjacent (in the D2Q9 8-neighborhood) to a GAS cell is converted
    into an INTERFACE cell to ensure an enclosing protective interface boundary layer.
    """
    types = np.copy(cell_types)
    is_gas = types == CellType.GAS
    is_liquid = types == CellType.LIQUID

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            gas_nbr = np.roll(np.roll(is_gas, shift=-dy, axis=0), shift=-dx, axis=1)
            touch_gas = is_liquid & gas_nbr
            types[touch_gas] = CellType.INTERFACE

    if solid_mask is not None:
        types[solid_mask] = CellType.SOLID

    return types


_NEIGHBOURS = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
)

FREE_SURFACE_SCHEMES = ("korner", "legacy")


def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    r"""Periodic lookup of the neighbour value :math:`a(x + d_x, y + d_y)` on the last two axes."""
    return np.roll(np.roll(a, -dy, axis=-2), -dx, axis=-1)


class VOFHydroSolver:
    r"""Two-dimensional D2Q9 Hybrid Volume-of-Fluid Free-Surface LBM Solver.

    Parameters
    ----------
    nx : int
        Number of grid nodes in the x-direction.
    ny : int
        Number of grid nodes in the y-direction.
    tau : float, optional
        BGK relaxation time :math:`\tau > 0.5`, default 0.8.
    gravity : float or tuple of (float, float), optional
        Gravitational acceleration vector :math:`(g_x, g_y)` or downward scalar :math:`g`.
    surface_tension : float, optional
        Macroscopic surface tension coefficient :math:`\sigma \ge 0`, default 0.0.
    gas_pressure : float, optional
        Ambient external gas pressure :math:`p_{\text{gas}}`, default 0.0.
    solid_mask : np.ndarray of bool, optional
        Boolean array of shape `(ny, nx)` where `True` indicates a solid obstacle.
    eps : float, optional
        Volume fraction tolerance for cell state classification, default :math:`10^{-5}`.
    dt : float, optional
        Time step size in lattice units, default 1.0.
    free_surface_scheme : {'korner', 'legacy'}, optional
        Mass-exchange and cell-conversion algorithm, default ``'korner'``.

        * ``'korner'``: Körner et al. (2005) / Thürey (2007). No mass is exchanged
          across faces touching GAS or SOLID cells; LIQUID faces have weight 1 and
          INTERFACE-INTERFACE faces weight :math:`(C + C_{nbr})/2`. Interface cells
          fill when :math:`m > (1+\kappa)\rho` and empty when :math:`m < -\kappa\rho`,
          and the excess mass is redistributed conservatively to neighbouring
          interface cells.
        * ``'legacy'``: the Phase 12-15 algorithm (fill-area-weighted exchange with
          all non-solid neighbours, threshold classification on :math:`C`). Kept
          for exact reproducibility of earlier results; it lets liquid mass leak
          into GAS cells, which smears the interface (see Phase 16 diary entry).
    conversion_hysteresis : float, optional
        Hysteresis :math:`\kappa` for interface filling/emptying in the
        ``'korner'`` scheme, default :math:`10^{-3}`.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        tau: float = 0.8,
        gravity: Optional[Union[float, Tuple[float, float]]] = None,
        surface_tension: float = 0.0,
        gas_pressure: float = 0.0,
        solid_mask: Optional[np.ndarray] = None,
        eps: float = 1e-5,
        dt: float = 1.0,
        free_surface_scheme: str = "korner",
        conversion_hysteresis: float = 1e-3,
    ) -> None:
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid dimensions must be positive, got nx={nx}, ny={ny}.")
        if tau <= 0.5:
            raise ValueError(f"Relaxation time must satisfy tau > 0.5, got {tau}.")
        if free_surface_scheme not in FREE_SURFACE_SCHEMES:
            raise ValueError(
                f"free_surface_scheme must be one of {FREE_SURFACE_SCHEMES}, "
                f"got '{free_surface_scheme}'."
            )
        if conversion_hysteresis < 0.0:
            raise ValueError("conversion_hysteresis must be non-negative.")
        self.free_surface_scheme = free_surface_scheme
        self.kappa = float(conversion_hysteresis)

        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        self.dt = float(dt)
        self.sigma = float(surface_tension)
        self.p_gas = float(gas_pressure)
        self.eps = float(eps)

        # Parse gravity
        if gravity is not None:
            if isinstance(gravity, (int, float)):
                self.gravity: Tuple[float, float] = (0.0, -float(gravity))
            else:
                self.gravity = (float(gravity[0]), float(gravity[1]))
        else:
            self.gravity = (0.0, 0.0)

        # Solid mask
        if solid_mask is not None:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
            if self.solid_mask.shape != (self.ny, self.nx):
                raise ValueError("solid_mask shape does not match grid dimensions.")
        else:
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)

        # -------------------------------------------------------------
        # 1. HYDRODYNAMIC KINETIC STATE
        # -------------------------------------------------------------
        self.f: np.ndarray = np.zeros((9, self.ny, self.nx), dtype=np.float64)
        self.rho: np.ndarray = np.ones((self.ny, self.nx), dtype=np.float64)
        self.u: np.ndarray = np.zeros((2, self.ny, self.nx), dtype=np.float64)
        self.force: np.ndarray = np.zeros((2, self.ny, self.nx), dtype=np.float64)

        # -------------------------------------------------------------
        # 2. GEOMETRIC VOF STATE
        # -------------------------------------------------------------
        self.mass: np.ndarray = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.fill_level: np.ndarray = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.cell_types: np.ndarray = np.full(
            (self.ny, self.nx), CellType.GAS, dtype=np.int32
        )
        self.nx_field: np.ndarray = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.ny_field: np.ndarray = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.curvature: np.ndarray = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.alphas: np.ndarray = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Mass tracking audit log
        self.initial_mass: float = 0.0
        self.initial_volume: float = 0.0
        self.time_step: int = 0

        if np.any(self.solid_mask):
            self.cell_types[self.solid_mask] = CellType.SOLID

    # -----------------------------------------------------------------
    # State Inspection & Diagnostic Properties
    # -----------------------------------------------------------------

    @property
    def fluid_mask(self) -> np.ndarray:
        r"""Active fluid cells: LIQUID or INTERFACE."""
        return (self.cell_types == CellType.LIQUID) | (
            self.cell_types == CellType.INTERFACE
        )

    @property
    def total_mass(self) -> float:
        r"""Total liquid mass inside the domain :math:`\sum M(\mathbf{x})`."""
        return float(np.sum(self.mass[~self.solid_mask]))

    @property
    def total_volume(self) -> float:
        r"""Total liquid volume inside the domain :math:`\sum C(\mathbf{x}) \Delta x^2`."""
        return float(np.sum(self.fill_level[~self.solid_mask]))

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        r"""Liquid center of mass :math:`(x_{\text{cm}}, y_{\text{cm}})`."""
        m_tot = self.total_mass
        if m_tot < 1e-12:
            return (0.0, 0.0)
        y_grid, x_grid = np.mgrid[0 : self.ny, 0 : self.nx]
        x_cm = float(np.sum(x_grid * self.mass) / m_tot)
        y_cm = float(np.sum(y_grid * self.mass) / m_tot)
        return (x_cm, y_cm)

    @property
    def kinetic_energy(self) -> float:
        r"""Total kinetic energy of active fluid cells :math:`\frac{1}{2}\sum \rho \|\mathbf{u}\|^2`."""
        active = self.fluid_mask
        u_sq = self.u[0, active] ** 2 + self.u[1, active] ** 2
        return 0.5 * float(np.sum(self.rho[active] * u_sq))

    # -----------------------------------------------------------------
    # Initialization Routines
    # -----------------------------------------------------------------

    def initialize_from_fill_level(
        self,
        fill_level: np.ndarray,
        rho_liquid: float = 1.0,
        u_init: Optional[np.ndarray] = None,
    ) -> None:
        r"""Initialize solver state from a prescribed volume fraction field :math:`C(\mathbf{x}) \in [0, 1]`."""
        C_init = np.clip(np.asarray(fill_level, dtype=np.float64), 0.0, 1.0)
        if self.solid_mask is not None:
            C_init[self.solid_mask] = 0.0

        self.fill_level = C_init
        raw_types = classify_cells(self.fill_level, self.solid_mask, eps=self.eps)
        self.cell_types = enforce_neighborhood_consistency(
            raw_types, self.solid_mask
        )

        # Base density & velocity
        self.rho = np.full((self.ny, self.nx), rho_liquid, dtype=np.float64)
        if u_init is None:
            self.u = np.zeros((2, self.ny, self.nx), dtype=np.float64)
        else:
            self.u = np.copy(np.asarray(u_init, dtype=np.float64))

        # Liquid mass: M = C * rho * dx^2
        self.mass = self.fill_level * self.rho

        # Initialize distribution functions to equilibrium in active fluid cells
        self.f = compute_equilibrium(self.rho, self.u)
        # Clear distributions in pure gas and solid cells
        inactive = (self.cell_types == CellType.GAS) | (
            self.cell_types == CellType.SOLID
        )
        self.f[:, inactive] = 0.0

        self.reconstruct_geometry()

        self.initial_mass = self.total_mass
        self.initial_volume = self.total_volume
        self.time_step = 0

    def initialize_droplet(
        self,
        radius: float,
        xc: Optional[float] = None,
        yc: Optional[float] = None,
        rho_liquid: float = 1.0,
        interface_width: float = 1.0,
    ) -> None:
        r"""Initialize a stationary circular liquid droplet surrounded by gas."""
        if xc is None:
            xc = self.nx / 2.0
        if yc is None:
            yc = self.ny / 2.0

        y, x = np.mgrid[0 : self.ny, 0 : self.nx]
        dx = np.minimum(np.abs(x - xc), self.nx - np.abs(x - xc))
        dy = np.minimum(np.abs(y - yc), self.ny - np.abs(y - yc))
        r = np.sqrt(dx**2 + dy**2)

        # Smooth fill level
        C_init = np.clip(0.5 - (r - radius) / max(interface_width, 1e-4), 0.0, 1.0)
        self.initialize_from_fill_level(C_init, rho_liquid=rho_liquid)

    def initialize_dam_break(
        self,
        column_width: int,
        column_height: int,
        rho_liquid: float = 1.0,
        add_walls: bool = True,
    ) -> None:
        r"""Initialize a 2D liquid column in the lower-left corner behind a vertical dam gate.

        Parameters
        ----------
        column_width : int
            Width of the liquid column in lattice cells.
        column_height : int
            Height of the liquid column in lattice cells.
        rho_liquid : float, optional
            Liquid density, default 1.0.
        add_walls : bool, optional
            Whether to enforce solid no-slip walls along the floor (y=0) and
            side boundaries (x=0, x=nx-1) if no solid_mask was supplied.
        """
        if add_walls and not np.any(self.solid_mask):
            self.solid_mask[0, :] = True
            self.solid_mask[:, 0] = True
            self.solid_mask[:, -1] = True
            self.cell_types[self.solid_mask] = CellType.SOLID

        fill = np.zeros((self.ny, self.nx), dtype=np.float64)
        y_max = min(self.ny - 1, column_height + 1)
        x_max = min(self.nx - 1, column_width + 1)

        fill[1:y_max, 1:x_max] = 1.0
        self.initialize_from_fill_level(fill, rho_liquid=rho_liquid)

    # -----------------------------------------------------------------
    # Geometric Interface Reconstruction & Curvature
    # -----------------------------------------------------------------

    def reconstruct_geometry(self) -> None:
        r"""Evaluate Youngs normal vectors, interface curvature, and PLIC line constants."""
        self.nx_field, self.ny_field = compute_normal_youngs(
            self.fill_level, self.solid_mask, periodic=True
        )

        # Interface curvature kappa = div(n) = dnx/dx + dny/dy for outward unit normal
        # Using 2nd-order central differences on normal field
        dnx_dx = (
            np.roll(self.nx_field, -1, axis=1) - np.roll(self.nx_field, 1, axis=1)
        ) / 2.0
        dny_dy = (
            np.roll(self.ny_field, -1, axis=0) - np.roll(self.ny_field, 1, axis=0)
        ) / 2.0

        # Curvature is physical only at interface cells
        iface_mask = self.cell_types == CellType.INTERFACE
        self.curvature = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.curvature[iface_mask] = dnx_dx[iface_mask] + dny_dy[iface_mask]

        # PLIC line constants
        if np.any(iface_mask):
            c_iface = self.fill_level[iface_mask]
            nx_iface = self.nx_field[iface_mask]
            ny_iface = self.ny_field[iface_mask]
            self.alphas[iface_mask] = plic_inverse_alpha(nx_iface, ny_iface, c_iface)

    # -----------------------------------------------------------------
    # Free-Surface Boundary Condition & Streaming
    # -----------------------------------------------------------------

    def step(self) -> None:
        r"""Execute one complete time step of the hybrid VOF-LBM algorithm."""
        active = self.fluid_mask

        # 1. Update effective gravity body force in active fluid
        self._compute_body_force(active)

        # 2. Collision step in active fluid cells (BGK + Guo forcing)
        f_eq = compute_equilibrium(self.rho, self.u)
        f_post = bgk_collision(
            self.f,
            f_eq,
            self.tau,
            force=self.force,
            u=self.u,
            dt=self.dt,
        )
        f_post[:, ~active] = 0.0

        # 3. State-aware streaming and solid bounce-back using validated library function
        f_streamed = stream_and_bounce(f_post, solid_mask=self.solid_mask)

        # 4. Free-surface gas boundary reconstruction (Körner et al. 2005)
        self._reconstruct_gas_populations(f_streamed, f_post)

        # 5. Directional mass exchange (pairwise anti-symmetric)
        self._exchange_mass(f_post)

        # 6. Cell State Transitions & Population Initialization
        self._handle_state_transitions(f_streamed)

        # 7. Macroscopic Moments on Active Cells
        self._update_macroscopic()

        # 8. Reconstruct Geometry for next time step
        self.reconstruct_geometry()
        self.time_step += 1

    # -----------------------------------------------------------------
    # Step Building Blocks (shared with HybridVOFStep)
    # -----------------------------------------------------------------

    def _compute_body_force(self, active: np.ndarray) -> None:
        r"""Gravity body force :math:`\mathbf{F} = \rho \mathbf{g}` on active fluid cells."""
        self.force = np.zeros((2, self.ny, self.nx), dtype=np.float64)
        if self.gravity != (0.0, 0.0):
            gx, gy = self.gravity
            self.force[0, active] = self.rho[active] * gx
            self.force[1, active] = self.rho[active] * gy

    def _update_macroscopic(self) -> None:
        r"""Recompute :math:`\rho` and :math:`\mathbf{u}` on active cells from ``self.f``."""
        active = self.fluid_mask
        if np.any(active):
            rho_all, u_all = compute_macroscopic(
                self.f,
                force=self.force,
                dt=self.dt,
            )
            self.rho[active] = rho_all[active]
            self.u[:, active] = u_all[:, active]
            self.u[:, ~active] = 0.0

    def _reconstruct_gas_populations(
        self, f_streamed: np.ndarray, f_post: np.ndarray
    ) -> None:
        r"""Reconstruct populations streaming in from GAS neighbours (in place).

        For an interface cell whose neighbour in direction :math:`\mathbf{c}_i` is
        gas, :math:`\bar f_{\bar i} = f^{eq}_{\bar i}(\rho_G, \mathbf{u}) +
        f^{eq}_i(\rho_G, \mathbf{u}) - f^*_i` with
        :math:`\rho_G = p_G / c_s^2 + 1` including the Young-Laplace jump.
        """
        iface_mask = self.cell_types == CellType.INTERFACE
        for i in range(1, 9):
            opp = OPPOSITE[i]
            cx, cy = C[i]
            neighbor_is_gas = np.roll(
                np.roll(self.cell_types == CellType.GAS, shift=-cy, axis=0),
                shift=-cx,
                axis=1,
            )
            reconstruct_mask = iface_mask & neighbor_is_gas

            if np.any(reconstruct_mask):
                # Interface pressure incorporating Young-Laplace capillary jump
                p_iface = self.p_gas + self.sigma * self.curvature[reconstruct_mask]
                rho_iface = p_iface / CS_SQ + 1.0  # reference density offset

                # Reconstruct incoming population f_bar_i = feq_bar_i + feq_i - f_i*
                u_iface = self.u[:, reconstruct_mask]
                feq = compute_equilibrium(rho_iface, u_iface)

                f_streamed[opp, reconstruct_mask] = (
                    feq[opp] + feq[i] - f_post[i, reconstruct_mask]
                )

    def _exchange_mass(self, f_post: np.ndarray) -> None:
        r"""Update ``self.mass`` with the pairwise anti-symmetric exchange
        :math:`\Delta m_i = w_i\,(f^*_{\bar i}(\mathbf{x}+\mathbf{c}_i) - f^*_i(\mathbf{x}))`.
        """
        if self.free_surface_scheme == "legacy":
            self._exchange_mass_legacy(f_post)
            return

        types = self.cell_types
        is_liq = types == CellType.LIQUID
        is_if = types == CellType.INTERFACE
        fluid = is_liq | is_if
        delta_m = np.zeros((self.ny, self.nx), dtype=np.float64)
        for i in range(1, 9):
            opp = OPPOSITE[i]
            cx, cy = int(C[i][0]), int(C[i][1])
            both = fluid & _shift(fluid, cx, cy)
            # w is symmetric under (x, i) <-> (x + c_i, opp): exact conservation.
            w = np.zeros((self.ny, self.nx), dtype=np.float64)
            w[both & (is_liq | _shift(is_liq, cx, cy))] = 1.0
            ii = both & is_if & _shift(is_if, cx, cy)
            c_nbr = _shift(self.fill_level, cx, cy)
            w[ii] = 0.5 * (self.fill_level[ii] + c_nbr[ii])
            delta_m += w * (_shift(f_post[opp], cx, cy) - f_post[i])
        self.mass[~self.solid_mask] += delta_m[~self.solid_mask]

    def _exchange_mass_legacy(self, f_post: np.ndarray) -> None:
        r"""Phase 12-15 exchange: weight :math:`(C + C_{nbr})/2` on every non-solid face."""
        delta_m = np.zeros((self.ny, self.nx), dtype=np.float64)

        for i in range(1, 9):
            opp = OPPOSITE[i]
            cx, cy = C[i]

            c_nbr = np.roll(
                np.roll(self.fill_level, shift=-cy, axis=0), shift=-cx, axis=1
            )
            solid_nbr = np.roll(
                np.roll(self.solid_mask, shift=-cy, axis=0), shift=-cx, axis=1
            )

            # Active fluid face between cell x and neighbor x + c_i
            fluid_face = (~self.solid_mask) & (~solid_nbr)

            # Symmetric face contact area: A_i(x) = A_opp(x + c_i)
            A_i = np.where(fluid_face, 0.5 * (self.fill_level + c_nbr), 0.0)

            # Population from neighbor moving in direction opp toward cell x
            f_nbr_opp = np.roll(
                np.roll(f_post[opp], shift=-cy, axis=0), shift=-cx, axis=1
            )

            # Net mass flux from neighbor into self: Delta M_i = A_i * (f_bar_i*(x+c_i) - f_i*(x))
            flux = A_i * (f_nbr_opp - f_post[i])
            delta_m += flux

        # Update liquid mass globally across all fluid cells
        self.mass[~self.solid_mask] += delta_m[~self.solid_mask]

    def run(self, num_steps: int) -> None:
        r"""Advance simulation by `num_steps` time steps."""
        for _ in range(num_steps):
            self.step()

    # -----------------------------------------------------------------
    # State Transition & Mass Redistribution Mechanics
    # -----------------------------------------------------------------

    def _handle_state_transitions(self, f_streamed: np.ndarray) -> None:
        r"""Handle interface filling/emptying, enforce neighborhood consistency, and initialize new interface cells."""
        if self.free_surface_scheme == "legacy":
            self._handle_state_transitions_legacy(f_streamed)
        else:
            self._handle_state_transitions_korner(f_streamed)

    def _handle_state_transitions_korner(self, f_streamed: np.ndarray) -> None:
        r"""Körner/Thürey cell conversions with conservative excess-mass redistribution.

        1. Interface cells fill if :math:`m > (1+\kappa)\rho` and empty if
           :math:`m < -\kappa\rho`. Interface cells without fluid neighbours are
           emptied and those without gas neighbours are filled. A cell next to a
           filling cell may not empty (this would create LIQUID-GAS contact).
        2. GAS neighbours of filled cells become INTERFACE, initialised with
           :math:`f^{eq}(\bar\rho, \bar{\mathbf{u}})` averaged over fluid
           neighbours. LIQUID neighbours of emptied cells become INTERFACE.
        3. The excess mass (:math:`m-\rho` for filled, :math:`m` for emptied cells)
           is shared equally among neighbouring INTERFACE cells. If a cell has no
           such neighbour its conversion is cancelled, so no mass is ever lost.
        """
        t = self.cell_types
        is_gas = t == CellType.GAS
        is_liq = t == CellType.LIQUID
        iface = t == CellType.INTERFACE
        fluid = is_liq | iface

        rho_new, u_new = compute_macroscopic(f_streamed, force=self.force, dt=self.dt)
        rho_new = np.where(fluid, rho_new, 1.0)

        n_fluid = np.zeros((self.ny, self.nx), dtype=np.int32)
        n_gas = np.zeros((self.ny, self.nx), dtype=np.int32)
        for dx, dy in _NEIGHBOURS:
            n_fluid += _shift(fluid, dx, dy)
            n_gas += _shift(is_gas, dx, dy)

        k = self.kappa
        emptied = iface & ((self.mass < -k * rho_new) | (n_fluid == 0))
        filled = iface & ((self.mass > (1.0 + k) * rho_new) | (n_gas == 0)) & ~emptied
        near_filled = np.zeros_like(filled)
        for dx, dy in _NEIGHBOURS:
            near_filled |= _shift(filled, dx, dy)
        emptied &= ~near_filled

        new_if = np.zeros_like(filled)
        liq_to_if = np.zeros_like(filled)
        for dx, dy in _NEIGHBOURS:
            new_if |= _shift(filled, dx, dy) & is_gas
            liq_to_if |= _shift(emptied, dx, dy) & is_liq

        if np.any(new_if):
            rho_sum = np.zeros((self.ny, self.nx), dtype=np.float64)
            u_sum = np.zeros((2, self.ny, self.nx), dtype=np.float64)
            count = np.zeros((self.ny, self.nx), dtype=np.float64)
            for dx, dy in _NEIGHBOURS:
                src = _shift(fluid, dx, dy)
                rho_sum += np.where(src, _shift(rho_new, dx, dy), 0.0)
                u_sum += np.where(src, _shift(u_new, dx, dy), 0.0)
                count += src
            count = np.maximum(count, 1.0)
            rho0 = rho_sum[new_if] / count[new_if]
            u0 = u_sum[:, new_if] / count[new_if]
            # Cap velocity to enforce low-Mach stability (as in the legacy scheme)
            u_mag = np.hypot(u0[0], u0[1])
            u0 = u0 * np.where(u_mag > 0.15, 0.15 / np.maximum(u_mag, 1e-300), 1.0)
            f_streamed[:, new_if] = compute_equilibrium(rho0, u0)
            rho_new[new_if] = rho0

        new_t = np.copy(t)
        new_t[new_if] = CellType.INTERFACE
        new_t[liq_to_if] = CellType.INTERFACE
        new_t[filled] = CellType.LIQUID
        new_t[emptied] = CellType.GAS

        excess = np.zeros((self.ny, self.nx), dtype=np.float64)
        excess[filled] = self.mass[filled] - rho_new[filled]
        self.mass[filled] = rho_new[filled]
        excess[emptied] = self.mass[emptied]
        self.mass[emptied] = 0.0

        donors = filled | emptied
        receivers = (new_t == CellType.INTERFACE) & ~donors
        n_recv = np.zeros((self.ny, self.nx), dtype=np.int32)
        for dx, dy in _NEIGHBOURS:
            n_recv += _shift(receivers, dx, dy)
        orphan = donors & (n_recv == 0)
        if np.any(orphan):
            self.mass[orphan] += excess[orphan]
            excess[orphan] = 0.0
            new_t[orphan] = CellType.INTERFACE
        share = np.where(n_recv > 0, excess / np.maximum(n_recv, 1), 0.0)
        for dx, dy in _NEIGHBOURS:
            self.mass += np.where(receivers, _shift(share, dx, dy), 0.0)

        new_t[self.solid_mask] = CellType.SOLID
        new_t = enforce_neighborhood_consistency(new_t, self.solid_mask)
        self.cell_types = new_t
        self.f = f_streamed
        inactive = (new_t == CellType.GAS) | (new_t == CellType.SOLID)
        self.f[:, inactive] = 0.0

        fill = np.zeros((self.ny, self.nx), dtype=np.float64)
        fill[new_t == CellType.LIQUID] = 1.0
        ifm = new_t == CellType.INTERFACE
        fill[ifm] = np.clip(self.mass[ifm] / np.maximum(rho_new[ifm], 1e-6), 0.0, 1.0)
        self.fill_level = fill

    def _handle_state_transitions_legacy(self, f_streamed: np.ndarray) -> None:
        r"""Phase 12-15 conversions: threshold classification on :math:`C = m/\rho`."""
        old_types = np.copy(self.cell_types)
        rho_safe = np.maximum(self.rho, 1e-6)

        # Recalculate fill level from mass
        self.fill_level[~self.solid_mask] = np.clip(
            self.mass[~self.solid_mask] / rho_safe[~self.solid_mask], 0.0, 1.0
        )

        # Discrete cell classifications based on thresholds
        raw_types = classify_cells(self.fill_level, self.solid_mask, eps=self.eps)

        # Enforce topological neighborhood completeness (no LIQUID-GAS contact)
        new_types = enforce_neighborhood_consistency(raw_types, self.solid_mask)

        # Detect newly converted interface cells from gas
        gas_to_iface = (old_types == CellType.GAS) & (new_types == CellType.INTERFACE)
        if np.any(gas_to_iface):
            y_idxs, x_idxs = np.where(gas_to_iface)
            for y, x in zip(y_idxs, x_idxs):
                u_sum = np.zeros(2, dtype=np.float64)
                count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny_c = (y + dy) % self.ny
                        nx_c = (x + dx) % self.nx
                        if old_types[ny_c, nx_c] in (
                            CellType.LIQUID,
                            CellType.INTERFACE,
                        ):
                            u_sum += self.u[:, ny_c, nx_c]
                            count += 1
                u_avg = u_sum / max(count, 1)
                u_mag = np.hypot(u_avg[0], u_avg[1])
                # Cap velocity to enforce low-Mach stability
                if u_mag > 0.15:
                    u_avg = u_avg * (0.15 / u_mag)
                f_streamed[:, y, x] = compute_equilibrium(1.0, u_avg)

        self.cell_types = new_types
        self.f = f_streamed

        # Zero distribution functions in inactive cells
        inactive = (self.cell_types == CellType.GAS) | (
            self.cell_types == CellType.SOLID
        )
        self.f[:, inactive] = 0.0

    # -----------------------------------------------------------------
    # High-Level Diagnostics
    # -----------------------------------------------------------------

    def mass_error(self) -> float:
        r"""Relative mass drift :math:`|M(t) - M(0)| / M(0)`."""
        if self.initial_mass < 1e-12:
            return 0.0
        return float(abs(self.total_mass - self.initial_mass) / self.initial_mass)

    def volume_error(self) -> float:
        r"""Relative volume drift :math:`|V(t) - V(0)| / V(0)`."""
        if self.initial_volume < 1e-12:
            return 0.0
        return float(abs(self.total_volume - self.initial_volume) / self.initial_volume)

    def front_position(self, threshold: float = 0.5) -> float:
        r"""Maximum x-coordinate of the advancing liquid front where :math:`C > \text{threshold}`."""
        x_coords = np.where(self.fill_level > threshold)[1]
        if len(x_coords) == 0:
            return 0.0
        return float(np.max(x_coords))

    def height_profile(self, threshold: float = 0.5) -> np.ndarray:
        r"""Column liquid height as a function of x coordinate."""
        h = np.zeros(self.nx, dtype=np.float64)
        for x in range(self.nx):
            y_active = np.where(self.fill_level[:, x] > threshold)[0]
            if len(y_active) > 0:
                h[x] = float(np.max(y_active))
        return h

    def pressure_field(self) -> np.ndarray:
        r"""Thermodynamic pressure field :math:`p = c_s^2 \rho`."""
        return CS_SQ * self.rho

    def velocity_magnitude(self) -> np.ndarray:
        r"""Velocity magnitude field :math:`\|\mathbf{u}\| = \sqrt{u_x^2 + u_y^2}`."""
        return np.sqrt(self.u[0] ** 2 + self.u[1] ** 2)
