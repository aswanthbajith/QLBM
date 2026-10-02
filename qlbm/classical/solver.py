"""High-level classical D2Q9 BGK Lattice Boltzmann solver.

This module provides the `D2Q9BGKSolver` class, managing the simulation cycle:
macroscopic moment evaluation, equilibrium computation, BGK collision with optional
body forcing (Guo scheme), streaming with halfway bounce-back, and diagnostic tracking.
"""

from typing import Optional, Union

import numpy as np

from qlbm.classical.d2q9 import (
    bgk_collision,
    compute_entropy_h,
    compute_equilibrium,
    compute_macroscopic,
    compute_nonequilibrium_stress,
    stream_and_bounce,
    viscosity_from_relaxation_time,
)


class D2Q9BGKSolver:
    r"""Two-dimensional 9-velocity (D2Q9) BGK Lattice Boltzmann Solver.

    Parameters
    ----------
    nx : int
        Number of grid nodes in the x-direction.
    ny : int
        Number of grid nodes in the y-direction.
    tau : float, optional
        Non-dimensional relaxation time :math:`\tau > 0.5`, default 1.0.
    solid_mask : np.ndarray of bool, optional
        Boolean mask of shape `(ny, nx)` where `True` indicates a solid obstacle.
    force : np.ndarray, optional
        External body force of shape `(2, ny, nx)` or `(2,)`, default None.
    wall_velocity : np.ndarray, optional
        Velocity vector or field at solid walls, default None.
    dt : float, optional
        Time step size, default 1.0.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        tau: float = 1.0,
        solid_mask: Optional[np.ndarray] = None,
        force: Optional[np.ndarray] = None,
        wall_velocity: Optional[np.ndarray] = None,
        dt: float = 1.0,
    ) -> None:
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid dimensions must be positive, got nx={nx}, ny={ny}.")
        if tau <= 0.5:
            raise ValueError(
                f"BGK relaxation time must satisfy tau > 0.5 for stability, got {tau}."
            )

        self.nx = nx
        self.ny = ny
        self.tau = float(tau)
        self.dt = float(dt)

        if solid_mask is not None:
            mask = np.asarray(solid_mask, dtype=bool)
            if mask.shape != (ny, nx):
                raise ValueError(
                    f"solid_mask shape {mask.shape} does not match (ny={ny}, nx={nx})."
                )
            self.solid_mask: Optional[np.ndarray] = mask
        else:
            self.solid_mask = None

        self.force = force
        self.wall_velocity = wall_velocity

        # State fields
        self.f: np.ndarray = np.zeros((9, ny, nx), dtype=np.float64)
        self.rho: np.ndarray = np.ones((ny, nx), dtype=np.float64)
        self.u: np.ndarray = np.zeros((2, ny, nx), dtype=np.float64)
        self.time_step: int = 0

    @property
    def viscosity(self) -> float:
        r"""Kinematic shear viscosity :math:`\nu`."""
        return viscosity_from_relaxation_time(self.tau, self.dt)

    @property
    def omega(self) -> float:
        r"""Collision frequency :math:`\omega = \Delta t / \tau`."""
        return self.dt / self.tau

    @property
    def fluid_mask(self) -> np.ndarray:
        r"""Boolean mask of fluid nodes."""
        if self.solid_mask is None:
            return np.ones((self.ny, self.nx), dtype=bool)
        return ~self.solid_mask

    @property
    def total_mass(self) -> float:
        r"""Total fluid mass inside the computational domain."""
        return float(np.sum(self.rho[self.fluid_mask]))

    @property
    def total_momentum(self) -> np.ndarray:
        r"""Total fluid momentum vector inside the computational domain."""
        px = np.sum(self.rho[self.fluid_mask] * self.u[0, self.fluid_mask])
        py = np.sum(self.rho[self.fluid_mask] * self.u[1, self.fluid_mask])
        return np.array([px, py], dtype=np.float64)

    @property
    def kinetic_energy(self) -> float:
        r"""Total fluid kinetic energy :math:`\frac{1}{2} \sum \rho |\mathbf{u}|^2`."""
        u_sq = self.u[0, self.fluid_mask] ** 2 + self.u[1, self.fluid_mask] ** 2
        return 0.5 * float(np.sum(self.rho[self.fluid_mask] * u_sq))

    def initialize(
        self,
        rho: Union[float, np.ndarray] = 1.0,
        u: Optional[np.ndarray] = None,
    ) -> None:
        r"""Initialize the distribution functions to equilibrium.

        Parameters
        ----------
        rho : float or np.ndarray, optional
            Initial density field, default 1.0.
        u : np.ndarray, optional
            Initial velocity field of shape `(2, ny, nx)` or `(2,)`, default zero.
        """
        if u is None:
            u_field = np.zeros((2, self.ny, self.nx), dtype=np.float64)
        else:
            u_arr = np.asarray(u, dtype=np.float64)
            if u_arr.ndim == 1:
                u_field = np.zeros((2, self.ny, self.nx), dtype=np.float64)
                u_field[0, :, :] = u_arr[0]
                u_field[1, :, :] = u_arr[1]
            else:
                u_field = u_arr

        if np.isscalar(rho):
            rho_field = np.full((self.ny, self.nx), rho, dtype=np.float64)
        else:
            rho_field = np.asarray(rho, dtype=np.float64)

        self.f = compute_equilibrium(rho_field, u_field)

        if self.solid_mask is not None:
            self.f[:, self.solid_mask] = 0.0

        self.rho, self.u = compute_macroscopic(
            self.f, force=self.force, dt=self.dt
        )
        self.time_step = 0

    def step(self) -> None:
        r"""Execute one complete time step of the Lattice Boltzmann Method.

        Step sequence:
        1. Compute equilibrium distribution :math:`f^{(eq)}(\rho, \mathbf{u})`.
        2. Perform BGK collision with optional Guo body forcing :math:`f^*`.
        3. Perform streaming and halfway bounce-back boundary reflections.
        4. Recompute macroscopic moments :math:`\rho, \mathbf{u}`.
        """
        # 1. Equilibrium
        f_eq = compute_equilibrium(self.rho, self.u)

        # 2. Collision
        f_post = bgk_collision(
            self.f,
            f_eq,
            self.tau,
            force=self.force,
            u=self.u,
            dt=self.dt,
        )

        # 3. Streaming and boundary bounce-back
        self.f = stream_and_bounce(
            f_post,
            solid_mask=self.solid_mask,
            wall_velocity=self.wall_velocity,
        )

        # 4. Macroscopic moments
        self.rho, self.u = compute_macroscopic(
            self.f,
            force=self.force,
            dt=self.dt,
        )

        self.time_step += 1

    def run(self, steps: int) -> None:
        r"""Run the simulation for a specified number of steps.

        Parameters
        ----------
        steps : int
            Number of time steps to advance.
        """
        for _ in range(steps):
            self.step()

    def get_velocity_magnitude(self) -> np.ndarray:
        r"""Compute velocity magnitude field :math:`|\mathbf{u}| = \sqrt{u_x^2 + u_y^2}`."""
        return np.sqrt(self.u[0] ** 2 + self.u[1] ** 2)

    def get_vorticity(self) -> np.ndarray:
        r"""Compute scalar vorticity :math:`\omega_z = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}`.

        Uses second-order central differences on the periodic domain.
        """
        # Central difference in x (axis 1): (u_y[:, x+1] - u_y[:, x-1]) / 2
        duy_dx = (np.roll(self.u[1], -1, axis=1) - np.roll(self.u[1], 1, axis=1)) / 2.0
        # Central difference in y (axis 0): (u_x[y+1, :] - u_x[y-1, :]) / 2
        dux_dy = (np.roll(self.u[0], -1, axis=0) - np.roll(self.u[0], 1, axis=0)) / 2.0
        return duy_dx - dux_dy

    def get_nonequilibrium_stress(self) -> np.ndarray:
        r"""Compute non-equilibrium momentum flux tensor \Pi_{ab}^{(neq)}."""
        f_eq = compute_equilibrium(self.rho, self.u)
        return compute_nonequilibrium_stress(self.f, f_eq=f_eq)

    def get_entropy_h(self) -> float:
        r"""Compute Boltzmann's H-function H = \sum_{x \in fluid, i} f_i \ln(f_i / w_i)."""
        return compute_entropy_h(self.f, mask=self.fluid_mask)
