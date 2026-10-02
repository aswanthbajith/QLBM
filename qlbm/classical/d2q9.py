"""Two-dimensional nine-velocity (D2Q9) Lattice Boltzmann reference model.

This module provides the core data structures, constants, and vectorized operators
for the classical D2Q9 LBM solver with BGK collision, Guo forcing, and halfway
bounce-back boundary conditions.
"""

from typing import Optional, Tuple, Union

import numpy as np

# Discrete velocity vectors c_i = (c_ix, c_iy) for i in 0..8
# Convention:
#   0: rest (0, 0)
#   1: +x (1, 0)
#   2: +y (0, 1)
#   3: -x (-1, 0)
#   4: -y (0, -1)
#   5: +x+y (1, 1)
#   6: -x+y (-1, 1)
#   7: -x-y (-1, -1)
#   8: +x-y (1, -1)
C: np.ndarray = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int32,
)

# Lattice weights w_i for D2Q9
WEIGHTS: np.ndarray = np.array(
    [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ],
    dtype=np.float64,
)

# Opposite direction indices: c_{OPPOSITE[i]} = -c_i
OPPOSITE: np.ndarray = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# Lattice speed of sound: c_s^2 = 1/3, c_s = 1/sqrt(3)
CS_SQ: float = 1.0 / 3.0
CS: float = 1.0 / np.sqrt(3.0)


def viscosity_from_relaxation_time(tau: float, dt: float = 1.0) -> float:
    r"""Compute kinematic shear viscosity from BGK relaxation time.

    From the Chapman-Enskog expansion:
    .. math::
        \nu = c_s^2 (\tau - 0.5) \Delta t = \frac{2\tau - 1}{6} \Delta t

    Parameters
    ----------
    tau : float
        Non-dimensional relaxation time :math:`\tau > 0.5`.
    dt : float, optional
        Time step size, default is 1.0.

    Returns
    -------
    float
        Kinematic shear viscosity :math:`\nu`.
    """
    return CS_SQ * (tau - 0.5) * dt


def relaxation_time_from_viscosity(nu: float, dt: float = 1.0) -> float:
    r"""Compute BGK relaxation time from kinematic shear viscosity.

    .. math::
        \tau = \frac{\nu}{c_s^2 \Delta t} + 0.5 = \frac{3 \nu}{\Delta t} + 0.5

    Parameters
    ----------
    nu : float
        Kinematic shear viscosity :math:`\nu > 0`.
    dt : float, optional
        Time step size, default is 1.0.

    Returns
    -------
    float
        Non-dimensional relaxation time :math:`\tau`.
    """
    return (nu / (CS_SQ * dt)) + 0.5


def compute_equilibrium(
    rho: Union[float, np.ndarray],
    u: np.ndarray,
) -> np.ndarray:
    r"""Calculate the Maxwell-Boltzmann second-order polynomial equilibrium.

    .. math::
        f_i^{(eq)}(\rho, \mathbf{u}) = w_i \rho \left[ 1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2}
        + \frac{(\mathbf{c}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\mathbf{u} \cdot \mathbf{u}}{2 c_s^2} \right]

    With :math:`c_s^2 = 1/3`:
    .. math::
        f_i^{(eq)}(\rho, \mathbf{u}) = w_i \rho \left[ 1 + 3 (\mathbf{c}_i \cdot \mathbf{u})
        + \frac{9}{2} (\mathbf{c}_i \cdot \mathbf{u})^2 - \frac{3}{2} |\mathbf{u}|^2 \right]

    Parameters
    ----------
    rho : float or np.ndarray
        Density field of shape `(ny, nx)` or scalar.
    u : np.ndarray
        Macroscopic velocity field of shape `(2, ny, nx)` or `(2,)`.
        `u[0]` is the x-velocity, `u[1]` is the y-velocity.

    Returns
    -------
    np.ndarray
        Equilibrium distribution `f_eq` of shape `(9, ny, nx)` or `(9,)`.
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)

    # u_sq = u_x^2 + u_y^2
    u_sq = u_arr[0] ** 2 + u_arr[1] ** 2

    # Output shape: (9, *rho_shape)
    if rho_arr.ndim == 0:
        f_eq = np.empty(9, dtype=np.float64)
        for i in range(9):
            cu = C[i, 0] * u_arr[0] + C[i, 1] * u_arr[1]
            f_eq[i] = WEIGHTS[i] * rho_arr * (
                1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq
            )
    else:
        f_eq = np.empty((9, *rho_arr.shape), dtype=np.float64)
        for i in range(9):
            cu = C[i, 0] * u_arr[0] + C[i, 1] * u_arr[1]
            f_eq[i] = WEIGHTS[i] * rho_arr * (
                1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq
            )

    return f_eq


def compute_macroscopic(
    f: np.ndarray,
    force: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute macroscopic density and velocity from distribution functions.

    .. math::
        \rho(\mathbf{x}, t) = \sum_{i=0}^8 f_i(\mathbf{x}, t)

    .. math::
        \rho \mathbf{u}(\mathbf{x}, t) = \sum_{i=0}^8 f_i(\mathbf{x}, t) \mathbf{c}_i + \frac{\Delta t}{2} \mathbf{F}

    Parameters
    ----------
    f : np.ndarray
        Distribution functions of shape `(9, ny, nx)`.
    force : np.ndarray, optional
        External body force field of shape `(2, ny, nx)` or `(2,)`, default None.
    dt : float, optional
        Time step size, default 1.0.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Density `rho` of shape `(ny, nx)` and velocity `u` of shape `(2, ny, nx)`.
    """
    rho = np.sum(f, axis=0)

    # Momentum components
    rho_u = np.empty((2, *rho.shape), dtype=np.float64)
    rho_u[0] = np.sum(C[:, 0, None, None] * f, axis=0)
    rho_u[1] = np.sum(C[:, 1, None, None] * f, axis=0)

    if force is not None:
        f_arr = np.asarray(force, dtype=np.float64)
        if f_arr.ndim == 1:
            rho_u[0] += 0.5 * dt * f_arr[0]
            rho_u[1] += 0.5 * dt * f_arr[1]
        else:
            rho_u += 0.5 * dt * f_arr

    u = np.zeros_like(rho_u)
    fluid_mask = rho > 1e-14
    u[0, fluid_mask] = rho_u[0, fluid_mask] / rho[fluid_mask]
    u[1, fluid_mask] = rho_u[1, fluid_mask] / rho[fluid_mask]

    return rho, u


def compute_guo_force(
    u: np.ndarray,
    force: np.ndarray,
    tau: float,
    dt: float = 1.0,
) -> np.ndarray:
    r"""Compute Guo forcing source terms for D2Q9 BGK collision.

    From Guo, Zheng, and Shi (Phys. Rev. E 65, 046308, 2002):
    .. math::
        F_i = \left( 1 - \frac{\Delta t}{2\tau} \right) w_i \left[
            \frac{\mathbf{c}_i - \mathbf{u}}{c_s^2}
            + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^4} \mathbf{c}_i
        \right] \cdot \mathbf{F}

    Parameters
    ----------
    u : np.ndarray
        Macroscopic velocity field of shape `(2, ny, nx)`.
    force : np.ndarray
        External body force of shape `(2, ny, nx)` or `(2,)`.
    tau : float
        Relaxation time :math:`\tau`.
    dt : float, optional
        Time step size, default 1.0.

    Returns
    -------
    np.ndarray
        Source term `F_i` of shape `(9, ny, nx)`.
    """
    f_arr = np.asarray(force, dtype=np.float64)
    prefactor = (1.0 - dt / (2.0 * tau))

    # Expand force if 1D
    if f_arr.ndim == 1:
        fx = f_arr[0]
        fy = f_arr[1]
    else:
        fx = f_arr[0]
        fy = f_arr[1]

    ux = u[0]
    uy = u[1]

    # (c_i - u) . F = c_ix * fx + c_iy * fy - (ux * fx + uy * fy)
    u_dot_f = ux * fx + uy * fy

    F_i = np.empty((9, *ux.shape), dtype=np.float64)
    for i in range(9):
        cix = C[i, 0]
        ciy = C[i, 1]
        ci_dot_f = cix * fx + ciy * fy
        ci_dot_u = cix * ux + ciy * uy

        # Guo formula with cs^2 = 1/3, cs^4 = 1/9:
        # term1 = 3 * (ci_dot_f - u_dot_f)
        # term2 = 9 * ci_dot_u * ci_dot_f
        bracket = 3.0 * (ci_dot_f - u_dot_f) + 9.0 * ci_dot_u * ci_dot_f
        F_i[i] = prefactor * WEIGHTS[i] * bracket

    return F_i


def bgk_collision(
    f: np.ndarray,
    f_eq: np.ndarray,
    tau: float,
    force: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    r"""Execute Bhatnagar-Gross-Krook (BGK) collision step.

    .. math::
        f_i^*(\mathbf{x}, t) = f_i(\mathbf{x}, t) - \frac{\Delta t}{\tau} \left[
            f_i(\mathbf{x}, t) - f_i^{(eq)}(\mathbf{x}, t)
        \right] + \Delta t F_i(\mathbf{x}, t)

    Parameters
    ----------
    f : np.ndarray
        Pre-collision distribution functions of shape `(9, ny, nx)`.
    f_eq : np.ndarray
        Equilibrium distribution functions of shape `(9, ny, nx)`.
    tau : float
        Relaxation time :math:`\tau`.
    force : np.ndarray, optional
        External body force of shape `(2, ny, nx)` or `(2,)`, default None.
    u : np.ndarray, optional
        Macroscopic velocity field of shape `(2, ny, nx)`, required if force is given.
    dt : float, optional
        Time step size, default 1.0.

    Returns
    -------
    np.ndarray
        Post-collision distribution functions `f_star` of shape `(9, ny, nx)`.
    """
    omega = dt / tau
    f_star = (1.0 - omega) * f + omega * f_eq

    if force is not None:
        if u is None:
            raise ValueError("Macroscopic velocity `u` must be provided when `force` is non-None.")
        F_i = compute_guo_force(u, force, tau, dt=dt)
        f_star += dt * F_i

    return f_star


def stream_and_bounce(
    f_post: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
    wall_velocity: Optional[np.ndarray] = None,
    rho_wall: float = 1.0,
) -> np.ndarray:
    r"""Execute streaming step with optional halfway bounce-back on solid obstacles.

    Standard streaming applies periodic boundaries:
    .. math::
        f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i^*(\mathbf{x}, t)

    If `solid_mask` is provided, halfway bounce-back is applied to all fluid-solid boundary links:
    For fluid node :math:`\mathbf{x}_f` with solid neighbor at :math:`\mathbf{x}_f + \mathbf{c}_i`:
    .. math::
        f_{\bar{i}}(\mathbf{x}_f, t + \Delta t) = f_i^*(\mathbf{x}_f, t) - 2 w_i \rho_w \frac{\mathbf{c}_i \cdot \mathbf{u}_w}{c_s^2}

    Parameters
    ----------
    f_post : np.ndarray
        Post-collision distribution functions of shape `(9, ny, nx)`.
    solid_mask : np.ndarray of bool, optional
        Boolean mask of shape `(ny, nx)` where `True` indicates a solid obstacle node.
    wall_velocity : np.ndarray, optional
        Velocity field at walls of shape `(2, ny, nx)` or `(2,)`, default None.
    rho_wall : float, optional
        Reference density for moving wall momentum exchange, default 1.0.

    Returns
    -------
    np.ndarray
        Streamed distribution functions `f_next` of shape `(9, ny, nx)`.
    """
    f_next = np.empty_like(f_post)

    # 1. Base periodic streaming for all directions
    for i in range(9):
        cix = C[i, 0]
        ciy = C[i, 1]
        f_next[i] = np.roll(f_post[i], shift=(ciy, cix), axis=(0, 1))

    # 2. Halfway bounce-back on solid boundaries
    if solid_mask is not None and np.any(solid_mask):
        fluid_mask = ~solid_mask

        for i in range(1, 9):
            cix = C[i, 0]
            ciy = C[i, 1]
            opp = OPPOSITE[i]

            # A fluid node has a solid neighbor in direction i if
            # solid_mask[y + ciy, x + cix] is True
            neighbor_solid = np.roll(solid_mask, shift=(-ciy, -cix), axis=(0, 1))
            hit_solid = fluid_mask & neighbor_solid

            # Reflect outgoing post-collision population f_i^*(x_f) back to f_{opp}(x_f)
            bounced = f_post[i, hit_solid]

            # Moving wall momentum adjustment
            if wall_velocity is not None:
                wv = np.asarray(wall_velocity, dtype=np.float64)
                if wv.ndim == 1:
                    cu_w = cix * wv[0] + ciy * wv[1]
                else:
                    # Wall velocity may be defined on solid nodes or fluid interface
                    wv_solid = np.roll(wv, shift=(-ciy, -cix), axis=(1, 2))
                    cu_w_solid = cix * wv_solid[0, hit_solid] + ciy * wv_solid[1, hit_solid]
                    cu_w_fluid = cix * wv[0, hit_solid] + ciy * wv[1, hit_solid]
                    cu_w = np.where(cu_w_solid != 0.0, cu_w_solid, cu_w_fluid)
                # 2 * w_i * rho_w * (c_i . u_w) / c_s^2 with c_s^2 = 1/3 -> 6 * w_i * rho_w * cu_w
                bounced = bounced - 6.0 * WEIGHTS[i] * rho_wall * cu_w

            f_next[opp, hit_solid] = bounced

        # Solid nodes are zeroed out
        f_next[:, solid_mask] = 0.0

    return f_next
