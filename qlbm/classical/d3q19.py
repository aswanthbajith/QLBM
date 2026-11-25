"""D3Q19 Lattice Boltzmann Model for 3D Incompressible Hydrodynamics.

Provides discrete velocity vectors, lattice weights, equilibrium distribution
computation, Guo second-order body forcing in 3D, macroscopic moment decoding,
and standard solid-wall bounce-back boundary conditions for the D3Q19 lattice.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# D3Q19 Discrete Lattice Constants
# -----------------------------------------------------------------------------

# 19 velocity vectors: c[i] = (cx, cy, cz)
# Shape: (19, 3)
C_LIST = [
    (0, 0, 0),     # 0: Rest
    (+1, 0, 0),    # 1: +x
    (-1, 0, 0),    # 2: -x
    (0, +1, 0),    # 3: +y
    (0, -1, 0),    # 4: -y
    (0, 0, +1),    # 5: +z (up)
    (0, 0, -1),    # 6: -z (down)
    (+1, +1, 0),   # 7: +x +y
    (-1, -1, 0),   # 8: -x -y
    (+1, -1, 0),   # 9: +x -y
    (-1, +1, 0),   # 10: -x +y
    (+1, 0, +1),   # 11: +x +z
    (-1, 0, -1),   # 12: -x -z
    (+1, 0, -1),   # 13: +x -z
    (-1, 0, +1),   # 14: -x +z
    (0, +1, +1),   # 15: +y +z
    (0, -1, -1),   # 16: -y -z
    (0, +1, -1),   # 17: +y -z
    (0, -1, +1),   # 18: -y +z
]

C = np.array(C_LIST, dtype=np.int64)

# Opposite velocity indices: c[OPPOSITE[i]] == -c[i]
OPPOSITE = np.array([
    0,   # 0 -> 0
    2,   # 1 -> 2
    1,   # 2 -> 1
    4,   # 3 -> 4
    3,   # 4 -> 3
    6,   # 5 -> 6
    5,   # 6 -> 5
    8,   # 7 -> 8
    7,   # 8 -> 7
    10,  # 9 -> 10
    9,   # 10 -> 9
    12,  # 11 -> 12
    11,  # 12 -> 11
    14,  # 13 -> 14
    13,  # 14 -> 13
    16,  # 15 -> 16
    15,  # 16 -> 15
    18,  # 17 -> 18
    17,  # 18 -> 17
], dtype=np.int64)

# Lattice weights: sum(WEIGHTS) == 1.0
WEIGHTS = np.array([
    1.0 / 3.0,   # 0: rest (12/36)
    1.0 / 18.0,  # 1: +x (2/36)
    1.0 / 18.0,  # 2: -x
    1.0 / 18.0,  # 3: +y
    1.0 / 18.0,  # 4: -y
    1.0 / 18.0,  # 5: +z
    1.0 / 18.0,  # 6: -z
    1.0 / 36.0,  # 7: +x +y (1/36)
    1.0 / 36.0,  # 8: -x -y
    1.0 / 36.0,  # 9: +x -y
    1.0 / 36.0,  # 10: -x +y
    1.0 / 36.0,  # 11: +x +z
    1.0 / 36.0,  # 12: -x -z
    1.0 / 36.0,  # 13: +x -z
    1.0 / 36.0,  # 14: -x +z
    1.0 / 36.0,  # 15: +y +z
    1.0 / 36.0,  # 16: -y -z
    1.0 / 36.0,  # 17: +y -z
    1.0 / 36.0,  # 18: -y +z
], dtype=np.float64)

# Lattice speed of sound: cs^2 = 1/3
CS_SQ = 1.0 / 3.0
CS = np.sqrt(CS_SQ)


# -----------------------------------------------------------------------------
# Hydrodynamic Equations & Functions
# -----------------------------------------------------------------------------

def compute_equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    r"""Compute the 3D D3Q19 second-order equilibrium distribution :math:`f_i^{\text{eq}}`.

    Parameters
    ----------
    rho : np.ndarray
        Density field of shape `(nz, ny, nx)`.
    u : np.ndarray
        Velocity vector field of shape `(3, nz, ny, nx)`.

    Returns
    -------
    np.ndarray
        Equilibrium distribution of shape `(19, nz, ny, nx)`.
    """
    u = np.asarray(u, dtype=np.float64)
    target_shape = u.shape[1:] if u.ndim >= 1 else ()
    rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), target_shape)

    # u_sq = ux^2 + uy^2 + uz^2
    u_sq = u[0]**2 + u[1]**2 + u[2]**2

    feq = np.empty((19, *target_shape), dtype=np.float64)

    # Rest particle (i=0)
    feq[0] = WEIGHTS[0] * rho * (1.0 - 0.5 * u_sq / CS_SQ)

    # Moving particles (i=1..18)
    for i in range(1, 19):
        cx, cy, cz = C[i]
        c_dot_u = cx * u[0] + cy * u[1] + cz * u[2]
        feq[i] = WEIGHTS[i] * rho * (
            1.0
            + (c_dot_u / CS_SQ)
            + 0.5 * (c_dot_u**2 / CS_SQ**2)
            - 0.5 * (u_sq / CS_SQ)
        )

    return feq


def compute_guo_force(
    u: np.ndarray,
    force: np.ndarray,
    tau: float,
    dt: float = 1.0,
) -> np.ndarray:
    r"""Compute the 3D Guo body forcing source term :math:`F_i` for D3Q19.

    Parameters
    ----------
    u : np.ndarray
        Velocity field of shape `(3, nz, ny, nx)`.
    force : np.ndarray
        Total body force per unit volume of shape `(3, nz, ny, nx)` or `(3,)`.
    tau : float
        BGK relaxation time.
    dt : float, optional
        Lattice time step, default 1.0.

    Returns
    -------
    np.ndarray
        Forcing source term of shape `(19, nz, ny, nx)`.
    """
    u = np.asarray(u, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)

    # Ensure force has shape (3, nz, ny, nx)
    if force.ndim == 1:
        force = force[:, None, None, None]

    prefactor = (1.0 - 0.5 * dt / tau)
    fx, fy, fz = force[0], force[1], force[2]
    ux, uy, uz = u[0], u[1], u[2]

    Fi = np.empty((19, *u.shape[1:]), dtype=np.float64)

    # Rest particle
    Fi[0] = prefactor * WEIGHTS[0] * (
        (-ux * fx - uy * fy - uz * fz) / CS_SQ
    )

    for i in range(1, 19):
        cx, cy, cz = C[i]
        c_dot_u = cx * ux + cy * uy + cz * uz
        c_minus_u_dot_F = (cx - ux) * fx + (cy - uy) * fy + (cz - uz) * fz
        c_dot_F = cx * fx + cy * fy + cz * fz

        term1 = c_minus_u_dot_F / CS_SQ
        term2 = (c_dot_u * c_dot_F) / (CS_SQ**2)

        Fi[i] = prefactor * WEIGHTS[i] * (term1 + term2)

    return Fi


def compute_macroscopic(
    f: np.ndarray,
    force: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute macroscopic density :math:`\rho` and momentum :math:`\mathbf{u}`.

    Parameters
    ----------
    f : np.ndarray
        Distribution functions of shape `(19, nz, ny, nx)`.
    force : np.ndarray, optional
        Body force per unit volume of shape `(3, nz, ny, nx)` or `(3,)`.
    dt : float, optional
        Lattice time step, default 1.0.

    Returns
    -------
    rho : np.ndarray
        Density field of shape `(nz, ny, nx)`.
    u : np.ndarray
        Velocity vector field of shape `(3, nz, ny, nx)`.
    """
    rho = np.sum(f, axis=0)

    # Momentum: rho * u = sum(c_i * f_i) + 0.5 * force * dt
    # Shape: (3, nz, ny, nx)
    mom_x = np.tensordot(C[:, 0], f, axes=(0, 0))
    mom_y = np.tensordot(C[:, 1], f, axes=(0, 0))
    mom_z = np.tensordot(C[:, 2], f, axes=(0, 0))

    if force is not None:
        force = np.asarray(force, dtype=np.float64)
        if force.ndim == 1:
            mom_x += 0.5 * force[0] * dt
            mom_y += 0.5 * force[1] * dt
            mom_z += 0.5 * force[2] * dt
        else:
            mom_x += 0.5 * force[0] * dt
            mom_y += 0.5 * force[1] * dt
            mom_z += 0.5 * force[2] * dt

    safe_rho = np.where(rho > 1e-14, rho, 1.0)
    ux = np.where(rho > 1e-14, mom_x / safe_rho, 0.0)
    uy = np.where(rho > 1e-14, mom_y / safe_rho, 0.0)
    uz = np.where(rho > 1e-14, mom_z / safe_rho, 0.0)

    u = np.stack([ux, uy, uz], axis=0)
    return rho, u


def bgk_collision(
    f: np.ndarray,
    feq: np.ndarray,
    tau: float,
    force: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    r"""Execute standard 3D BGK collision with Guo body forcing.

    Parameters
    ----------
    f : np.ndarray
        Pre-collision distribution functions of shape `(19, nz, ny, nx)`.
    feq : np.ndarray
        Equilibrium distribution functions of shape `(19, nz, ny, nx)`.
    tau : float
        Relaxation time.
    force : np.ndarray, optional
        Body force field.
    u : np.ndarray, optional
        Velocity field needed for Guo forcing.
    dt : float, optional
        Time step.

    Returns
    -------
    np.ndarray
        Post-collision distribution functions of shape `(19, nz, ny, nx)`.
    """
    omega = dt / tau
    f_post = (1.0 - omega) * f + omega * feq

    if force is not None and u is not None:
        Fi = compute_guo_force(u, force, tau, dt=dt)
        f_post += dt * Fi

    return f_post


def stream_and_bounce(
    f: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Execute 3D spatial streaming and standard half-way solid wall bounce-back.

    Parameters
    ----------
    f : np.ndarray
        Distribution functions of shape `(19, nz, ny, nx)`.
    solid_mask : np.ndarray, optional
        Boolean solid wall mask of shape `(nz, ny, nx)`.

    Returns
    -------
    np.ndarray
        Streamed and reflected distribution functions of shape `(19, nz, ny, nx)`.
    """
    f_streamed = np.empty_like(f)

    # Rest particle
    f_streamed[0] = f[0]

    # Shift each direction in 3D: axis 0 is z, axis 1 is y, axis 2 is x
    for i in range(1, 19):
        cx, cy, cz = C[i]
        # Particles move in direction +c, so at position x they receive from x - c
        # In np.roll, shift = +cz moves array elements forward (from z-cz to z)
        f_streamed[i] = np.roll(
            np.roll(np.roll(f[i], shift=cz, axis=0), shift=cy, axis=1),
            shift=cx,
            axis=2,
        )

    # Solid bounce-back reflection
    if solid_mask is not None and np.any(solid_mask):
        f_post_bounce = np.copy(f_streamed)
        for i in range(1, 19):
            opp = OPPOSITE[i]
            cx, cy, cz = C[i]
            # Identify fluid cells where neighbor along +c is solid
            solid_nbr = np.roll(
                np.roll(np.roll(solid_mask, shift=-cz, axis=0), shift=-cy, axis=1),
                shift=-cx,
                axis=2,
            )
            bounce_mask = (~solid_mask) & solid_nbr
            # In fluid cell, population streaming into solid bounces back into opposite direction
            f_post_bounce[opp, bounce_mask] = f[i, bounce_mask]

        # Zero out inside solid cells
        f_post_bounce[:, solid_mask] = 0.0
        return f_post_bounce

    return f_streamed
