"""3D Volume-of-Fluid (VOF) Free-Surface Interface Tracking for D3Q19 LBM.

Implements 3D cell classification (GAS, INTERFACE, LIQUID, SOLID), 3D isotropic
gradient normal vector estimation, anti-symmetric directional mass fluxes,
free-surface gas pressure reconstruction, and topological cell conversions.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np

from qlbm.classical.d3q19 import C, CS_SQ, OPPOSITE, WEIGHTS, compute_equilibrium


class CellType(IntEnum):
    """3D VOF topological cell classification."""
    GAS = 0        # Empty cell (not calculated)
    INTERFACE = 1  # Free-surface boundary layer containing liquid and gas
    LIQUID = 2     # Fully saturated fluid cell governed by Navier-Stokes
    SOLID = 3      # Boundary wall, obstacle, or gate


def classify_cells_3d(
    fill_level: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-4,
) -> np.ndarray:
    r"""Classify each grid node into GAS, INTERFACE, LIQUID, or SOLID.

    Parameters
    ----------
    fill_level : np.ndarray
        Volume fraction field :math:`C(\mathbf{x}) \in [0, 1]` of shape `(nz, ny, nx)`.
    solid_mask : np.ndarray, optional
        Boolean mask marking solid obstacles of shape `(nz, ny, nx)`.
    eps : float, optional
        Numerical threshold tolerance, default 1e-4.

    Returns
    -------
    np.ndarray
        Array of :class:`CellType` of shape `(nz, ny, nx)`.
    """
    types = np.full(fill_level.shape, CellType.INTERFACE, dtype=np.int8)

    # Pure gas cells
    types[fill_level <= eps] = CellType.GAS

    # Pure liquid cells
    types[fill_level >= 1.0 - eps] = CellType.LIQUID

    # Solid cells override fluid
    if solid_mask is not None:
        types[solid_mask] = CellType.SOLID

    return enforce_neighborhood_consistency_3d(types, solid_mask)


def enforce_neighborhood_consistency_3d(
    cell_types: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Enforce the topological rule that LIQUID cells must never touch GAS cells directly in 3D.

    Any LIQUID cell adjacent (in the D3Q19 18-neighborhood) to a GAS cell is converted
    into an INTERFACE cell to ensure an enclosing protective interface boundary layer.
    """
    types = np.copy(cell_types)
    is_gas = (types == CellType.GAS)
    is_liquid = (types == CellType.LIQUID)

    has_gas_nbr = np.zeros_like(types, dtype=bool)
    for i in range(1, 19):
        cx, cy, cz = C[i]
        gas_nbr = (
            np.roll(
                np.roll(np.roll(is_gas, shift=-cz, axis=0), shift=-cy, axis=1),
                shift=-cx,
                axis=2,
            )
        )
        has_gas_nbr |= gas_nbr

    touch_gas = is_liquid & has_gas_nbr
    types[touch_gas] = CellType.INTERFACE

    if solid_mask is not None:
        types[solid_mask] = CellType.SOLID

    return types


def compute_normals_3d(
    fill_level: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute 3D interface normal vectors :math:`\mathbf{n} = -\nabla C` using isotropic D3Q19 gradient stencil.

    Parameters
    ----------
    fill_level : np.ndarray
        Fill fraction field :math:`C` of shape `(nz, ny, nx)`.
    solid_mask : np.ndarray, optional
        Solid obstacle mask.

    Returns
    -------
    normals : np.ndarray
        Outward unit normal vectors :math:`\hat{\mathbf{n}}` of shape `(3, nz, ny, nx)`.
    magnitudes : np.ndarray
        Gradient magnitude :math:`\|\nabla C\|_2` of shape `(nz, ny, nx)`.
    """
    c_field = np.copy(fill_level)
    if solid_mask is not None:
        # Avoid artificial gradients across solid walls
        c_field[solid_mask] = 0.5

    grad_x = np.zeros_like(c_field)
    grad_y = np.zeros_like(c_field)
    grad_z = np.zeros_like(c_field)

    inv_cs_sq = 1.0 / CS_SQ

    for i in range(1, 19):
        cx, cy, cz = C[i]
        # Sample neighbor C(x + c_i)
        c_nbr = np.roll(
            np.roll(np.roll(c_field, shift=-cz, axis=0), shift=-cy, axis=1),
            shift=-cx,
            axis=2,
        )
        weight = WEIGHTS[i] * inv_cs_sq
        grad_x += weight * cx * c_nbr
        grad_y += weight * cy * c_nbr
        grad_z += weight * cz * c_nbr

    # Outward normal pointing into gas: n = -grad(C)
    nx_field = -grad_x
    ny_field = -grad_y
    nz_field = -grad_z

    mag = np.sqrt(nx_field**2 + ny_field**2 + nz_field**2)
    safe_mag = np.where(mag > 1e-12, mag, 1.0)

    unit_nx = np.where(mag > 1e-12, nx_field / safe_mag, 0.0)
    unit_ny = np.where(mag > 1e-12, ny_field / safe_mag, 0.0)
    unit_nz = np.where(mag > 1e-12, nz_field / safe_mag, -1.0)  # default down if flat

    normals = np.stack([unit_nx, unit_ny, unit_nz], axis=0)
    return normals, mag


def compute_directional_mass_fluxes_3d(
    f: np.ndarray,
    fill_level: np.ndarray,
    cell_types: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    r"""Compute strictly anti-symmetric pairwise mass fluxes :math:`\Delta M_i` across interface links.

    Uses the canonical Körner / Thürey face contact area:
      - :math:`w = 1.0` if either cell is pure liquid.
      - :math:`w = \frac{1}{2}(C(\mathbf{x}) + C(\mathbf{x}+\mathbf{c}_i))` between two interface cells.
      - :math:`w = 0.0` otherwise.

    Exact anti-symmetry :math:`\Delta M_i(\mathbf{x}) \equiv -\Delta M_{\bar{i}}(\mathbf{x} + \mathbf{c}_i)`
    guarantees global mass conservation to machine precision.

    Parameters
    ----------
    f : np.ndarray
        Post-collision distribution functions of shape `(19, nz, ny, nx)`.
    fill_level : np.ndarray
        Fill fraction field of shape `(nz, ny, nx)`.
    cell_types : np.ndarray
        Cell type classification array of shape `(nz, ny, nx)`.
    solid_mask : np.ndarray, optional
        Boolean solid obstacle mask of shape `(nz, ny, nx)`.
    dt : float, optional
        Lattice time step, default 1.0.

    Returns
    -------
    np.ndarray
        Array of directional mass fluxes of shape `(19, nz, ny, nx)`.
    """
    delta_m = np.zeros_like(f)
    if solid_mask is None:
        solid_mask = (cell_types == CellType.SOLID)

    fluid = ((cell_types == CellType.LIQUID) | (cell_types == CellType.INTERFACE)) & (~solid_mask)
    is_liq = (cell_types == CellType.LIQUID) & (~solid_mask)
    is_if = (cell_types == CellType.INTERFACE) & (~solid_mask)

    for i in range(1, 19):
        opp = OPPOSITE[i]
        cx, cy, cz = C[i]

        fluid_nbr = np.roll(
            np.roll(np.roll(fluid, shift=-cz, axis=0), shift=-cy, axis=1),
            shift=-cx,
            axis=2,
        )
        liq_nbr = np.roll(
            np.roll(np.roll(is_liq, shift=-cz, axis=0), shift=-cy, axis=1),
            shift=-cx,
            axis=2,
        )
        if_nbr = np.roll(
            np.roll(np.roll(is_if, shift=-cz, axis=0), shift=-cy, axis=1),
            shift=-cx,
            axis=2,
        )
        c_nbr = np.roll(
            np.roll(np.roll(fill_level, shift=-cz, axis=0), shift=-cy, axis=1),
            shift=-cx,
            axis=2,
        )
        f_opp_nbr = np.roll(
            np.roll(np.roll(f[opp], shift=-cz, axis=0), shift=-cy, axis=1),
            shift=-cx,
            axis=2,
        )

        both = fluid & fluid_nbr
        w = np.zeros_like(fill_level)
        w[both & (is_liq | liq_nbr)] = 1.0
        ii = both & is_if & if_nbr
        w[ii] = 0.5 * (fill_level[ii] + c_nbr[ii])
        delta_m[i] = w * (f_opp_nbr - f[i]) * dt

    return delta_m


def reconstruct_gas_boundaries_3d(
    f_post_stream: np.ndarray,
    f_post_collide: np.ndarray,
    cell_types: np.ndarray,
    rho_gas: float = 1.0,
    u: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Reconstruct missing distribution functions entering interface cells from empty gas cells.

    .. math::
        f_{\bar{i}}(\mathbf{x}, t + \Delta t) = f_i^{\text{eq}}(\rho_{\text{gas}}, \mathbf{u}) + f_{\bar{i}}^{\text{eq}}(\rho_{\text{gas}}, \mathbf{u}) - f_i^*(\mathbf{x}, t)

    Parameters
    ----------
    f_post_stream : np.ndarray
        Streamed distribution functions of shape `(19, nz, ny, nx)`.
    f_post_collide : np.ndarray
        Post-collision distribution functions of shape `(19, nz, ny, nx)`.
    cell_types : np.ndarray
        Cell type classification of shape `(nz, ny, nx)`.
    rho_gas : float, optional
        Gas reference density, default 1.0.
    u : np.ndarray, optional
        Local fluid velocity field of shape `(3, nz, ny, nx)`.

    Returns
    -------
    np.ndarray
        Reconstructed distribution functions of shape `(19, nz, ny, nx)`.
    """
    f_recon = np.copy(f_post_stream)
    is_interface = (cell_types == CellType.INTERFACE)

    if not np.any(is_interface):
        return f_recon

    if u is None:
        u_field = np.zeros((3, *cell_types.shape), dtype=np.float64)
    else:
        u_field = u

    rho_field = np.full(cell_types.shape, rho_gas, dtype=np.float64)
    feq_gas = compute_equilibrium(rho_field, u_field)

    for i in range(1, 19):
        opp = OPPOSITE[i]
        cx, cy, cz = C[i]

        # Neighbor along direction +c_i
        nbr_type = np.roll(
            np.roll(np.roll(cell_types, shift=-cz, axis=0), shift=-cy, axis=1),
            shift=-cx,
            axis=2,
        )

        # Reconstruct missing population f[opp] arriving from a GAS neighbor
        recon_mask = is_interface & (nbr_type == CellType.GAS)

        # f_opp_recon = feq[opp] + feq[i] - f_post_collide[i]
        val = feq_gas[opp] + feq_gas[i] - f_post_collide[i]
        f_recon[opp, recon_mask] = val[recon_mask]

    return f_recon


def update_cell_states_3d(
    mass: np.ndarray,
    rho: np.ndarray,
    cell_types: np.ndarray,
    solid_mask: np.ndarray,
    delta_m: np.ndarray,
    conversion_hysteresis: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Update mass, compute new fill fractions, and execute topological cell transitions in 3D.

    Implements canonical 3D Körner / Thürey cell state transitions:
      1. Advect mass using anti-symmetric fluxes :math:`\Delta M_i`.
      2. Interface cells that overfill (:math:`M > (1+\kappa)\rho` or no gas neighbors) become LIQUID.
         All neighboring GAS cells become INTERFACE cells.
      3. Interface cells that empty (:math:`M < -\kappa\rho` or no fluid neighbors) become GAS.
         All neighboring LIQUID cells become INTERFACE cells.
      4. Excess mass is conserved and shared among active neighboring INTERFACE cells.
      5. Enforce neighborhood consistency (no LIQUID directly touches GAS).

    Parameters
    ----------
    mass : np.ndarray
        Current liquid mass in each cell of shape `(nz, ny, nx)`.
    rho : np.ndarray
        Current macroscopic density of shape `(nz, ny, nx)`.
    cell_types : np.ndarray
        Current cell types of shape `(nz, ny, nx)`.
    solid_mask : np.ndarray
        Solid obstacle mask of shape `(nz, ny, nx)`.
    delta_m : np.ndarray
        Directional mass fluxes of shape `(19, nz, ny, nx)`.
    conversion_hysteresis : float, optional
        Transition buffer threshold :math:`\kappa`, default 0.05.

    Returns
    -------
    new_mass : np.ndarray
        Updated mass field.
    new_fill : np.ndarray
        Updated fill fraction field :math:`C(\mathbf{x}) \in [0, 1]`.
    new_types : np.ndarray
        Updated cell types array.
    """
    net_flux = np.sum(delta_m, axis=0)

    new_mass = np.copy(mass)
    new_mass[~solid_mask] += net_flux[~solid_mask]
    new_mass[solid_mask] = 0.0

    safe_rho = np.where(rho > 1e-12, rho, 1.0)
    is_fluid = ((cell_types == CellType.LIQUID) | (cell_types == CellType.INTERFACE)) & (~solid_mask)
    safe_rho = np.where(is_fluid, safe_rho, 1.0)

    is_gas = (cell_types == CellType.GAS) & (~solid_mask)
    is_iface = (cell_types == CellType.INTERFACE) & (~solid_mask)
    is_liq = (cell_types == CellType.LIQUID) & (~solid_mask)

    # 1. Neighbor counts in D3Q19 stencil
    n_fluid = np.zeros_like(mass, dtype=np.int32)
    n_gas = np.zeros_like(mass, dtype=np.int32)

    for i in range(1, 19):
        cx, cy, cz = C[i]
        f_nbr = np.roll(np.roll(np.roll(is_fluid, shift=-cz, axis=0), shift=-cy, axis=1), shift=-cx, axis=2)
        g_nbr = np.roll(np.roll(np.roll(is_gas, shift=-cz, axis=0), shift=-cy, axis=1), shift=-cx, axis=2)
        n_fluid += f_nbr
        n_gas += g_nbr

    # 2. Conversion candidates
    kappa = float(conversion_hysteresis)
    emptied = is_iface & ((new_mass < -kappa * safe_rho) | (n_fluid == 0))
    filled = is_iface & ((new_mass > (1.0 + kappa) * safe_rho) | (n_gas == 0)) & (~emptied)

    # Avoid emptying next to a filling cell
    near_filled = np.zeros_like(filled, dtype=bool)
    for i in range(1, 19):
        cx, cy, cz = C[i]
        near_filled |= np.roll(np.roll(np.roll(filled, shift=-cz, axis=0), shift=-cy, axis=1), shift=-cx, axis=2)
    emptied &= ~near_filled

    # 3. Gas neighbors of filled cells become INTERFACE
    #    Liquid neighbors of emptied cells become INTERFACE
    new_if = np.zeros_like(filled, dtype=bool)
    liq_to_if = np.zeros_like(filled, dtype=bool)

    for i in range(1, 19):
        cx, cy, cz = C[i]
        fil_nbr = np.roll(np.roll(np.roll(filled, shift=-cz, axis=0), shift=-cy, axis=1), shift=-cx, axis=2)
        emp_nbr = np.roll(np.roll(np.roll(emptied, shift=-cz, axis=0), shift=-cy, axis=1), shift=-cx, axis=2)
        new_if |= fil_nbr & is_gas
        liq_to_if |= emp_nbr & is_liq

    new_types = np.copy(cell_types)
    new_types[new_if] = CellType.INTERFACE
    new_types[liq_to_if] = CellType.INTERFACE
    new_types[filled] = CellType.LIQUID
    new_types[emptied] = CellType.GAS

    # 4. Conservative excess mass distribution
    excess = np.zeros_like(new_mass, dtype=np.float64)
    excess[filled] = new_mass[filled] - safe_rho[filled]
    new_mass[filled] = safe_rho[filled]
    excess[emptied] = new_mass[emptied]
    new_mass[emptied] = 0.0

    donors = filled | emptied
    receivers = (new_types == CellType.INTERFACE) & (~donors) & (~solid_mask)
    n_recv = np.zeros_like(new_mass, dtype=np.int32)
    for i in range(1, 19):
        cx, cy, cz = C[i]
        n_recv += np.roll(np.roll(np.roll(receivers, shift=-cz, axis=0), shift=-cy, axis=1), shift=-cx, axis=2)

    orphan = donors & (n_recv == 0)
    if np.any(orphan):
        new_mass[orphan] += excess[orphan]
        excess[orphan] = 0.0
        new_types[orphan] = CellType.INTERFACE

    share = np.where(n_recv > 0, excess / np.maximum(n_recv, 1), 0.0)
    for i in range(1, 19):
        cx, cy, cz = C[i]
        sh_nbr = np.roll(np.roll(np.roll(share, shift=-cz, axis=0), shift=-cy, axis=1), shift=-cx, axis=2)
        new_mass += np.where(receivers, sh_nbr, 0.0)

    # 5. Enforce solid & neighborhood consistency
    new_types[solid_mask] = CellType.SOLID
    new_types = enforce_neighborhood_consistency_3d(new_types, solid_mask)
    new_mass[solid_mask] = 0.0

    # 6. Fill fractions
    new_fill = np.zeros_like(new_mass, dtype=np.float64)
    new_fill[new_types == CellType.LIQUID] = 1.0
    ifm = (new_types == CellType.INTERFACE) & (~solid_mask)
    new_fill[ifm] = np.clip(new_mass[ifm] / np.maximum(safe_rho[ifm], 1e-6), 0.0, 1.0)
    new_fill[new_types == CellType.GAS] = 0.0
    new_fill[solid_mask] = 0.0

    return new_mass, new_fill, new_types
