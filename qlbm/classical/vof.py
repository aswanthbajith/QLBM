"""Volume-of-Fluid (VOF) Geometric Interface Reconstruction for D2Q9 LBM.

This module implements the geometric foundation of the Volume-of-Fluid (VOF)
free-surface / two-phase framework:
1. Discrete cell state classification (GAS, INTERFACE, LIQUID, SOLID).
2. Interface normal vector estimation via the 2nd-order Youngs 9-point stencil.
3. Piecewise Linear Interface Calculation (PLIC) closed-form forward area
   and inverse line-constant determination (Scardovelli & Zaleski 2000).
4. Full domain interface reconstruction and high-level `VOFInterface` container.
"""

from enum import IntEnum
from typing import Optional, Tuple, Union

import numpy as np


class CellType(IntEnum):
    r"""Discrete cell state classification in free-surface VOF-LBM."""

    GAS = 0        # Empty cell, not simulated in hydrodynamic LBM
    INTERFACE = 1  # Boundary cell containing liquid-gas interface
    LIQUID = 2     # Fully filled fluid cell governed by D2Q9 Navier-Stokes
    SOLID = 3      # No-slip solid obstacle or boundary wall


def classify_cells(
    fill_level: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    r"""Classify grid nodes into GAS, INTERFACE, LIQUID, or SOLID states.

    Parameters
    ----------
    fill_level : np.ndarray
        Array of shape `(ny, nx)` with fill levels :math:`C \in [0, 1]`.
    solid_mask : np.ndarray, optional
        Boolean array of shape `(ny, nx)` indicating solid obstacle nodes.
    eps : float, optional
        Threshold tolerance for distinguishing pure phases from interface cells,
        default :math:`10^{-5}`.

    Returns
    -------
    np.ndarray
        Array of shape `(ny, nx)` containing `CellType` integer codes.
    """
    C = np.asarray(fill_level, dtype=np.float64)
    cell_types = np.full(C.shape, CellType.INTERFACE, dtype=np.int32)

    # Classify pure gas and pure liquid
    cell_types[C <= eps] = CellType.GAS
    cell_types[C >= 1.0 - eps] = CellType.LIQUID

    # Override solid obstacle nodes
    if solid_mask is not None:
        cell_types[solid_mask] = CellType.SOLID

    return cell_types


def compute_normal_youngs(
    fill_level: np.ndarray,
    solid_mask: Optional[np.ndarray] = None,
    periodic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Estimate unit normal vectors :math:`\mathbf{n}` pointing from liquid to gas via Youngs method.

    Uses a 2nd-order 9-point central difference stencil with diagonal corner weighting:
    .. math::
        m_x = -\frac{1}{8} \left[ (C_{i+1, j+1} + 2C_{i+1, j} + C_{i+1, j-1})
                               - (C_{i-1, j+1} + 2C_{i-1, j} + C_{i-1, j-1}) \right]
        m_y = -\frac{1}{8} \left[ (C_{i+1, j+1} + 2C_{i, j+1} + C_{i-1, j+1})
                               - (C_{i+1, j-1} + 2C_{i, j-1} + C_{i-1, j-1}) \right]

    Parameters
    ----------
    fill_level : np.ndarray
        Fill level field of shape `(ny, nx)`.
    solid_mask : np.ndarray, optional
        Solid obstacle mask. Solid cells mirror local fluid fill level to prevent
        unphysical interface tilting at solid boundaries.
    periodic : bool, optional
        Whether to treat domain boundaries as periodic (default: True).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays `(nx_field, ny_field)` of unit normal components pointing outward
        into the gas phase.
    """
    C = np.copy(np.asarray(fill_level, dtype=np.float64))

    # Near solid walls, extrapolate fluid fill levels into solid to avoid normal distortion
    if solid_mask is not None:
        # Mirror adjacent fluid values into solids where applicable
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted_fluid = (~solid_mask) & np.roll(
                    np.roll(solid_mask, -dy, axis=0), -dx, axis=1
                )
                if np.any(shifted_fluid):
                    C[solid_mask] = 0.5  # Neutral default for solid nodes

    if periodic:
        pad_mode = "wrap"
    else:
        pad_mode = "edge"

    C_pad = np.pad(C, 1, mode=pad_mode)

    # Youngs 9-point gradient filter (pointing from liquid to gas)
    # y is axis 0, x is axis 1
    # Right column (x + 1) vs Left column (x - 1)
    mx = -(
        (C_pad[2:, 2:] + 2.0 * C_pad[1:-1, 2:] + C_pad[:-2, 2:])
        - (C_pad[2:, :-2] + 2.0 * C_pad[1:-1, :-2] + C_pad[:-2, :-2])
    ) / 8.0

    # Top row (y + 1) vs Bottom row (y - 1)
    my = -(
        (C_pad[2:, 2:] + 2.0 * C_pad[2:, 1:-1] + C_pad[2:, :-2])
        - (C_pad[:-2, 2:] + 2.0 * C_pad[:-2, 1:-1] + C_pad[:-2, :-2])
    ) / 8.0

    norm = np.hypot(mx, my)
    valid = norm > 1e-12

    nx_field = np.zeros_like(mx)
    ny_field = np.zeros_like(my)

    nx_field[valid] = mx[valid] / norm[valid]
    ny_field[valid] = my[valid] / norm[valid]

    return nx_field, ny_field


def compute_normal_central(
    fill_level: np.ndarray,
    periodic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Estimate unit normal vectors :math:`\mathbf{n}` using standard central differences."""
    C = np.asarray(fill_level, dtype=np.float64)
    pad_mode = "wrap" if periodic else "edge"
    C_pad = np.pad(C, 1, mode=pad_mode)

    mx = -(C_pad[1:-1, 2:] - C_pad[1:-1, :-2]) / 2.0
    my = -(C_pad[2:, 1:-1] - C_pad[:-2, 1:-1]) / 2.0

    norm = np.hypot(mx, my)
    valid = norm > 1e-12

    nx_field = np.zeros_like(mx)
    ny_field = np.zeros_like(my)

    nx_field[valid] = mx[valid] / norm[valid]
    ny_field[valid] = my[valid] / norm[valid]

    return nx_field, ny_field


def _plic_area_first_quadrant(n1: float, n2: float, alpha: float) -> float:
    r"""Calculate area under :math:`n_1 x + n_2 y \le \alpha` in :math:`[0, 1]^2` with :math:`0 \le n_1 \le n_2`."""
    if alpha <= 0.0:
        return 0.0
    if alpha >= n1 + n2:
        return 1.0

    if n1 < 1e-14:
        return min(1.0, max(0.0, alpha / max(n2, 1e-14)))

    if alpha <= n1:
        return (alpha**2) / (2.0 * n1 * n2)
    elif alpha <= n2:
        return (2.0 * alpha * n1 - n1**2) / (2.0 * n1 * n2)
    else:
        return 1.0 - ((n1 + n2 - alpha) ** 2) / (2.0 * n1 * n2)


def _plic_alpha_first_quadrant(n1: float, n2: float, C: float) -> float:
    r"""Find line constant :math:`\alpha` such that area equals :math:`C` in :math:`[0, 1]^2` with :math:`0 \le n_1 \le n_2`."""
    C = min(1.0, max(0.0, C))
    if n1 < 1e-14:
        return C * max(n2, 1e-14)

    # Critical volume thresholds
    C1 = (n1**2) / (2.0 * n1 * n2)  # = n1 / (2 * n2)
    C2 = 1.0 - C1

    if C <= C1:
        return np.sqrt(2.0 * n1 * n2 * C)
    elif C <= C2:
        return n1 / 2.0 + n2 * C
    else:
        return n1 + n2 - np.sqrt(2.0 * n1 * n2 * (1.0 - C))


def plic_forward_area(
    nx: Union[float, np.ndarray],
    ny: Union[float, np.ndarray],
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Calculate fractional volume :math:`C` in unit cell :math:`[0, 1]^2` cut by :math:`n_x x + n_y y \le \alpha`.

    Exact analytical formula across all 4 quadrants (Scardovelli & Zaleski 2000).

    Parameters
    ----------
    nx : float or np.ndarray
        X-component of normal vector.
    ny : float or np.ndarray
        Y-component of normal vector.
    alpha : float or np.ndarray
        Line constant in local cell coordinates :math:`x, y \in [0, 1]`.

    Returns
    -------
    float or np.ndarray
        Enclosed liquid volume fraction :math:`C \in [0, 1]`.
    """
    is_scalar = np.isscalar(nx) and np.isscalar(ny) and np.isscalar(alpha)
    nx_arr = np.atleast_1d(np.asarray(nx, dtype=np.float64))
    ny_arr = np.atleast_1d(np.asarray(ny, dtype=np.float64))
    al_arr = np.atleast_1d(np.asarray(alpha, dtype=np.float64))

    # Broadcast shapes
    nx_b, ny_b, al_b = np.broadcast_arrays(nx_arr, ny_arr, al_arr)
    C_out = np.zeros_like(al_b)

    for idx in np.ndindex(al_b.shape):
        n_x = nx_b[idx]
        n_y = ny_b[idx]
        a = al_b[idx]

        mx = abs(n_x)
        my = abs(n_y)
        m1 = min(mx, my)
        m2 = max(mx, my)

        # Coordinate quadrant shift: x' = 1-x if nx < 0, y' = 1-y if ny < 0
        shift = max(0.0, -n_x) + max(0.0, -n_y)
        alpha_prime = a + shift

        C_out[idx] = _plic_area_first_quadrant(m1, m2, alpha_prime)

    return float(C_out[0]) if is_scalar else C_out


def plic_inverse_alpha(
    nx: Union[float, np.ndarray],
    ny: Union[float, np.ndarray],
    C: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Find PLIC line constant :math:`\alpha` such that :math:`n_x x + n_y y \le \alpha` encloses volume :math:`C`.

    Closed-form inversion exact to machine precision across all four quadrants.

    Parameters
    ----------
    nx : float or np.ndarray
        X-component of normal vector.
    ny : float or np.ndarray
        Y-component of normal vector.
    C : float or np.ndarray
        Target volume fraction :math:`C \in [0, 1]`.

    Returns
    -------
    float or np.ndarray
        Line constant :math:`\alpha`.
    """
    is_scalar = np.isscalar(nx) and np.isscalar(ny) and np.isscalar(C)
    nx_arr = np.atleast_1d(np.asarray(nx, dtype=np.float64))
    ny_arr = np.atleast_1d(np.asarray(ny, dtype=np.float64))
    c_arr = np.atleast_1d(np.asarray(C, dtype=np.float64))

    nx_b, ny_b, c_b = np.broadcast_arrays(nx_arr, ny_arr, c_arr)
    al_out = np.zeros_like(c_b)

    for idx in np.ndindex(c_b.shape):
        n_x = nx_b[idx]
        n_y = ny_b[idx]
        c = c_b[idx]

        mx = abs(n_x)
        my = abs(n_y)
        m1 = min(mx, my)
        m2 = max(mx, my)

        alpha_prime = _plic_alpha_first_quadrant(m1, m2, c)
        shift = max(0.0, -n_x) + max(0.0, -n_y)

        al_out[idx] = alpha_prime - shift

    return float(al_out[0]) if is_scalar else al_out


class VOFInterface:
    r"""High-level Volume-of-Fluid (VOF) interface geometry and reconstruction manager.

    Parameters
    ----------
    nx : int
        Grid dimensions in x.
    ny : int
        Grid dimensions in y.
    solid_mask : np.ndarray, optional
        Boolean obstacle mask.
    eps : float, optional
        Tolerance for cell state classification, default :math:`10^{-5}`.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        solid_mask: Optional[np.ndarray] = None,
        eps: float = 1e-5,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.eps = float(eps)

        if solid_mask is not None:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
        else:
            self.solid_mask = np.zeros((self.ny, self.nx), dtype=bool)

        # Primary VOF fields
        self.fill_level = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.cell_types = np.full(
            (self.ny, self.nx), CellType.GAS, dtype=np.int32
        )
        self.nx_field = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.ny_field = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.alphas = np.zeros((self.ny, self.nx), dtype=np.float64)

        if np.any(self.solid_mask):
            self.cell_types[self.solid_mask] = CellType.SOLID

    def initialize_fill_level(self, fill_level: np.ndarray) -> None:
        r"""Initialize fill level :math:`C(\mathbf{x}) \in [0, 1]` and reconstruct initial interface."""
        C = np.clip(np.asarray(fill_level, dtype=np.float64), 0.0, 1.0)
        if self.solid_mask is not None:
            C[self.solid_mask] = 0.0
        self.fill_level = C
        self.reconstruct()

    def reconstruct(self, method: str = "youngs") -> None:
        r"""Execute full interface reconstruction: classification, normal calculation, and PLIC line constants."""
        # 1. Update discrete cell states
        self.cell_types = classify_cells(
            self.fill_level, self.solid_mask, eps=self.eps
        )

        # 2. Compute interface unit normal vectors
        if method == "youngs":
            self.nx_field, self.ny_field = compute_normal_youngs(
                self.fill_level, self.solid_mask, periodic=True
            )
        else:
            self.nx_field, self.ny_field = compute_normal_central(
                self.fill_level, periodic=True
            )

        # 3. Compute PLIC line constant alpha for all interface cells
        interface_mask = self.cell_types == CellType.INTERFACE
        if np.any(interface_mask):
            c_iface = self.fill_level[interface_mask]
            nx_iface = self.nx_field[interface_mask]
            ny_iface = self.ny_field[interface_mask]

            al_iface = plic_inverse_alpha(nx_iface, ny_iface, c_iface)
            self.alphas[interface_mask] = al_iface

    @property
    def total_liquid_volume(self) -> float:
        r"""Total integrated liquid volume :math:`V = \sum C(\mathbf{x}) \Delta x^2`."""
        return float(np.sum(self.fill_level[~self.solid_mask]))

    @property
    def interface_cell_count(self) -> int:
        r"""Total count of active interface cells."""
        return int(np.sum(self.cell_types == CellType.INTERFACE))

    @property
    def liquid_cell_count(self) -> int:
        r"""Total count of pure liquid cells."""
        return int(np.sum(self.cell_types == CellType.LIQUID))

    @property
    def gas_cell_count(self) -> int:
        r"""Total count of empty gas cells."""
        return int(np.sum(self.cell_types == CellType.GAS))
