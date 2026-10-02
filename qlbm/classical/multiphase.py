"""Shan-Chen pseudopotential multiphase model for D2Q9 LBM.

This module implements the non-local inter-particle interaction force,
pseudopotential functions (exponential and Carnahan-Starling), non-ideal
equations of state, and the specialized `ShanChenMultiphaseSolver` class.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from qlbm.classical.d2q9 import C, CS_SQ, WEIGHTS
from qlbm.classical.solver import D2Q9BGKSolver


def compute_pseudopotential(
    rho: np.ndarray,
    form: str = "exponential",
    rho_0: float = 1.0,
    **kwargs,
) -> np.ndarray:
    r"""Compute the effective pseudopotential \psi(\rho).

    Parameters
    ----------
    rho : np.ndarray
        Density field of shape `(ny, nx)` or scalar.
    form : str, optional
        Form of the pseudopotential: 'exponential' (default) or 'carnahan_starling'.
    rho_0 : float, optional
        Reference density scale for exponential potential, default 1.0.
    kwargs : dict
        Additional parameters for non-ideal equations of state (a, b, R, T, G).

    Returns
    -------
    np.ndarray
        Pseudopotential field \psi(\rho) of same shape as rho.
    """
    rho_arr = np.asarray(rho, dtype=np.float64)

    if form == "exponential":
        # Classical Shan & Chen (1993): psi(rho) = rho_0 * [1 - exp(-rho / rho_0)]
        return rho_0 * (1.0 - np.exp(-rho_arr / rho_0))

    elif form == "carnahan_starling":
        # Carnahan-Starling real-gas equation of state:
        # p_CS = rho * R * T * (1 + eta + eta^2 - eta^3) / (1 - eta)^3 - a * rho^2
        # where eta = b * rho / 4
        a = kwargs.get("a", 1.0)
        b = kwargs.get("b", 4.0)
        R = kwargs.get("R", 1.0)
        T = kwargs.get("T", 0.09)
        G = kwargs.get("G", -1.0)

        eta = b * rho_arr / 4.0
        # Protect against non-physical packing
        eta = np.clip(eta, 0.0, 0.99)
        p_CS = (
            rho_arr * R * T * (1.0 + eta + eta**2 - eta**3) / ((1.0 - eta) ** 3)
            - a * (rho_arr**2)
        )

        # Inversion: psi = sqrt(2 * (p_CS - c_s^2 * rho) / (c_s^2 * G))
        delta_p = p_CS - CS_SQ * rho_arr
        radicand = 2.0 * delta_p / (CS_SQ * G)
        radicand = np.maximum(radicand, 0.0)
        return np.sqrt(radicand)

    else:
        raise ValueError(
            f"Unknown pseudopotential form '{form}'. Supported: 'exponential', 'carnahan_starling'."
        )


def compute_shan_chen_force(
    psi: np.ndarray,
    G: float = -5.0,
) -> np.ndarray:
    r"""Compute the inter-particle Shan-Chen interaction force field.

    .. math::
        \mathbf{F}_{\text{SC}}(\mathbf{x}) = -G \psi(\mathbf{x}) \sum_{i=1}^8 w_i \psi(\mathbf{x} + \mathbf{c}_i \Delta t) \mathbf{c}_i

    Parameters
    ----------
    psi : np.ndarray
        Pseudopotential field of shape `(ny, nx)`.
    G : float, optional
        Interaction strength parameter. Attractive forces require :math:`G < 0`, default -5.0.

    Returns
    -------
    np.ndarray
        Interaction force vector field of shape `(2, ny, nx)`.
        `F[0]` is F_x and `F[1]` is F_y.
    """
    F_sc = np.zeros((2, *psi.shape), dtype=np.float64)

    for i in range(1, 9):
        cix = C[i, 0]
        ciy = C[i, 1]
        # Neighbor at x + c_i: roll with (-ciy, -cix)
        psi_neighbor = np.roll(psi, shift=(-ciy, -cix), axis=(0, 1))
        F_sc[0] += WEIGHTS[i] * psi_neighbor * cix
        F_sc[1] += WEIGHTS[i] * psi_neighbor * ciy

    prefactor = -G * psi
    F_sc[0] *= prefactor
    F_sc[1] *= prefactor

    return F_sc


def shan_chen_pressure(
    rho: np.ndarray,
    G: float = -5.0,
    rho_0: float = 1.0,
) -> np.ndarray:
    r"""Compute non-ideal Shan-Chen equation of state pressure.

    .. math::
        p(\rho) = c_s^2 \rho + \frac{c_s^2 G}{2} \psi^2(\rho) = \frac{1}{3} \rho + \frac{G}{6} \psi^2(\rho)

    Parameters
    ----------
    rho : np.ndarray
        Density field.
    G : float, optional
        Interaction parameter, default -5.0.
    rho_0 : float, optional
        Reference density, default 1.0.

    Returns
    -------
    np.ndarray
        Thermodynamic pressure field.
    """
    psi = compute_pseudopotential(rho, form="exponential", rho_0=rho_0)
    return CS_SQ * rho + 0.5 * CS_SQ * G * (psi**2)


class ShanChenMultiphaseSolver(D2Q9BGKSolver):
    r"""Two-dimensional multiphase Lattice Boltzmann solver using the Shan-Chen model.

    Couples the D2Q9 hydrodynamic engine with the non-local inter-particle
    pseudopotential interaction force :math:`\mathbf{F}_{\text{SC}}`.

    Parameters
    ----------
    nx : int
        Grid size in x.
    ny : int
        Grid size in y.
    tau : float, optional
        Relaxation time :math:`\tau > 0.5`, default 1.0.
    G : float, optional
        Shan-Chen interaction strength. :math:`G < -4.0` triggers spontaneous phase separation.
    form : str, optional
        Pseudopotential form ('exponential' or 'carnahan_starling'), default 'exponential'.
    rho_0 : float, optional
        Reference density scale, default 1.0.
    gravity : float or tuple, optional
        Gravitational acceleration, default None.
    solid_mask : np.ndarray of bool, optional
        Obstacle mask, default None.
    dt : float, optional
        Time step size, default 1.0.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        tau: float = 1.0,
        G: float = -5.0,
        form: str = "exponential",
        rho_0: float = 1.0,
        gravity: Optional[Union[float, Tuple[float, float]]] = None,
        solid_mask: Optional[np.ndarray] = None,
        dt: float = 1.0,
    ) -> None:
        self.G = float(G)
        self.form = form
        self.rho_0 = float(rho_0)

        # Dynamic force callback integrating Shan-Chen interaction force
        def total_interaction_force(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
            psi_field = compute_pseudopotential(
                rho, form=self.form, rho_0=self.rho_0
            )
            return compute_shan_chen_force(psi_field, G=self.G)

        super().__init__(
            nx=nx,
            ny=ny,
            tau=tau,
            solid_mask=solid_mask,
            force=total_interaction_force,
            gravity=gravity,
            dt=dt,
        )

    @property
    def psi(self) -> np.ndarray:
        r"""Current pseudopotential field :math:`\psi(\rho)`."""
        return compute_pseudopotential(
            self.rho, form=self.form, rho_0=self.rho_0
        )

    @property
    def interaction_force(self) -> np.ndarray:
        r"""Current Shan-Chen interaction force field :math:`\mathbf{F}_{\text{SC}}`."""
        return compute_shan_chen_force(self.psi, G=self.G)

    @property
    def liquid_density(self) -> float:
        r"""Maximum density in fluid domain (representative of liquid phase)."""
        return float(np.max(self.rho[self.fluid_mask]))

    @property
    def vapor_density(self) -> float:
        r"""Minimum density in fluid domain (representative of vapor phase)."""
        return float(np.min(self.rho[self.fluid_mask]))

    @property
    def density_ratio(self) -> float:
        r"""Coexisting density ratio :math:`\rho_L / \rho_G`."""
        vap = self.vapor_density
        if vap < 1e-12:
            return float("inf")
        return self.liquid_density / vap

    def initialize_droplet(
        self,
        radius: float,
        xc: Optional[float] = None,
        yc: Optional[float] = None,
        rho_liquid: float = 1.9,
        rho_vapor: float = 0.12,
        interface_width: float = 1.5,
    ) -> None:
        r"""Initialize a 2D circular droplet centered in the domain."""
        initialize_circular_droplet(
            self,
            radius=radius,
            xc=xc,
            yc=yc,
            rho_liquid=rho_liquid,
            rho_vapor=rho_vapor,
            interface_width=interface_width,
        )

    def measure_droplet(
        self,
        xc: Optional[float] = None,
        yc: Optional[float] = None,
    ) -> dict:
        r"""Measure geometric, thermodynamic, and spurious velocity properties of the droplet."""
        return measure_droplet_properties(self, xc=xc, yc=yc)


@dataclass
class LaplaceFitResult:
    r"""Results of linear regression for the 2D Young-Laplace law :math:`\Delta p = \sigma / R + c_0`."""

    sigma: float
    intercept: float
    r_squared: float
    radii: np.ndarray
    delta_pressures: np.ndarray


def initialize_circular_droplet(
    solver: ShanChenMultiphaseSolver,
    radius: float,
    xc: Optional[float] = None,
    yc: Optional[float] = None,
    rho_liquid: float = 1.9,
    rho_vapor: float = 0.12,
    interface_width: float = 1.5,
) -> None:
    r"""Initialize a 2D circular liquid droplet in a surrounding vapor phase.

    Parameters
    ----------
    solver : ShanChenMultiphaseSolver
        Target multiphase solver instance.
    radius : float
        Nominal droplet radius in lattice units.
    xc : float, optional
        Center x-coordinate. Default: domain midpoint `nx / 2.0`.
    yc : float, optional
        Center y-coordinate. Default: domain midpoint `ny / 2.0`.
    rho_liquid : float, optional
        Initial dense phase density, default 1.9.
    rho_vapor : float, optional
        Initial dilute phase density, default 0.12.
    interface_width : float, optional
        Diffuse interface transition thickness, default 1.5.
    """
    nx, ny = solver.nx, solver.ny
    if xc is None:
        xc = nx / 2.0
    if yc is None:
        yc = ny / 2.0

    y, x = np.mgrid[0:ny, 0:nx]
    # Periodic distance to center
    dx = np.minimum(np.abs(x - xc), nx - np.abs(x - xc))
    dy = np.minimum(np.abs(y - yc), ny - np.abs(y - yc))
    r = np.sqrt(dx**2 + dy**2)

    # Smooth hyperbolic tangent profile
    rho_init = rho_vapor + 0.5 * (rho_liquid - rho_vapor) * (
        1.0 - np.tanh((r - radius) / max(interface_width, 1e-6))
    )
    solver.initialize(rho=rho_init)


def measure_droplet_properties(
    solver: ShanChenMultiphaseSolver,
    xc: Optional[float] = None,
    yc: Optional[float] = None,
) -> dict:
    r"""Measure geometric and thermodynamic properties of a relaxed stationary droplet.

    Parameters
    ----------
    solver : ShanChenMultiphaseSolver
        Solver instance containing relaxed droplet state.
    xc : float, optional
        Center x-coordinate. Default `nx / 2.0`.
    yc : float, optional
        Center y-coordinate. Default `ny / 2.0`.

    Returns
    -------
    dict
        Dictionary containing:
        - 'rho_in': density at droplet center
        - 'rho_out': density in far-field vapor (corner)
        - 'p_in': thermodynamic pressure at center
        - 'p_out': thermodynamic pressure in vapor
        - 'delta_p': Laplace pressure jump p_in - p_out
        - 'effective_radius': R_eff = sqrt(Area / pi)
        - 'u_max': maximum spurious velocity magnitude
        - 'u_bulk_in': velocity magnitude at droplet center
        - 'u_bulk_out': velocity magnitude at domain corner
    """
    nx, ny = solver.nx, solver.ny
    if xc is None:
        xc = nx / 2.0
    if yc is None:
        yc = ny / 2.0

    ix = int(round(xc)) % nx
    iy = int(round(yc)) % ny

    rho_in = float(solver.rho[iy, ix])
    rho_out = float(solver.rho[0, 0])

    p_in = float(shan_chen_pressure(rho_in, G=solver.G, rho_0=solver.rho_0))
    p_out = float(shan_chen_pressure(rho_out, G=solver.G, rho_0=solver.rho_0))
    delta_p = p_in - p_out

    # Effective radius from area of nodes above midpoint density
    rho_mid = 0.5 * (rho_in + rho_out)
    droplet_mask = solver.rho > rho_mid
    area = float(np.sum(droplet_mask))
    effective_radius = float(np.sqrt(area / np.pi))

    u_mag = np.sqrt(solver.u[0] ** 2 + solver.u[1] ** 2)
    u_max = float(np.max(u_mag))
    u_bulk_in = float(u_mag[iy, ix])
    u_bulk_out = float(u_mag[0, 0])

    return {
        "rho_in": rho_in,
        "rho_out": rho_out,
        "p_in": p_in,
        "p_out": p_out,
        "delta_p": delta_p,
        "effective_radius": effective_radius,
        "u_max": u_max,
        "u_bulk_in": u_bulk_in,
        "u_bulk_out": u_bulk_out,
    }


def fit_laplace_law(
    radii: Union[list, np.ndarray],
    delta_pressures: Union[list, np.ndarray],
) -> LaplaceFitResult:
    r"""Perform linear regression on 2D Young-Laplace relation :math:`\Delta p = \sigma \cdot (1/R) + c_0`.

    Parameters
    ----------
    radii : list or np.ndarray
        List of droplet effective radii :math:`R`.
    delta_pressures : list or np.ndarray
        Corresponding pressure differences :math:`\Delta p = p_{\text{in}} - p_{\text{out}}`.

    Returns
    -------
    LaplaceFitResult
        Fitted surface tension :math:`\sigma`, intercept, and :math:`R^2`.
    """
    r_arr = np.asarray(radii, dtype=np.float64)
    dp_arr = np.asarray(delta_pressures, dtype=np.float64)

    if len(r_arr) < 2:
        raise ValueError("At least 2 data points required for Laplace linear fit.")

    inv_r = 1.0 / r_arr
    slope, intercept = np.polyfit(inv_r, dp_arr, 1)

    corr = np.corrcoef(inv_r, dp_arr)[0, 1]
    r_squared = float(corr**2)

    return LaplaceFitResult(
        sigma=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        radii=r_arr,
        delta_pressures=dp_arr,
    )

