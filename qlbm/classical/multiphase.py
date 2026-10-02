"""Shan-Chen pseudopotential multiphase model for D2Q9 LBM.

This module implements the non-local inter-particle interaction force,
pseudopotential functions (exponential and Carnahan-Starling), non-ideal
equations of state, and the specialized `ShanChenMultiphaseSolver` class.
"""

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
