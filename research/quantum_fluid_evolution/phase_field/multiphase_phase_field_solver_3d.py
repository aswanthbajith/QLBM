"""3D Multiphase Quantum Phase-Field Solver (Conservative Allen-Cahn + D3Q19/D3Q7).

Couples hydrodynamic Navier-Stokes evolution with Conservative Allen-Cahn (CAC)
diffuse interface tracking:
1. Hydrodynamics: D3Q19 lattice distribution f_i(x, t) for fluid momentum and velocity u.
2. Phase-Field:   D3Q7 lattice distribution g_i(x, t) for order parameter phi in [-1, +1]
   (phi > 0: liquid/water, phi < 0: gas/air, phi = 0: free-surface interface).

Mathematical Formulation:
-------------------------
- Order parameter:       phi(x) = sum_{i=0}^6 g_i(x)
- CAC equilibrium:      g_i^eq = w_i * phi + w_i * (c_i . (phi * u)) / c_sg^2
- Normal vector:        n = grad(phi) / (|grad(phi)| + eps)
- Curvature:            kappa = div(n)
- Surface tension force: F_s = sigma * kappa * grad(phi)  (Continuum Surface Force)
- Gravity body force:   F_g = rho(phi) * g, with rho(phi) = (rho_L + rho_G)/2 + (rho_L - rho_G)/2 * phi
- Total force:          F_total = F_s + F_g
- D3Q19 Hydrodynamics:  collide f_i with relaxation tau_f, source F_total, and stream across periodic/wall domain.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C as C_D3Q19, CS_SQ, OPPOSITE as OPP_D3Q19, WEIGHTS as W_D3Q19
from research.quantum_fluid_evolution.phase_field.quantum_phase_field_prototype import (
    C_D3Q7,
    CS2_D3Q7,
    WEIGHTS_D3Q7,
    QuantumPhaseFieldPrototype,
)

# Opposite indices for D3Q7:
# 0 -> 0, 1 (+x) <-> 2 (-x), 3 (+y) <-> 4 (-y), 5 (+z) <-> 6 (-z)
OPP_D3Q7 = np.array([0, 2, 1, 4, 3, 6, 5], dtype=np.int64)


class MultiphasePhaseFieldSolver3D:
    """3D Multiphase Fluid Solver coupling D3Q19 hydrodynamics with D3Q7 Conservative Allen-Cahn."""

    def __init__(
        self,
        nx: int = 16,
        ny: int = 8,
        nz: int = 16,
        tau_f: float = 0.8,
        tau_phi: float = 0.8,
        sigma: float = 0.02,
        interface_width: float = 1.2,
        rho_liquid: float = 1.0,
        rho_gas: float = 0.1,
        g: Tuple[float, float, float] = (0.0, 0.0, -0.005),
        solid_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.tau_f = float(tau_f)
        self.tau_phi = float(tau_phi)
        self.sigma = float(sigma)
        self.xi = float(interface_width)
        self.rho_L = float(rho_liquid)
        self.rho_G = float(rho_gas)
        self.g = np.asarray(g, dtype=np.float64)

        if solid_mask is None:
            self.solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        else:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)

        # Quantum Phase-Field prototype helper for block-encoding
        self.q_prototype = QuantumPhaseFieldPrototype(
            nx=min(self.nx, 4), ny=min(self.ny, 4), nz=min(self.nz, 8),
            tau_phi=self.tau_phi,
        )

        # Initialize distribution arrays
        self.f = np.zeros((19, self.nz, self.ny, self.nx), dtype=np.float64)
        self.g_dist = np.zeros((7, self.nz, self.ny, self.nx), dtype=np.float64)

    # -----------------------------------------------------------------
    # Macroscopic Moment Calculations
    # -----------------------------------------------------------------

    def compute_phi(self, g_arr: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute order parameter phi = sum_i g_i in [-1, 1]."""
        if g_arr is None:
            g_arr = self.g_dist
        return np.sum(g_arr, axis=0)

    def compute_density(self, phi: np.ndarray) -> np.ndarray:
        """Compute smooth physical density field rho(phi)."""
        return 0.5 * (self.rho_L + self.rho_G) + 0.5 * (self.rho_L - self.rho_G) * phi

    def compute_velocity(
        self, f_arr: Optional[np.ndarray] = None, force: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute macroscopic density rho and velocity u with force correction."""
        if f_arr is None:
            f_arr = self.f

        # Hydrodynamic density is strictly conserved sum of populations
        rho = np.sum(f_arr, axis=0)

        # Momentum: rho * u = sum(f_i * c_i) + 0.5 * force
        mom = np.tensordot(C_D3Q19, f_arr, axes=([0], [0]))  # (3, nz, ny, nx)

        if force is not None:
            mom += 0.5 * force

        u = np.zeros_like(mom)
        fluid_mask = ~self.solid_mask
        for d in range(3):
            u[d, fluid_mask] = mom[d, fluid_mask] / np.maximum(rho[fluid_mask], 1e-6)

        return rho, u

    # -----------------------------------------------------------------
    # Gradient, Normal & Curvature Stencils
    # -----------------------------------------------------------------

    def compute_gradient(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D central difference gradient: grad = (grad_x, grad_y, grad_z)."""
        gz = 0.5 * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))
        gy = 0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1))
        gx = 0.5 * (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2))
        return np.stack([gx, gy, gz], axis=0)

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D isotropic discrete Laplacian."""
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) - 6.0 * field
        )

    def compute_total_force(self, phi: np.ndarray) -> np.ndarray:
        """Compute combined interfacial surface tension and gravity force.

        Uses the smooth, unconditionally non-singular chemical potential
        formulation: F_s = mu * grad(phi), where:
          mu = (3*sigma / (2*xi)) * (phi^3 - phi) - (3*sigma*xi / 4) * laplacian(phi)
        """
        phi_clip = np.clip(phi, -1.0, 1.0)
        A_pot = 3.0 * self.sigma / (2.0 * self.xi)
        kappa_pot = 3.0 * self.sigma * self.xi / 4.0

        lap_phi = self.compute_laplacian(phi_clip)
        mu_phi = A_pot * (phi_clip**3 - phi_clip) - kappa_pot * lap_phi
        grad_phi = self.compute_gradient(phi_clip)

        # Continuum surface tension force: F_s = mu * grad(phi)
        F_s = mu_phi[None, :, :, :] * grad_phi

        # Gravity: F_g = rho(phi) * g
        rho = self.compute_density(phi_clip)
        F_g = rho[None, :, :, :] * self.g[:, None, None, None]

        F_total = F_s + F_g
        # Zero out force inside solid obstacles
        F_total[:, self.solid_mask] = 0.0
        return F_total

    # -----------------------------------------------------------------
    # Conservative Allen-Cahn (CAC) Phase-Field Step
    # -----------------------------------------------------------------

    def step_phase_field(self, u: np.ndarray) -> np.ndarray:
        """Advance phase-field distributions g_i by one collide-stream step."""
        phi = self.compute_phi()
        phi_clip = np.clip(phi, -1.0, 1.0)

        # Robust interface normal vector: n = grad(phi) / |grad(phi)|
        grad_phi = self.compute_gradient(phi_clip)
        grad_norm = np.sqrt(np.sum(grad_phi**2, axis=0))
        # Mask strictly at true interface to prevent spurious nucleation in ambient air
        mask_interface = (grad_norm > 0.02) & (np.abs(phi_clip) < 0.95) & (~self.solid_mask)

        unit_n = np.zeros_like(grad_phi)
        for d in range(3):
            unit_n[d, mask_interface] = grad_phi[d, mask_interface] / grad_norm[mask_interface]

        # Equilibrium distribution: g_i^eq = w_i * phi + w_i * (c_i . (phi * u)) / c_s^2
        cu = np.tensordot(C_D3Q7, u, axes=([1], [0]))  # (7, nz, ny, nx)
        cn = np.tensordot(C_D3Q7, unit_n, axes=([1], [0]))  # (7, nz, ny, nx)

        g_eq = WEIGHTS_D3Q7[:, None, None, None] * (
            phi[None, :, :, :] + (phi[None, :, :, :] * cu) / CS2_D3Q7
        )

        # Counter-diffusion source term restoring hyperbolic-tangent profile:
        # S_i = (1 - 1/(2*tau_phi)) * w_i * (c_i . n) * (1 - phi_clip^2) / (2 * xi * c_s^2)
        source_factor = (1.0 - 0.5 / self.tau_phi) * (1.0 - phi_clip**2) / (2.0 * self.xi * CS2_D3Q7)
        source_factor[~mask_interface] = 0.0
        S_i = WEIGHTS_D3Q7[:, None, None, None] * cn * source_factor[None, :, :, :]

        # Collision: g* = g - (g - g_eq)/tau_phi + S_i
        g_star = self.g_dist - (self.g_dist - g_eq) / self.tau_phi + S_i

        # Streaming on D3Q7 lattice
        g_next = np.zeros_like(self.g_dist)
        for i in range(7):
            cx, cy, cz = C_D3Q7[i]
            # Stream periodic fluid
            streamed_i = np.roll(np.roll(np.roll(g_star[i], cz, axis=0), cy, axis=1), cx, axis=2)
            g_next[i] = streamed_i

        # True halfway bounce-back on solid boundary interfaces
        fluid_mask = ~self.solid_mask
        for i in range(1, 7):
            cx, cy, cz = C_D3Q7[i]
            opp = OPP_D3Q7[i]
            nbr_solid = np.roll(np.roll(np.roll(self.solid_mask, -cz, axis=0), -cy, axis=1), -cx, axis=2)
            bounce_mask = fluid_mask & nbr_solid
            g_next[opp, bounce_mask] = g_star[i, bounce_mask]

        g_next[:, self.solid_mask] = WEIGHTS_D3Q7[:, None] * (-1.0)
        self.g_dist = g_next
        return self.compute_phi()

    # -----------------------------------------------------------------
    # D3Q19 Hydrodynamic Step
    # -----------------------------------------------------------------

    def step_hydrodynamics(self, force: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Advance hydrodynamic distributions f_i by one collide-stream step with Guo forcing."""
        phi = self.compute_phi()
        rho, u = self.compute_velocity(self.f, force=force)

        # D3Q19 Equilibrium: f_eq
        cu = np.tensordot(C_D3Q19, u, axes=([1], [0]))  # (19, nz, ny, nx)
        u_sq = np.sum(u**2, axis=0)  # (nz, ny, nx)

        feq = W_D3Q19[:, None, None, None] * rho[None, :, :, :] * (
            1.0 + cu / CS_SQ + 0.5 * (cu**2) / (CS_SQ**2) - 0.5 * u_sq[None, :, :, :] / CS_SQ
        )

        # Guo Forcing Term:
        # F_i = (1 - 1/(2*tau)) * w_i * [ (c_i - u)/c_s^2 + (c_i . u)/(c_s^4) * c_i ] . force
        cf = np.tensordot(C_D3Q19, force, axes=([1], [0]))
        uf = np.sum(u * force, axis=0)
        F_guo = (1.0 - 0.5 / self.tau_f) * W_D3Q19[:, None, None, None] * (
            (cf - uf[None, :, :, :]) / CS_SQ + (cu * cf) / (CS_SQ**2)
        )

        # Collision: f* = f - (f - feq)/tau + F_guo
        f_star = self.f - (self.f - feq) / self.tau_f + F_guo

        # Streaming on D3Q19
        f_next = np.zeros_like(self.f)
        for i in range(19):
            cx, cy, cz = C_D3Q19[i]
            streamed_i = np.roll(np.roll(np.roll(f_star[i], cz, axis=0), cy, axis=1), cx, axis=2)
            f_next[i] = streamed_i

        # True halfway bounce-back on solid obstacles
        fluid_mask = ~self.solid_mask
        for i in range(1, 19):
            cx, cy, cz = C_D3Q19[i]
            opp = OPP_D3Q19[i]
            nbr_solid = np.roll(np.roll(np.roll(self.solid_mask, -cz, axis=0), -cy, axis=1), -cx, axis=2)
            bounce_mask = fluid_mask & nbr_solid
            f_next[opp, bounce_mask] = f_star[i, bounce_mask]

        f_next[:, self.solid_mask] = W_D3Q19[:, None] * 1.0
        self.f = f_next
        return rho, u

    # -----------------------------------------------------------------
    # Unified Step & Initialization
    # -----------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """Execute one complete coupled multiphase fluid evolution step."""
        # 1. Compute forces based on current interface and gravity
        phi = self.compute_phi()
        force = self.compute_total_force(phi)

        # 2. Advance hydrodynamics
        rho, u = self.step_hydrodynamics(force)

        # 3. Advance phase field
        phi_next = self.step_phase_field(u)

        # Metrics
        liquid_vol = float(np.sum(phi_next > 0.0))
        total_order = float(np.sum(phi_next))
        max_u = float(np.max(np.sqrt(np.sum(u**2, axis=0))))
        jz = float(np.sum(rho * u[2]))

        return {
            "phi": phi_next,
            "rho": rho,
            "u": u,
            "liquid_volume": liquid_vol,
            "total_order_mass": total_order,
            "max_velocity": max_u,
            "vertical_momentum_jz": jz,
        }

    def initialize_droplet(self, center: Tuple[float, float, float], radius: float) -> None:
        """Initialize a spherical liquid droplet in gas."""
        z0, y0, x0 = center
        z = np.arange(self.nz, dtype=np.float64)[:, None, None]
        y = np.arange(self.ny, dtype=np.float64)[None, :, None]
        x = np.arange(self.nx, dtype=np.float64)[None, None, :]

        dist = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        # Hyperbolic tangent diffuse interface: phi = -tanh((dist - R)/(2*xi))
        phi = -np.tanh((dist - radius) / (2.0 * self.xi))
        phi[self.solid_mask] = -1.0

        # Initialize g_dist
        for i in range(7):
            self.g_dist[i] = WEIGHTS_D3Q7[i] * phi

        # Initialize f_dist to resting reference density
        for i in range(19):
            self.f[i] = W_D3Q19[i] * 1.0

    def initialize_waterfall_reservoir(
        self, z_shelf: int = 8, x_cliff: int = 6, fill_height: Optional[int] = None
    ) -> None:
        """Initialize liquid column resting on an elevated reservoir shelf."""
        phi = np.full((self.nz, self.ny, self.nx), -1.0, dtype=np.float64)

        if fill_height is None:
            # Leave generous headspace air cushion above the water to prevent ceiling shear
            fill_height = min(self.nz - 4, z_shelf + 1 + max(3, (self.nz - 1 - z_shelf) // 2))

        # Liquid in reservoir above shelf
        for z in range(z_shelf + 1, fill_height + 1):
            for y in range(1, self.ny - 1):
                for x in range(1, x_cliff):
                    if not self.solid_mask[z, y, x]:
                        phi[z, y, x] = 1.0

        for i in range(7):
            self.g_dist[i] = WEIGHTS_D3Q7[i] * phi

        # Initialize f_dist to resting reference density
        for i in range(19):
            self.f[i] = W_D3Q19[i] * 1.0
