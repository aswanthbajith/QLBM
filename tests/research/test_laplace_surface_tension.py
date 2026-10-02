"""Tests for Surface Tension & 2D Young-Laplace Law Validation (Phase 10).

Validates:
1. Circular droplet initialization via smooth hyperbolic tangent profile.
2. Exact mass conservation during multiphase droplet relaxation.
3. 2D Young-Laplace relation: Delta p = p_in - p_out = sigma / R_eff.
4. Linear regression quality: R^2 > 0.98 and intercept near zero.
5. Monotonic scaling of surface tension sigma with interaction parameter |G|.
6. Spatial localization of spurious/parasitic currents strictly at the interface,
   vanishing to machine precision in bulk liquid and vapor.
7. Exponential growth of spurious currents with |G| (and density ratio),
   formally justifying the architectural selection of VOF for dam-break flow.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qlbm.classical.multiphase import (
    ShanChenMultiphaseSolver,
    fit_laplace_law,
    initialize_circular_droplet,
    measure_droplet_properties,
    shan_chen_pressure,
)


class TestDropletInitializationAndConservation:
    """Validate initial profile generation and discrete mass invariance."""

    def test_circular_droplet_tanh_profile(self):
        """Profile smoothly transitions between vapor and liquid densities."""
        nx, ny = 32, 32
        solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=-5.0)
        solver.initialize_droplet(radius=8.0, rho_liquid=1.9, rho_vapor=0.12)

        # Center should be dense liquid
        assert np.isclose(solver.rho[ny // 2, nx // 2], 1.9, atol=1e-3)
        # Corners should be dilute vapor
        assert np.isclose(solver.rho[0, 0], 0.12, atol=1e-3)
        # All densities strictly within bounds
        assert np.all(solver.rho >= 0.12 - 1e-6)
        assert np.all(solver.rho <= 1.9 + 1e-6)

    def test_droplet_mass_conservation_during_relaxation(self):
        """Discrete mass remains invariant during droplet relaxation to machine precision."""
        nx, ny = 32, 32
        solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=-5.0)
        solver.initialize_droplet(radius=8.0)

        m0 = solver.total_mass
        solver.run(300)
        m1 = solver.total_mass

        assert np.isclose(m1, m0, atol=1e-11)


class TestYoungLaplaceLaw:
    """Validate the Young-Laplace relation Delta p = sigma / R in 2D."""

    def test_laplace_linear_scaling_and_surface_tension(self):
        """Delta p vs 1/R exhibits strict linear proportionality with R^2 > 0.98."""
        nx, ny = 48, 48
        G = -5.0
        radii_nominal = [8.0, 10.0, 12.0, 14.0, 16.0]

        effective_radii = []
        delta_pressures = []

        for R0 in radii_nominal:
            solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=G)
            solver.initialize_droplet(radius=R0)
            solver.run(500)

            props = solver.measure_droplet()
            effective_radii.append(props["effective_radius"])
            delta_pressures.append(props["delta_p"])

        # Perform linear regression Delta p = sigma / R + c_0
        fit = fit_laplace_law(effective_radii, delta_pressures)

        # 1. High linearity (R^2 > 0.98)
        assert fit.r_squared > 0.98, f"Expected R^2 > 0.98, got {fit.r_squared:.4f}"

        # 2. Positive physical surface tension sigma ~ 0.032 - 0.036
        assert 0.025 < fit.sigma < 0.040, f"Expected sigma in [0.025, 0.040], got {fit.sigma:.5f}"

        # 3. Intercept passing close to origin (|c_0| < 5e-4)
        assert abs(fit.intercept) < 5e-4, f"Expected intercept near zero, got {fit.intercept:.6f}"

    def test_surface_tension_monotonicity_with_g(self):
        """Surface tension sigma increases monotonically with interaction strength |G|."""
        nx, ny = 48, 48
        R0 = 12.0
        g_values = [-4.5, -5.0, -5.5]
        sigmas = []

        for G in g_values:
            solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=G)
            solver.initialize_droplet(radius=R0)
            solver.run(500)

            props = solver.measure_droplet()
            # Estimate local sigma = Delta p * R_eff
            sigma_est = props["delta_p"] * props["effective_radius"]
            sigmas.append(sigma_est)

        # Check strict monotonic increase: sigma(-5.5) > sigma(-5.0) > sigma(-4.5)
        assert sigmas[0] < sigmas[1] < sigmas[2], f"Surface tension non-monotonic: {sigmas}"
        # Ratio between G=-5.5 and G=-4.5 is at least 4x
        assert sigmas[2] / sigmas[0] > 4.0


class TestSpuriousCurrentsAndParasiticVortices:
    """Analyze parasitic interfacial velocities and density-ratio stability limits."""

    def test_spurious_currents_spatial_localization(self):
        """Spurious currents peak strictly at the interface and vanish in bulk phases."""
        nx, ny = 48, 48
        solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=-5.0)
        solver.initialize_droplet(radius=12.0)
        solver.run(500)

        props = solver.measure_droplet()

        # Maximum velocity occurs at diffuse interface
        assert props["u_max"] > 1e-3, f"Expected non-zero interface spurious current, got {props['u_max']}"

        # Bulk liquid velocity at center vanishes to machine precision
        assert props["u_bulk_in"] < 1e-14, f"Bulk liquid velocity not zero: {props['u_bulk_in']}"

        # Bulk vapor velocity in far-field corner vanishes to machine precision
        assert props["u_bulk_out"] < 1e-14, f"Bulk vapor velocity not zero: {props['u_bulk_out']}"

    def test_spurious_currents_scaling_with_interaction_strength(self):
        """Spurious velocity magnitude grows rapidly with interaction strength |G|."""
        nx, ny = 48, 48
        R0 = 12.0
        g_values = [-4.5, -5.0, -5.5]
        u_max_values = []

        for G in g_values:
            solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=G)
            solver.initialize_droplet(radius=R0)
            solver.run(500)

            props = solver.measure_droplet()
            u_max_values.append(props["u_max"])

        # Spurious currents grow strictly with |G|
        assert u_max_values[0] < u_max_values[1] < u_max_values[2], (
            f"Spurious currents did not grow monotonically: {u_max_values}"
        )

        # Spurious velocity at G=-5.5 is an order of magnitude larger than at G=-4.5
        assert u_max_values[2] / u_max_values[0] > 8.0, (
            f"Expected rapid growth in spurious currents, got ratio {u_max_values[2] / u_max_values[0]}"
        )
