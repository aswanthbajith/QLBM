"""Tests for Gravity and External Forcing in D2Q9 LBM (Phase 7).

Validates:
1. Gravitational acceleration configuration (scalar downward, 2D vector).
2. Hydrostatic equilibrium benchmark in a closed vertical fluid column:
   dp/dy = -rho * g  ==>  rho(y) = rho_mid * exp(-3 * g * (y - y_mid)).
3. Linear hydrostatic pressure head Delta p = rho * g * h.
4. Spurious velocity suppression in hydrostatic equilibrium.
5. Uniform free-fall acceleration at exact rate g per time step.
6. Dynamic callable force callback support for multiphase coupling.
7. Exact mass conservation in gravity-driven bounded columns.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qlbm.classical.d2q9 import CS_SQ
from qlbm.classical.solver import D2Q9BGKSolver


class TestGravityConfiguration:
    """Validate gravity vector initialization and effective force evaluation."""

    def test_scalar_downward_gravity(self):
        """Scalar gravity g sets downward vector (0, -g)."""
        solver = D2Q9BGKSolver(nx=4, ny=4, gravity=1e-4)
        assert solver.gravity == (0.0, -1e-4)
        solver.initialize(rho=1.2)
        eff_force = solver._get_effective_force()
        assert eff_force.shape == (2, 4, 4)
        assert np.allclose(eff_force[0], 0.0, atol=1e-15)
        assert np.allclose(eff_force[1], -1.2 * 1e-4, atol=1e-15)

    def test_vector_gravity(self):
        """Tuple gravity (gx, gy) sets full 2D gravity vector."""
        solver = D2Q9BGKSolver(nx=4, ny=4, gravity=(2e-5, -8e-5))
        assert solver.gravity == (2e-5, -8e-5)
        solver.initialize(rho=1.0)
        eff_force = solver._get_effective_force()
        assert np.allclose(eff_force[0], 2e-5, atol=1e-15)
        assert np.allclose(eff_force[1], -8e-5, atol=1e-15)

    def test_callable_force_callback(self):
        """Callable force function force(rho, u) is evaluated dynamically."""
        def custom_drag(rho, u):
            # Quadratic drag: -0.1 * rho * |u| * u
            speed = np.sqrt(u[0] ** 2 + u[1] ** 2)
            return -0.1 * rho * speed * u

        solver = D2Q9BGKSolver(nx=4, ny=4, force=custom_drag)
        u_init = np.zeros((2, 4, 4))
        u_init[0] = 0.05
        solver.initialize(rho=1.0, u=u_init)

        eff_force = solver._get_effective_force()
        assert np.allclose(eff_force[0], -0.1 * 1.0 * 0.05 * 0.05, atol=1e-15)
        assert np.allclose(eff_force[1], 0.0, atol=1e-15)


class TestHydrostaticEquilibrium:
    """Validate hydrostatic column benchmark: dp/dy = -rho * g."""

    def test_hydrostatic_density_profile(self):
        r"""In a vertical fluid column under gravity g with no-slip walls at bottom and top:

        At steady state: dp/dy = -rho * g ==> rho(y) = rho_mid * exp(-3 * g * (y - y_mid)).
        """
        nx, ny = 4, 33
        g = 5e-5
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True       # solid bottom wall
        solid_mask[ny - 1, :] = True  # solid top wall

        solver = D2Q9BGKSolver(
            nx=nx,
            ny=ny,
            tau=0.8,
            solid_mask=solid_mask,
            gravity=g,
        )
        solver.initialize(rho=1.0)

        # Run to steady hydrostatic equilibrium
        solver.run(3000)

        # Extract fluid vertical profile
        y_fluid = np.arange(1, ny - 1, dtype=np.float64)
        rho_sim = solver.u  # placeholder
        rho_sim = solver.rho[1 : ny - 1, nx // 2]
        u_sim = solver.u[:, 1 : ny - 1, nx // 2]

        # Analytical barometric distribution centered at middle node
        mid = len(y_fluid) // 2
        y_mid = y_fluid[mid]
        rho_mid = rho_sim[mid]
        rho_analytical = rho_mid * np.exp(-3.0 * g * (y_fluid - y_mid))

        # Relative L2 error must be below 0.01%
        l2_err = np.linalg.norm(rho_sim - rho_analytical) / np.linalg.norm(rho_analytical)
        assert l2_err < 1e-4, f"Hydrostatic profile L2 error {l2_err:.4e} exceeds 0.01%"

        # Spurious velocity must be suppressed to near zero
        max_u = np.max(np.abs(u_sim))
        assert max_u < 1e-4, f"Spurious velocity {max_u:.4e} too large in hydrostatic state"

    def test_hydrostatic_pressure_gradient(self):
        r"""Verify discrete pressure gradient matches dp/dy = -rho * g."""
        nx, ny = 4, 25
        g = 4e-5
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[ny - 1, :] = True

        solver = D2Q9BGKSolver(
            nx=nx,
            ny=ny,
            tau=1.0,
            solid_mask=solid_mask,
            gravity=g,
        )
        solver.initialize(rho=1.0)
        solver.run(3000)

        # Discrete pressure gradient via central differences in interior nodes
        p = solver.get_pressure()
        dp_dy = (p[3 : ny - 2, 2] - p[1 : ny - 4, 2]) / 2.0
        rho_interior = solver.rho[2 : ny - 3, 2]
        expected_dp_dy = -rho_interior * g

        # Relative gradient error
        grad_error = np.linalg.norm(dp_dy - expected_dp_dy) / np.linalg.norm(expected_dp_dy)
        assert grad_error < 0.001, f"Pressure gradient error {grad_error:.4e} exceeds 0.1%"

    def test_hydrostatic_mass_conservation(self):
        """Gravity in a closed domain must preserve total fluid mass to machine precision."""
        nx, ny = 4, 21
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[ny - 1, :] = True

        solver = D2Q9BGKSolver(nx=nx, ny=ny, tau=0.8, solid_mask=solid_mask, gravity=1e-4)
        solver.initialize(rho=1.0)

        m0 = solver.total_mass
        solver.run(500)
        m1 = solver.total_mass

        rel_mass_err = abs(m1 - m0) / m0
        assert rel_mass_err < 1e-13


class TestGravityDynamicAcceleration:
    """Validate dynamic velocity acceleration under uniform gravity."""

    def test_uniform_free_fall_rate(self):
        r"""In an unconstrained periodic domain, fluid accelerates at exact rate g."""
        N = 8
        g = 1e-4
        solver = D2Q9BGKSolver(nx=N, ny=N, tau=0.8, gravity=g)
        solver.initialize(rho=1.0)

        steps = 50
        solver.run(steps)

        # Because velocity is defined at t + 0.5 dt, after N steps u_y = -(N + 0.5) * g
        expected_uy = -(steps + 0.5) * g
        sim_uy = np.mean(solver.u[1])

        assert np.isclose(sim_uy, expected_uy, atol=1e-12)
