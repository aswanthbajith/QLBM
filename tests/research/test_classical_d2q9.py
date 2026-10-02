"""Tests for Classical D2Q9 Lattice Boltzmann Model (Phase 4).

Validates:
1. Lattice tensor symmetries and moments (0th through 4th order).
2. Maxwell-Boltzmann equilibrium moments (density, momentum, pressure tensor).
3. BGK collision conservation and Guo forcing exactness.
4. Numerical mass conservation on periodic domains and with solid obstacle walls.
5. Invariance of stationary and uniform advection states.
6. Analytical benchmark: Poiseuille channel flow parabolic profile.
7. Analytical benchmark: 2D Taylor-Green vortex viscous decay rate.
8. Lid-driven cavity vortex generation with moving wall momentum exchange.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qlbm.classical.d2q9 import (
    C,
    CS,
    CS_SQ,
    OPPOSITE,
    WEIGHTS,
    bgk_collision,
    compute_equilibrium,
    compute_guo_force,
    compute_macroscopic,
    relaxation_time_from_viscosity,
    stream_and_bounce,
    viscosity_from_relaxation_time,
)
from qlbm.classical.solver import D2Q9BGKSolver


class TestD2Q9LatticeProperties:
    """Validate mathematical properties and tensor symmetries of the D2Q9 lattice."""

    def test_weight_sum(self):
        """Sum of lattice weights must equal 1."""
        assert np.isclose(np.sum(WEIGHTS), 1.0, atol=1e-15)

    def test_first_moment_zero(self):
        r"""First moment of weights must vanish: \sum w_i c_i = 0."""
        first_moment = np.sum(WEIGHTS[:, None] * C, axis=0)
        assert np.allclose(first_moment, [0.0, 0.0], atol=1e-15)

    def test_second_moment_isotropy(self):
        r"""Second moment of weights must equal c_s^2 \delta_{ab}."""
        second_moment = np.einsum("i,ia,ib->ab", WEIGHTS, C, C)
        expected = CS_SQ * np.eye(2)
        assert np.allclose(second_moment, expected, atol=1e-15)

    def test_third_moment_zero(self):
        r"""Third moment of weights must vanish due to lattice parity symmetry."""
        third_moment = np.einsum("i,ia,ib,ic->abc", WEIGHTS, C, C, C)
        assert np.allclose(third_moment, 0.0, atol=1e-15)

    def test_fourth_moment_isotropy(self):
        r"""Fourth moment of weights must satisfy isotropic Navier-Stokes closure:

        \sum w_i c_{ia} c_{ib} c_{ic} c_{id} = c_s^4 (\delta_{ab}\delta_{cd} + \delta_{ac}\delta_{bd} + \delta_{ad}\delta_{bc})
        """
        fourth_moment = np.einsum("i,ia,ib,ic,id->abcd", WEIGHTS, C, C, C, C)
        delta = np.eye(2)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        expected = (CS_SQ**2) * (
                            delta[a, b] * delta[c, d]
                            + delta[a, c] * delta[b, d]
                            + delta[a, d] * delta[b, c]
                        )
                        assert np.isclose(fourth_moment[a, b, c, d], expected, atol=1e-15)

    def test_opposite_direction_involution(self):
        """Opposite indices must satisfy c_{opp[i]} = -c_i and opp[opp[i]] = i."""
        for i in range(9):
            opp = OPPOSITE[i]
            assert np.array_equal(C[opp], -C[i])
            assert OPPOSITE[opp] == i


class TestD2Q9EquilibriumAndMoments:
    """Validate moment reproduction of the polynomial equilibrium distribution."""

    def test_equilibrium_moments(self):
        """Equilibrium must reproduce rho, rho*u, and correct pressure tensor."""
        rho_val = 1.25
        u_val = np.array([0.08, -0.05], dtype=np.float64)

        f_eq = compute_equilibrium(rho_val, u_val)

        # 0th moment -> rho
        assert np.isclose(np.sum(f_eq), rho_val, atol=1e-14)

        # 1st moment -> rho * u
        rho_u = np.sum(f_eq[:, None] * C, axis=0)
        assert np.allclose(rho_u, rho_val * u_val, atol=1e-14)

        # 2nd moment -> P delta_ab + rho u_a u_b (with P = rho * cs^2)
        pi = np.einsum("i,ia,ib->ab", f_eq, C, C)
        expected_pi = rho_val * CS_SQ * np.eye(2) + rho_val * np.outer(u_val, u_val)
        assert np.allclose(pi, expected_pi, atol=1e-14)

    def test_macroscopic_inversion(self):
        """compute_macroscopic must invert equilibrium to exact rho and u."""
        ny, nx = 8, 8
        rho_field = np.ones((ny, nx)) * 1.05
        u_field = np.zeros((2, ny, nx))
        u_field[0] = 0.05
        u_field[1] = -0.02

        f = compute_equilibrium(rho_field, u_field)
        rho_calc, u_calc = compute_macroscopic(f)

        assert np.allclose(rho_calc, rho_field, atol=1e-14)
        assert np.allclose(u_calc, u_field, atol=1e-14)


class TestBGKCollisionAndForcing:
    """Validate BGK collision operator and Guo forcing scheme."""

    def test_unforced_collision_conservation(self):
        """Unforced BGK collision must conserve mass and momentum locally."""
        ny, nx = 4, 4
        rho = np.ones((ny, nx))
        u = np.zeros((2, ny, nx))
        u[0] = 0.02
        f = compute_equilibrium(rho, u)
        # Perturb f non-trivially
        f_perturbed = f + 0.01 * (np.random.RandomState(42).rand(*f.shape) - 0.5)

        # Recompute moments of perturbed state
        rho_p, u_p = compute_macroscopic(f_perturbed)
        f_eq = compute_equilibrium(rho_p, u_p)

        # Collision with tau = 0.8
        f_star = bgk_collision(f_perturbed, f_eq, tau=0.8)

        # Check post-collision moments match pre-collision moments
        rho_star, u_star = compute_macroscopic(f_star)
        assert np.allclose(rho_star, rho_p, atol=1e-14)
        assert np.allclose(u_star, u_p, atol=1e-14)

    def test_guo_forcing_moments(self):
        """Guo source terms must inject zero mass and exact momentum F."""
        ny, nx = 4, 4
        u = np.zeros((2, ny, nx))
        u[0] = 0.05
        u[1] = 0.01
        force = np.array([0.001, -0.002])
        tau = 0.75

        F_i = compute_guo_force(u, force, tau)

        # Mass moment of Guo force is zero: sum F_i = 0
        sum_F = np.sum(F_i, axis=0)
        assert np.allclose(sum_F, 0.0, atol=1e-15)

        # Momentum moment of Guo force equals (1 - 1/(2*tau)) * F
        p_F = np.sum(F_i[:, None, :, :] * C[:, :, None, None], axis=0)
        expected_pF = (1.0 - 1.0 / (2.0 * tau)) * force
        assert np.allclose(p_F[0], expected_pF[0], atol=1e-15)
        assert np.allclose(p_F[1], expected_pF[1], atol=1e-15)


class TestConservationAndInvariance:
    """Validate global mass and momentum conservation in D2Q9BGKSolver."""

    def test_periodic_mass_and_momentum_conservation(self):
        """In a periodic domain without external forcing, mass and momentum are exactly conserved."""
        solver = D2Q9BGKSolver(nx=16, ny=16, tau=0.8)
        u_init = np.zeros((2, 16, 16))
        u_init[0, 6:10, 6:10] = 0.05  # localized velocity pulse
        solver.initialize(rho=1.0, u=u_init)

        m0 = solver.total_mass
        p0 = solver.total_momentum

        solver.run(50)

        m1 = solver.total_mass
        p1 = solver.total_momentum

        assert np.isclose(m1, m0, atol=1e-12)
        assert np.allclose(p1, p0, atol=1e-12)

    def test_solid_obstacle_mass_conservation(self):
        """Halfway bounce-back must conserve total fluid mass to machine precision."""
        nx, ny = 16, 16
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[6:10, 6:10] = True  # 4x4 square obstacle in center

        solver = D2Q9BGKSolver(nx=nx, ny=ny, tau=0.8, solid_mask=solid_mask)
        u_init = np.zeros((2, ny, nx))
        u_init[0] = 0.03  # uniform flow impinging on obstacle
        solver.initialize(rho=1.0, u=u_init)

        initial_mass = solver.total_mass
        solver.run(100)
        final_mass = solver.total_mass

        relative_mass_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_mass_error < 1e-13

    def test_stationary_uniform_field_invariance(self):
        """Resting fluid (u=0, rho=1) must remain invariant over time."""
        solver = D2Q9BGKSolver(nx=8, ny=8, tau=1.0)
        solver.initialize(rho=1.0, u=np.zeros((2, 8, 8)))

        solver.run(100)

        assert np.allclose(solver.rho, 1.0, atol=1e-14)
        assert np.allclose(solver.u, 0.0, atol=1e-14)


class TestAnalyticalBenchmarks:
    """Validate classical D2Q9 solver against exact analytical Navier-Stokes solutions."""

    def test_poiseuille_channel_flow(self):
        r"""Poiseuille flow between parallel solid plates.

        Analytical parabolic steady-state:
        u_x(y) = \frac{F_x}{2 \rho \nu} (y - y_{bot}) (y_{top} - y)
        With halfway bounce-back on solid plates at y=0 and y=ny-1:
        y_{bot} = 0.5, y_{top} = ny - 1.5, H = ny - 2.
        """
        nx = 4
        ny = 17
        tau = 0.8
        nu = viscosity_from_relaxation_time(tau)
        rho_0 = 1.0
        fx = 1e-5
        force = np.array([fx, 0.0])

        # Solid walls at y=0 (bottom) and y=ny-1 (top)
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[ny - 1, :] = True

        solver = D2Q9BGKSolver(
            nx=nx, ny=ny, tau=tau, solid_mask=solid_mask, force=force
        )
        solver.initialize(rho=rho_0, u=None)

        # Run until steady state
        solver.run(3000)

        # Extract fluid velocity profile along vertical centerline
        y_fluid = np.arange(1, ny - 1, dtype=np.float64)
        u_sim = solver.u[0, 1 : ny - 1, nx // 2]

        # Analytical solution with halfway bounceback (wall at y=0.5 and y=ny-1.5)
        y_bot = 0.5
        y_top = ny - 1.5
        u_analytical = (fx / (2.0 * rho_0 * nu)) * (y_fluid - y_bot) * (y_top - y_fluid)

        # Relative L2 error
        l2_error = np.linalg.norm(u_sim - u_analytical) / np.linalg.norm(u_analytical)
        assert l2_error < 0.005, f"Poiseuille L2 error {l2_error:.4e} exceeds 0.5%"

        # Centerline maximum velocity check
        mid_idx = len(y_fluid) // 2
        assert np.isclose(u_sim[mid_idx], u_analytical[mid_idx], rtol=0.005)

    def test_taylor_green_vortex_decay(self):
        r"""2D Taylor-Green vortex decay rate benchmark.

        Analytical solution:
        u_x(x, y, t) = -u_0 \cos(kx) \sin(ky) e^{-2 \nu k^2 t}
        u_y(x, y, t) =  u_0 \sin(kx) \cos(ky) e^{-2 \nu k^2 t}
        Kinetic energy decay:
        E(t) = E(0) e^{-4 \nu k^2 t}
        """
        N = 32
        tau = 0.8
        nu = viscosity_from_relaxation_time(tau)
        k = 2.0 * np.pi / N
        u0 = 0.03

        x = np.arange(N, dtype=np.float64)
        y = np.arange(N, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        u_init = np.empty((2, N, N), dtype=np.float64)
        u_init[0] = -u0 * np.cos(k * X) * np.sin(k * Y)
        u_init[1] = u0 * np.sin(k * X) * np.cos(k * Y)

        # Initial pressure/density: P = P0 - 0.25 * rho0 * u0^2 * (cos(2kx) + cos(2ky))
        rho_init = 1.0 - (0.25 * u0**2 / CS_SQ) * (np.cos(2.0 * k * X) + np.cos(2.0 * k * Y))

        solver = D2Q9BGKSolver(nx=N, ny=N, tau=tau)
        solver.initialize(rho=rho_init, u=u_init)

        E0 = solver.kinetic_energy
        steps = 200
        solver.run(steps)
        E_t = solver.kinetic_energy

        # Theoretical decay: E(t) / E(0) = exp(-4 * nu * k^2 * t)
        decay_expected = np.exp(-4.0 * nu * (k**2) * steps)
        decay_sim = E_t / E0

        relative_decay_error = abs(decay_sim - decay_expected) / decay_expected
        assert relative_decay_error < 0.02, (
            f"Taylor-Green decay error {relative_decay_error:.4e} exceeds 2% "
            f"(sim: {decay_sim:.5f}, expected: {decay_expected:.5f})"
        )

    def test_lid_driven_cavity_vortex_generation(self):
        """Lid-driven cavity flow generates primary recirculation vortex."""
        N = 25
        tau = 0.8
        u_lid = 0.08

        # Boundaries on all 4 sides
        solid_mask = np.zeros((N, N), dtype=bool)
        solid_mask[0, :] = True       # bottom
        solid_mask[:, 0] = True       # left
        solid_mask[:, N - 1] = True   # right
        solid_mask[N - 1, :] = True   # top (lid)

        # Top wall velocity
        wall_velocity = np.zeros((2, N, N), dtype=np.float64)
        wall_velocity[0, N - 1, :] = u_lid

        solver = D2Q9BGKSolver(
            nx=N,
            ny=N,
            tau=tau,
            solid_mask=solid_mask,
            wall_velocity=wall_velocity,
        )
        solver.initialize(rho=1.0)

        # Run 500 steps to develop vortex
        solver.run(500)

        # In interior fluid, check that top fluid layer moves in +x direction
        u_top_interior = solver.u[0, N - 2, N // 2]
        assert u_top_interior > 0.02, f"Expected positive u_x near lid, got {u_top_interior}"

        # And bottom fluid layer moves in -x direction (recirculation)
        u_bottom_interior = solver.u[0, 2, N // 2]
        assert u_bottom_interior < 0.0, f"Expected negative u_x near bottom, got {u_bottom_interior}"

        # Vorticity in center should be negative (clockwise primary vortex)
        vorticity = solver.get_vorticity()
        center_vorticity = vorticity[N // 2, N // 2]
        assert center_vorticity < 0.0, f"Expected clockwise vortex, got {center_vorticity}"
