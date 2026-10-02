"""Tests for D2Q9 BGK Collision Dynamics and Relaxation Spectrum (Phase 5).

Validates:
1. Orthogonal moment transformation matrix properties and roundtrip invertibility.
2. Eigenvalues and relaxation spectrum of the BGK collision operator.
3. Stability boundary: asymptotic stability for tau > 0.5 and instability for tau < 0.5.
4. Non-equilibrium stress tensor scaling and viscous Couette shear stress recovery.
5. Discrete H-theorem: monotonic decrease of H-function during homogeneous relaxation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qlbm.classical.d2q9 import (
    C,
    CS_SQ,
    M_INV,
    M_MATRIX,
    WEIGHTS,
    bgk_collision,
    compute_distributions_from_moments,
    compute_entropy_h,
    compute_equilibrium,
    compute_macroscopic,
    compute_moments,
    compute_nonequilibrium_stress,
    viscosity_from_relaxation_time,
)
from qlbm.classical.solver import D2Q9BGKSolver


class TestMomentMatrixProperties:
    """Validate algebraic properties of the Gram-Schmidt orthogonal moment basis M."""

    def test_moment_matrix_orthogonality(self):
        """M @ M^T must be strictly diagonal (orthogonal moment vectors)."""
        G = M_MATRIX @ M_MATRIX.T
        off_diagonal = G - np.diag(np.diag(G))
        assert np.allclose(off_diagonal, 0.0, atol=1e-14)

    def test_analytical_inverse(self):
        """Analytical inverse M_INV must equal np.linalg.inv(M_MATRIX)."""
        inv_numeric = np.linalg.inv(M_MATRIX)
        assert np.allclose(M_INV, inv_numeric, atol=1e-14)
        assert np.allclose(M_MATRIX @ M_INV, np.eye(9), atol=1e-14)

    def test_moment_roundtrip_transformation(self):
        """f -> m -> f must preserve arbitrary distribution functions to machine precision."""
        rng = np.random.RandomState(42)
        f_random = rng.uniform(0.01, 1.0, size=(9, 4, 4))
        m = compute_moments(f_random)
        f_recovered = compute_distributions_from_moments(m)
        assert np.allclose(f_recovered, f_random, atol=1e-14)

    def test_conserved_moment_mapping(self):
        """Moments m_0, m_3, m_5 must exactly correspond to rho, rho*ux, rho*uy."""
        rho_val = 1.15
        u_val = np.array([0.04, -0.03])
        f_eq = compute_equilibrium(rho_val, u_val)
        m = compute_moments(f_eq)

        # m[0] = rho
        assert np.isclose(m[0], rho_val, atol=1e-14)
        # m[3] = j_x
        assert np.isclose(m[3], rho_val * u_val[0], atol=1e-14)
        # m[5] = j_y
        assert np.isclose(m[5], rho_val * u_val[1], atol=1e-14)


class TestBGKEigenvaluesAndStability:
    """Validate spectral decomposition and linear numerical stability of the BGK operator."""

    @pytest.mark.parametrize("tau", [0.6, 0.8, 1.0, 1.5, 2.0])
    def test_collision_eigenvalues(self, tau):
        r"""Linearized post-collision operator C = I + Omega must have eigenvalues:

        - 3 eigenvalues = 1.0 (conserved modes: rho, jx, jy)
        - 6 eigenvalues = 1 - 1/tau (decaying modes)
        """
        omega = 1.0 / tau
        # Jacobian of equilibrium with respect to f around rest state rho=1, u=0
        # For small perturbations around rho=1, u=0:
        # f_eq_i(f) = w_i * [ sum(f_j) + 3 c_ia sum(f_j c_ja) ]
        # P_eq_{ij} = w_i * [ 1 + 3 (c_ix * c_jx + c_iy * c_jy) ]
        P_eq = np.zeros((9, 9), dtype=np.float64)
        for i in range(9):
            for j in range(9):
                ci_dot_cj = C[i, 0] * C[j, 0] + C[i, 1] * C[j, 1]
                P_eq[i, j] = WEIGHTS[i] * (1.0 + 3.0 * ci_dot_cj)

        # Post-collision operator in distribution space: C = (1 - omega)*I + omega*P_eq
        C_op = (1.0 - omega) * np.eye(9) + omega * P_eq

        eigenvalues = np.sort(np.linalg.eigvals(C_op))

        # Expected: three 1.0 and six (1 - omega)
        expected = np.sort(np.array([1.0, 1.0, 1.0] + [1.0 - omega] * 6))
        assert np.allclose(eigenvalues, expected, atol=1e-13)

    def test_asymptotic_stability_tau_greater_than_half(self):
        """For tau > 0.5, non-equilibrium perturbations decay exponentially."""
        tau = 0.75
        omega = 1.0 / tau
        rho0 = 1.0
        u0 = np.array([0.0, 0.0])
        f_eq = compute_equilibrium(rho0, u0)

        # Add a non-equilibrium, zero-mass, zero-momentum perturbation
        perturbation = np.array([0.0, 0.01, -0.01, 0.01, -0.01, 0.005, -0.005, 0.005, -0.005])
        f = f_eq + perturbation

        decay_rates = []
        for step in range(5):
            f_star = bgk_collision(f, f_eq, tau)
            norm_before = np.linalg.norm(f - f_eq)
            norm_after = np.linalg.norm(f_star - f_eq)
            decay_rates.append(norm_after / norm_before)
            f = f_star

        # Decay factor per step must exactly equal |1 - omega|
        expected_rate = abs(1.0 - omega)
        for rate in decay_rates:
            assert np.isclose(rate, expected_rate, atol=1e-14)

    def test_instability_tau_less_than_half(self):
        """For tau < 0.5 (omega > 2), perturbations grow without bound (|1 - omega| > 1)."""
        tau = 0.4
        omega = 1.0 / tau  # 2.5
        rho0 = 1.0
        u0 = np.array([0.0, 0.0])
        f_eq = compute_equilibrium(rho0, u0)

        perturbation = np.array([0.0, 0.01, -0.01, 0.01, -0.01, 0.005, -0.005, 0.005, -0.005])
        f = f_eq + perturbation

        norm0 = np.linalg.norm(f - f_eq)
        f_star = bgk_collision(f, f_eq, tau)
        norm1 = np.linalg.norm(f_star - f_eq)

        # Amplification factor = |1 - 2.5| = 1.5 > 1
        amplification = norm1 / norm0
        assert amplification > 1.49
        assert np.isclose(amplification, abs(1.0 - omega), atol=1e-14)


class TestNonEquilibriumStressAndCouette:
    """Validate non-equilibrium stress tensor and viscous shear stress recovery."""

    def test_nonequilibrium_stress_zero_at_equilibrium(self):
        """At equilibrium, non-equilibrium stress tensor must be identically zero."""
        rho = np.ones((8, 8)) * 1.0
        u = np.zeros((2, 8, 8))
        u[0] = 0.05
        u[1] = -0.02
        f_eq = compute_equilibrium(rho, u)

        pi_neq = compute_nonequilibrium_stress(f_eq, f_eq=f_eq)
        assert np.allclose(pi_neq, 0.0, atol=1e-15)

    def test_couette_shear_stress_relation(self):
        r"""In steady Couette flow, non-equilibrium stress \Pi_{xy}^{(neq)} satisfies:

        \Pi_{xy}^{(neq)} = -\frac{\tau}{3} \rho \dot{\gamma}
        and physical shear stress is:
        \sigma_{xy} = -(1 - 1/(2*tau)) \Pi_{xy}^{(neq)} = \rho \nu \dot{\gamma}
        """
        N = 17
        tau = 0.8
        nu = viscosity_from_relaxation_time(tau)
        u_lid = 0.04

        # Channel with stationary bottom at y=0 and moving top at y=N-1
        solid_mask = np.zeros((N, 4), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[N - 1, :] = True

        wall_velocity = np.zeros((2, N, 4), dtype=np.float64)
        wall_velocity[0, N - 1, :] = u_lid

        solver = D2Q9BGKSolver(
            nx=4,
            ny=N,
            tau=tau,
            solid_mask=solid_mask,
            wall_velocity=wall_velocity,
        )
        solver.initialize(rho=1.0)

        # Run until linear Couette profile establishes
        solver.run(2000)

        # Effective channel height between halfway walls: H = N - 2
        H = N - 2.0
        shear_rate = u_lid / H

        # Extract non-equilibrium stress at center of channel (y = N // 2)
        pi_neq = solver.get_nonequilibrium_stress()
        pi_xy_sim = pi_neq[0, 1, N // 2, 2]

        # Analytical non-equilibrium stress: Pi_xy^{(neq)} = - (tau / 3) * rho * shear_rate
        expected_pi_xy = -(tau / 3.0) * 1.0 * shear_rate

        assert np.isclose(pi_xy_sim, expected_pi_xy, rtol=0.015), (
            f"Simulated Pi_xy {pi_xy_sim:.6e} vs expected {expected_pi_xy:.6e}"
        )

        # Physical shear stress recovery: sigma_xy = -(1 - 1/(2*tau)) * Pi_xy
        sigma_xy_recovered = -(1.0 - 1.0 / (2.0 * tau)) * pi_xy_sim
        expected_sigma_xy = 1.0 * nu * shear_rate

        assert np.isclose(sigma_xy_recovered, expected_sigma_xy, rtol=0.015)


class TestDiscreteHTheoremAndLyapunov:
    """Validate thermodynamic consistency, Lyapunov functionals, and H-function behavior."""

    @pytest.mark.parametrize("tau", [0.6, 0.8, 1.0, 1.5])
    def test_quadratic_lyapunov_monotonic_decrease(self, tau):
        r"""The quadratic distance \Phi(f) = \sum (f_i - f_i^{(eq)})^2 / w_i must decay strictly monotonically.

        \Phi(f^*) = (1 - \omega)^2 \Phi(f) with |1 - \omega| < 1 for all tau > 0.5.
        """
        omega = 1.0 / tau
        rho0 = 1.0
        u0 = np.array([0.02, 0.01])
        f_eq = compute_equilibrium(rho0, u0)

        # Non-equilibrium perturbation
        rng = np.random.RandomState(42)
        perturbation = rng.uniform(-0.02, 0.02, size=9)
        f = f_eq + perturbation

        for step in range(10):
            phi_before = np.sum(((f - f_eq) ** 2) / WEIGHTS)
            f_star = bgk_collision(f, f_eq, tau)
            phi_after = np.sum(((f_star - f_eq) ** 2) / WEIGHTS)

            expected_phi = ((1.0 - omega) ** 2) * phi_before
            assert np.isclose(phi_after, expected_phi, atol=1e-14)
            if phi_before > 1e-14:
                assert phi_after < phi_before, f"Lyapunov functional failed to decrease at step {step}"
            else:
                assert phi_after <= 1e-14
            f = f_star

    def test_logarithmic_h_convergence_to_equilibrium(self):
        """As f converges to f_eq under BGK collision, Boltzmann's H converges to H_eq."""
        tau = 0.8
        rho0 = 1.0
        u0 = np.array([0.02, 0.01])
        f_eq = compute_equilibrium(rho0, u0)
        h_eq = compute_entropy_h(f_eq)

        rng = np.random.RandomState(42)
        perturbation = rng.uniform(-0.02, 0.02, size=9)
        f = f_eq + perturbation

        # After 25 collision steps, perturbation is damped by (1 - 1.25)^25 = (-0.25)^25 ~ 10^-15
        for _ in range(25):
            f = bgk_collision(f, f_eq, tau)

        h_final = compute_entropy_h(f)
        assert np.isclose(h_final, h_eq, atol=1e-12)
