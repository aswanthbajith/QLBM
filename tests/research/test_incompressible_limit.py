"""Tests for Macroscopic Moments and the Incompressible Limit in D2Q9 (Phase 6).

Validates:
1. Isothermal ideal gas equation of state: p = c_s^2 rho and gauge pressure.
2. Density fluctuation scaling with Mach number: delta_rho / rho_0 ~ O(Ma^2).
3. Velocity divergence error scaling: ||div(u)||_L2 ~ O(Ma^2).
4. Physical acoustic wave propagation at sound speed c_s = 1/sqrt(3).
5. Low-Mach number operational criterion for gravity-driven flows (Torricelli limit).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qlbm.classical.d2q9 import CS, CS_SQ
from qlbm.classical.solver import D2Q9BGKSolver


class TestEquationOfState:
    """Validate thermodynamic pressure and gauge pressure recovery in D2Q9."""

    def test_absolute_pressure_relation(self):
        """p = c_s^2 * rho with c_s^2 = 1/3."""
        solver = D2Q9BGKSolver(nx=8, ny=8)
        solver.initialize(rho=1.2)
        p = solver.get_pressure()
        assert np.allclose(p, (1.0 / 3.0) * 1.2, atol=1e-15)

    def test_gauge_pressure_relation(self):
        """p_gauge = c_s^2 * (rho - rho_0)."""
        solver = D2Q9BGKSolver(nx=8, ny=8)
        solver.initialize(rho=1.05)
        p_gauge = solver.get_pressure(reference_density=1.0)
        assert np.allclose(p_gauge, (1.0 / 3.0) * 0.05, atol=1e-15)


class TestIncompressibleScaling:
    """Validate quadratic O(Ma^2) scaling of density fluctuations and velocity divergence."""

    def test_density_fluctuation_quadratic_mach_scaling(self):
        r"""Density variations \delta\rho / \rho_0 must scale strictly as O(Ma^2) = O(u_0^2)."""
        N = 32
        k = 2.0 * np.pi / N
        x = np.arange(N, dtype=np.float64)
        y = np.arange(N, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        u_amplitudes = [0.01, 0.02, 0.04, 0.08]
        drho_max = []

        for u0 in u_amplitudes:
            solver = D2Q9BGKSolver(nx=N, ny=N, tau=0.8)
            u_init = np.empty((2, N, N), dtype=np.float64)
            u_init[0] = -u0 * np.cos(k * X) * np.sin(k * Y)
            u_init[1] = u0 * np.sin(k * X) * np.cos(k * Y)
            solver.initialize(rho=1.0, u=u_init)
            solver.run(10)

            drho = np.max(np.abs(solver.rho - 1.0))
            drho_max.append(drho)

        # Check quadratic scaling: drho / u0^2 must be constant to within 1%
        ratios = [drho_max[i] / (u_amplitudes[i] ** 2) for i in range(len(u_amplitudes))]
        for r in ratios[1:]:
            assert np.isclose(r, ratios[0], rtol=0.01), (
                f"Non-quadratic density scaling: ratios={ratios}"
            )

    def test_divergence_error_quadratic_mach_scaling(self):
        r"""Divergence error ||div(u)||_L2 must scale strictly as O(Ma^2) = O(u_0^2)."""
        N = 32
        k = 2.0 * np.pi / N
        x = np.arange(N, dtype=np.float64)
        y = np.arange(N, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        u_amplitudes = [0.01, 0.02, 0.04, 0.08]
        div_norms = []

        for u0 in u_amplitudes:
            solver = D2Q9BGKSolver(nx=N, ny=N, tau=0.8)
            u_init = np.empty((2, N, N), dtype=np.float64)
            u_init[0] = -u0 * np.cos(k * X) * np.sin(k * Y)
            u_init[1] = u0 * np.sin(k * X) * np.cos(k * Y)
            solver.initialize(rho=1.0, u=u_init)
            solver.run(10)

            div = solver.get_divergence()
            l2_div = np.linalg.norm(div) / N
            div_norms.append(l2_div)

        # Check quadratic scaling: div_norm / u0^2 must be constant to within 1%
        ratios = [div_norms[i] / (u_amplitudes[i] ** 2) for i in range(len(u_amplitudes))]
        for r in ratios[1:]:
            assert np.isclose(r, ratios[0], rtol=0.01), (
                f"Non-quadratic divergence scaling: ratios={ratios}"
            )


class TestAcousticWavePropagation:
    """Validate artificial compressibility sound speed c_s = 1/sqrt(3)."""

    def test_acoustic_pulse_speed(self):
        r"""A localized density pulse splits into symmetric sound waves traveling at c_s = 1/\sqrt{3}."""
        Nx = 128
        Ny = 4
        tau = 0.6
        solver = D2Q9BGKSolver(nx=Nx, ny=Ny, tau=tau)

        x0 = 64
        sigma = 4.0
        x = np.arange(Nx, dtype=np.float64)
        rho_1d = 1.0 + 0.02 * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
        rho_2d = np.tile(rho_1d, (Ny, 1))

        solver.initialize(rho=rho_2d)

        # Propagate for T = 40 time steps
        T = 40
        solver.run(T)

        # Locate right-traveling acoustic peak
        rho_t = solver.rho[Ny // 2]
        peak_right = np.argmax(rho_t[x0:]) + x0
        simulated_distance = peak_right - x0
        simulated_speed = simulated_distance / T

        # Expected sound speed c_s = 1 / sqrt(3) ~ 0.57735
        expected_speed = CS
        relative_speed_error = abs(simulated_speed - expected_speed) / expected_speed

        assert relative_speed_error < 0.01, (
            f"Simulated speed {simulated_speed:.4f} vs c_s={expected_speed:.4f}, error={relative_speed_error:.4e}"
        )


class TestDamBreakLowMachCriterion:
    """Validate Mach number constraint on gravity parameterization for dam-break flows."""

    def test_torricelli_mach_number_bound(self):
        r"""To ensure Ma <= 0.1 for a collapsing liquid column of height H,

        the Torricelli velocity U_torr = \sqrt{2 g H} must satisfy:
        Ma_torr = \sqrt{6 g H} <= 0.1  ==>  g <= 0.01 / (6 H).
        """
        H = 20.0  # 20 lattice nodes column height
        max_allowed_g = 0.01 / (6.0 * H)

        # For g = max_allowed_g, Torricelli Mach number is exactly 0.1
        u_peak = np.sqrt(2.0 * max_allowed_g * H)
        mach_peak = u_peak / CS

        assert np.isclose(mach_peak, 0.1, rtol=1e-5)
        # Verify that max_allowed_g is of order 8.3e-5 (confirming small-g requirement)
        assert 8.0e-5 < max_allowed_g < 8.5e-5
