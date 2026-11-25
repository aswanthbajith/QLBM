"""Automated Verification Suite for Deterministic Dissipation & 3D Multiphase Benchmarks.

Covers:
1. Deterministic signed two-rail CPTP quantum channel (arXiv:2604.25429).
2. Four-paradigm quantum collision scorecard and scaling invariants.
3. Benchmark 1: Static 3D Droplet & Laplace Law (Tolman diffuse curvature correction).
4. Benchmark 2: Translating 3D Droplet Advection & Exact Mass Conservation.
5. Benchmark 3: Falling 3D Droplet Free-Fall Acceleration under Gravity.
6. Benchmark 4: 3D Waterfall Step Detachment, Cliff Plunge & Obstacle Collision.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from research.quantum_fluid_evolution.collision.deterministic_dissipation_channel import (
    DeterministicDissipationChannel,
    SignedTwoRailDensityMatrix,
)
from research.quantum_fluid_evolution.collision.collision_benchmarks import (
    run_collision_comparison_benchmark,
)
from research.quantum_fluid_evolution.phase_field.multiphase_benchmarks import (
    MultiphaseBenchmarkSuite,
)


class TestDeterministicDissipationChannel:
    """Verifies the signed two-rail CPTP dissipative quantum channel."""

    def test_kraus_completeness_and_trace_preservation(self):
        """Kraus operators must satisfy sum_k K_k^dag K_k = I_2 to machine precision."""
        for tau in [0.6, 0.8, 1.0, 1.5, 2.0]:
            channel = DeterministicDissipationChannel(tau=tau)
            K0, K1 = channel.kraus_operators()
            
            completeness = K0.conj().T @ K0 + K1.conj().T @ K1
            np.testing.assert_allclose(
                completeness, np.eye(2, dtype=np.complex128),
                atol=1e-14,
                err_msg=f"Kraus completeness failed for tau={tau}",
            )

    def test_signed_two_rail_density_matrix_properties(self):
        """Signed two-rail encoding must produce unit-trace Hermitian density matrices."""
        channel = DeterministicDissipationChannel(tau=0.85)

        # Test both positive and negative values of delta
        for delta in [-0.25, -0.05, 0.0, 0.12, 0.35]:
            two_rail = SignedTwoRailDensityMatrix(delta=delta, delta_max=0.5)
            rho = two_rail.to_density_matrix()

            assert rho.shape == (2, 2)
            assert np.isclose(np.trace(rho), 1.0, atol=1e-14)
            np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
            # Eigenvalues must be non-negative (positive semi-definite)
            eigvals = np.linalg.eigvalsh(rho)
            assert np.all(eigvals >= -1e-14)

    def test_deterministic_relaxation_evolution(self):
        """Channel action E(rho) must deterministically dissipate delta without shot loss."""
        tau = 0.8
        channel = DeterministicDissipationChannel(tau=tau)

        delta_in = 0.20
        two_rail_in = SignedTwoRailDensityMatrix(delta=delta_in, delta_max=0.5)
        
        rho_out, delta_out = channel.apply_channel(two_rail_in)

        # Trace of output state must be exactly 1.0 (deterministic, no postselection)
        assert np.isclose(np.trace(rho_out), 1.0, atol=1e-14)
        
        # Verify analytical relaxation: delta_out = (1 - 1/tau) * delta_in
        expected_delta_out = (1.0 - 1.0 / tau) * delta_in
        assert np.isclose(delta_out, expected_delta_out, atol=1e-12)

    def test_multi_step_survival_probability_scaling(self):
        """Multi-step CPTP evolution must have P_succ identically 1.0 for all step counts."""
        channel = DeterministicDissipationChannel(tau=0.8)
        
        for steps in [1, 2, 4, 10, 50, 100]:
            p_succ = channel.success_probability(steps=steps)
            assert p_succ == 1.0


class TestFourParadigmCollisionScorecard:
    """Verifies the four-paradigm comparative collision benchmark."""

    def test_collision_comparison_suite_invariants(self):
        results = run_collision_comparison_benchmark(tau=0.85)

        assert len(results) == 4
        assert "Linearized Block-Encoding" in results
        assert "Denoising Projector" in results
        assert "Unitary SO(2) Rotation" in results
        assert "Deterministic CPTP Channel" in results

        # Check deterministic channel entry
        cptp = results["Deterministic CPTP Channel"]
        assert cptp["success_probability"] == 1.00
        assert cptp["mass_conservation_error"] < 1e-12
        assert cptp["momentum_conservation_error"] < 1e-12
        assert cptp["ancilla_qubits"] == 0


class TestCanonicalMultiphaseBenchmarks:
    """Automated tests for the four canonical 3D multiphase quantum benchmarks."""

    @pytest.fixture(scope="class")
    @classmethod
    def suite(cls) -> MultiphaseBenchmarkSuite:
        return MultiphaseBenchmarkSuite()

    def test_benchmark_1_static_droplet_laplace_law(self, suite: MultiphaseBenchmarkSuite):
        """Benchmark 1: Static 3D droplet satisfies Laplace-Tolman relation within 10%."""
        res = suite.run_static_droplet_laplace_test(r0=3.5, sigma=0.02, num_steps=35)
        
        assert res["passed"] is True
        assert res["relative_error"] < 0.10  # Tolman-corrected relative error
        assert res["max_parasitic_velocity"] < 0.02
        assert res["delta_p_measured"] > 0.010

    def test_benchmark_2_translating_droplet_advection(self, suite: MultiphaseBenchmarkSuite):
        """Benchmark 2: Translating 3D droplet preserves order parameter mass exactly."""
        res = suite.run_translating_droplet_test(
            u0=(0.04, 0.0, 0.0),
            grid_size=(8, 8, 16),
            num_steps=20,
        )

        assert res["passed"] is True
        # Exact mass conservation in CAC LBM
        assert res["mass_relative_error"] < 1e-12
        # Displacement closely matches advection u0 * num_steps
        assert res["displacement_error"] < 0.10

    def test_benchmark_3_falling_droplet_gravity(self, suite: MultiphaseBenchmarkSuite):
        """Benchmark 3: Droplet in free-fall under gravity accelerates downwards."""
        res = suite.run_falling_droplet_test(
            gz=-0.004,
            grid_size=(16, 8, 8),
            num_steps=20,
        )

        assert res["passed"] is True
        assert res["vel_downward"] is True
        # Free-fall trajectory Delta z ~ 0.5 * g * t^2
        assert res["freefall_z_error"] < 0.5

    def test_benchmark_4_waterfall_detachment_and_obstacle(self, suite: MultiphaseBenchmarkSuite):
        """Benchmark 4: Water flows over cliff, enters free fall, impacts lower basin."""
        res = suite.run_waterfall_step_detachment_test(
            grid_size=(16, 8, 16),
            num_steps=35,
        )

        assert res["passed"] is True
        assert res["water_detached_over_cliff"] is True
        assert res["downward_momentum_accumulated"] is True
        assert res["solid_penetration_count"] == 0.0
