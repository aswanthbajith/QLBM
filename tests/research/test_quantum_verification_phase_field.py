"""Comprehensive Verification Suite for Quantum Fluid Verification & Phase-Field Model.

Validates the research advancements addressing the five core feedback areas:
1. Independent Classical D3Q19 Solver consistency with Quantum Stepper (< 1e-10)
2. Collision discrepancy resolution and analytical low-Mach bound (< 5/(6*tau) * u^2)
3. Physical wall boundary face-flux verification (exact zero fluid penetration)
4. Transpilation gate explosion and hardware-basis metric verification
5. Circuit-executable multi-step Repeat-Until-Success (RUS) scaling (1 to 4 steps)
6. Quantum Phase-Field (Cahn-Hilliard D3Q19 + D3Q7) dissipative block-encoding
7. Analytical equilibrium hyperbolic-tangent interface and surface tension recovery
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from qlbm.classical.d3q19 import WEIGHTS
from qlbm.components.vof.quantum_stepper_3d import QuantumFluid3DStepper
from research.quantum_fluid_evolution.boundaries.wall_flux_verification import WallFaceFluxVerifier
from research.quantum_fluid_evolution.collision.collision_benchmarks import (
    run_collision_comparison_benchmark,
)
from research.quantum_fluid_evolution.evolution.circuit_transpilation_analysis import (
    analyze_circuit_transpilation,
)
from research.quantum_fluid_evolution.evolution.shot_based_quantum_stepper import ShotBasedQuantumFluidStepper
from research.quantum_fluid_evolution.phase_field.quantum_phase_field_prototype import (
    QuantumPhaseFieldPrototype,
    WEIGHTS_D3Q7,
)
from research.quantum_fluid_evolution.validation.independent_classical_d3q19 import (
    IndependentClassicalD3Q19Solver,
)


class TestIndependentClassicalValidation:
    """Verifies that the independent classical D3Q19 reference validates quantum dynamics."""

    def test_classical_linearized_matches_quantum_stepper(self):
        """Independent classical solver must match QuantumFluid3DStepper to machine precision."""
        nx, ny, nz = 4, 4, 8
        tau = 0.8
        gz = -0.01

        # Solid floor at z = 0
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, :, :] = True

        f0 = np.zeros((19, nz, ny, nx), dtype=np.float64)
        for i in range(19):
            f0[i, ~solid] = WEIGHTS[i] * 1.0

        q_stepper = QuantumFluid3DStepper(
            nx=nx, ny=ny, nz=nz, tau=tau, g=(0.0, 0.0, gz), solid_mask=solid
        )
        c_solver = IndependentClassicalD3Q19Solver(
            nx=nx, ny=ny, nz=nz, tau=tau, g=(0.0, 0.0, gz), solid_mask=solid
        )

        num_steps = 4
        q_res = q_stepper.run_quantum_evolution(f0, num_steps=num_steps)
        c_res = c_solver.run(f0, num_steps=num_steps, linearized=True)

        diff = np.max(np.abs(q_res["f_final"] - c_res["f_final"]))
        assert diff < 1e-10, f"Discrepancy vs independent classical solver = {diff:.2e}"

    def test_collision_discrepancy_bound(self):
        """Discrepancy between linearized and non-linear BGK must strictly obey O(Ma^2) bound."""
        results = run_collision_comparison_benchmark(tau=0.8, u_char=0.08)
        lin_res = results["Linearized Block-Encoding"]
        err = lin_res["max_difference_vs_bgk"]
        bound = lin_res["analytical_error_bound"]

        assert err < bound, f"Collision error {err} exceeds theoretical bound {bound}"
        assert lin_res["bound_satisfied"] is True


class TestWallFaceFluxVerification:
    """Verifies strict non-penetration condition across stationary floor, cliff, and gate."""

    def test_floor_face_flux(self):
        """Fluid streaming followed by bounce-back on flat basin floor must yield zero net penetration."""
        verifier = WallFaceFluxVerifier(nx=8, ny=4, nz=8)
        res = verifier.test_stationary_floor_flux()

        assert res["net_penetration_flux"] == 0.0 or abs(res["net_penetration_flux"]) < 1e-14
        assert res["max_face_discrepancy"] < 1e-14
        assert res["status"] == "PASSED"

    def test_shelf_and_cliff_face_flux(self):
        """Elevated shelf and vertical drop faces must prevent fluid leaking into solid geometry."""
        verifier = WallFaceFluxVerifier(nx=8, ny=4, nz=8)
        res = verifier.test_elevated_shelf_and_cliff_flux()

        assert abs(res["net_penetration_flux"]) < 1e-14
        assert res["max_face_discrepancy"] < 1e-14
        assert res["status"] == "PASSED"

    def test_closed_gate_face_flux(self):
        """Closed vertical gate must hold zero fluid penetration flux before opening."""
        verifier = WallFaceFluxVerifier(nx=8, ny=4, nz=8)
        res = verifier.test_closed_gate_flux()

        assert abs(res["net_penetration_flux"]) < 1e-14
        assert res["max_face_discrepancy"] < 1e-14
        assert res["status"] == "PASSED"


class TestTranspilationMetrics:
    """Verifies transpilation into physical basis gates (['cx', 'u']) and gate accounting."""

    def test_micro_grid_transpilation_metrics(self):
        """Transpiling 11-qubit circuit must yield valid depth and CNOT gate counts."""
        res = analyze_circuit_transpilation(nx=2, ny=2, nz=4)

        assert res["total_qubits"] == 11
        assert res["transpiled_cx_count"] > 1000
        assert res["transpiled_depth"] > 1000
        assert res["transpiled_cx_count"] == 5238 or res["transpiled_cx_count"] > 4000


class TestMultiStepScalingAndRUS:
    """Verifies circuit-executable measurement and Repeat-Until-Success multi-step scaling."""

    def test_multi_step_scaling_1_to_4(self):
        """Multi-step evolution must maintain mass conservation and high survival probability."""
        stepper = ShotBasedQuantumFluidStepper(nx=4, ny=4, nz=8, gz=-0.01)
        results = stepper.run_multi_step_scaling_experiment(step_counts=[1, 2, 4])

        for r in results:
            assert r["mass_error"] < 1e-8, f"Mass error {r['mass_error']} too large at step {r['steps']}"
            assert r["cumulative_p_succ"] > 0.85, f"Cumulative P_succ dropped to {r['cumulative_p_succ']}"
            assert r["err_vs_classical_linear"] < 1e-10, "Linearized mismatch"
            assert r["expected_shot_cost"] < 1.30, "Excessive shot multiplier"


class TestQuantumPhaseFieldPrototype:
    """Verifies the two-distribution Quantum Phase-Field (Cahn-Hilliard) prototype."""

    def test_dissipative_block_encoding_unitarity(self):
        """Block-encoding unitary U_block must satisfy U^dag U = I to machine precision."""
        model = QuantumPhaseFieldPrototype(nx=4, ny=4, nz=8, tau_phi=0.8)
        U = model.U_block

        diff = np.max(np.abs(U @ U.conj().T - np.eye(16)))
        assert diff < 1e-12, f"Unitarity error {diff:.2e}"

    def test_cahn_hilliard_planar_equilibrium(self):
        """Analytical hyperbolic tangent profile must satisfy chemical equilibrium."""
        model = QuantumPhaseFieldPrototype(nx=4, ny=4, nz=16, tau_phi=0.8, A=0.04, kappa=0.02)
        phi_eq = model.create_planar_equilibrium_profile()
        eq = model.verify_equilibrium_state(phi_eq)

        assert eq["equilibrium_verified"], "Equilibrium state verification failed"
        assert eq["max_abs_mu"] < 0.05
        assert eq["sigma_relative_error"] < 0.20

    def test_quantum_dissipative_step_equilibrium_invariance(self):
        """Equilibrium state must remain stationary under quantum dissipative collision."""
        model = QuantumPhaseFieldPrototype(nx=4, ny=4, nz=16, tau_phi=0.8)
        phi_eq = model.create_planar_equilibrium_profile()

        g0 = np.zeros((7, 16, 4, 4), dtype=np.float64)
        for i in range(7):
            g0[i, :, :, :] = WEIGHTS_D3Q7[i] * phi_eq

        g_coll, p_succ = model.run_quantum_dissipative_step(g0)
        phi_after = np.sum(g_coll, axis=0)

        drift = float(np.max(np.abs(phi_after - phi_eq)))
        assert drift < 1e-12, f"Equilibrium drift {drift:.2e} exceeds tolerance"
        assert p_succ > 0.70, f"Ancilla success probability {p_succ:.4f} too low"
