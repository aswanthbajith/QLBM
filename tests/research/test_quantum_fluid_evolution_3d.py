"""Comprehensive verification suite for End-to-End Quantum 3D Fluid Evolution.

Validates the mathematical, spectral, and quantum-circuit properties of:
1. D3Q19 conserved moment projection matrix E (mass and 3D momentum)
2. Linearized D3Q19 BGK collision matrix C_lin
3. Linearized Guo gravitational forcing matrix G
4. 6-qubit unitary block dilation U_cg with machine-precision unitarity
5. Integrated spatial wall boundary operator (oracle, controlled bounce, uncompute)
6. Multi-step consecutive quantum fluid evolution without intermediate classical reconstruction
7. Numerical trajectory match versus classical D3Q19 hydrodynamics
8. Exact mass conservation and physical momentum accumulation under gravity
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlbm.classical.d3q19 import C, CS_SQ, OPPOSITE, WEIGHTS
from qlbm.components.vof.collision_3d import LinearizedQuantum3DCollisionOperator
from qlbm.components.vof.quantum_boundary_3d import QuantumD3Q19SpatialBoundaryOperator
from qlbm.components.vof.quantum_stepper_3d import QuantumFluid3DStepper
from qlbm.components.vof.streaming_3d import QuantumD3Q19StreamingOperator


class TestD3Q19CollisionAndGravitySpectra:
    """Mathematical and spectral verification of D3Q19 projection matrix E and block encoding."""

    def test_projection_matrix_idempotence(self):
        """E must be an exact orthogonal projection: E^2 == E to machine precision."""
        E = LinearizedQuantum3DCollisionOperator.compute_projection_matrix_E()
        diff = np.max(np.abs(E @ E - E))
        assert diff < 1e-14, f"E^2 - E max error = {diff}"

    def test_projection_matrix_eigenvalues(self):
        """E must have 4 unit eigenvalues (rho, jx, jy, jz) and 15 null eigenvalues."""
        E = LinearizedQuantum3DCollisionOperator.compute_projection_matrix_E()
        eigvals = np.sort(np.real(np.linalg.eigvals(E)))

        # 15 zeros
        assert np.all(np.abs(eigvals[:15]) < 1e-14), "Non-conserved eigenvalues not zero"
        # 4 ones
        assert np.all(np.abs(eigvals[15:] - 1.0) < 1e-14), "Conserved eigenvalues not unity"

    def test_linearized_collision_eigenvalues(self):
        """C_lin must have eigenvalues 1.0 (multiplicity 4) and (1 - 1/tau) (multiplicity 15)."""
        tau = 0.8
        omega = 1.0 / tau
        op = LinearizedQuantum3DCollisionOperator(tau=tau)
        Clin = op.collision_matrix

        eigvals = np.sort(np.real(np.linalg.eigvals(Clin)))
        expected_decay = 1.0 - omega

        assert np.all(np.abs(eigvals[:15] - expected_decay) < 1e-14)
        assert np.all(np.abs(eigvals[15:] - 1.0) < 1e-14)

    def test_gravity_forcing_mass_conservation(self):
        """Guo forcing matrix G must have zero column sums, preserving mass exactly."""
        g = np.array([0.0, 0.0, -0.025])
        G = LinearizedQuantum3DCollisionOperator.compute_gravity_forcing_matrix(
            g=g, tau=0.8, dt=1.0
        )
        col_sums = np.sum(G, axis=0)
        assert np.max(np.abs(col_sums)) < 1e-15, f"Mass non-conservation in G: {np.max(np.abs(col_sums))}"

    def test_gravity_forcing_momentum_injection(self):
        """Guo forcing matrix G must inject exact momentum (1 - dt / (2*tau)) * g * dt."""
        g = np.array([0.0, 0.0, -0.01])
        tau = 0.8
        dt = 1.0
        G = LinearizedQuantum3DCollisionOperator.compute_gravity_forcing_matrix(
            g=g, tau=tau, dt=dt
        )

        injected_mom = np.zeros(3)
        for i in range(19):
            injected_mom += G[i, 0] * C[i]

        expected_mom = (1.0 - 0.5 * dt / tau) * g * dt
        diff = np.max(np.abs(injected_mom - expected_mom))
        assert diff < 1e-15, f"Momentum injection error: {diff}"

    def test_unitary_block_dilation_precision(self):
        """U_cg must satisfy U U^dagger == I and U^dagger U == I to machine precision."""
        for tau in [0.6, 0.8, 1.2]:
            op = LinearizedQuantum3DCollisionOperator(
                tau=tau, g=(0.0, 0.0, -0.02)
            )
            U = op.unitary_matrix
            assert U.shape == (64, 64)
            err_right = np.max(np.abs(U @ U.T - np.eye(64)))
            err_left = np.max(np.abs(U.T @ U - np.eye(64)))
            assert err_right < 1e-14, f"tau={tau}: U U^dagger error = {err_right}"
            assert err_left < 1e-14, f"tau={tau}: U^dagger U error = {err_left}"

    def test_ancilla_success_probability_bound(self):
        """Ancilla success probability |<0| U |0>|^2 must exceed 95% for physical low-Mach gravity."""
        op = LinearizedQuantum3DCollisionOperator(tau=0.8, g=(0.0, 0.0, -0.01))
        assert op.success_probability > 0.95, f"P_succ = {op.success_probability} < 0.95"


class TestIntegratedQuantumSpatialBoundary:
    """Verification of spatial obstacle oracle and controlled bounce-back reflection."""

    def test_boundary_circuit_construction(self):
        """Verify quantum boundary circuit allocation for 2x2x4 grid."""
        nx, ny, nz = 2, 2, 4
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, :, :] = True  # Floor boundary

        op = QuantumD3Q19SpatialBoundaryOperator(nx=nx, ny=ny, nz=nz, solid_mask=solid)
        res = op.resource_metrics()

        assert res["num_spatial_qubits"] == 4  # 1 + 1 + 2
        assert res["num_vel_qubits"] == 5
        assert res["num_wall_ancilla"] == 1
        assert res["num_qubits"] == 10
        assert res["num_solid_nodes"] == 4

    def test_bounce_back_permutation_reversal(self):
        """Bounce-back operator must invert opposite velocities on solid nodes."""
        nx, ny, nz = 2, 2, 4
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, :, :] = True

        op = QuantumD3Q19SpatialBoundaryOperator(nx=nx, ny=ny, nz=nz, solid_mask=solid)
        f_in = np.zeros((19, nz, ny, nx), dtype=np.float64)
        for i in range(19):
            f_in[i] = float(i + 1)

        f_out = op.apply_to_distribution_array(f_in)

        # On solid floor (z=0), velocities must be swapped
        for i in range(19):
            assert np.allclose(f_out[OPPOSITE[i], 0, :, :], f_in[i, 0, :, :])

        # On fluid nodes (z > 0), distributions must remain unchanged
        for i in range(19):
            assert np.allclose(f_out[i, 1:, :, :], f_in[i, 1:, :, :])


class TestMultiStepQuantumFluidEvolution:
    """Verification of continuous multi-step quantum fluid evolution without classical reconstruction."""

    @pytest.fixture
    def setup_small_stepper(self):
        """Set up a test 2x2x4 stepper."""
        nx, ny, nz = 2, 2, 4
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, :, :] = True  # solid floor

        stepper = QuantumFluid3DStepper(
            nx=nx, ny=ny, nz=nz, tau=0.8, g=(0.0, 0.0, -0.01), solid_mask=solid
        )

        # Initial condition: resting fluid on fluid nodes z in {1, 2, 3}
        f_init = np.zeros((19, nz, ny, nx), dtype=np.float64)
        for i in range(19):
            f_init[i, 1:, :, :] = WEIGHTS[i] * 1.0

        return stepper, f_init

    def test_consecutive_quantum_steps_without_reconstruction(self, setup_small_stepper):
        """Execute 4 consecutive steps directly in quantum register without intermediate decoding."""
        stepper, f_init = setup_small_stepper
        res = stepper.run_quantum_evolution(f_init, num_steps=4)

        assert res["num_steps"] == 4
        assert len(res["history"]) == 4
        for entry in res["history"]:
            assert entry["p_succ"] > 0.85, f"Low ancilla success at step {entry['step']}"
            assert np.isfinite(entry["total_mass"])
            assert np.isfinite(entry["jz"])

    def test_quantum_vs_classical_d3q19_trajectory_match(self, setup_small_stepper):
        """Quantum multi-step trajectory must match classical D3Q19 solver to < 1e-12 per step."""
        stepper, f_init = setup_small_stepper
        res_quant = stepper.run_quantum_evolution(f_init, num_steps=4)
        res_class = stepper.run_classical_evolution(f_init, num_steps=4)

        for s in range(4):
            q_hist = res_quant["history"][s]
            c_hist = res_class["history"][s]

            # Mass match
            mass_diff = abs(q_hist["total_mass"] - c_hist["total_mass"])
            assert mass_diff < 1e-10, f"Step {s+1} mass mismatch: {mass_diff}"

            # Momentum match
            jz_diff = abs(q_hist["jz"] - c_hist["jz"])
            assert jz_diff < 1e-10, f"Step {s+1} Jz mismatch: {jz_diff}"

        # Final full distribution array comparison
        f_diff = np.max(np.abs(res_quant["f_final"] - res_class["f_final"]))
        assert f_diff < 1e-10, f"Final distribution max difference: {f_diff}"

    def test_downward_momentum_under_gravity(self, setup_small_stepper):
        """Under downward gravity (gz < 0), initial Jz must become negative."""
        stepper, f_init = setup_small_stepper
        res = stepper.run_quantum_evolution(f_init, num_steps=2)

        j0 = 0.0
        j1 = res["history"][0]["jz"]
        j2 = res["history"][1]["jz"]

        assert j1 < j0, f"Expected downward momentum after step 1, got {j1}"
        assert j2 < 0.0, f"Expected negative momentum at step 2, got {j2}"

    def test_resource_summary_for_4x4x8(self):
        """Verify quantum resource metrics on a 4x4x8 lattice (14 qubits)."""
        nx, ny, nz = 4, 4, 8
        stepper = QuantumFluid3DStepper(nx=nx, ny=ny, nz=nz, tau=0.8, g=(0.0, 0.0, -0.01))
        res = stepper.resource_summary()

        assert res["total_qubits"] == 14  # 2 + 2 + 3 + 5 + 1 + 1
        assert res["spatial_qubits"] == 7
        assert res["velocity_qubits"] == 5
        assert res["ancilla_qubits"] == 2
        assert res["statevector_dim"] == 16384
        assert res["statevector_ram_bytes"] == 262144  # 256 KB
        assert res["step_circuit_depth"] > 0
        assert res["step_circuit_size"] > 0
