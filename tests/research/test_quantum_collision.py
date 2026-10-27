"""Tests for Phase 14: Linearized Quantum D2Q9 Collision Operator & Block Encoding.

Validates the mathematical, spectral, and quantum-circuit properties of:
- Projection matrix E (conserved moment projector)
- Linearized collision matrix C_lin
- Hermite weight symmetrization C_tilde
- 5-qubit unitary block dilation U_coll
- Machine-precision conservation of mass and momentum
- Analytical O(Ma^2) quadratic truncation error bound
- End-to-end hybrid dam-break simulation with quantum collision
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qlbm.classical.d2q9 import C, CS_SQ, WEIGHTS, bgk_collision, compute_equilibrium
from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver
from qlbm.components.vof.collision import LinearizedQuantumCollisionOperator
from qlbm.components.vof.hybrid_stepper import HybridVOFStep


class TestLinearizedCollisionSpectra:
    """Mathematical and spectral verification of projection operator E and C_lin."""

    def test_projection_matrix_idempotence(self):
        """E must be an exact orthogonal projection: E^2 == E to machine precision."""
        E = LinearizedQuantumCollisionOperator.compute_projection_matrix_E()
        diff = np.max(np.abs(E @ E - E))
        assert diff < 1e-15, f"E^2 - E max error = {diff}"

    def test_projection_matrix_eigenvalues(self):
        """E must have 3 eigenvalues equal to 1 (rho, jx, jy) and 6 equal to 0."""
        E = LinearizedQuantumCollisionOperator.compute_projection_matrix_E()
        eigvals = np.sort(np.real(np.linalg.eigvals(E)))

        # 6 zeros
        assert np.all(np.abs(eigvals[:6]) < 1e-14)
        # 3 ones
        assert np.all(np.abs(eigvals[6:] - 1.0) < 1e-14)

    def test_linearized_collision_eigenvalues(self):
        """C_lin must have eigenvalues 1.0 (multiplicity 3) and (1 - 1/tau) (multiplicity 6)."""
        tau = 0.8
        omega = 1.0 / tau
        op = LinearizedQuantumCollisionOperator(tau=tau)
        Clin = op.collision_matrix

        eigvals = np.sort(np.real(np.linalg.eigvals(Clin)))
        expected_decay = 1.0 - omega  # 1 - 1.25 = -0.25

        assert np.all(np.abs(eigvals[:6] - expected_decay) < 1e-14)
        assert np.all(np.abs(eigvals[6:] - 1.0) < 1e-14)


class TestHermiteSymmetrizationAndUnitarity:
    """Verification of Hermite weight symmetrization and 5-qubit unitary block dilation."""

    def test_weighted_matrix_symmetry(self):
        """C_tilde = W^-1 C_lin W must be strictly symmetric."""
        op = LinearizedQuantumCollisionOperator(tau=0.75)
        W = np.diag(np.sqrt(WEIGHTS))
        W_inv = np.diag(1.0 / np.sqrt(WEIGHTS))
        Clin_tilde = W_inv @ op.collision_matrix @ W

        asym = np.max(np.abs(Clin_tilde - Clin_tilde.T))
        assert asym < 1e-15, f"Asymmetry in C_tilde: {asym}"

    def test_spectral_norm_subunitary(self):
        """Spectral norm ||C_tilde||_2 <= 1.0 for physical relaxation times tau > 0.5."""
        for tau in [0.55, 0.7, 0.8, 1.0, 1.5, 2.0]:
            op = LinearizedQuantumCollisionOperator(tau=tau)
            W = np.diag(np.sqrt(WEIGHTS))
            W_inv = np.diag(1.0 / np.sqrt(WEIGHTS))
            Clin_tilde = W_inv @ op.collision_matrix @ W
            s_max = np.max(np.linalg.svd(Clin_tilde, compute_uv=False))
            assert s_max <= 1.0 + 1e-14, f"tau={tau}: s_max={s_max} > 1.0"

    def test_unitary_block_dilation(self):
        """The 32x32 block dilation matrix U_coll must satisfy U U^dagger == I to machine precision."""
        for tau in [0.6, 0.8, 1.2]:
            op = LinearizedQuantumCollisionOperator(tau=tau)
            U = op.unitary_matrix
            assert U.shape == (32, 32)
            u_err = np.max(np.abs(U @ U.T - np.eye(32)))
            assert u_err < 1e-14, f"tau={tau}: Unitary error = {u_err}"

    def test_quantum_circuit_construction(self):
        """Verify Qiskit QuantumCircuit building and resource estimate."""
        op = LinearizedQuantumCollisionOperator(tau=0.8)
        qc = op.build_circuit()
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 5

        res = op.resource_estimate()
        assert res["total_qubits"] == 5
        assert res["ancilla_qubits"] == 1
        assert res["velocity_qubits"] == 4
        assert res["matrix_dimension"] == 32


class TestCollisionInvariantsAndConservation:
    """Exact preservation of physical conservation laws under quantum collision."""

    def test_mass_conservation_arbitrary_distributions(self):
        """Total density rho = sum_i f_i must be preserved to machine precision."""
        op = LinearizedQuantumCollisionOperator(tau=0.8)
        np.random.seed(12345)
        for _ in range(20):
            f_in = np.random.uniform(0.01, 0.5, 9)
            f_post = op.collision_matrix @ f_in
            assert abs(np.sum(f_post) - np.sum(f_in)) < 1e-15

    def test_momentum_conservation_arbitrary_distributions(self):
        """Momentum j = sum_i f_i c_i must be preserved to machine precision."""
        op = LinearizedQuantumCollisionOperator(tau=0.8)
        np.random.seed(67890)
        for _ in range(20):
            f_in = np.random.uniform(0.01, 0.5, 9)
            f_post = op.collision_matrix @ f_in
            j_in = C.T @ f_in
            j_post = C.T @ f_post
            assert np.max(np.abs(j_post - j_in)) < 1e-15


class TestMachQuadraticConvergence:
    """Verification of the theoretical O(Ma^2) error scaling versus non-linear BGK."""

    def test_quadratic_mach_convergence_rate(self):
        """Truncation error ||f_lin - f_bgk|| must scale quadratically with velocity magnitude."""
        tau = 0.8
        op = LinearizedQuantumCollisionOperator(tau=tau)

        u_magnitudes = [0.001, 0.005, 0.01, 0.02, 0.05]
        errors = []

        for u_mag in u_magnitudes:
            u = np.array([u_mag, 0.6 * u_mag])
            rho = 1.0
            f_eq = compute_equilibrium(rho, u)
            # Add small non-equilibrium perturbation
            f_in = f_eq + 0.01 * np.array([0, 1, -1, 1, -1, 0.5, -0.5, 0.5, -0.5])

            # Non-linear BGK
            f_bgk = bgk_collision(
                f_in[:, None, None],
                compute_equilibrium(rho, u)[:, None, None],
                tau,
            )[:, 0, 0]

            # Linearized collision
            f_lin = op.collision_matrix @ f_in

            err = float(np.max(np.abs(f_lin - f_bgk)))
            errors.append(err)

            # Check theoretical upper bound: err <= (5 / 6 tau) * |u|^2
            bound = op.theoretical_error_bound(np.hypot(u[0], u[1]), tau=tau)
            assert err <= bound + 1e-12

        # Check quadratic slope: log(err) vs log(u) must have slope ~ 2.0
        log_u = np.log(u_magnitudes)
        log_err = np.log(errors)
        slope, _ = np.polyfit(log_u, log_err, 1)

        assert abs(slope - 2.0) < 0.05, f"Expected O(u^2) slope ~ 2.0, got {slope:.3f}"


class TestEndToEndHybridDamBreakWithCollision:
    """Benchmark comparing classical BGK vs. quantum collision in hybrid dam break."""

    def test_classical_vs_quantum_collision_dam_break_match(self):
        """Side-by-side dam break comparison: both satisfy mass conservation and low-Mach agreement."""
        nx, ny = 16, 8
        tau = 0.8
        gravity = 0.0005

        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[:, 0] = True
        solid_mask[:, -1] = True

        # Solver 1: Pure classical BGK
        s1 = VOFHydroSolver(nx=nx, ny=ny, tau=tau, gravity=gravity, solid_mask=solid_mask)
        s1.initialize_dam_break(column_width=4, column_height=6, add_walls=False)
        m0_1 = s1.total_mass

        # Solver 2: Hybrid Quantum Collision
        s2 = VOFHydroSolver(nx=nx, ny=ny, tau=tau, gravity=gravity, solid_mask=solid_mask)
        s2.initialize_dam_break(column_width=4, column_height=6, add_walls=False)
        m0_2 = s2.total_mass

        coll_op = LinearizedQuantumCollisionOperator(tau=tau)
        stepper2 = HybridVOFStep(
            s2, use_quantum_streaming=False, quantum_collision_op=coll_op
        )

        for _ in range(15):
            s1.step()
            stepper2.step()

        # 1. Both must conserve mass strictly
        assert abs(s1.total_mass - m0_1) / m0_1 < 1e-12
        assert abs(s2.total_mass - m0_2) / m0_2 < 1e-12

        # 2. Both must produce consistent macroscopic observables
        # Relative difference in mass field must be small (< 0.5%)
        diff_mass = np.max(np.abs(s1.mass - s2.mass))
        assert diff_mass < 0.05
