"""Unit and integration tests for quantum VOF components and hybrid stepping."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlbm.classical.d2q9 import C, OPPOSITE, stream_and_bounce
from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver
from qlbm.classical.vof import CellType
from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.vof import (
    HybridVOFStep,
    QuantumCollisionOperator,
    QuantumD2Q9BounceBackOperator,
    QuantumD2Q9StreamingOperator,
    QuantumVOFComponent,
)


class DummyCollisionOperator(QuantumCollisionOperator):
    """Minimal concrete collision operator for testing the abstract interface."""

    def build_circuit(
        self, lattice: Any, model_parameters: Dict[str, Any]
    ) -> QuantumCircuit:
        return QuantumCircuit(4)

    def resource_estimate(
        self, lattice: Any, model_parameters: Dict[str, Any]
    ) -> Dict[str, int]:
        return {"num_qubits": 4, "depth": 0, "size": 0}


class TestQuantumVOFComponents:
    """Level 1: Verification of component interfaces, circuit construction, and register sizing."""

    def test_quantum_collision_operator_interface(self):
        """QuantumCollisionOperator enforces abstract method implementation."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class directly
            QuantumCollisionOperator()  # type: ignore

        dummy = DummyCollisionOperator()
        qc = dummy.build_circuit(None, {"tau": 0.8})
        assert isinstance(qc, QuantumCircuit)
        res = dummy.resource_estimate(None, {})
        assert res["num_qubits"] == 4

    def test_d2q9_bounce_back_operator_construction(self):
        """QuantumD2Q9BounceBackOperator builds correct circuit for D2Q9."""
        bb_op = QuantumD2Q9BounceBackOperator(encoding=ABEncodingType.AB)
        assert bb_op.width() == 4
        assert "QuantumD2Q9BounceBackOperator" in str(bb_op)

        # Test controlled bounce-back
        bb_ctrl = QuantumD2Q9BounceBackOperator(
            encoding=ABEncodingType.AB, controlled=True
        )
        assert bb_ctrl.width() == 5

    def test_d2q9_bounce_back_permutation_correctness(self):
        """Bounce-back unitary strictly swaps opposite velocity states for D2Q9."""
        bb_op = QuantumD2Q9BounceBackOperator(encoding=ABEncodingType.AB)
        circ = bb_op.circuit

        for v in range(9):
            psi_in = Statevector.from_int(v, 16)
            psi_out = psi_in.evolve(circ)
            probs = psi_out.probabilities()
            v_out = int(np.argmax(probs))
            assert v_out == OPPOSITE[v], (
                f"Velocity {v} bounced to {v_out}, expected {OPPOSITE[v]}"
            )
            assert abs(probs[v_out] - 1.0) < 1e-12

    def test_d2q9_streaming_operator_circuit_allocation(self):
        """Streaming operator allocates exact log2(nx) + log2(ny) + 4 + 1 qubits."""
        nx, ny = 4, 8
        streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)
        metrics = streamer.resource_metrics()
        # 2 qubits for x, 3 qubits for y, 4 for velocity, 1 for obstacle ancilla = 10
        assert metrics["num_qubits"] == 10
        assert metrics["num_x_qubits"] == 2
        assert metrics["num_y_qubits"] == 3
        assert metrics["num_vel_qubits"] == 4
        assert metrics["size"] > 0
        assert metrics["depth"] > 0


class TestQuantumStreamingEquivalence:
    """Level 2: Strict numerical equivalence between quantum circuit and classical streaming."""

    def test_all_9_discrete_velocities_stream_correctly(self):
        """Every single D2Q9 velocity shifts coordinates by (cx, cy) modulo grid size."""
        nx, ny = 4, 4
        streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)
        circ = streamer.circuit

        x_in, y_in = 1, 2
        for v in range(9):
            cx, cy = C[v]
            # State indexing in ABLattice: x | (y << num_x) | (vel << (num_x + num_y))
            state_idx = x_in | (y_in << 2) | (v << 4)
            psi = Statevector.from_int(state_idx, 2**circ.num_qubits).evolve(circ)
            probs = psi.probabilities()
            out_idx = int(np.argmax(probs))

            x_out = out_idx & 3
            y_out = (out_idx >> 2) & 3
            v_out = (out_idx >> 4) & 15

            expected_x = (x_in + cx) % nx
            expected_y = (y_in + cy) % ny

            assert x_out == expected_x, f"v={v}: x_out={x_out} != expected {expected_x}"
            assert y_out == expected_y, f"v={v}: y_out={y_out} != expected {expected_y}"
            assert v_out == v, f"v={v}: v_out={v_out} != {v}"
            assert abs(probs[out_idx] - 1.0) < 1e-12

    def test_distribution_array_streaming_matches_classical(self):
        """QuantumD2Q9StreamingOperator.stream_distribution_array matches NumPy streaming to machine precision."""
        nx, ny = 4, 4
        streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)

        rng = np.random.default_rng(12345)
        f_in = rng.uniform(0.01, 1.0, size=(9, ny, nx))

        f_quantum = streamer.stream_distribution_array(f_in)
        f_classical = stream_and_bounce(f_in, solid_mask=None)

        max_diff = float(np.max(np.abs(f_quantum - f_classical)))
        assert max_diff < 1e-12, f"Discrepancy {max_diff:.2e} exceeds tolerance 1e-12"


class TestHybridVOFStep:
    """Level 3: End-to-end integration and conservation of HybridVOFStep."""

    def test_hybrid_step_matches_classical_step(self):
        """Hybrid step with quantum streaming produces identical macroscopic state to classical solver."""
        nx, ny = 8, 8
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[:, 0] = True
        solid_mask[:, -1] = True

        # Initialize two identical solvers
        solver_classical = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.001, solid_mask=solid_mask
        )
        solver_classical.initialize_dam_break(column_width=3, column_height=4)

        solver_hybrid = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.001, solid_mask=solid_mask
        )
        solver_hybrid.initialize_dam_break(column_width=3, column_height=4)

        stepper = HybridVOFStep(solver_hybrid, use_quantum_streaming=True)

        # Run 5 steps
        for _ in range(5):
            solver_classical.step()
            stepper.step()

        # Compare hydrodynamic and geometric fields
        diff_mass = float(np.max(np.abs(solver_classical.mass - solver_hybrid.mass)))
        diff_rho = float(np.max(np.abs(solver_classical.rho - solver_hybrid.rho)))
        diff_u = float(np.max(np.abs(solver_classical.u - solver_hybrid.u)))
        diff_fill = float(
            np.max(np.abs(solver_classical.fill_level - solver_hybrid.fill_level))
        )

        assert diff_mass < 1e-11, f"Mass mismatch: {diff_mass:.2e}"
        assert diff_rho < 1e-11, f"Density mismatch: {diff_rho:.2e}"
        assert diff_u < 1e-11, f"Velocity mismatch: {diff_u:.2e}"
        assert diff_fill < 1e-11, f"Fill level mismatch: {diff_fill:.2e}"

    def test_hybrid_step_mass_conservation_strict(self):
        """HybridVOFStep preserves liquid mass to machine precision across time."""
        nx, ny = 8, 8
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, gravity=0.001)
        solver.initialize_dam_break(column_width=3, column_height=4)

        stepper = HybridVOFStep(solver, use_quantum_streaming=True)

        for _ in range(10):
            stepper.step()
            assert solver.mass_error() < 1e-12
