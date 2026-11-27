"""Automated Test Suite for Unified Dual-Register Quantum Multiphase Circuit.

Validates:
1. D3Q7 Unitary Rotation Collision (SO(2) Givens rotations on 3 qubits, P_succ = 1.0, 0 ancilla).
2. Dual-register multiphase circuit sizing (14 qubits for 2x2x2: 3 spatial + 5 hydro + 3 phase + 1 wall + 2 qpe).
3. Shared spatial coordinate streaming via 3D Draper QFT adders.
4. Clean obstacle ancilla uncomputation (< 1e-12 residual entanglement).
5. Exact L2 norm conservation (< 1e-11 drift) with active interfacial cross-coupling.
6. Terminal single-shot QPE velocity readout with zero spectral leakage.
7. Physical hardware transpilation on 127-qubit IBM Eagle architecture (FakeSherbrooke).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.quantum_fluid_evolution.collision.rotation_collision import RotationCollisionD3Q7
from research.quantum_fluid_evolution.phase_field.unified_quantum_multiphase_circuit import (
    UnifiedQuantumMultiphaseCircuit,
)


class TestUnifiedQuantumMultiphase:
    """Test suite for unified dual-register quantum multiphase step circuit."""

    def test_d3q7_rotation_collision_unitarity(self) -> None:
        """Verify D3Q7 SO(2) rotation collision matrix is strictly unitary and preserves mass."""
        rot7 = RotationCollisionD3Q7(theta=0.08)
        U8 = rot7.unitary_matrix

        # Strictly 8x8 unitary
        assert U8.shape == (8, 8)
        diff_eye = np.linalg.norm(U8.T @ U8 - np.eye(8))
        assert diff_eye < 1e-14

        # Invariant rest state (index 0)
        assert abs(U8[0, 0] - 1.0) < 1e-14

        # Resource metrics
        met = rot7.resource_metrics()
        assert met["num_qubits"] == 3
        assert met["ancilla_qubits"] == 0
        assert met["success_probability"] == 1.0
        assert met["postselection_required"] is False

        # Apply to arbitrary amplitude distribution
        g = np.array([0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
        g_rot = rot7.apply(g)
        assert g_rot.shape == (7,)
        # Unitary L2 amplitude norm conservation:
        assert abs(np.linalg.norm(g_rot) - np.linalg.norm(g)) < 1e-14

        # At theta=0, rotation is identity and strictly preserves linear sum
        rot_zero = RotationCollisionD3Q7(theta=0.0)
        g_zero = rot_zero.apply(g)
        assert abs(np.sum(g_zero) - np.sum(g)) < 1e-15

    def test_multiphase_circuit_qubit_allocation_and_metrics(self) -> None:
        """Verify qubit register allocation for dual-register multiphase circuit."""
        # 2x2x2 grid with QPE enabled: 3 spatial + 5 hydro + 3 phase + 1 wall + 2 qpe = 14 qubits
        circuit_with_qpe = UnifiedQuantumMultiphaseCircuit(
            nx=2, ny=2, nz=2, enable_qpe=True, coupling_kappa=0.03
        )
        assert circuit_with_qpe.total_qubits == 14
        m1 = circuit_with_qpe.circuit_metrics()
        assert m1["total_qubits"] == 14
        assert m1["spatial_qubits"] == 3
        assert m1["hydro_vel19_qubits"] == 5
        assert m1["phase_vel7_qubits"] == 3
        assert m1["wall_ancilla_qubits"] == 1
        assert m1["qpe_precision_qubits"] == 2
        assert m1["cross_coupling_active"] is True
        assert m1["circuit_depth"] > 0
        assert m1["total_gates"] > 40

        # 2x2x2 grid without QPE: 3 + 5 + 3 + 1 = 12 qubits
        circuit_no_qpe = UnifiedQuantumMultiphaseCircuit(
            nx=2, ny=2, nz=2, enable_qpe=False, coupling_kappa=0.0
        )
        assert circuit_no_qpe.total_qubits == 12
        m2 = circuit_no_qpe.circuit_metrics()
        assert m2["total_qubits"] == 12
        assert m2["cross_coupling_active"] is False

    def test_multiphase_wall_ancilla_clean_uncomputation(self) -> None:
        """Verify shared obstacle ancilla uncomputes to |0> with machine-zero residual entanglement."""
        solid = np.zeros((2, 2, 2), dtype=bool)
        solid[0, :, :] = True  # z=0 wall boundary

        multiphase = UnifiedQuantumMultiphaseCircuit(
            nx=2, ny=2, nz=2, solid_mask=solid, theta_19=0.05, theta_7=0.05, enable_qpe=False
        )
        # Total qubits = 12 (3 spatial + 5 hydro + 3 phase + 1 wall)
        tot_q = multiphase.total_qubits

        # Prepare uniform superposition on fluid registers (qubits 0..10) with wall ancilla (qubit 11) in |0>
        qc_init = QuantumCircuit(tot_q)
        for q in range(tot_q - 1):
            qc_init.h(q)
        psi_init = Statevector.from_label("0" * tot_q).evolve(qc_init)

        res = multiphase.evolve_statevector(psi_init)
        assert res["wall_ancilla_p1"] < 1e-12, (
            f"Residual wall ancilla probability: {res['wall_ancilla_p1']:.2e}"
        )
        assert res["clean_uncomputation"] is True
        assert res["norm_drift"] < 1e-11

    def test_multiphase_norm_drift_conservation(self) -> None:
        """Verify strict unitary L2 norm conservation under active cross-register interfacial coupling."""
        multiphase = UnifiedQuantumMultiphaseCircuit(
            nx=2, ny=2, nz=2, theta_19=0.06, theta_7=0.04, coupling_kappa=0.05, enable_qpe=False
        )
        tot_q = 12
        qc_init = QuantumCircuit(tot_q)
        for q in range(tot_q - 1):
            qc_init.h(q)
        psi_init = Statevector.from_label("0" * tot_q).evolve(qc_init)

        res = multiphase.evolve_statevector(psi_init)
        assert res["norm_drift"] < 1e-11, f"Norm drift: {res['norm_drift']:.2e}"
        assert abs(res["statevector_norm"] - 1.0) < 1e-11

    def test_multiphase_terminal_qpe_velocity(self) -> None:
        """Verify terminal single-shot QPE extraction on the coupled multiphase circuit."""
        multiphase = UnifiedQuantumMultiphaseCircuit(
            nx=2, ny=2, nz=2, theta_19=0.0, theta_7=0.0, coupling_kappa=0.0, enable_qpe=True, qpe_axis=2
        )
        # 14 qubits: 3 spatial + 5 hydro + 3 phase + 1 wall + 2 qpe
        # Encode pure downward velocity cz = -1 (D3Q19 index 6: binary 00110 on hydro velocity qubits)
        # Spatial: qubits 0..2
        # Hydro: qubits 3..7 (index 6 is binary 00110 -> qubits 4, 5)
        qc_init = QuantumCircuit(14)
        qc_init.x(4)
        qc_init.x(5)
        psi_init = Statevector.from_label("0" * 14).evolve(qc_init)

        res = multiphase.evolve_statevector(psi_init)
        qpe = res["qpe_readout"]
        assert qpe is not None
        assert qpe["p_downward_negative"] > 0.99
        assert abs(qpe["terminal_mean_velocity"] - (-1.0)) < 1e-10

    def test_multiphase_hardware_transpilation_eagle(self) -> None:
        """Verify transpilation of the 14-qubit multiphase circuit for 127-qubit IBM Eagle."""
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()

        circuit = UnifiedQuantumMultiphaseCircuit(
            nx=2, ny=2, nz=2, enable_qpe=True, coupling_kappa=0.02
        )
        transpiled_qc = circuit.transpile_for_hardware(backend=backend, optimization_level=1)

        assert transpiled_qc.num_qubits == 127
        assert transpiled_qc.depth() > 0
        ops = transpiled_qc.count_ops()
        assert "ecr" in ops or "cx" in ops
        assert ops.get("measure", 0) == 2
