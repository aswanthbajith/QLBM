"""Automated Test Suite for the Systematic 3-Tier QCFD Framework.

Validates:
1. Unified 100% unitary quantum step circuit architecture (Collision + Streaming + Boundary + QPE).
2. Clean uncomputation of obstacle ancillas (residual entanglement < 1e-14).
3. Zero-spectral-leakage QPE interferometry across all 3 Cartesian axes (X, Y, Z).
4. Full execution of the 3-tier comparative benchmark pipeline (Classical vs Ideal vs Noisy Hardware).
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

from qlbm.classical.d3q19 import C, CS_SQ, WEIGHTS
from research.quantum_fluid_evolution.evolution.quantum_phase_estimation_velocity import (
    QuantumPhaseEstimationVelocityProfiler,
)
from research.quantum_fluid_evolution.evolution.unified_quantum_step_circuit import (
    UnifiedQuantumFluidStepCircuit,
)
from ASWANTH.executables.run_systematic_qcfd_benchmark import (
    execute_systematic_benchmark,
    run_tier1_classical_baseline,
    run_tier2_ideal_quantum,
)


class TestSystematicQCFDFramework:
    """Test suite for systematic QCFD verification and 3-tier benchmark."""

    def test_unified_quantum_circuit_creation(self) -> None:
        """Verify unified circuit allocates expected qubits, registers, and sub-circuits."""
        solid = np.zeros((2, 2, 2), dtype=bool)
        solid[0, :, :] = True

        unified = UnifiedQuantumFluidStepCircuit(
            nx=2, ny=2, nz=2, solid_mask=solid, theta=0.06, enable_qpe=True, qpe_axis=2
        )
        metrics = unified.circuit_metrics()

        assert metrics["total_qubits"] == 11  # 3 spatial + 5 vel + 1 wall + 2 qpe
        assert metrics["spatial_qubits"] == 3
        assert metrics["velocity_qubits"] == 5
        assert metrics["wall_ancilla_qubits"] == 1
        assert metrics["qpe_precision_qubits"] == 2
        assert metrics["uncomputed_wall_ancilla"] is True
        assert metrics["zero_spectral_leakage_qpe"] is True
        assert metrics["circuit_depth"] > 0
        assert metrics["total_gates"] > 50

    def test_wall_ancilla_zero_residual_entanglement(self) -> None:
        """Verify wall boundary ancilla uncomputes to |0> with exact machine-zero residual."""
        solid = np.zeros((2, 2, 2), dtype=bool)
        solid[0, :, :] = True

        unified = UnifiedQuantumFluidStepCircuit(
            nx=2, ny=2, nz=2, solid_mask=solid, theta=0.05, enable_qpe=False
        )
        qc = unified.circuit

        # Prepare uniform superposition on fluid qubits with wall ancilla in |0>
        tot_q = unified.total_qubits  # 9 qubits (3 spatial + 5 vel + 1 wall)
        qc_init = QuantumCircuit(tot_q)
        for q in range(unified.total_spatial_qubits + unified.num_vel_qubits):
            qc_init.h(q)

        psi_start = Statevector.from_label("0" * tot_q).evolve(qc_init)
        psi_out = psi_start.evolve(qc)
        probs = psi_out.probabilities()

        # Wall ancilla is qubit index 8 (highest qubit): basis states >= 256
        prob_wall_1 = sum(probs[256:])
        assert prob_wall_1 < 1e-14, f"Residual entanglement on wall ancilla: {prob_wall_1:.2e}"

    def test_qpe_zero_spectral_leakage_all_axes(self) -> None:
        """Verify zero spectral leakage and exact moment equivalence across X, Y, and Z axes."""
        cs2 = 1.0 / 3.0
        # Test targets for each axis
        axis_targets = [(0, 0.08), (1, -0.06), (2, -0.15)]

        for axis, u_target in axis_targets:
            profiler = QuantumPhaseEstimationVelocityProfiler(
                num_spatial_qubits=0,
                lattice_type="D3Q19",
                num_precision_qubits=2,
                axis=axis,
            )
            f_test = np.copy(WEIGHTS) * 1.0
            for i in range(19):
                f_test[i] += WEIGHTS[i] * (C[i, axis] * u_target / cs2)

            res = profiler.profile_from_distribution(f_test)
            assert res["quantum_vs_classical_error"] < 1e-12
            assert abs(res["mean_velocity"] - u_target) < 1e-10
            # Check directional probability dominance
            if u_target > 0:
                assert res["p_upward_positive"] > res["p_downward_negative"]
            else:
                assert res["p_downward_negative"] > res["p_upward_positive"]

    def test_systematic_benchmark_pipeline(self, tmp_path: Path) -> None:
        """Verify the full 3-tier benchmark execution and JSON report generation."""
        out_dir = str(tmp_path / "benchmark_test_output")
        report = execute_systematic_benchmark(
            nx=2, ny=2, nz=2, theta=0.05, shots=512, output_dir=out_dir
        )

        assert report["lattice"]["nx"] == 2
        assert "tier1_classical" in report
        assert "tier2_ideal_quantum" in report
        assert "tier3_noisy_hardware" in report
        assert "comparative_metrics" in report

        comp = report["comparative_metrics"]
        assert comp["mission_status"] == "SUCCESS"
        assert comp["error_ideal_quantum"] < 0.05

        # Check JSON file written
        json_file = Path(out_dir) / "systematic_benchmark_report.json"
        assert json_file.exists()
        assert json_file.stat().st_size > 0
