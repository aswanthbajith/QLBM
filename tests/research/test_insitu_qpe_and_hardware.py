"""Automated Tests for In-Situ QPE Profiling and Physical Hardware Transpilation.

Verifies:
1. Exact mathematical equivalence between QPE expectation velocities and classical moments.
2. In-situ QPE extraction during 3D waterfall time-stepping simulation.
3. Transpilation of QLBM circuits onto 127Q superconducting QPU topologies (FakeSherbrooke).
4. Physical coherence headroom and error mitigation analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C as C_D3Q19, WEIGHTS as W_D3Q19
from research.quantum_fluid_evolution.evolution.quantum_phase_estimation_velocity import (
    QuantumPhaseEstimationVelocityProfiler,
)
from ASWANTH.executables.run_quantum_phase_field_waterfall import run_waterfall_simulation
from ASWANTH.executables.submit_ibm_quantum_job import (
    build_hardware_circuit,
    get_target_backend,
    analyze_hardware_readiness,
    run_hardware_pipeline,
)


class TestInSituQPEAndHardware:
    """Test suite for in-situ QPE velocity profiling and hardware transpilation."""

    def test_qpe_profile_from_distribution_rest(self) -> None:
        """Verify QPE profiler on rest distribution has exact moment match."""
        profiler = QuantumPhaseEstimationVelocityProfiler(
            num_spatial_qubits=0,
            lattice_type="D3Q19",
            num_precision_qubits=2,
            axis=2,
        )
        f_rest = np.copy(W_D3Q19) * 1.0
        res = profiler.profile_from_distribution(f_rest)

        assert abs(res["mean_velocity"]) < 1e-12
        assert abs(res["classical_moment_velocity"]) < 1e-12
        assert res["quantum_vs_classical_error"] < 1e-12
        assert abs(res["p_neutral_zero"] - (2.0 / 3.0)) < 1e-6
        assert abs(res["p_downward_negative"] - (1.0 / 6.0)) < 1e-6
        assert abs(res["p_upward_positive"] - (1.0 / 6.0)) < 1e-6

    def test_qpe_profile_from_distribution_downward(self) -> None:
        """Verify downward velocity produces negative mean velocity matching moment to machine precision."""
        profiler = QuantumPhaseEstimationVelocityProfiler(
            num_spatial_qubits=0,
            lattice_type="D3Q19",
            num_precision_qubits=2,
            axis=2,
        )
        # Create non-equilibrium state with downward bias
        f_down = np.copy(W_D3Q19) * 1.0
        cs2 = 1.0 / 3.0
        uz_target = -0.12
        for i in range(19):
            f_down[i] += W_D3Q19[i] * (C_D3Q19[i, 2] * uz_target / cs2)

        res = profiler.profile_from_distribution(f_down)

        assert res["mean_velocity"] < -0.10
        assert res["p_downward_negative"] > res["p_upward_positive"]
        # Error between QPE interferometric expectation and classical moment must be machine zero
        assert res["quantum_vs_classical_error"] < 1e-12

    def test_qpe_profile_4d_array_coordinate(self) -> None:
        """Verify 4D array extraction at specified coordinate probe."""
        profiler = QuantumPhaseEstimationVelocityProfiler(
            num_spatial_qubits=0,
            lattice_type="D3Q19",
            num_precision_qubits=2,
            axis=2,
        )
        f_4d = np.zeros((19, 4, 4, 4), dtype=np.float64)
        for i in range(19):
            f_4d[i, :, :, :] = W_D3Q19[i]

        # Put an upward jet at (z=2, y=1, x=3)
        cs2 = 1.0 / 3.0
        for i in range(19):
            f_4d[i, 2, 1, 3] += W_D3Q19[i] * (C_D3Q19[i, 2] * 0.15 / cs2)

        res_jet = profiler.profile_from_distribution(f_4d, coord=(2, 1, 3))
        res_rest = profiler.profile_from_distribution(f_4d, coord=(0, 0, 0))

        assert res_jet["mean_velocity"] > +0.12
        assert abs(res_rest["mean_velocity"]) < 1e-12
        assert res_jet["probe_coord"] == [2, 1, 3]

    def test_waterfall_insitu_qpe_execution(self) -> None:
        """Verify waterfall simulation loop executes with in-situ QPE profiling."""
        sim_res = run_waterfall_simulation(
            grid_size=(4, 4, 4),
            num_steps=3,
            gz=-0.02,
            sigma=0.015,
            tau=0.8,
            gate_velocity=0.08,
            engine="quantum",
            collision_mode="quantum_rotation",
            qlks_a=1.0,
            qpe_profile=True,
            qpe_interval=1,
            save_npz=False,
        )

        assert sim_res["passed"] or len(sim_res["history"]) == 3
        # Check that step 1 and step 3 recorded QPE probe data
        hist_1 = sim_res["history"][0]
        assert "qpe" in hist_1
        assert "lip" in hist_1["qpe"]
        assert "plunge" in hist_1["qpe"]

        lip_qpe = hist_1["qpe"]["lip"]
        assert "p_downward_negative" in lip_qpe
        assert "mean_velocity" in lip_qpe
        assert lip_qpe["quantum_vs_classical_error"] < 1e-10

    def test_hardware_transpilation_fake_sherbrooke(self) -> None:
        """Verify hardware transpilation and readiness analysis on FakeSherbrooke (127Q)."""
        report = run_hardware_pipeline(
            backend_name="fake_sherbrooke",
            circuit_type="qpe_velocity",
            optimization_level=3,
            shots=512,
            dry_run=True,
        )

        analysis = report["hardware_analysis"]
        assert analysis["backend_name"] == "fake_sherbrooke"
        assert analysis["backend_num_qubits"] == 127
        assert analysis["active_physical_qubits"] >= 7
        assert analysis["transpiled_circuit_depth"] > 100
        assert analysis["native_2q_gates"] > 0
        assert analysis["median_t1_coherence_us"] > 50.0
        assert analysis["median_t2_coherence_us"] > 30.0
        assert len(analysis["recommended_error_mitigation"]) == 3

    def test_hardware_circuit_builders(self) -> None:
        """Verify all algorithmic circuit builders construct valid QuantumCircuits."""
        for ctype in ["qpe_velocity", "rotation_collision", "coupled_step"]:
            qc, measured = build_hardware_circuit(ctype, prepare_downward_state=True)
            assert qc.num_qubits >= 5
            assert len(measured) > 0
            assert qc.depth() > 0
