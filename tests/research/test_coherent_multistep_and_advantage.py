"""Automated Test Suite for Coherent Multi-Step Quantum Fluid Stepper and Asymptotic Advantage.

Validates:
1. Multi-step continuous quantum coherence without intermediate wavepacket collapse.
2. Clean ancilla uncomputation (< 1e-12 residual) across multi-step execution.
3. Exact L2 norm conservation (drift < 1e-11) over chained fluid time steps.
4. Single-shot terminal QPE velocity extraction at step T.
5. Asymptotic quantum advantage scaling, crossover grid resolution (N=512), and surface code footprints.
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

from qlbm.classical.d3q19 import C, WEIGHTS
from research.quantum_fluid_evolution.evolution.coherent_multistep_quantum_stepper import (
    CoherentMultiStepQuantumStepper,
)
from research.quantum_fluid_evolution.evolution.quantum_advantage_analyzer import (
    QuantumFluidResourceScalingEstimator,
)


class TestCoherentMultiStepQuantumStepper:
    """Test suite for coherent continuous quantum fluid evolution."""

    def test_multistep_circuit_construction_and_scaling(self) -> None:
        """Verify circuit construction, qubit mapping, and linear gate scaling with steps."""
        stepper_1 = CoherentMultiStepQuantumStepper(nx=2, ny=2, nz=2, num_steps=1, enable_terminal_qpe=True)
        stepper_2 = CoherentMultiStepQuantumStepper(nx=2, ny=2, nz=2, num_steps=2, enable_terminal_qpe=True)
        stepper_4 = CoherentMultiStepQuantumStepper(nx=2, ny=2, nz=2, num_steps=4, enable_terminal_qpe=True)

        # 3 spatial + 5 velocity + 1 wall + 2 qpe = 11 qubits
        assert stepper_1.total_qubits == 11
        assert stepper_2.total_qubits == 11
        assert stepper_4.total_qubits == 11

        m1 = stepper_1.circuit_metrics()
        m2 = stepper_2.circuit_metrics()
        m4 = stepper_4.circuit_metrics()

        assert m1["total_qubits"] == 11
        assert m2["total_qubits"] == 11
        assert m4["total_qubits"] == 11

        # Circuit depth and total gates must scale with number of steps
        assert m2["total_gates"] > m1["total_gates"]
        assert m4["total_gates"] > m2["total_gates"]
        assert m1["ancilla_recycled"] is True

    def test_coherent_multistep_norm_conservation(self) -> None:
        """Verify exact unitary L2 norm conservation (drift < 1e-11) across T = 1, 2, 4 steps."""
        tot_q = 11  # 3 spatial + 5 vel + 1 wall + 2 qpe

        # Prepare uniform superposition over fluid registers (8 spatial * 19 vel states)
        qc_init = QuantumCircuit(tot_q)
        for q in range(8):  # 3 spatial + 5 velocity
            qc_init.h(q)
        psi_init = Statevector.from_label("0" * tot_q).evolve(qc_init)

        for T in [1, 2, 4]:
            stepper = CoherentMultiStepQuantumStepper(
                nx=2, ny=2, nz=2, num_steps=T, theta=0.04, enable_terminal_qpe=False
            )
            # When enable_terminal_qpe=False, total_qubits is 9 (3 spatial + 5 vel + 1 wall)
            qc_sub = QuantumCircuit(9)
            for q in range(8):
                qc_sub.h(q)
            psi_sub = Statevector.from_label("0" * 9).evolve(qc_sub)

            res = stepper.evolve_statevector(psi_sub)
            assert res["num_steps"] == T
            assert res["norm_drift"] < 1e-11, f"Norm drift at T={T} was {res['norm_drift']:.2e}"
            assert abs(res["statevector_norm"] - 1.0) < 1e-11

    def test_coherent_multistep_wall_uncomputation(self) -> None:
        """Verify obstacle ancilla uncomputes back to |0> across multi-step execution."""
        solid = np.zeros((2, 2, 2), dtype=bool)
        solid[0, :, :] = True  # z=0 wall boundary

        for T in [1, 2, 3]:
            stepper = CoherentMultiStepQuantumStepper(
                nx=2, ny=2, nz=2, num_steps=T, solid_mask=solid, theta=0.05, enable_terminal_qpe=False
            )
            # 9 qubits: 3 spatial + 5 vel + 1 wall
            qc_init = QuantumCircuit(9)
            for q in range(8):
                qc_init.h(q)
            psi_init = Statevector.from_label("0" * 9).evolve(qc_init)

            res = stepper.evolve_statevector(psi_init)
            assert res["wall_ancilla_p1"] < 1e-12, (
                f"Residual wall ancilla probability at T={T}: {res['wall_ancilla_p1']:.2e}"
            )
            assert res["clean_uncomputation"] is True

    def test_coherent_multistep_terminal_qpe(self) -> None:
        """Verify terminal single-shot QPE readout extracts downward velocity with zero spectral leakage."""
        # Setup stepper with terminal QPE on Z axis (axis 2)
        stepper = CoherentMultiStepQuantumStepper(
            nx=2, ny=2, nz=2, num_steps=2, theta=0.0, enable_terminal_qpe=True, qpe_axis=2
        )
        # Total qubits: 11 (3 spatial + 5 vel + 1 wall + 2 qpe)
        # Initialize state with pure downward velocity cz = -1 (D3Q19 index 6: c=(0, 0, -1))
        # Binary of 6 in 5 velocity qubits: 00110 -> qubits 3, 4 are set (within reg_vel)
        qc_init = QuantumCircuit(11)
        # spatial in |0,0,0>
        # velocity = 6 -> binary 00110 on qubits [3, 4, 5, 6, 7]
        # qubit index 3 + 1 = 4, index 3 + 2 = 5
        vel_qubit_base = 3
        qc_init.x(vel_qubit_base + 1)
        qc_init.x(vel_qubit_base + 2)

        psi_init = Statevector.from_label("0" * 11).evolve(qc_init)
        res = stepper.evolve_statevector(psi_init)

        qpe = res["qpe_readout"]
        assert qpe is not None
        # Should detect downward negative velocity
        assert qpe["p_downward_negative"] > 0.99
        assert abs(qpe["terminal_mean_velocity"] - (-1.0)) < 1e-10


class TestQuantumAdvantageScalingAnalyzer:
    """Test suite for asymptotic scaling, fault-tolerant resources, and quantum advantage."""

    def test_asymptotic_resource_and_crossover(self) -> None:
        """Verify exponential spatial compression and N=512 crossover."""
        estimator = QuantumFluidResourceScalingEstimator(gpu_glups_h100=8.5)

        # Small grid (8x8x8): Classical GPU wins
        res_8 = estimator.analyze_grid_size(N=8, num_steps=1000)
        assert res_8["quantum_lbm"]["total_logical_qubits"] == 17  # 3*3 + 5 + 1 + 2 = 17
        assert res_8["quantum_advantage"]["achieves_speedup"] is False

        # Intermediate grid (256x256x256): Classical GPU still ahead
        res_256 = estimator.analyze_grid_size(N=256, num_steps=1000)
        assert res_256["quantum_lbm"]["total_logical_qubits"] == 32  # 3*8 + 8 = 32
        assert res_256["quantum_advantage"]["achieves_speedup"] is False

        # Crossover grid (512x512x512): Quantum speedup achieved
        res_512 = estimator.analyze_grid_size(N=512, num_steps=1000)
        assert res_512["quantum_lbm"]["total_logical_qubits"] == 35  # 3*9 + 8 = 35
        assert res_512["quantum_advantage"]["achieves_speedup"] is True
        assert res_512["quantum_advantage"]["speedup_ratio_vs_h100"] > 2.0

        # High-resolution grid (1024x1024x1024): Substantial speedup
        res_1024 = estimator.analyze_grid_size(N=1024, num_steps=1000)
        assert res_1024["quantum_lbm"]["total_logical_qubits"] == 38  # 3*10 + 8 = 38
        assert res_1024["classical_lbm"]["memory_ram_gb"] > 150.0  # 152 GB
        assert res_1024["quantum_advantage"]["achieves_speedup"] is True
        assert res_1024["quantum_advantage"]["speedup_ratio_vs_h100"] > 15.0

    def test_scaling_curve_generation(self) -> None:
        """Verify scaling curve generation across power-of-two grids."""
        estimator = QuantumFluidResourceScalingEstimator()
        curve = estimator.generate_scaling_curve(grid_sizes=[16, 64, 256, 1024], num_steps=500)

        assert len(curve) == 4
        # Verify logical qubits increase logarithmically: 20, 26, 32, 38
        qubit_counts = [pt["quantum_lbm"]["total_logical_qubits"] for pt in curve]
        assert qubit_counts == [20, 26, 32, 38]

        # Verify physical qubits in surface code are computed
        for pt in curve:
            ft = pt["fault_tolerant_surface_code"]
            assert ft["code_distance_d"] >= 3
            assert ft["total_system_physical_qubits"] > 0
            assert ft["total_t_count"] > 0
