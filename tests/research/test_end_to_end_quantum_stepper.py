"""Tests for End-to-End Quantum Multiphase Stepper and QPE Velocity Profiler.

Verifies:
1. CoupledQuantumMultiphase3DStepper in 'quantum_rotation' mode preserves mass & order.
2. CoupledQuantumMultiphase3DStepper in 'quantum_vqc' mode preserves mass & order.
3. Resource summary accurately reports quantum collision circuit metrics.
4. Deep VQC training (num_layers=4) converges with low loss.
5. QPE velocity profiler correctly identifies rest state (uz = 0).
6. QPE velocity profiler correctly measures downward free-fall (uz = -1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import WEIGHTS as W_D3Q19
from qlbm.components.vof.streaming_d3q7 import WEIGHTS_D3Q7
from research.quantum_fluid_evolution.collision.variational_collision_trainer import (
    VariationalCollisionTrainer,
)
from research.quantum_fluid_evolution.evolution.quantum_phase_estimation_velocity import (
    QuantumPhaseEstimationVelocityProfiler,
)
from research.quantum_fluid_evolution.phase_field.coupled_quantum_multiphase_stepper import (
    CoupledQuantumMultiphase3DStepper,
)


def _setup_box(nx: int = 4, ny: int = 4, nz: int = 8, **kwargs):
    solid = np.zeros((nz, ny, nx), dtype=bool)
    solid[0, :, :] = True          # Floor
    solid[nz - 1, :, :] = True     # Ceiling
    solid[:, :, 0] = True          # Left wall
    solid[:, :, nx - 1] = True     # Right wall
    stepper = CoupledQuantumMultiphase3DStepper(
        nx=nx, ny=ny, nz=nz, solid_mask=solid, **kwargs
    )
    return stepper, solid


class TestEndToEndQuantumStepper:
    """End-to-end quantum multiphase simulation verification."""

    def test_stepper_quantum_rotation_collision(self):
        """Stepper with SO(2) Givens rotation collision operator preserves mass and order."""
        stepper, solid = _setup_box(
            nx=4, ny=4, nz=8,
            g=(0.0, 0.0, -0.01),
            sigma=0.01,
            rho_liquid=1.0,
            rho_gas=0.2,
            collision_mode="quantum_rotation",
        )
        nz, ny, nx = solid.shape

        f = np.stack([W_D3Q19[i] * np.ones((nz, ny, nx)) for i in range(19)])
        f[:, solid] = 0.0
        g = np.stack([WEIGHTS_D3Q7[i] * np.ones((nz, ny, nx)) for i in range(7)])
        g[:, solid] = -WEIGHTS_D3Q7[:, None]

        init_mass = float(np.sum(np.sum(f, axis=0)[~solid]))
        init_order = float(np.sum(g))

        for _ in range(15):
            f, g, metrics = stepper.step(f, g)

        final_mass = metrics["fluid_mass"]
        final_order = metrics["total_order_mass"]

        assert abs(final_mass - init_mass) < 1e-9, f"Mass drift: {abs(final_mass - init_mass):.3e}"
        assert abs(final_order - init_order) < 1e-9, f"Order drift: {abs(final_order - init_order):.3e}"

        res = stepper.resource_summary()
        assert res["collision_mode"] == "quantum_rotation"
        assert res["collision_quantum"] is True
        assert res["collision_circuit_qubits"] == 5

    def test_stepper_quantum_vqc_collision(self):
        """Stepper with Variational Quantum Collision circuit preserves mass."""
        stepper, solid = _setup_box(
            nx=4, ny=4, nz=8,
            g=(0.0, 0.0, -0.01),
            sigma=0.01,
            collision_mode="quantum_vqc",
        )
        nz, ny, nx = solid.shape

        f = np.stack([W_D3Q19[i] * np.ones((nz, ny, nx)) for i in range(19)])
        f[:, solid] = 0.0
        g = np.stack([WEIGHTS_D3Q7[i] * np.ones((nz, ny, nx)) for i in range(7)])
        g[:, solid] = -WEIGHTS_D3Q7[:, None]

        init_mass = float(np.sum(np.sum(f, axis=0)[~solid]))

        for _ in range(10):
            f, g, metrics = stepper.step(f, g)

        final_mass = metrics["fluid_mass"]
        assert abs(final_mass - init_mass) < 1e-9, f"Mass drift: {abs(final_mass - init_mass):.3e}"

        res = stepper.resource_summary()
        assert res["collision_mode"] == "quantum_vqc"
        assert res["collision_quantum"] is True
        assert res["collision_circuit_qubits"] == 5

    def test_deep_vqc_training_l4(self):
        """Train deep 4-layer VQC collision ansatz for D3Q7."""
        trainer = VariationalCollisionTrainer(
            lattice_type="D3Q7",
            tau=0.8,
            num_layers=4,
        )
        assert trainer.num_parameters == 4 * 3 + 3  # 15 parameters
        report = trainer.train(max_iter=30, seed=123)

        assert report["final_loss"] < report["initial_loss"]
        assert report["mean_state_fidelity"] > 0.85
        assert report["bound_circuit"].depth() > 0

    def test_qpe_velocity_rest_state(self):
        """QPE on fluid at rest must measure dominant neutral probability P(uz = 0)."""
        profiler = QuantumPhaseEstimationVelocityProfiler(
            num_spatial_qubits=0,
            lattice_type="D3Q19",
            num_precision_qubits=2,
            axis=2,  # z-axis
        )

        # Equilibrium state at rest (zero net velocity)
        f_rest = np.copy(W_D3Q19)
        psi_rest = np.zeros(32, dtype=np.complex128)
        psi_rest[:19] = np.sqrt(f_rest)
        psi_rest /= np.linalg.norm(psi_rest)

        res = profiler.profile_velocity_spectrum(Statevector(psi_rest))

        assert res["p_neutral_zero"] > 0.60, f"Expected high neutral prob, got {res['p_neutral_zero']}"
        assert abs(res["mean_velocity"]) < 0.05, f"Expected near-zero velocity, got {res['mean_velocity']}"

    def test_qpe_velocity_downward_flow(self):
        """QPE on downward-biased velocity distribution must measure P(uz = -1) > P(uz = +1)."""
        profiler = QuantumPhaseEstimationVelocityProfiler(
            num_spatial_qubits=0,
            lattice_type="D3Q19",
            num_precision_qubits=2,
            axis=2,
        )

        # Downward biased distribution (enhanced cz = -1: index 6 in D3Q19)
        f_down = np.copy(W_D3Q19)
        f_down[6] += 0.20   # cz = -1
        f_down[5] = max(0.0, f_down[5] - 0.05)  # cz = +1
        f_down /= np.sum(f_down)

        psi_down = np.zeros(32, dtype=np.complex128)
        psi_down[:19] = np.sqrt(f_down)
        psi_down /= np.linalg.norm(psi_down)

        res = profiler.profile_velocity_spectrum(Statevector(psi_down))

        assert res["p_downward_negative"] > res["p_upward_positive"]
        assert res["mean_velocity"] < 0.0, f"Expected downward velocity, got {res['mean_velocity']}"

    def test_qpe_d3q7_velocity_profiling(self):
        """QPE profiler on D3Q7 phase-field register correctly extracts z-axis velocity."""
        profiler = QuantumPhaseEstimationVelocityProfiler(
            num_spatial_qubits=0,
            lattice_type="D3Q7",
            num_precision_qubits=2,
            axis=2,
        )

        # Rest state for D3Q7: w_zero = 0.75
        g_rest = np.copy(WEIGHTS_D3Q7)
        psi_g = np.zeros(8, dtype=np.complex128)
        psi_g[:7] = np.sqrt(g_rest)
        psi_g /= np.linalg.norm(psi_g)

        res = profiler.profile_velocity_spectrum(Statevector(psi_g))
        assert res["p_neutral_zero"] > 0.70
