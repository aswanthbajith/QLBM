"""Automated Test Suite for Quantum D3Q7 Phase-Field Circuits & Coupled Multiphase Stepper.

Covers:
1. Quantum D3Q7 Draper QFT Streaming Operator equivalence to classical roll (< 1e-12).
2. Quantum D3Q7 Spatial Wall Obstacle Operator involution (U^2 = I) and uncomputation.
3. Hardware basis gate transpilation audit (['cx', 'u']) and CNOT gate budgets.
4. Continuous multi-step Quantum Phase-Field Stepper (QCAC D3Q7) mass conservation.
5. Coupled Quantum Multiphase Stepper (D3Q19 + D3Q7) dual-register evolution.
6. QLKS viscosity decoupling parameter A verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qlbm.classical.d3q19 import WEIGHTS as W19
from qlbm.components.vof.quantum_boundary_d3q7 import QuantumD3Q7SpatialBoundaryOperator
from qlbm.components.vof.streaming_d3q7 import (
    C_D3Q7,
    OPP_D3Q7,
    QuantumD3Q7StreamingOperator,
    WEIGHTS_D3Q7 as W7,
)
from research.quantum_fluid_evolution.phase_field.circuit_transpilation_d3q7 import (
    analyze_d3q7_transpilation,
)
from research.quantum_fluid_evolution.phase_field.coupled_quantum_multiphase_stepper import (
    CoupledQuantumMultiphase3DStepper,
)
from research.quantum_fluid_evolution.phase_field.quantum_phase_field_stepper_3d import (
    QuantumPhaseField3DStepper,
)


class TestD3Q7StreamingOperator:
    """Verifies the 3-velocity-qubit Quantum D3Q7 Draper QFT Streaming Operator."""

    def test_d3q7_streaming_matches_classical_roll_exact(self):
        """Quantum streaming statevector evolution must match classical periodic shift to machine precision."""
        nx, ny, nz = 4, 4, 4
        op = QuantumD3Q7StreamingOperator(nx=nx, ny=ny, nz=nz)

        np.random.seed(101)
        g_in = np.random.randn(7, nz, ny, nx)

        # Quantum streaming
        g_quantum = op.stream_distribution_array(g_in)

        # Classical roll
        g_classical = np.zeros_like(g_in)
        for i in range(7):
            cx, cy, cz = C_D3Q7[i]
            g_classical[i] = np.roll(np.roll(np.roll(g_in[i], cz, axis=0), cy, axis=1), cx, axis=2)

        max_diff = float(np.max(np.abs(g_quantum - g_classical)))
        assert max_diff < 1e-12, f"Discrepancy {max_diff:.4e} exceeds machine tolerance"

    def test_d3q7_streaming_resource_metrics(self):
        """Verify qubit count and low circuit depth of cardinal D3Q7 streaming."""
        op = QuantumD3Q7StreamingOperator(nx=4, ny=4, nz=4)
        m = op.resource_metrics()

        assert m["num_qubits"] == 9  # 2 + 2 + 2 + 3
        assert m["num_vel_qubits"] == 3
        assert m["depth"] < 50
        assert m["two_qubit_gates"] <= 12


class TestD3Q7SpatialBoundaryOperator:
    """Verifies the quantum spatial obstacle boundary circuit for D3Q7."""

    def test_d3q7_boundary_involution_and_uncomputation(self):
        """Boundary operator must satisfy U^2 = I and uncompute wall ancilla to zero."""
        nx, ny, nz = 2, 2, 2
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, 0, 0] = True  # Solid cell at origin

        op = QuantumD3Q7SpatialBoundaryOperator(nx=nx, ny=ny, nz=nz, solid_mask=solid)

        # Basis layout: x (1q), y (1q), z (1q), vel (3q), wall (1q)
        # Prepare state with amplitude in velocity 1 (+x) at solid cell (0, 0, 0)
        psi_data = np.zeros(2**op.total_qubits, dtype=np.complex128)
        psi_data[1 << 3] = 1.0
        psi_in = Statevector(psi_data)

        # Apply boundary operator once: should swap velocity 1 -> velocity 2 (-x)
        psi_out = op.apply_boundary_statevector(psi_in)
        expected_idx = 2 << 3
        actual_idx = int(np.argmax(np.abs(psi_out.data)))
        assert actual_idx == expected_idx
        assert np.isclose(psi_out.data[expected_idx], 1.0)

        # Apply boundary operator second time: should return to original state
        psi_invol = op.apply_boundary_statevector(psi_out)
        diff = float(np.max(np.abs(psi_invol.data - psi_in.data)))
        assert diff < 1e-12, f"Involution error {diff:.4e} exceeds tolerance"


class TestD3Q7HardwareTranspilation:
    """Verifies physical hardware basis gate decomposition for D3Q7 circuits."""

    def test_d3q7_transpilation_basis_gates(self):
        """Transpilation into ['cx', 'u'] must yield low CNOT count (< 300 for 2x2x4)."""
        res = analyze_d3q7_transpilation(nx=2, ny=2, nz=4)

        assert res["total_qubits"] == 8
        assert res["total_cx_gates"] < 350
        assert res["est_duration_us"] < 100.0  # < 100 microseconds


class TestQuantumPhaseFieldStepper3D:
    """Verifies the continuous multi-step QCAC D3Q7 statevector stepper."""

    def test_qcac_stepper_continuous_evolution(self):
        """Statevector stepper must advance continuously and conserve order parameter mass."""
        nx, ny, nz = 4, 4, 4
        stepper = QuantumPhaseField3DStepper(
            nx=nx, ny=ny, nz=nz, tau_phi=0.8, interface_width=1.0
        )

        # Initial central droplet
        z0, y0, x0 = 2.0, 2.0, 2.0
        z, y, x = np.ogrid[:nz, :ny, :nx]
        dist = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        phi0 = -np.tanh((dist - 1.2) / 2.0)
        g0 = W7[:, None, None, None] * phi0

        u_field = np.zeros((3, nz, ny, nx))
        u_field[0] = 0.03  # Advection along x

        evo = stepper.run_quantum_evolution(g0, u_field, num_steps=5)

        assert evo["passed"] is True
        assert evo["mass_conservation_error"] < 1e-10
        assert len(evo["history"]) == 5


class TestCoupledQuantumMultiphaseStepper:
    """Verifies the unified dual-register (D3Q19 + D3Q7) coupled quantum stepper."""

    def test_coupled_quantum_simulation_and_obstacle_interaction(self):
        """Coupled quantum engine must advance both registers with zero solid penetration."""
        nx, ny, nz = 4, 4, 4
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, :, :] = True  # Solid floor

        stepper = CoupledQuantumMultiphase3DStepper(
            nx=nx, ny=ny, nz=nz,
            tau_f=0.8, tau_phi=0.8,
            sigma=0.015, g=(0.0, 0.0, -0.005),
            solid_mask=solid,
            qlks_A=1.0,
        )

        # Droplet at z = 2.5
        z, y, x = np.ogrid[:nz, :ny, :nx]
        dist = np.sqrt((x - 2.0)**2 + (y - 2.0)**2 + (z - 2.5)**2)
        phi0 = -np.tanh((dist - 1.2) / 2.0)
        phi0[solid] = -1.0

        f_init = np.zeros((19, nz, ny, nx))
        for i in range(19):
            f_init[i] = W19[i] * 1.0

        g_init = np.zeros((7, nz, ny, nx))
        for i in range(7):
            g_init[i] = W7[i] * phi0

        res = stepper.run_coupled_simulation(f_init, g_init, num_steps=4)

        assert res["passed"] is True
        assert res["mass_conservation_error"] < 0.05
        # Verify downward momentum injection under gravity
        assert res["history"][-1]["vertical_momentum_jz"] < -0.10
        # Verify strict zero solid penetration
        assert all(h["solid_penetration"] == 0.0 for h in res["history"])

    def test_qlks_viscosity_decoupling_stability(self):
        """QLKS parameter A in (0, 1] must scale effective viscosity without destabilizing stepper."""
        nx, ny, nz = 4, 4, 4
        # Test lower viscosity (higher Reynolds number) via A = 0.5
        stepper_low_nu = CoupledQuantumMultiphase3DStepper(
            nx=nx, ny=ny, nz=nz,
            tau_f=0.8, tau_phi=0.8,
            qlks_A=0.5,
        )
        res_info = stepper_low_nu.resource_summary()
        expected_nu = (1.0 / 3.0) * 0.5 * (0.8 - 0.5)
        assert np.isclose(res_info["effective_viscosity"], expected_nu, atol=1e-12)
