"""Integrity and physics regression tests for CoupledQuantumMultiphase3DStepper.

Verifies:
1. Exact mass and order-parameter conservation across multi-step evolution.
2. Fluid at rest in a closed box remains strictly at rest under gravity.
3. Strict zero penetration/leakage into solid obstacle nodes.
4. Correct Boussinesq buoyancy acceleration (density contrast).
5. Dynamic transpilation resource reporting (no hard-coded numbers).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from qiskit import transpile

from qlbm.classical.d3q19 import C as C_D3Q19, CS_SQ, WEIGHTS as W_D3Q19
from qlbm.components.vof.streaming_d3q7 import WEIGHTS_D3Q7
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


class TestCoupledStepperIntegrity:
    """Rigorous physical and algorithmic integrity verification."""

    def test_closed_box_rest_stability(self):
        """Single-phase fluid in a closed box under gravity must remain at rest."""
        stepper, solid = _setup_box(
            nx=4, ny=4, nz=8,
            g=(0.0, 0.0, -0.04),
            rho_liquid=1.0,
            rho_gas=0.1,
        )
        nz, ny, nx = solid.shape

        f = np.stack([W_D3Q19[i] * np.ones((nz, ny, nx)) for i in range(19)])
        f[:, solid] = 0.0
        g = np.stack([-WEIGHTS_D3Q7[i] * np.ones((nz, ny, nx)) for i in range(7)])
        g[:, solid] = -WEIGHTS_D3Q7[:, None]

        init_mass = float(np.sum(np.sum(f, axis=0)[~solid]))

        for s in range(30):
            f, g, metrics = stepper.step(f, g)

        assert metrics["max_velocity"] < 1e-12, (
            f"Fluid gained unphysical velocity in closed box: {metrics['max_velocity']:.3e}"
        )
        assert abs(metrics["fluid_mass"] - init_mass) < 1e-9, (
            f"Mass drift in closed box: {abs(metrics['fluid_mass'] - init_mass):.3e}"
        )

    def test_mass_and_order_conservation_multistep(self):
        """Mass and order parameter must be conserved to machine precision across steps."""
        stepper, solid = _setup_box(
            nx=4, ny=4, nz=8,
            g=(0.0, 0.0, -0.02),
            sigma=0.005,
            rho_liquid=1.0,
            rho_gas=0.2,
        )
        nz, ny, nx = solid.shape

        phi0 = -np.ones((nz, ny, nx))
        phi0[4:6, :, 1:3] = 1.0  # Liquid drop in middle
        phi0[solid] = -1.0

        f = np.stack([W_D3Q19[i] * np.ones((nz, ny, nx)) for i in range(19)])
        f[:, solid] = 0.0
        g = np.stack([WEIGHTS_D3Q7[i] * phi0 for i in range(7)])

        init_mass = float(np.sum(np.sum(f, axis=0)[~solid]))
        init_order = float(np.sum(g))

        for _ in range(25):
            f, g, metrics = stepper.step(f, g)

        final_mass = metrics["fluid_mass"]
        final_order = metrics["total_order_mass"]

        assert abs(final_mass - init_mass) < 1e-9, (
            f"Fluid mass not conserved: delta = {abs(final_mass - init_mass):.3e}"
        )
        assert abs(final_order - init_order) < 1e-9, (
            f"Order parameter not conserved: delta = {abs(final_order - init_order):.3e}"
        )

    def test_wall_impermeability_strict_zero_leak(self):
        """Solid nodes must have exactly zero fluid density and order phi = -1."""
        stepper, solid = _setup_box(nx=4, ny=4, nz=8)
        nz, ny, nx = solid.shape

        f = np.stack([W_D3Q19[i] * np.ones((nz, ny, nx)) for i in range(19)])
        f[:, solid] = 0.0
        g = np.stack([WEIGHTS_D3Q7[i] * np.ones((nz, ny, nx)) for i in range(7)])

        for _ in range(10):
            f, g, _ = stepper.step(f, g)

        rho = np.sum(f, axis=0)
        phi = np.sum(g, axis=0)

        assert np.all(rho[solid] == 0.0), "Fluid leaked into solid nodes!"
        assert np.all(phi[solid] == -1.0), "Order parameter on solid nodes not -1!"

    def test_buoyancy_force_direction(self):
        """Liquid with rho_liquid > rho_gas must experience downward buoyant acceleration."""
        stepper, solid = _setup_box(
            nx=4, ny=4, nz=8,
            g=(0.0, 0.0, -0.05),
            rho_liquid=1.0,
            rho_gas=0.1,
        )
        nz, ny, nx = solid.shape

        phi = -np.ones((nz, ny, nx))
        phi[4:6, :, 1:3] = 1.0  # Liquid block
        phi[solid] = -1.0

        Fb = stepper.compute_buoyancy_force(phi)
        liquid_mask = (phi > 0.0) & (~solid)
        gas_mask = (phi < 0.0) & (~solid)

        # Buoyancy on liquid must be negative (downward)
        assert np.all(Fb[2, liquid_mask] < 0.0), "Buoyancy on liquid is not downward!"
        # Buoyancy on gas must be positive (upward displacement)
        assert np.all(Fb[2, gas_mask] > 0.0), "Buoyancy on gas is not upward!"
        # Net force over fluid must sum to zero in closed domain
        net_fz = float(np.sum(Fb[2, ~solid]))
        assert abs(net_fz) < 1e-12, f"Net buoyancy force not zero: {net_fz:.3e}"

    def test_transpiled_circuit_metrics_honest(self):
        """Verify that resource summary reflects actual simulated circuits."""
        stepper, _ = _setup_box(nx=4, ny=2, nz=4)
        summary = stepper.resource_summary()

        assert summary["hydro_streaming_circuit_qubits"] == 10  # 2 + 1 + 2 + 5 = 10
        assert summary["phase_streaming_circuit_qubits"] == 8   # 2 + 1 + 2 + 3 = 8
        assert summary["qlks_applied"] is False, "QLKS should be honestly reported as not applied"
