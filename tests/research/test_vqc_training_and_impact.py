"""PyTest Suite for VQC Variational Training, 27Q Transpilation & Waterfall Impact.

Validates:
1. Variational Quantum Collision (VQC) parameter calibration engine for D3Q19 and D3Q7.
2. Hardware transpilation of the coupled 27-qubit multiphase circuit across:
   - IBM superconducting native (CX + 1Q)
   - Neutral atom CZ native (CZ + 1Q)
   - Fault-tolerant Clifford+T synthesis and logical T-count estimation
3. Full multi-step quantum waterfall detachment and lower basin impact dynamics.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from qiskit import QuantumCircuit

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ASWANTH.executables.run_quantum_phase_field_waterfall import run_waterfall_simulation
from research.quantum_fluid_evolution.collision.variational_collision_trainer import (
    VariationalCollisionTrainer,
)
from research.quantum_fluid_evolution.evolution.transpilation_analysis_27q import (
    run_coupled_27q_transpilation_analysis,
)


# =====================================================================
# 1. Variational Quantum Collision Parameter Training Tests
# =====================================================================

def test_vqc_parameter_training_d3q19():
    """Verify classical VQE parameter calibration for 5-qubit D3Q19 collision ansatz."""
    trainer = VariationalCollisionTrainer(lattice_type="D3Q19", tau=0.8, num_layers=2)
    res = trainer.train(max_iter=30)

    assert res["lattice_type"] == "D3Q19"
    assert res["num_parameters"] == 15
    assert res["mean_state_fidelity"] > 0.70, f"Fidelity {res['mean_state_fidelity']:.4f} below threshold"
    assert res["final_loss"] < res["initial_loss"], "Training loss failed to decrease"

    bound_qc = res["bound_circuit"]
    assert isinstance(bound_qc, QuantumCircuit)
    assert bound_qc.num_parameters == 0  # Fully bound


def test_vqc_parameter_training_d3q7():
    """Verify classical VQE parameter calibration for 3-qubit D3Q7 phase-field collision."""
    trainer = VariationalCollisionTrainer(lattice_type="D3Q7", tau=0.8, num_layers=2)
    res = trainer.train(max_iter=30)

    assert res["lattice_type"] == "D3Q7"
    assert res["num_parameters"] == 9
    assert res["mean_state_fidelity"] > 0.85, f"Fidelity {res['mean_state_fidelity']:.4f} below threshold"
    assert res["final_loss"] < res["initial_loss"]

    bound_qc = res["bound_circuit"]
    assert isinstance(bound_qc, QuantumCircuit)
    assert bound_qc.num_parameters == 0


# =====================================================================
# 2. Coupled 27-Qubit Hardware Transpilation & Synthesis Tests
# =====================================================================

def test_coupled_27q_transpilation_all_architectures():
    """Verify hardware gate decomposition and Clifford+T estimation for coupled 27-qubit system."""
    res = run_coupled_27q_transpilation_analysis(nx=4, ny=2, nz=4)
    archs = res["architectures"]

    # 1. IBM Superconducting
    ibm = archs["ibm_superconducting"]
    assert "cx" in ibm["basis_gates"]
    assert ibm["total_cx_count"] > 1000
    assert ibm["estimated_runtime_us"] > 0.0

    # 2. Neutral Atom CZ
    cz_arch = archs["neutral_atom_cz"]
    assert "cz" in cz_arch["basis_gates"]
    assert cz_arch["total_cz_count"] > 1000
    assert cz_arch["estimated_runtime_us"] > 0.0

    # 3. Fault-Tolerant Clifford+T
    ft = archs["fault_tolerant_clifford_t"]
    assert ft["estimated_logical_t_count"] > 10000
    assert ft["synthesis_epsilon"] == 1e-3


# =====================================================================
# 3. Quantum Waterfall Detachment & Basin Impact Physics
# =====================================================================

def test_quantum_waterfall_detachment_and_impact():
    """Verify 20-step quantum waterfall simulates detachment and downward acceleration with zero penetration."""
    res = run_waterfall_simulation(
        grid_size=(8, 4, 8),
        num_steps=20,
        engine="quantum",
    )
    assert res["passed"] is True, "Waterfall physical validation failed"

    hist = res["history"]
    # Check that detachment past cliff face occurred
    assert any(h["air_falling_volume"] > 0 for h in hist), "Water failed to detach over cliff"

    # Check downward momentum under gravity
    assert min(h["jz"] for h in hist) < -1.0, "Failed to develop significant downward momentum"

    # Check strict obstacle containment (zero penetration)
    assert all(h["solid_penetration"] == 0 for h in hist), "Fluid leaked into solid obstacles"
