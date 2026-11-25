"""Comprehensive PyTest Suite for Variational Quantum Collision & QND Coupling.

Validates:
1. Variational Quantum Collision (VQC) Ansatz (PRE Accepted Nov 2026, 10.1103/d93h-bp5b):
   - Strict matrix unitarity (U^dag U - I < 1e-14)
   - Exact mass and momentum conservation (< 1e-13)
   - Deterministic success probability P_succ == 1.0 (zero ancilla)
   - D3Q19 and D3Q7 parameterized ansatz circuits
2. Quantum Non-Demolition (QND) Moment Extraction:
   - Macroscopic momentum expectation values (Jx, Jy, Jz)
   - Single-ancilla Hadamard test interferometry without spatial wavepacket collapse
   - Total mass and phase order parameter extraction
3. Scorecard Comparison:
   - 5-paradigm collision comparison (Classical, Denoising, CPTP, Block-Encoding, VQC)
4. Coupled Quantum Multiphase Waterfall Physics:
   - Water detachment over cliff edge into air column
   - Downward acceleration under vertical gravity (Jz < 0)
   - Strict zero solid penetration (quantum obstacle oracle)
   - Order parameter mass conservation
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C as C_D3Q19, CS_SQ, WEIGHTS as W_D3Q19
from qlbm.components.vof.streaming_d3q7 import C_D3Q7, WEIGHTS_D3Q7 as W_D3Q7
from research.quantum_fluid_evolution.collision.collision_benchmarks import (
    run_collision_comparison_benchmark,
)
from research.quantum_fluid_evolution.collision.variational_collision_pre import (
    VariationalQuantumCollisionD3Q19,
    VariationalQuantumCollisionD3Q7,
)
from research.quantum_fluid_evolution.evolution.qnd_moment_extraction import (
    QuantumNonDemolitionMomentExtractor,
)
from research.quantum_fluid_evolution.phase_field.coupled_quantum_multiphase_stepper import (
    CoupledQuantumMultiphase3DStepper,
)


# =====================================================================
# 1. Variational Quantum Collision (VQC D3Q19) Tests
# =====================================================================

def test_vqc_d3q19_unitarity_and_invariants():
    """Verify VQC D3Q19 operator has exact unitarity and preserves conserved moments."""
    tau = 0.85
    vqc = VariationalQuantumCollisionD3Q19(tau=tau, dt=1.0, num_layers=2)

    # 1. Check matrix unitarity: U^dag U == I
    U = vqc.unitary_matrix
    eye32 = np.eye(32, dtype=np.float64)
    unitarity_err = float(np.max(np.abs(U.T @ U - eye32)))
    assert unitarity_err < 1e-14, f"Unitarity error {unitarity_err:.2e} exceeds threshold"

    # 2. Check collision on arbitrary perturbed distributions
    rng = np.random.default_rng(42)
    rho_target = 1.25
    u_target = np.array([0.05, -0.03, 0.08], dtype=np.float64)

    # Equilibrium + random perturbation
    f_in = np.zeros(19, dtype=np.float64)
    for i in range(19):
        cu = np.dot(C_D3Q19[i], u_target)
        f_in[i] = W_D3Q19[i] * rho_target * (1.0 + cu / CS_SQ)
    f_in += 0.01 * (rng.random(19) - 0.5)

    rho_in = float(np.sum(f_in))
    j_in = np.tensordot(C_D3Q19, f_in, axes=([0], [0]))

    f_out = vqc.apply(f_in)
    rho_out = float(np.sum(f_out))
    j_out = np.tensordot(C_D3Q19, f_out, axes=([0], [0]))

    # Exact conservation laws
    mass_err = abs(rho_out - rho_in)
    mom_err = float(np.max(np.abs(j_out - j_in)))
    assert mass_err < 1e-13, f"Mass conservation error {mass_err:.2e} exceeds 1e-13"
    assert mom_err < 1e-13, f"Momentum conservation error {mom_err:.2e} exceeds 1e-13"

    # Check 4D domain application
    nz, ny, nx = 4, 2, 4
    f_field = np.zeros((19, nz, ny, nx), dtype=np.float64)
    for i in range(19):
        f_field[i] = W_D3Q19[i] * 1.0
    f_field_coll = vqc.apply(f_field)
    assert f_field_coll.shape == (19, nz, ny, nx)
    field_mass_err = float(np.max(np.abs(np.sum(f_field_coll, axis=0) - 1.0)))
    assert field_mass_err < 1e-13


def test_vqc_d3q7_unitarity_and_invariants():
    """Verify VQC D3Q7 operator has exact unitarity and preserves order parameter."""
    tau_phi = 0.75
    vqc_phi = VariationalQuantumCollisionD3Q7(tau=tau_phi, dt=1.0, num_layers=2)

    # 1. Check matrix unitarity: U^dag U == I
    U = vqc_phi.unitary_matrix
    eye8 = np.eye(8, dtype=np.float64)
    unitarity_err = float(np.max(np.abs(U.T @ U - eye8)))
    assert unitarity_err < 1e-14, f"Unitarity error {unitarity_err:.2e} exceeds threshold"

    # 2. Check conservation of order parameter phi = sum g_i
    phi_val = 0.65
    g_in = W_D3Q7 * phi_val
    # Add non-conserved perturbation
    g_in[1] += 0.05
    g_in[2] -= 0.05
    order_in = float(np.sum(g_in))

    g_out = vqc_phi.apply(g_in)
    order_out = float(np.sum(g_out))

    order_err = abs(order_out - order_in)
    assert order_err < 1e-13, f"Order parameter conservation error {order_err:.2e} exceeds 1e-13"


def test_vqc_circuit_ansatz_properties():
    """Verify VQC circuit structure, parameter count, and zero ancilla resource metrics."""
    vqc19 = VariationalQuantumCollisionD3Q19(tau=0.8, num_layers=3)
    metrics19 = vqc19.resource_metrics()
    assert metrics19["ancilla_qubits"] == 0
    assert metrics19["success_probability"] == 1.0
    assert metrics19["num_qubits"] == 5
    assert vqc19.circuit.num_parameters == (3 * 5 + 5)  # 20 parameters

    vqc7 = VariationalQuantumCollisionD3Q7(tau=0.8, num_layers=2)
    metrics7 = vqc7.resource_metrics()
    assert metrics7["ancilla_qubits"] == 0
    assert metrics7["success_probability"] == 1.0
    assert metrics7["num_qubits"] == 3
    assert vqc7.circuit.num_parameters == (2 * 3 + 3)  # 9 parameters


# =====================================================================
# 2. Quantum Non-Demolition (QND) Moment Extraction Tests
# =====================================================================

def test_qnd_momentum_extraction_accuracy():
    """Verify QND extractor correctly extracts macroscopic momentum from a quantum statevector."""
    # 2x2x2 lattice = 8 spatial nodes -> 3 spatial qubits
    num_spatial = 3
    extractor = QuantumNonDemolitionMomentExtractor(num_spatial_qubits=num_spatial, lattice_type="D3Q19")

    # Construct physical distribution with known momentum
    nz, ny, nx = 2, 2, 2
    u_exact = np.array([0.08, -0.04, -0.06], dtype=np.float64)
    f_field = np.zeros((19, nz, ny, nx), dtype=np.float64)
    for i in range(19):
        cu = np.dot(C_D3Q19[i], u_exact)
        f_field[i] = W_D3Q19[i] * 1.0 * (1.0 + cu / CS_SQ)

    total_jx_exact = float(np.sum(C_D3Q19[:, 0, None, None, None] * f_field))
    total_jy_exact = float(np.sum(C_D3Q19[:, 1, None, None, None] * f_field))
    total_jz_exact = float(np.sum(C_D3Q19[:, 2, None, None, None] * f_field))
    total_mass_exact = float(np.sum(f_field))

    # Convert to statevector: state has dim 2^(3 + 5) = 256
    # Amplitudes are sqrt(f_i(x))
    dim_total = 2**(num_spatial + 5)
    amplitudes = np.zeros(dim_total, dtype=np.complex128)

    # Layout: velocity mode i in upper 5 qubits, spatial coordinate x in lower 3 qubits
    f_flat = f_field.reshape(19, 8)
    for i in range(19):
        for x in range(8):
            state_idx = (i << num_spatial) | x
            amplitudes[state_idx] = np.sqrt(max(f_flat[i, x], 0.0))

    norm = float(np.linalg.norm(amplitudes))
    amplitudes /= norm
    psi = Statevector(amplitudes)

    # Extract via QND
    mom_qnd = extractor.extract_momentum_expectation(psi, norm=norm)
    mass_qnd = extractor.extract_mass_expectation(psi, norm=norm)

    assert abs(mom_qnd["jx"] - total_jx_exact) < 1e-12
    assert abs(mom_qnd["jy"] - total_jy_exact) < 1e-12
    assert abs(mom_qnd["jz"] - total_jz_exact) < 1e-12
    assert abs(mass_qnd - total_mass_exact) < 1e-12


def test_qnd_hadamard_test_circuit():
    """Verify single-ancilla Hadamard test circuit executes correctly on Qiskit statevector."""
    num_spatial = 1  # 2 spatial nodes, 1 spatial qubit
    extractor = QuantumNonDemolitionMomentExtractor(num_spatial_qubits=num_spatial, lattice_type="D3Q19")

    # Simple normalized test state
    dim = 2**(num_spatial + 5)  # 64
    amp = np.zeros(dim, dtype=np.complex128)
    # Put amplitude in downward velocity mode c_6 = (0, 0, -1)
    # Mode 6 has c_z = -1
    amp[(6 << num_spatial) | 0] = 1.0
    psi = Statevector(amp)

    hadamard_res = extractor.verify_qnd_hadamard_test_circuit(psi, direction="z", theta=0.04)
    assert hadamard_res["circuit_qubits"] == (num_spatial + 5 + 1)  # 7 qubits
    assert hadamard_res["expected_J"] == -1.0
    # For theta = 0.04 and eigenvalue -1: <exp(i*theta*J)> = exp(-i*0.04)
    # Ancilla Z = Re(exp(-i*0.04)) = cos(-0.04) = 0.999200
    expected_ancilla_Z = np.cos(-0.04)
    assert abs(hadamard_res["ancilla_Z"] - expected_ancilla_Z) < 1e-6


# =====================================================================
# 3. Scorecard Benchmarks (5 Paradigms) Tests
# =====================================================================

def test_five_paradigm_collision_benchmarks():
    """Verify that comparative benchmark suite includes all 5 paradigms and VQC has P_succ=1.0."""
    results = run_collision_comparison_benchmark()
    assert len(results) >= 5, f"Expected 5 paradigms in scorecard, found {len(results)}"

    assert "Linearized Block-Encoding" in results
    assert "Denoising Projector" in results
    assert "Unitary SO(2) Rotation" in results
    assert "Deterministic CPTP Channel" in results
    assert "Variational VQC (PRE Nov 2026)" in results

    # Verify VQC entries
    vqc_res = results["Variational VQC (PRE Nov 2026)"]
    assert vqc_res["ancilla_qubits"] == 0
    assert vqc_res["success_probability"] == 1.0
    assert vqc_res["mass_conservation_error"] < 1e-12
    assert vqc_res["momentum_conservation_error"] < 1e-12


# =====================================================================
# 4. Coupled Quantum Multiphase Waterfall Physics Verification
# =====================================================================

def test_coupled_quantum_multiphase_waterfall_detachment():
    """Verify 3D quantum waterfall demonstrates detachment, downward momentum, and zero penetration."""
    nz, ny, nx = 8, 4, 8
    z_shelf = 4
    x_cliff = 2

    solid = np.zeros((nz, ny, nx), dtype=bool)
    solid[0, :, :] = True  # Basin floor
    solid[: z_shelf + 1, :, : x_cliff + 1] = True  # Shelf & cliff face
    solid[: max(2, nz // 4), :, (x_cliff + 3):(x_cliff + 6)] = True  # Basin obstacle

    # Initial liquid reservoir on shelf reaching cliff edge
    phi_init = np.full((nz, ny, nx), -1.0, dtype=np.float64)
    for z_i in range(z_shelf + 1, nz - 1):
        for y_i in range(1, ny - 1):
            for x_i in range(1, x_cliff + 1):
                if not solid[z_i, y_i, x_i]:
                    phi_init[z_i, y_i, x_i] = 1.0

    init_liquid_vol = int(np.sum(phi_init > 0.0))
    init_total_order = float(np.sum(phi_init))
    assert init_liquid_vol > 0, "Initial liquid reservoir must not be empty"

    stepper = CoupledQuantumMultiphase3DStepper(
        nx=nx, ny=ny, nz=nz,
        tau_f=0.8, tau_phi=0.8,
        sigma=0.015, g=(0.0, 0.0, -0.010),
        solid_mask=solid,
    )

    gate_velocity = 0.10
    f_curr = np.zeros((19, nz, ny, nx), dtype=np.float64)
    for i in range(19):
        f_curr[i] = W_D3Q19[i] * 1.0 * (1.0 + C_D3Q19[i, 0] * gate_velocity / CS_SQ)

    g_curr = np.zeros((7, nz, ny, nx), dtype=np.float64)
    for i in range(7):
        g_curr[i] = W_D3Q7[i] * phi_init

    history = []
    for s in range(1, 13):
        f_curr, g_curr, res = stepper.step(f_curr, g_curr)
        phi = np.sum(g_curr, axis=0)
        # Liquid in open air column past cliff edge (x > x_cliff)
        air_liquid = int(np.sum(phi[:, :, x_cliff + 1:] > 0.0))
        # Solid penetration (any fluid inside solid obstacle)
        solid_pen = int(np.sum(phi[solid] > -0.5))

        history.append({
            "step": s,
            "air_falling_volume": air_liquid,
            "jz": res["vertical_momentum_jz"],
            "solid_penetration": solid_pen,
            "total_order_mass": res["total_order_mass"],
        })

    # Assertions
    # 1. Jet detachment: liquid must spill into air column past cliff edge
    max_air_falling = max(h["air_falling_volume"] for h in history)
    assert max_air_falling > 0, f"Water failed to detach over cliff edge: max air volume {max_air_falling}"

    # 2. Downward acceleration: vertical momentum Jz must become negative and significant
    min_jz = min(h["jz"] for h in history)
    assert min_jz < -1.0, f"Expected downward acceleration Jz < -1.0, got {min_jz:.4f}"

    # 3. Boundary containment: zero penetration into solid walls or obstacles
    assert all(h["solid_penetration"] == 0 for h in history), "Detected fluid penetration into solid walls"

    # 4. Conservative order parameter mass preservation
    final_order = history[-1]["total_order_mass"]
    order_err = abs(final_order - init_total_order) / abs(init_total_order)
    assert order_err < 0.05, f"Order parameter relative error {order_err:.2e} exceeded 5%"
