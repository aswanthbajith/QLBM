"""Quantitative Benchmark of D3Q19 Quantum Collision Operators vs Classical BGK.

Compares three quantum collision paradigms against classical non-linear BGK:
1. Linearized Block-Encoding BGK (6 qubits: 5 velocity + 1 ancilla)
2. Denoising Projection Collision (6 qubits: 5 velocity + 1 ancilla, arXiv:2604.09997)
3. Unitary SO(2) Rotation Collision (5 qubits: 0 ancilla, arXiv:2609.16721)

Produces a comparative scorecard evaluating:
- Mass & momentum conservation
- Relaxation accuracy vs non-linear BGK
- Unitarity precision ||U U^dagger - I||
- Ancilla overhead and success probability P_succ
- Circuit resources (qubits, depth, gates)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C, bgk_collision, compute_equilibrium, compute_macroscopic
from qlbm.components.vof.collision_3d import LinearizedQuantum3DCollisionOperator
from research.quantum_fluid_evolution.collision.denoising_collision import DenoisingCollision3D
from research.quantum_fluid_evolution.collision.rotation_collision import RotationCollision3D


def run_collision_comparison_benchmark(
    tau: float = 0.8,
    u_char: float = 0.05,
) -> Dict[str, Any]:
    """Execute comparative benchmark of all three quantum collision operators."""
    # 1. Create a representative non-equilibrium distribution with velocity u_char
    rho0 = 1.0
    u0 = np.array([u_char, 0.0, -u_char], dtype=np.float64)
    feq = compute_equilibrium(np.array([rho0]), u0[:, None])[:, 0]

    # Add non-equilibrium perturbation (viscous stress mode)
    perturb = np.zeros(19, dtype=np.float64)
    perturb[1] = 0.02
    perturb[2] = -0.02
    perturb[5] = -0.015
    perturb[6] = 0.015
    f_in = feq + perturb

    rho_field, u_field = compute_macroscopic(f_in[:, None, None, None])
    rho_in = float(rho_field[0, 0, 0])
    u_in = u_field[:, 0, 0, 0]
    j_in = rho_in * u_in

    # 2. Classical Non-linear BGK Reference
    feq_actual = compute_equilibrium(rho_field, u_field)
    f_bgk = bgk_collision(f_in[:, None, None, None], feq_actual, tau=tau)[:, 0, 0, 0]
    rho_bgk = float(np.sum(f_bgk))
    j_bgk = np.sum(f_bgk[:, None] * C, axis=0)

    # Analytical low-Mach quadratic error bound: ||f_lin - f_bgk|| <= 5/(6*tau) * ||u||^2
    u_mag_sq = float(np.sum(u_in**2))
    analytical_error_bound = float((5.0 / (6.0 * tau)) * u_mag_sq)

    # 3. Instantiate Quantum Operators
    op_lin = LinearizedQuantum3DCollisionOperator(tau=tau, g=(0, 0, 0), dt=1.0)
    op_denoise = DenoisingCollision3D(tau=tau, dt=1.0)
    op_rot = RotationCollision3D(theta=0.05)

    # 4. Apply Operators
    f_lin = op_lin.apply(f_in)
    f_denoise = op_denoise.apply(f_in)
    f_rot = op_rot.apply(f_in)

    # 5. Evaluate Invariants & Discrepancies
    results = {}

    for name, f_out, op, is_unitary in [
        ("Linearized Block-Encoding", f_lin, op_lin, True),
        ("Denoising Projector", f_denoise, op_denoise, True),
        ("Unitary SO(2) Rotation", f_rot, op_rot, True),
    ]:
        rho_out = float(np.sum(f_out))
        j_out = np.sum(f_out[:, None] * C, axis=0)

        mass_err = abs(rho_out - rho_in)
        mom_err = float(np.linalg.norm(j_out - j_in))
        bgk_diff = float(np.max(np.abs(f_out - f_bgk)))

        U = op.unitary_matrix
        dim = U.shape[0]
        u_err = float(np.max(np.abs(U @ U.T - np.eye(dim))))

        res_metrics = op.resource_metrics() if hasattr(op, "resource_metrics") else op.resource_estimate()

        results[name] = {
            "mass_conservation_error": mass_err,
            "momentum_conservation_error": mom_err,
            "max_difference_vs_bgk": bgk_diff,
            "analytical_error_bound": analytical_error_bound,
            "bound_satisfied": bool(bgk_diff <= analytical_error_bound),
            "unitarity_error": u_err,
            "unitary_dimension": dim,
            "qubits": res_metrics.get("num_qubits", 6),
            "ancilla_qubits": res_metrics.get("ancilla_qubits", 0),
            "success_probability": res_metrics.get("success_probability", 1.0),
        }

    return results


def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print clean formatted scorecard table comparing collision models."""
    print("=" * 105)
    print("  QUANTUM D3Q19 COLLISION OPERATOR SCORECARD (vs Classical Non-linear BGK)")
    print("=" * 105)
    header = f"{'Collision Model':<28} | {'Qubits':^6} | {'Ancilla':^7} | {'P_succ':^8} | {'Mass Err':^11} | {'Mom Err':^11} | {'vs BGK Err':^11} | {'Unitary Err':^11}"
    print(header)
    print("-" * 105)

    for name, m in results.items():
        row = (
            f"{name:<28} | "
            f"{m['qubits']:^6d} | "
            f"{m['ancilla_qubits']:^7d} | "
            f"{m['success_probability']:^8.4f} | "
            f"{m['mass_conservation_error']:^11.2e} | "
            f"{m['momentum_conservation_error']:^11.2e} | "
            f"{m['max_difference_vs_bgk']:^11.4e} | "
            f"{m['unitarity_error']:^11.2e}"
        )
        print(row)
    print("=" * 105)


if __name__ == "__main__":
    res = run_collision_comparison_benchmark()
    print_comparison_table(res)
