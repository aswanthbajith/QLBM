"""Variational Quantum Collision (VQC) Parameter Training Engine.

Based on:
  Physical Review E (Accepted 20 November 2026), DOI: 10.1103/d93h-bp5b;
  Möller & Lăcătuș (arXiv:2507.12256).

Implements classical VQE-style variational parameter optimization for the layered
VQC ansatz circuits on D3Q19 and D3Q7 lattices:
  min_{theta} L(theta) = 1/K sum_{k=1}^K (1 - |<psi_target^{(k)}| V(theta) |psi_in^{(k)}>|^2)
                         + lambda_cons * Conservation_Penalty(theta)

Calibrates the single-qubit rotation angles theta directly for arbitrary relaxation
times tau (or Reynolds numbers Re), outputting concrete, fully bound QuantumCircuit
objects executable on quantum backends without intermediate state reconstruction.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C as C_D3Q19, CS_SQ, WEIGHTS as W_D3Q19
from qlbm.components.vof.streaming_d3q7 import C_D3Q7, WEIGHTS_D3Q7 as W_D3Q7
from research.quantum_fluid_evolution.collision.variational_collision_pre import (
    VariationalQuantumCollisionD3Q19,
    VariationalQuantumCollisionD3Q7,
)


class VariationalCollisionTrainer:
    """Variational Quantum Collision Parameter Training & Calibration Engine."""

    def __init__(
        self,
        lattice_type: str = "D3Q19",
        tau: float = 0.8,
        num_layers: int = 2,
        lambda_cons: float = 5.0,
    ) -> None:
        self.lattice_type = lattice_type.upper()
        self.tau = float(tau)
        self.num_layers = int(num_layers)
        self.lambda_cons = float(lambda_cons)

        if self.lattice_type == "D3Q19":
            self.vqc = VariationalQuantumCollisionD3Q19(tau=self.tau, num_layers=self.num_layers)
            self.num_qubits = 5
            self.dim = 32
            self.num_vels = 19
            self.C = C_D3Q19
            self.W = W_D3Q19
        elif self.lattice_type == "D3Q7":
            self.vqc = VariationalQuantumCollisionD3Q7(tau=self.tau, num_layers=self.num_layers)
            self.num_qubits = 3
            self.dim = 8
            self.num_vels = 7
            self.C = C_D3Q7
            self.W = W_D3Q7
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

        self.circuit_template = self.vqc.circuit
        self.num_parameters = self.circuit_template.num_parameters

        # Generate representative training dataset
        self.training_pairs = self._generate_training_dataset()

    def _generate_training_dataset(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """Generate diverse hydrodynamic test distributions and their analytical collision targets."""
        pairs = []
        u_samples = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.06, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.08]),
            np.array([0.05, 0.05, -0.05]),
            np.array([-0.04, 0.06, 0.02]),
        ]

        for u in u_samples:
            if self.lattice_type == "D3Q19":
                f_in = np.zeros(19, dtype=np.float64)
                rho_in = 1.0
                for i in range(19):
                    cu = np.dot(self.C[i], u)
                    f_in[i] = self.W[i] * rho_in * (1.0 + cu / CS_SQ)
                # Add non-equilibrium viscous perturbation
                f_in[1] += 0.015
                f_in[2] -= 0.015
                f_in[5] += 0.010
                f_in[6] -= 0.010

                f_target = self.vqc.apply(f_in)
                rho_val = float(np.sum(f_in))
                j_val = np.tensordot(self.C, f_in, axes=([0], [0]))

                # Convert to normalized statevectors of dimension 32
                psi_in = np.zeros(self.dim, dtype=np.complex128)
                psi_target = np.zeros(self.dim, dtype=np.complex128)
                psi_in[:19] = np.sqrt(np.maximum(f_in, 0.0))
                psi_target[:19] = np.sqrt(np.maximum(f_target, 0.0))

                norm_in = float(np.linalg.norm(psi_in))
                norm_tgt = float(np.linalg.norm(psi_target))
                psi_in /= norm_in
                psi_target /= norm_tgt

                pairs.append((psi_in, psi_target, rho_val, j_val))

            else:
                # D3Q7 phase-field distributions
                phi_vals = [-0.8, -0.2, 0.0, 0.5, 0.9]
                for phi_val in phi_vals:
                    g_in = self.W * phi_val
                    g_in[1] += 0.02
                    g_in[2] -= 0.02
                    g_target = self.vqc.apply(g_in)

                    psi_in = np.zeros(self.dim, dtype=np.complex128)
                    psi_target = np.zeros(self.dim, dtype=np.complex128)
                    # Signed amplitude representation for order parameter
                    psi_in[:7] = np.sign(g_in) * np.sqrt(np.abs(g_in) + 1e-12)
                    psi_target[:7] = np.sign(g_target) * np.sqrt(np.abs(g_target) + 1e-12)

                    norm_in = float(np.linalg.norm(psi_in))
                    norm_tgt = float(np.linalg.norm(psi_target))
                    psi_in /= norm_in
                    psi_target /= norm_tgt

                    pairs.append((psi_in, psi_target, float(np.sum(g_in)), np.zeros(3)))

        return pairs

    def loss(self, params: np.ndarray) -> float:
        """Compute training loss combining state infidelity and conservation penalties."""
        bound_qc = self.circuit_template.assign_parameters(params)
        U_bound = Operator(bound_qc).data  # (dim, dim)

        total_infidelity = 0.0
        total_cons_penalty = 0.0

        for psi_in, psi_target, rho_expected, j_expected in self.training_pairs:
            psi_out = U_bound @ psi_in
            overlap = np.abs(np.vdot(psi_target, psi_out))**2
            infidelity = 1.0 - overlap
            total_infidelity += infidelity

            # Conservation penalty: check conservation subspace projection
            # In velocity computational basis:
            if self.lattice_type == "D3Q19":
                f_out_sq = np.abs(psi_out[:19])**2
                rho_measured = float(np.sum(f_out_sq))
                delta_rho = rho_measured - 1.0
                j_measured = np.tensordot(self.C, f_out_sq, axes=([0], [0]))
                total_cons_penalty += delta_rho**2 + float(np.sum(j_measured**2))
            else:
                delta_norm = float(np.sum(np.abs(psi_out[:7])**2)) - 1.0
                total_cons_penalty += delta_norm**2

        K = len(self.training_pairs)
        mean_infidelity = total_infidelity / K
        mean_cons = total_cons_penalty / K
        return float(mean_infidelity + self.lambda_cons * mean_cons)

    def train(
        self,
        max_iter: int = 80,
        method: str = "BFGS",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Execute classical optimization to calibrate VQC parameter angles theta."""
        rng = np.random.default_rng(seed)
        # Small perturbation around identity
        init_params = 0.05 * (rng.random(self.num_parameters) - 0.5)

        t0 = time.perf_counter()
        res = minimize(
            self.loss,
            init_params,
            method=method,
            options={"maxiter": max_iter, "disp": False},
        )
        elapsed = time.perf_counter() - t0

        optimal_params = res.x
        final_loss = float(res.fun)
        bound_qc = self.circuit_template.assign_parameters(optimal_params)

        # Evaluate mean test fidelity across dataset
        U_bound = Operator(bound_qc).data
        fidelities = []
        for psi_in, psi_target, _, _ in self.training_pairs:
            psi_out = U_bound @ psi_in
            fid = float(np.abs(np.vdot(psi_target, psi_out))**2)
            fidelities.append(fid)

        mean_fidelity = float(np.mean(fidelities))

        return {
            "lattice_type": self.lattice_type,
            "tau": self.tau,
            "num_layers": self.num_layers,
            "num_parameters": self.num_parameters,
            "training_time_seconds": elapsed,
            "iterations": int(res.nit) if hasattr(res, "nit") else max_iter,
            "initial_loss": float(self.loss(init_params)),
            "final_loss": final_loss,
            "mean_state_fidelity": mean_fidelity,
            "optimal_parameters": optimal_params.tolist(),
            "bound_circuit": bound_qc,
            "optimization_success": bool(res.success),
        }
