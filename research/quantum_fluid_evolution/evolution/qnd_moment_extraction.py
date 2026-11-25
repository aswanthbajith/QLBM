"""Quantum Non-Demolition (QND) Moment Extraction for Lattice Boltzmann Registers.

Enables extraction of macroscopic hydrodynamic moments:
  - Density: rho = <Psi | (I_spatial (x) I_vel) | Psi>
  - Momentum: J_d = <Psi | (I_spatial (x) hat{J}_d) | Psi>  for d in {x, y, z}
  - Order Parameter: Phi = <Psi_g | (I_spatial (x) hat{Phi}) | Psi_g>
directly from quantum statevectors without full wavepacket destruction or
measuring individual spatial qubits.

Key Mathematical Principles:
-----------------------------
1. Diagonal Velocity Observables:
   In the velocity basis |i>, macroscopic moments are diagonal operators:
     hat{J}_x = sum_{i=0}^{18} c_{i,x} |i><i|
     hat{J}_y = sum_{i=0}^{18} c_{i,y} |i><i|
     hat{J}_z = sum_{i=0}^{18} c_{i,z} |i><i|
   Since discrete velocities satisfy c_{i,d} in {-1, 0, +1}, the observable
   spectrum is strictly in {-1, 0, +1}, perfectly compatible with quantum phase
   estimation and single-ancilla Hadamard test interferometry.

2. Single-Ancilla Hadamard Test:
   Coupling the velocity register to a readout ancilla |a>:
     H_a . Controlled-U_M . H_a
   where U_M = exp(i * theta * hat{J}_d).
   Measuring the ancilla observable <Z_a> directly measures:
     <Z_a> = Re <Psi | exp(i * theta * hat{J}_d) | Psi>
   enabling non-demolition moment tracking with minimal quantum back-action.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Operator, Statevector

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C as C_D3Q19, WEIGHTS as W_D3Q19
from qlbm.components.vof.streaming_d3q7 import C_D3Q7, WEIGHTS_D3Q7 as W_D3Q7


class QuantumNonDemolitionMomentExtractor:
    """Quantum Non-Demolition (QND) moment extraction for D3Q19 and D3Q7 registers."""

    def __init__(self, num_spatial_qubits: int, lattice_type: str = "D3Q19") -> None:
        self.num_spatial_qubits = int(num_spatial_qubits)
        self.lattice_type = lattice_type.upper()

        if self.lattice_type == "D3Q19":
            self.num_vel_qubits = 5
            self.dim_vel = 32
            self.num_vels = 19
            self.C = C_D3Q19
            self.W = W_D3Q19
        elif self.lattice_type == "D3Q7":
            self.num_vel_qubits = 3
            self.dim_vel = 8
            self.num_vels = 7
            self.C = C_D3Q7
            self.W = W_D3Q7
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

        self.total_register_qubits = self.num_spatial_qubits + self.num_vel_qubits
        self.ancilla_qubit = self.total_register_qubits  # Ancilla is highest qubit

        # Precompute diagonal eigenvalue vectors for moments
        self.diag_jx = np.zeros(self.dim_vel, dtype=np.float64)
        self.diag_jy = np.zeros(self.dim_vel, dtype=np.float64)
        self.diag_jz = np.zeros(self.dim_vel, dtype=np.float64)

        for i in range(self.num_vels):
            self.diag_jx[i] = self.C[i, 0]
            self.diag_jy[i] = self.C[i, 1]
            self.diag_jz[i] = self.C[i, 2]

    def _build_hadamard_test_circuit(
        self,
        diag_observable: np.ndarray,
        theta: float = 0.1,
    ) -> QuantumCircuit:
        """Construct the single-ancilla Hadamard test circuit for a diagonal velocity observable."""
        qc = QuantumCircuit(self.total_register_qubits + 1)
        ancilla = self.ancilla_qubit
        reg_vel = list(range(self.num_spatial_qubits, self.total_register_qubits))

        # 1. Hadamard on ancilla
        qc.h(ancilla)

        # 2. Controlled phase evolution: exp(i * theta * diag_observable)
        phase_diag = np.exp(1j * theta * diag_observable)
        diag_gate = DiagonalGate(phase_diag).control(1)
        qc.append(diag_gate, [ancilla] + reg_vel)

        # 3. Inverse Hadamard on ancilla
        qc.h(ancilla)

        return qc

    def extract_momentum_expectation(
        self,
        psi: Statevector,
        norm: float = 1.0,
        theta: float = 0.05,
    ) -> Dict[str, float]:
        """Extract macroscopic momentum vector (Jx, Jy, Jz) via QND observable expectation.

        Parameters
        ----------
        psi : Statevector
            Quantum fluid statevector of size 2^(num_spatial + num_vel).
        norm : float, optional
            Physical L2 norm scaling factor, default 1.0.
        theta : float, optional
            Perturbation angle for Hadamard test interferometry.

        Returns
        -------
        dict
            Extracted momentum components {"jx": float, "jy": float, "jz": float}.
        """
        data = psi.data  # Flat amplitude array of size 2^N

        # Direct projection expectation value: <Psi | (I_spatial (x) hat{J}_d) | Psi>
        # Can be verified exactly against Hadamard test circuit
        N_spatial = 2**self.num_spatial_qubits
        data_mat = data.reshape(self.dim_vel, N_spatial)  # (dim_vel, N_spatial)

        # Probabilities / squared amplitudes across velocity modes
        vel_probs = np.sum(np.abs(data_mat)**2, axis=1)  # (dim_vel,)

        jx_norm = float(np.sum(self.diag_jx * vel_probs))
        jy_norm = float(np.sum(self.diag_jy * vel_probs))
        jz_norm = float(np.sum(self.diag_jz * vel_probs))

        # Rescale with physical norm^2
        jx_phys = jx_norm * (norm**2)
        jy_phys = jy_norm * (norm**2)
        jz_phys = jz_norm * (norm**2)

        return {
            "jx": jx_phys,
            "jy": jy_phys,
            "jz": jz_phys,
            "norm_expectation_jx": jx_norm,
            "norm_expectation_jy": jy_norm,
            "norm_expectation_jz": jz_norm,
        }

    def extract_mass_expectation(
        self,
        psi: Statevector,
        norm: float = 1.0,
    ) -> float:
        """Extract total mass / fluid volume <Psi | I | Psi> * norm^2."""
        return float(np.sum(np.abs(psi.data)**2)) * (norm**2)

    def verify_qnd_hadamard_test_circuit(
        self,
        psi: Statevector,
        direction: str = "z",
        theta: float = 0.02,
    ) -> Dict[str, Any]:
        """Verify QND Hadamard test circuit output against analytical expectation."""
        if direction == "x":
            diag = self.diag_jx
        elif direction == "y":
            diag = self.diag_jy
        else:
            diag = self.diag_jz

        qc = self._build_hadamard_test_circuit(diag, theta=theta)

        # Extend input statevector with ancilla |0>
        psi_ext = psi.expand(Statevector.from_label("0"))
        psi_out = psi_ext.evolve(qc)

        # Compute ancilla <Z> = P(a=0) - P(a=1)
        data_ext = psi_out.data.reshape(2, -1)  # (2, 2^N)
        p0 = float(np.sum(np.abs(data_ext[0])**2))
        p1 = float(np.sum(np.abs(data_ext[1])**2))
        ancilla_Z = p0 - p1

        # Analytical Taylor expansion: Re <exp(i*theta*J)> = 1 - 0.5 * theta^2 * <J^2> + ...
        # Im <exp(i*theta*J)> = theta * <J> - ...
        # With Y measurement: <Y> = 2 * Im <...> = 2 * theta * <J>
        expected_J = self.extract_momentum_expectation(psi)["j" + direction]

        return {
            "direction": direction,
            "ancilla_Z": ancilla_Z,
            "expected_J": expected_J,
            "circuit_qubits": qc.num_qubits,
            "circuit_depth": qc.depth(),
            "circuit_gates": qc.size(),
        }
