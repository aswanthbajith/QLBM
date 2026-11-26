r"""Quantum Phase Estimation (QPE) Velocity Spectral Profiler for Lattice Boltzmann Registers.

Enables direct quantum extraction of the macroscopic velocity distribution:
  P(u_z = -1), P(u_z = 0), P(u_z = +1)
directly from quantum fluid statevectors without full wavepacket destruction or
measuring individual spatial qubits.

Mathematical Principles:
-----------------------
1. Diagonal Velocity Hamiltonian:
   In the computational velocity basis |i>, the directional velocity operator is diagonal:
     \hat{J}_z = \sum_{i=0}^{18} c_{i,z} |i><i|
   with discrete spectrum c_{i,z} \in {-1, 0, +1}.

2. Unitary Phase Encoding:
   We construct the unitary operator:
     U = \exp(i \cdot 2\pi \cdot \phi_0 \cdot \hat{J}_z)
   Choosing \phi_0 = 1/4 (phase scale 2*pi/4):
     - c_{i,z} =  0 -> phase = 0.00 -> binary fraction .00_2 -> ancilla state |00>
     - c_{i,z} = +1 -> phase = 0.25 -> binary fraction .01_2 -> ancilla state |01>
     - c_{i,z} = -1 -> phase = 0.75 -> binary fraction .11_2 -> ancilla state |11>

3. Quantum Phase Estimation (QPE):
   Precision register |0>^{\otimes m} (m >= 2 ancilla qubits):
     |0>^{\otimes m} |v>
       --[ H^{\otimes m} ]-->
       --[ Controlled-U^{2^k} ]-->
       --[ QFT_m^\dagger ]-->
     Measure precision register to obtain the exact velocity eigenvalue spectrum!
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, QFTGate
from qiskit.quantum_info import Statevector

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C as C_D3Q19, WEIGHTS as W_D3Q19
from qlbm.components.vof.streaming_d3q7 import C_D3Q7, WEIGHTS_D3Q7 as W_D3Q7


class QuantumPhaseEstimationVelocityProfiler:
    """Quantum Phase Estimation (QPE) profiler for discrete lattice velocity observables."""

    def __init__(
        self,
        num_spatial_qubits: int = 0,
        lattice_type: str = "D3Q19",
        num_precision_qubits: int = 2,
        axis: int = 2,  # 0: x, 1: y, 2: z
    ) -> None:
        self.num_spatial_qubits = int(num_spatial_qubits)
        self.lattice_type = lattice_type.upper()
        self.num_precision_qubits = int(num_precision_qubits)
        self.axis = int(axis)

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

        self.total_target_qubits = self.num_spatial_qubits + self.num_vel_qubits
        self.total_qubits = self.total_target_qubits + self.num_precision_qubits

        # Precompute diagonal velocity eigenvalues
        self.diag_c = np.zeros(self.dim_vel, dtype=np.float64)
        for i in range(self.num_vels):
            self.diag_c[i] = self.C[i, self.axis]

        # Phase base scale: 1/4 maps {-1, 0, +1} -> {0.75, 0.0, 0.25}
        self.phase_scale = 0.25

        # Construct full QPE quantum circuit
        self.circuit = self._construct_qpe_circuit()

    def _construct_qpe_circuit(self) -> QuantumCircuit:
        """Build the complete QPE circuit with controlled-U powers and inverse QFT."""
        qc = QuantumCircuit(self.total_qubits, name=f"QPE_Velocity_{self.lattice_type}")

        precision_qubits = list(
            range(self.total_target_qubits, self.total_qubits)
        )
        vel_qubits = list(
            range(self.num_spatial_qubits, self.total_target_qubits)
        )

        # 1. Initialize precision register in uniform superposition
        for q in precision_qubits:
            qc.h(q)

        # 2. Controlled unitary evolution powers U^(2^k)
        for k, ctrl in enumerate(precision_qubits):
            power = 2**k
            phases = np.exp(1j * 2.0 * np.pi * self.phase_scale * power * self.diag_c)
            gate = DiagonalGate(phases).control(1)
            qc.append(gate, [ctrl] + vel_qubits)

        # 3. Inverse QFT on precision register
        inv_qft = QFTGate(self.num_precision_qubits).inverse()
        qc.append(inv_qft, precision_qubits)

        return qc

    def profile_velocity_spectrum(
        self,
        psi: Statevector,
    ) -> Dict[str, Any]:
        """Profile the directional velocity spectrum P(c) from a quantum statevector.

        Parameters
        ----------
        psi : Statevector
            Target quantum statevector on `total_target_qubits`.

        Returns
        -------
        dict with probabilities P(c = -1), P(c = 0), P(c = +1), and mean velocity.
        """
        # Prepare composite state on total_qubits:
        # In Qiskit's little-endian ordering:
        # target register on qubits 0 .. (total_target_qubits-1)
        # precision register on qubits total_target_qubits .. (total_qubits-1)
        dim_prec = 2**self.num_precision_qubits
        psi_prec0 = Statevector.from_label("0" * self.num_precision_qubits)
        full_in = psi.expand(psi_prec0)

        # Evolve under QPE circuit
        full_out = full_in.evolve(self.circuit)
        probs = full_out.probabilities()

        # Group probabilities by precision measurement outcome
        dim_target = 2**self.total_target_qubits

        spectrum_raw = np.zeros(dim_prec, dtype=np.float64)
        for prec_idx in range(dim_prec):
            # Sum over all target basis states for this precision outcome:
            # basis state index = target_idx + prec_idx * dim_target
            start_idx = prec_idx * dim_target
            end_idx = (prec_idx + 1) * dim_target
            spectrum_raw[prec_idx] = float(np.sum(probs[start_idx:end_idx]))

        # Map precision outcomes to physical discrete velocities:
        # With phase_scale = 0.25 and m = 2 precision qubits:
        # - outcome 0 (bin 00) -> phase 0.00 -> velocity c = 0
        # - outcome 1 (bin 01) -> phase 0.25 -> velocity c = +1
        # - outcome 2 (bin 10) -> phase 0.50 -> unused (symmetric middle)
        # - outcome 3 (bin 11) -> phase 0.75 -> velocity c = -1
        p_neg = 0.0
        p_zero = 0.0
        p_pos = 0.0

        for k, p in enumerate(spectrum_raw):
            phase = (k / dim_prec) % 1.0
            d_zero = min(phase, 1.0 - phase)
            d_pos = abs(phase - 0.25)
            d_neg = abs(phase - 0.75)

            closest = np.argmin([d_zero, d_pos, d_neg])
            if closest == 0:
                p_zero += p
            elif closest == 1:
                p_pos += p
            else:
                p_neg += p

        mean_velocity = float(p_pos * (+1.0) + p_neg * (-1.0))

        return {
            "p_downward_negative": float(p_neg),
            "p_neutral_zero": float(p_zero),
            "p_upward_positive": float(p_pos),
            "mean_velocity": mean_velocity,
            "raw_precision_distribution": spectrum_raw.tolist(),
            "circuit_qubits": self.circuit.num_qubits,
            "circuit_depth": self.circuit.depth(),
        }
