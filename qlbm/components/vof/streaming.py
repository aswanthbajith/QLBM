"""Quantum D2Q9 streaming operator for the VOF-LBM framework."""

from __future__ import annotations

import time
from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing_extensions import override

from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.vof.base import QuantumVOFComponent
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.tools.utils import is_two_pow


class QuantumD2Q9StreamingOperator(QuantumVOFComponent):
    """Quantum streaming operator for 2D D2Q9 lattice distributions.

    Implements conditional coordinate translation along the 9 discrete
    velocities using Draper QFT adders on spatial coordinate registers:
        |x> |y> |c_i> -> |(x + c_{ix}) mod Nx> |(y + c_{iy}) mod Ny> |c_i>

    Parameters
    ----------
    nx : int
        Grid nodes in x-direction (must be a power of two).
    ny : int
        Grid nodes in y-direction (must be a power of two).
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        if not is_two_pow(nx) or not is_two_pow(ny):
            raise ValueError(f"Grid dimensions must be powers of two, got nx={nx}, ny={ny}")

        self.nx = int(nx)
        self.ny = int(ny)
        self.num_x_qubits = int(np.log2(self.nx))
        self.num_y_qubits = int(np.log2(self.ny))
        self.num_vel_qubits = 4  # ceil(log2(9)) = 4

        # Precompute flat basis indices for fast vectorized amplitude encoding/decoding
        # Layout in ABLattice:
        # q0 .. q(num_x-1): x register
        # q(num_x) .. q(num_x+num_y-1): y register
        # q(num_x+num_y) .. q(num_x+num_y+3): velocity register
        y_grid, x_grid = np.mgrid[0 : self.ny, 0 : self.nx]
        self._basis_indices = np.empty((9, self.ny, self.nx), dtype=np.int64)
        for i in range(9):
            self._basis_indices[i] = (
                x_grid
                | (y_grid << self.num_x_qubits)
                | (i << (self.num_x_qubits + self.num_y_qubits))
            )

        # Instantiate ABLattice for D2Q9
        self.lattice = ABLattice(
            {
                "lattice": {"dim": {"x": self.nx, "y": self.ny}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        super().__init__(name="quantum_d2q9_streaming", logger=logger)
        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Construct the QFT Draper adder streaming quantum circuit."""
        ab_streamer = ABStreamingOperator(self.lattice, logger=self.logger)
        return ab_streamer.circuit

    def resource_metrics(self) -> Dict[str, int]:
        """Compute quantum circuit resource metrics."""
        ops = self.circuit.count_ops()
        two_qubit_gates = sum(
            count for gate, count in ops.items() if gate in ("cx", "cz", "cp", "crz", "swap", "mcx")
        )
        return {
            "num_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "size": self.circuit.size(),
            "two_qubit_gates": two_qubit_gates,
            "num_x_qubits": self.num_x_qubits,
            "num_y_qubits": self.num_y_qubits,
            "num_vel_qubits": self.num_vel_qubits,
            "statevector_dim": 2**self.circuit.num_qubits,
            "statevector_bytes": 16 * (2**self.circuit.num_qubits),
        }

    def stream_statevector(self, statevector: Statevector) -> Statevector:
        """Evolve an arbitrary input statevector under the streaming circuit."""
        return statevector.evolve(self.circuit)

    def stream_distribution_array(
        self,
        f: np.ndarray,
        return_timing: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Apply quantum streaming to a classical D2Q9 distribution array.

        This method converts the normalized classical distribution array
        f of shape (9, ny, nx) into quantum probability amplitudes, evolves
        them under the unitary streaming circuit, and decodes the resulting
        statevector back into streamed classical distributions.

        Parameters
        ----------
        f : np.ndarray
            Input post-collision distributions of shape (9, ny, nx).
        return_timing : bool, optional
            Whether to return fine-grained phase execution timings.

        Returns
        -------
        np.ndarray or tuple of (np.ndarray, dict)
            Streamed distributions of shape (9, ny, nx), optionally with timing metrics.
        """
        if f.shape != (9, self.ny, self.nx):
            raise ValueError(
                f"Expected shape (9, {self.ny}, {self.nx}), got {f.shape}"
            )
        if np.any(f < 0.0):
            raise ValueError(
                f"Negative populations detected in input distribution: min(f) = {np.min(f):.6e}"
            )

        t_start = time.perf_counter()

        f_total = float(np.sum(f))
        if f_total < 1e-15:
            res = np.zeros_like(f)
            if return_timing:
                return res, {
                    "t_state_prep": 0.0,
                    "t_circuit_evolution": 0.0,
                    "t_readout_decoding": 0.0,
                    "t_total": time.perf_counter() - t_start,
                }
            return res

        # 1. State preparation (vectorized amplitude encoding)
        t_prep_start = time.perf_counter()
        total_dim = 2**self.circuit.num_qubits
        amplitudes = np.zeros(total_dim, dtype=np.complex128)
        amplitudes[self._basis_indices] = np.sqrt(f / f_total)
        psi_in = Statevector(amplitudes)
        t_prep = time.perf_counter() - t_prep_start

        # 2. Circuit evolution
        t_evolve_start = time.perf_counter()
        psi_out = self.stream_statevector(psi_in)
        t_evolve = time.perf_counter() - t_evolve_start

        # 3. Readout and probability extraction (vectorized decoding)
        t_read_start = time.perf_counter()
        probs = psi_out.probabilities()
        f_streamed = probs[self._basis_indices] * f_total
        t_read = time.perf_counter() - t_read_start

        t_total = time.perf_counter() - t_start

        if return_timing:
            timing = {
                "t_state_prep": t_prep,
                "t_circuit_evolution": t_evolve,
                "t_readout_decoding": t_read,
                "t_total": t_total,
            }
            return f_streamed, timing

        return f_streamed
