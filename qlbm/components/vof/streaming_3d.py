"""Quantum D3Q19 streaming operator for the 3D VOF-LBM framework."""

from __future__ import annotations

import time
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.common.adders import PhaseShift
from qlbm.components.vof.base import QuantumVOFComponent
from qlbm.tools.utils import get_qubits_to_invert, is_two_pow


class QuantumD3Q19StreamingOperator(QuantumVOFComponent):
    """Quantum streaming operator for 3D D3Q19 lattice distributions.

    Implements conditional 3D coordinate translations along the 19 discrete
    velocities using Draper QFT adders on spatial coordinate registers:
        |x> |y> |z> |c_i> -> |(x + c_{ix}) mod Nx> |(y + c_{iy}) mod Ny> |(z + c_{iz}) mod Nz> |c_i>

    The velocity register requires ceil(log2(19)) = 5 qubits.
    Spatial registers require:
        num_x_qubits = log2(nx)
        num_y_qubits = log2(ny)
        num_z_qubits = log2(nz)
    Total qubits = num_x_qubits + num_y_qubits + num_z_qubits + 5.

    Parameters
    ----------
    nx : int
        Grid nodes in x-direction (must be a power of two).
    ny : int
        Grid nodes in y-direction (must be a power of two).
    nz : int
        Grid nodes in z-direction (must be a power of two).
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        if not is_two_pow(nx) or not is_two_pow(ny) or not is_two_pow(nz):
            raise ValueError(
                f"Grid dimensions must be powers of two, got nx={nx}, ny={ny}, nz={nz}"
            )

        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.num_x_qubits = int(np.log2(self.nx))
        self.num_y_qubits = int(np.log2(self.ny))
        self.num_z_qubits = int(np.log2(self.nz))
        self.num_vel_qubits = 5  # ceil(log2(19)) = 5

        self.total_spatial_qubits = (
            self.num_x_qubits + self.num_y_qubits + self.num_z_qubits
        )
        self.total_qubits = self.total_spatial_qubits + self.num_vel_qubits

        # Precompute flat basis indices for fast vectorized amplitude encoding/decoding
        # Layout:
        # q0 .. q(num_x-1): x register
        # q(num_x) .. q(num_x+num_y-1): y register
        # q(num_x+num_y) .. q(num_x+num_y+num_z-1): z register
        # q(total_spatial) .. q(total_qubits-1): velocity register
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        self._basis_indices = np.empty((19, self.nz, self.ny, self.nx), dtype=np.int64)
        for i in range(19):
            self._basis_indices[i] = (
                x_grid
                | (y_grid << self.num_x_qubits)
                | (z_grid << (self.num_x_qubits + self.num_y_qubits))
                | (i << self.total_spatial_qubits)
            )

        super().__init__(name="quantum_d3q19_streaming", logger=logger)
        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Construct the 3D QFT Draper adder streaming quantum circuit."""
        qc = QuantumCircuit(self.total_qubits)

        reg_x = list(range(0, self.num_x_qubits))
        reg_y = list(range(self.num_x_qubits, self.num_x_qubits + self.num_y_qubits))
        reg_z = list(
            range(
                self.num_x_qubits + self.num_y_qubits,
                self.total_spatial_qubits,
            )
        )
        reg_v = list(
            range(self.total_spatial_qubits, self.total_qubits)
        )

        dim_regs = [reg_x, reg_y, reg_z]
        # Velocity shifts for D3Q19 in each spatial dimension:
        # x: +1 -> [1, 7, 9, 11, 13],  -1 -> [2, 8, 10, 12, 14]
        # y: +1 -> [3, 7, 10, 15, 17], -1 -> [4, 8, 9, 16, 18]
        # z: +1 -> [5, 11, 14, 15, 18], -1 -> [6, 12, 13, 16, 17]
        dim_shifts = [
            ([1, 7, 9, 11, 13], [2, 8, 10, 12, 14]),
            ([3, 7, 10, 15, 17], [4, 8, 9, 16, 18]),
            ([5, 11, 14, 15, 18], [6, 12, 13, 16, 17]),
        ]

        for dim, (pos_vels, neg_vels) in enumerate(dim_shifts):
            grid_qubits = dim_regs[dim]
            n_dim = len(grid_qubits)
            if n_dim == 0:
                continue

            # Forward QFT into Fourier momentum basis
            qc.compose(QFT(n_dim), qubits=grid_qubits, inplace=True)

            for positive, vels in [(True, pos_vels), (False, neg_vels)]:
                ps = PhaseShift(
                    num_qubits=n_dim,
                    positive=positive,
                    logger=self.logger,
                )
                ps_gate = ps.circuit.control(self.num_vel_qubits).decompose()

                for vi in vels:
                    to_invert = [
                        reg_v[q]
                        for q in get_qubits_to_invert(vi, self.num_vel_qubits)
                    ]
                    if to_invert:
                        qc.x(to_invert)

                    qc.compose(
                        ps_gate,
                        qubits=reg_v + grid_qubits,
                        inplace=True,
                    )

                    if to_invert:
                        qc.x(to_invert)

            # Inverse QFT back to computational position basis
            qc.compose(QFT(n_dim, inverse=True), qubits=grid_qubits, inplace=True)

        return qc

    def resource_metrics(self) -> Dict[str, int]:
        """Compute quantum circuit resource metrics."""
        ops = self.circuit.count_ops()
        two_qubit_gates = sum(
            count
            for gate, count in ops.items()
            if gate in ("cx", "cz", "cp", "crz", "swap", "mcx")
        )
        return {
            "num_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "size": self.circuit.size(),
            "two_qubit_gates": two_qubit_gates,
            "num_x_qubits": self.num_x_qubits,
            "num_y_qubits": self.num_y_qubits,
            "num_z_qubits": self.num_z_qubits,
            "num_vel_qubits": self.num_vel_qubits,
            "statevector_dim": 2**self.circuit.num_qubits,
            "statevector_bytes": 16 * (2**self.circuit.num_qubits),
        }

    def stream_statevector(self, statevector: Statevector) -> Statevector:
        """Evolve an arbitrary input statevector under the 3D streaming circuit."""
        return statevector.evolve(self.circuit)

    def stream_distribution_array(
        self,
        f: np.ndarray,
        return_timing: bool = False,
        signed: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Apply quantum 3D streaming to a classical D3Q19 distribution array.

        Note: this executes the streaming circuit on a classical statevector
        simulator with exact amplitude readout. The streaming circuit is a
        permutation (periodic shift), so the result equals ``np.roll``.

        Parameters
        ----------
        f : np.ndarray
            Input post-collision distributions of shape (19, nz, ny, nx).
        return_timing : bool, optional
            Whether to return fine-grained phase execution timings.
        signed : bool, optional
            If False (default), use probability encoding ``sqrt(f / sum f)``,
            which requires non-negative input. If True, use signed L2 amplitude
            encoding ``f / ||f||`` and real-part readout, which is exact for the
            streaming permutation and accepts negative populations.

        Returns
        -------
        np.ndarray or tuple of (np.ndarray, dict)
            Streamed distributions of shape (19, nz, ny, nx), optionally with timing metrics.
        """
        if f.shape != (19, self.nz, self.ny, self.nx):
            raise ValueError(
                f"Expected shape (19, {self.nz}, {self.ny}, {self.nx}), got {f.shape}"
            )
        if not signed and np.any(f < 0.0):
            raise ValueError(
                f"Negative populations detected in input distribution: min(f) = {np.min(f):.6e}"
            )

        t_start = time.perf_counter()

        f_total = float(np.linalg.norm(f)) if signed else float(np.sum(f))
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
        if signed:
            amplitudes[self._basis_indices] = f / f_total
        else:
            amplitudes[self._basis_indices] = np.sqrt(f / f_total)
        psi_in = Statevector(amplitudes)
        t_prep = time.perf_counter() - t_prep_start

        # 2. Circuit evolution
        t_evolve_start = time.perf_counter()
        psi_out = self.stream_statevector(psi_in)
        t_evolve = time.perf_counter() - t_evolve_start

        # 3. Readout and decoding (vectorized)
        t_read_start = time.perf_counter()
        if signed:
            f_streamed = np.real(psi_out.data[self._basis_indices]) * f_total
        else:
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
