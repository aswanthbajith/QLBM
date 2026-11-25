"""Quantum D3Q7 Streaming Operator for 3D Conservative Allen-Cahn Phase-Field Dynamics.

Implements conditional 3D coordinate translations along the 7 discrete velocities
of the D3Q7 lattice using Draper QFT adders on spatial coordinate registers:
    |x> |y> |z> |c_i> -> |(x + c_{ix}) mod Nx> |(y + c_{iy}) mod Ny> |(z + c_{iz}) mod Nz> |c_i>

Unlike D3Q19 where velocities have diagonal components, D3Q7 velocities lie
strictly along the cardinal axes (+-e_x, +-e_y, +-e_z), requiring only 3 velocity
qubits and significantly reducing circuit depth and CNOT count.
"""

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

C_D3Q7 = np.array(
    [
        [0, 0, 0],   # 0: Rest
        [1, 0, 0],   # 1: +x
        [-1, 0, 0],  # 2: -x
        [0, 1, 0],   # 3: +y
        [0, -1, 0],  # 4: -y
        [0, 0, 1],   # 5: +z
        [0, 0, -1],  # 6: -z
    ],
    dtype=np.int64,
)

WEIGHTS_D3Q7 = np.array(
    [0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
    dtype=np.float64,
)

OPP_D3Q7 = np.array([0, 2, 1, 4, 3, 6, 5], dtype=np.int64)


class QuantumD3Q7StreamingOperator(QuantumVOFComponent):
    """Quantum streaming operator for 3D D3Q7 phase-field distributions.

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
        self.num_vel_qubits = 3  # ceil(log2(7)) = 3

        self.total_spatial_qubits = (
            self.num_x_qubits + self.num_y_qubits + self.num_z_qubits
        )
        self.total_qubits = self.total_spatial_qubits + self.num_vel_qubits

        # Precompute flat basis indices for fast vectorized amplitude encoding/decoding
        # Layout:
        # q0 .. q(num_x-1): x register
        # q(num_x) .. q(num_x+num_y-1): y register
        # q(num_x+num_y) .. q(num_x+num_y+num_z-1): z register
        # q(total_spatial) .. q(total_qubits-1): velocity register (3 qubits)
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        self._basis_indices = np.empty((7, self.nz, self.ny, self.nx), dtype=np.int64)
        for i in range(7):
            self._basis_indices[i] = (
                x_grid
                | (y_grid << self.num_x_qubits)
                | (z_grid << (self.num_x_qubits + self.num_y_qubits))
                | (i << self.total_spatial_qubits)
            )

        super().__init__(name="quantum_d3q7_streaming", logger=logger)
        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Construct the 3D QFT Draper adder streaming quantum circuit for D3Q7."""
        qc = QuantumCircuit(self.total_qubits)

        reg_x = list(range(0, self.num_x_qubits))
        reg_y = list(range(self.num_x_qubits, self.num_x_qubits + self.num_y_qubits))
        reg_z = list(
            range(
                self.num_x_qubits + self.num_y_qubits,
                self.total_spatial_qubits,
            )
        )
        reg_v = list(range(self.total_spatial_qubits, self.total_qubits))

        dim_regs = [reg_x, reg_y, reg_z]
        # Velocity shifts for D3Q7 along cardinal axes:
        # x: +1 -> [1],  -1 -> [2]
        # y: +1 -> [3],  -1 -> [4]
        # z: +1 -> [5],  -1 -> [6]
        dim_shifts = [
            ([1], [2]),
            ([3], [4]),
            ([5], [6]),
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

    def resource_metrics(self) -> Dict[str, Any]:
        """Compute quantum circuit resource metrics."""
        ops = self.circuit.count_ops()
        two_qubit_gates = sum(
            count
            for gate, count in ops.items()
            if gate in ("cx", "cz", "cp", "crz", "swap", "mcx")
        )
        return {
            "lattice_type": "D3Q7",
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
        """Evolve input statevector under the D3Q7 3D streaming circuit."""
        return statevector.evolve(self.circuit)

    def stream_distribution_array(
        self,
        g: np.ndarray,
        return_timing: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Apply quantum 3D streaming to a D3Q7 phase-field distribution array.

        Parameters
        ----------
        g : np.ndarray
            Input distributions of shape `(7, nz, ny, nx)`.
        return_timing : bool, optional
            Whether to return execution timings.

        Returns
        -------
        np.ndarray or tuple of (np.ndarray, dict)
            Streamed distributions of shape `(7, nz, ny, nx)`.
        """
        if g.shape != (7, self.nz, self.ny, self.nx):
            raise ValueError(
                f"Expected shape (7, {self.nz}, {self.ny}, {self.nx}), got {g.shape}"
            )

        t_start = time.perf_counter()

        # Signed L2 norm normalization: amplitudes alpha_k = g_i(x) / norm
        norm = float(np.linalg.norm(g))
        if norm < 1e-15:
            res = np.zeros_like(g)
            if return_timing:
                return res, {
                    "t_state_prep": 0.0,
                    "t_circuit_evolution": 0.0,
                    "t_state_decode": 0.0,
                    "t_total": 0.0,
                }
            return res

        # 1. State Preparation
        state_data = np.zeros(2**self.total_qubits, dtype=np.complex128)
        state_data[self._basis_indices] = g / norm
        psi_in = Statevector(state_data)
        t_prep = time.perf_counter()

        # 2. Circuit Evolution
        psi_out = self.stream_statevector(psi_in)
        t_evolve = time.perf_counter()

        # 3. Readout & Rescaling
        out_data = psi_out.data
        g_out = np.real(out_data[self._basis_indices]) * norm
        t_decode = time.perf_counter()

        if return_timing:
            return g_out, {
                "t_state_prep": t_prep - t_start,
                "t_circuit_evolution": t_evolve - t_prep,
                "t_state_decode": t_decode - t_evolve,
                "t_total": t_decode - t_start,
            }
        return g_out
