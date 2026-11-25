"""Integrated Quantum Wall Boundary Operator for 3D D3Q7 Phase-Field Lattices.

Implements solid-wall bounce-back boundary conditions for the Conservative
Allen-Cahn D3Q7 phase-field lattice directly within the quantum computational register:
1. Spatial Boundary Oracle O_wall: Flips an obstacle ancilla qubit |wall>
   conditioned on the spatial coordinate register |x, y, z> in Omega_solid.
2. Controlled D3Q7 Bounce-Back Unitary U_bounce: Swaps opposite discrete
   velocities (c_1 <-> c_2, c_3 <-> c_4, c_5 <-> c_6) conditioned on |wall> = |1>.
3. Uncomputation Oracle O_wall^dagger: Restores the obstacle ancilla to |0>,
   leaving no residual entanglement between phase-field registers and the wall ancilla.
"""

from __future__ import annotations

from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from typing_extensions import override

from qlbm.components.vof.base import QuantumVOFComponent
from qlbm.components.vof.streaming_d3q7 import OPP_D3Q7
from qlbm.tools.utils import get_qubits_to_invert, is_two_pow


class QuantumD3Q7SpatialBoundaryOperator(QuantumVOFComponent):
    """Quantum spatial wall bounce-back operator for 3D D3Q7 lattices.

    Parameters
    ----------
    nx : int
        Grid nodes in x-direction (must be a power of two).
    ny : int
        Grid nodes in y-direction (must be a power of two).
    nz : int
        Grid nodes in z-direction (must be a power of two).
    solid_mask : np.ndarray, optional
        Boolean array of shape `(nz, ny, nx)` marking solid obstacle nodes.
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        solid_mask: Optional[np.ndarray] = None,
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
        self.total_spatial_qubits = (
            self.num_x_qubits + self.num_y_qubits + self.num_z_qubits
        )
        self.num_vel_qubits = 3  # ceil(log2(7)) = 3
        self.num_wall_ancilla = 1

        self.total_qubits = (
            self.total_spatial_qubits + self.num_vel_qubits + self.num_wall_ancilla
        )

        if solid_mask is None:
            self.solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        else:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
            if self.solid_mask.shape != (self.nz, self.ny, self.nx):
                raise ValueError(
                    f"solid_mask shape {self.solid_mask.shape} does not match ({self.nz}, {self.ny}, {self.nx})"
                )

        super().__init__(name="quantum_d3q7_spatial_boundary", logger=logger)
        self.circuit = self.create_circuit()

    def _build_bounce_unitary_8(self) -> np.ndarray:
        """Construct the 8x8 3-qubit bounce-back permutation unitary."""
        dim = 2**self.num_vel_qubits  # 8
        U = np.zeros((dim, dim), dtype=np.float64)
        for i in range(7):
            U[OPP_D3Q7[i], i] = 1.0
        U[7, 7] = 1.0  # Invariant padding state
        return U

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Construct the complete quantum circuit for spatial wall bounce-back."""
        qc = QuantumCircuit(self.total_qubits)

        reg_spatial = list(range(self.total_spatial_qubits))
        reg_vel = list(range(self.total_spatial_qubits, self.total_spatial_qubits + self.num_vel_qubits))
        wall_ancilla = self.total_qubits - 1

        # 1. Oracle O_wall: detect solid nodes and flip wall_ancilla
        solid_indices = np.argwhere(self.solid_mask)
        for z, y, x in solid_indices:
            x_int = int(x)
            y_int = int(y)
            z_int = int(z)
            spatial_state = (
                x_int
                | (y_int << self.num_x_qubits)
                | (z_int << (self.num_x_qubits + self.num_y_qubits))
            )
            to_invert = [
                reg_spatial[q]
                for q in get_qubits_to_invert(spatial_state, self.total_spatial_qubits)
            ]
            if to_invert:
                qc.x(to_invert)

            qc.mcx(reg_spatial, wall_ancilla)

            if to_invert:
                qc.x(to_invert)

        # 2. Controlled D3Q7 Bounce-Back Unitary
        U_bounce = self._build_bounce_unitary_8()
        bounce_gate = UnitaryGate(U_bounce, label="D3Q7_Bounce").control(1)
        qc.append(bounce_gate, [wall_ancilla] + reg_vel)

        # 3. Uncomputation Oracle O_wall^dag
        for z, y, x in solid_indices:
            x_int = int(x)
            y_int = int(y)
            z_int = int(z)
            spatial_state = (
                x_int
                | (y_int << self.num_x_qubits)
                | (z_int << (self.num_x_qubits + self.num_y_qubits))
            )
            to_invert = [
                reg_spatial[q]
                for q in get_qubits_to_invert(spatial_state, self.total_spatial_qubits)
            ]
            if to_invert:
                qc.x(to_invert)

            qc.mcx(reg_spatial, wall_ancilla)

            if to_invert:
                qc.x(to_invert)

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
            "lattice_type": "D3Q7_Boundary",
            "num_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "size": self.circuit.size(),
            "two_qubit_gates": two_qubit_gates,
            "solid_cells": int(np.sum(self.solid_mask)),
            "statevector_dim": 2**self.circuit.num_qubits,
        }

    def apply_boundary_statevector(self, statevector: Statevector) -> Statevector:
        """Apply the spatial wall bounce-back circuit to an input statevector."""
        return statevector.evolve(self.circuit)
