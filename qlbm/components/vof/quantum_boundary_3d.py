"""Integrated Quantum Wall Boundary Operator for 3D D3Q19 Lattices.

Implements solid-wall bounce-back boundary conditions directly within the
quantum computational register:
1. Spatial Boundary Oracle O_wall: Flips an obstacle ancilla qubit |wall>
   conditioned on the spatial coordinate register |x, y, z> in Omega_solid.
2. Controlled D3Q19 Bounce-Back Unitary U_bounce: Swaps opposite discrete
   velocities (c_i <-> c_opposite[i]) conditioned on |wall> = |1>.
3. Uncomputation Oracle O_wall^dagger: Restores the obstacle ancilla to |0>,
   leaving no residual entanglement between fluid registers and the wall ancilla.
"""

from __future__ import annotations

from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from typing_extensions import override

from qlbm.classical.d3q19 import OPPOSITE
from qlbm.components.vof.base import QuantumVOFComponent
from qlbm.tools.utils import get_qubits_to_invert, is_two_pow


class QuantumD3Q19SpatialBoundaryOperator(QuantumVOFComponent):
    """Quantum spatial wall bounce-back operator for 3D D3Q19 lattices.

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
        If None, an empty obstacle mask (all fluid) is used.
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
        self.num_vel_qubits = 5  # ceil(log2(19)) = 5
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

        super().__init__(name="quantum_d3q19_spatial_boundary", logger=logger)
        self.circuit = self.create_circuit()

    def _build_bounce_unitary_32(self) -> np.ndarray:
        """Construct the 32x32 5-qubit bounce-back permutation unitary."""
        dim = 2**self.num_vel_qubits  # 32
        U = np.zeros((dim, dim), dtype=np.float64)
        for i in range(19):
            U[OPPOSITE[i], i] = 1.0
        for i in range(19, 32):
            U[i, i] = 1.0
        return U

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Construct the complete quantum circuit for spatial wall bounce-back."""
        reg_x = list(range(0, self.num_x_qubits))
        reg_y = list(range(self.num_x_qubits, self.num_x_qubits + self.num_y_qubits))
        reg_z = list(
            range(
                self.num_x_qubits + self.num_y_qubits,
                self.total_spatial_qubits,
            )
        )
        reg_v = list(
            range(self.total_spatial_qubits, self.total_spatial_qubits + self.num_vel_qubits)
        )
        wall_anc = self.total_qubits - 1

        qc = QuantumCircuit(self.total_qubits, name="quantum_d3q19_wall_boundary")

        # 1. Spatial Obstacle Oracle O_wall
        solid_coords = np.argwhere(self.solid_mask)
        if len(solid_coords) > 0:
            for z, y, x in solid_coords:
                # Target integer in spatial register: x | (y << nx_q) | (z << (nx_q + ny_q))
                coord_idx = int(
                    x
                    | (y << self.num_x_qubits)
                    | (z << (self.num_x_qubits + self.num_y_qubits))
                )
                to_invert = [
                    q for q in get_qubits_to_invert(coord_idx, self.total_spatial_qubits)
                ]
                if to_invert:
                    qc.x(to_invert)
                # Multi-controlled NOT onto wall ancilla
                spatial_qubits = list(range(self.total_spatial_qubits))
                if spatial_qubits:
                    qc.mcx(spatial_qubits, wall_anc)
                else:
                    qc.x(wall_anc)
                if to_invert:
                    qc.x(to_invert)

            # 2. Controlled Bounce-Back on velocity register
            u32 = self._build_bounce_unitary_32()
            bounce_gate = UnitaryGate(u32, label="D3Q19Bounce").control(1)
            qc.append(bounce_gate, [wall_anc] + reg_v)

            # 3. Uncompute Spatial Obstacle Oracle O_wall^dagger
            for z, y, x in reversed(solid_coords):
                coord_idx = int(
                    x
                    | (y << self.num_x_qubits)
                    | (z << (self.num_x_qubits + self.num_y_qubits))
                )
                to_invert = [
                    q for q in get_qubits_to_invert(coord_idx, self.total_spatial_qubits)
                ]
                if to_invert:
                    qc.x(to_invert)
                spatial_qubits = list(range(self.total_spatial_qubits))
                if spatial_qubits:
                    qc.mcx(spatial_qubits, wall_anc)
                else:
                    qc.x(wall_anc)
                if to_invert:
                    qc.x(to_invert)

        return qc

    def resource_metrics(self) -> Dict[str, Any]:
        """Compute circuit resource metrics."""
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
            "num_spatial_qubits": self.total_spatial_qubits,
            "num_vel_qubits": self.num_vel_qubits,
            "num_wall_ancilla": self.num_wall_ancilla,
            "num_solid_nodes": int(np.sum(self.solid_mask)),
        }

    def apply_to_distribution_array(self, f: np.ndarray) -> np.ndarray:
        """Apply bounce-back reflection to distribution array on solid nodes.

        Parameters
        ----------
        f : np.ndarray
            Distribution array of shape `(19, nz, ny, nx)`.

        Returns
        -------
        np.ndarray
            Post-reflection distribution array.
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape != (19, self.nz, self.ny, self.nx):
            raise ValueError(
                f"Expected shape (19, {self.nz}, {self.ny}, {self.nx}), got {f.shape}"
            )

        f_out = f.copy()
        for i in range(19):
            f_out[OPPOSITE[i], self.solid_mask] = f[i, self.solid_mask]
        return f_out
