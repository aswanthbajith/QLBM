"""Quantum solid-wall bounce-back reflection operators for D3Q19 lattices."""

from __future__ import annotations

from logging import Logger, getLogger
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from typing_extensions import override

from qlbm.classical.d3q19 import OPPOSITE
from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.vof.base import QuantumVOFComponent


class QuantumD3Q19BounceBackOperator(QuantumVOFComponent):
    """Quantum solid boundary bounce-back operator for D3Q19 discrete velocities.

    In lattice Boltzmann hydrodynamics, standard no-slip wall reflection
    swaps opposite discrete velocities:
        c_i -> c_{opposite[i]}
    For D3Q19, the 9 opposite pairs are:
        (1 <-> 2), (3 <-> 4), (5 <-> 6), (7 <-> 8), (9 <-> 10),
        (11 <-> 12), (13 <-> 14), (15 <-> 16), (17 <-> 18),
    with 0 (rest) remaining unchanged.

    This operator implements the exact unitary permutation on the velocity
    register (encoded with 5 qubits in amplitude-based encoding or 19 qubits
    in one-hot encoding).

    Parameters
    ----------
    encoding : ABEncodingType, optional
        Encoding type (AB for ceil(log2(19))=5 qubits, OH for 19 qubits). Default is AB.
    controlled : bool, optional
        Whether to condition the reflection on an external boundary-marker qubit.
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        encoding: ABEncodingType = ABEncodingType.AB,
        controlled: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        self.encoding = encoding
        self.controlled = controlled
        self.num_velocity_qubits = 5 if encoding == ABEncodingType.AB else 19
        super().__init__(name="quantum_d3q19_bounce_back", logger=logger)
        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Construct the D3Q19 bounce-back reflection quantum circuit."""
        opposite_pairs = [
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
            (17, 18),
        ]

        if self.encoding == ABEncodingType.OH:
            circuit = QuantumCircuit(self.num_velocity_qubits)
            for a, b in opposite_pairs:
                circuit.swap(a, b)
        else:
            # 5-qubit permutation matrix
            dim = 2**self.num_velocity_qubits  # 32
            pattern = list(range(dim))
            for a, b in opposite_pairs:
                pattern[a], pattern[b] = pattern[b], pattern[a]

            U = np.zeros((dim, dim), dtype=complex)
            for j, k in enumerate(pattern):
                U[k, j] = 1.0

            gate = UnitaryGate(U, label="D3Q19BounceBack")
            circuit = QuantumCircuit(self.num_velocity_qubits)
            circuit.append(gate, range(self.num_velocity_qubits))

        if not self.controlled:
            return circuit

        # If controlled, add an obstacle marker control qubit
        ctrl_reg = QuantumRegister(1, name="wall_ctrl")
        vel_reg = QuantumRegister(self.num_velocity_qubits, name="vel")
        ctrl_circuit = QuantumCircuit(ctrl_reg, vel_reg)

        controlled_gate = circuit.to_gate(label="BounceBack").control(1)
        ctrl_circuit.append(controlled_gate, [ctrl_reg[0]] + list(vel_reg))
        return ctrl_circuit
