"""Quantum solid-wall bounce-back reflection operators for D2Q9 lattices."""

from __future__ import annotations

from logging import Logger, getLogger
from typing import Optional

from qiskit import QuantumCircuit, QuantumRegister
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.reflection.common import ABBounceBackReflectionPermutation
from qlbm.components.vof.base import QuantumVOFComponent
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization


class QuantumD2Q9BounceBackOperator(QuantumVOFComponent):
    """Quantum solid boundary bounce-back operator for D2Q9 discrete velocities.

    In lattice Boltzmann hydrodynamics, standard no-slip wall reflection
    swaps opposite discrete velocities:
        c_i -> c_{opposite[i]}
    For D2Q9, opposite pairs are:
        (1 <-> 3), (2 <-> 4), (5 <-> 7), (6 <-> 8), with 0 (rest) unchanged.

    This operator implements the exact unitary permutation on the velocity
    register (encoded with 4 qubits in amplitude-based encoding or 9 qubits
    in one-hot encoding).

    Parameters
    ----------
    encoding : ABEncodingType, optional
        Encoding type (AB for log2(q)=4 qubits, OH for q=9 qubits). Default is AB.
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
        self.num_velocity_qubits = 4 if encoding == ABEncodingType.AB else 9
        super().__init__(name="quantum_d2q9_bounce_back", logger=logger)
        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Construct the D2Q9 bounce-back reflection quantum circuit."""
        perm = ABBounceBackReflectionPermutation(
            num_qubits=self.num_velocity_qubits,
            discretization=LatticeDiscretization.D2Q9,
            encoding=self.encoding,
            logger=self.logger,
        )
        base_circuit = perm.circuit

        if not self.controlled:
            return base_circuit

        # If controlled, add an obstacle marker control qubit
        ctrl_reg = QuantumRegister(1, name="wall_ctrl")
        vel_reg = QuantumRegister(self.num_velocity_qubits, name="vel")
        circuit = QuantumCircuit(ctrl_reg, vel_reg)

        controlled_gate = base_circuit.to_gate(label="BounceBack").control(1)
        circuit.append(controlled_gate, [ctrl_reg[0]] + list(vel_reg))
        return circuit
