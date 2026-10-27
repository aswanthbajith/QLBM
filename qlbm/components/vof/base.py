"""Base classes and research interfaces for quantum Volume-of-Fluid (VOF) components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import QuantumComponent


class QuantumVOFComponent(QuantumComponent):
    """Base class for quantum circuits implementing operations within the VOF-LBM framework.

    Subclasses wrap a :class:`qiskit.QuantumCircuit` targeting discrete velocity
    transport, boundary conditions, or candidate collision approximations.
    """

    def __init__(
        self,
        name: str = "quantum_vof_component",
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.name = name

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', qubits={self.width()})"


class QuantumCollisionOperator(ABC):
    """Abstract research interface for quantum collision operators.

    This interface defines the contract for candidate quantum implementations
    of hydrodynamic collision dynamics (e.g. linearized BGK around equilibrium,
    Carleman embeddings, or block-encoded dissipative relaxation).

    A valid quantum collision operator must specify:
    1. The quantum encoding of the distribution (amplitude-based or multi-speed).
    2. The target mathematical transformation (dissipative relaxation).
    3. The required ancilla registers and circuit width.
    4. Error bounds and resource measures (depth, 2-qubit gate count).
    """

    @abstractmethod
    def build_circuit(
        self,
        lattice: Any,
        model_parameters: Dict[str, Any],
    ) -> QuantumCircuit:
        """Construct the quantum circuit implementing the candidate collision model.

        Parameters
        ----------
        lattice : Any
            The quantum lattice providing coordinate and velocity register layouts.
        model_parameters : dict
            Physical and numerical parameters (e.g. relaxation time tau, equilibrium
            reference state, error tolerance, truncation order).

        Returns
        -------
        QuantumCircuit
            The compiled quantum collision circuit.
        """
        pass

    @abstractmethod
    def resource_estimate(
        self,
        lattice: Any,
        model_parameters: Dict[str, Any],
    ) -> Dict[str, int]:
        """Return the estimated quantum resources (qubits, depth, CNOT count)."""
        pass
