"""Linearized Quantum D2Q9 Collision Operator with Ancilla Block-Encoding.

This module implements the quantum representation of the D2Q9 hydrodynamic
collision operator for low-Mach flows (such as dam-break problems):
1. Projection onto conserved moments: E = M^-1 M_cons M.
2. Low-Mach linearization: C_lin = (1 - omega) * I + omega * E.
3. Hermite/weight-basis symmetrization: C_tilde = W^-1 * C_lin * W.
4. Unitary dilation / block encoding into a 5-qubit quantum circuit (1 ancilla + 4 velocity qubits).
5. Exact theoretical O(Ma^2) quadratic truncation error bound.
"""

from __future__ import annotations

from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from typing_extensions import override

from qlbm.classical.d2q9 import C, CS_SQ, WEIGHTS
from qlbm.components.vof.base import QuantumCollisionOperator, QuantumVOFComponent


class LinearizedQuantumCollisionOperator(QuantumVOFComponent, QuantumCollisionOperator):
    r"""Linearized Quantum D2Q9 Collision Operator using Ancilla Block Encoding.

    In the low-Mach limit (:math:`\|\mathbf{u}\| \ll c_s`), the non-linear D2Q9 BGK
    equilibrium distribution :math:`f_i^{\text{eq}}(\rho, \mathbf{u})` can be rigorously
    linearized around rest state :math:`(\rho_0 = 1.0, \mathbf{u}_0 = 0)`:
    .. math::
        f_i^{\text{eq, lin}} = w_i \sum_j f_j + \frac{w_i}{c_s^2} \mathbf{c}_i \cdot \sum_j f_j \mathbf{c}_j

    In matrix form, :math:`\mathbf{f}^{\text{eq, lin}} = \mathbf{E} \mathbf{f}`, where :math:`\mathbf{E}`
    is the orthogonal projector onto the conserved subspace spanned by :math:`\{\rho, j_x, j_y\}`.
    The linearized BGK collision is:
    .. math::
        \mathbf{f}^* = \mathbf{f} - \omega (\mathbf{f} - \mathbf{E}\mathbf{f}) = \mathbf{C}_{\text{lin}} \mathbf{f}
        \quad \text{where } \mathbf{C}_{\text{lin}} = (1 - \omega)\mathbf{I} + \omega \mathbf{E}

    In the Hermite-weighted basis :math:`\tilde{\mathbf{f}} = \mathbf{W}^{-1} \mathbf{f}`,
    the operator :math:`\tilde{\mathbf{C}}_{\text{lin}} = \mathbf{W}^{-1} \mathbf{C}_{\text{lin}} \mathbf{W}`
    is strictly symmetric and contractive (:math:`\|\tilde{\mathbf{C}}_{\text{lin}}\|_2 \le 1.0` for all
    physical :math:`\tau > 0.5`). It is block-encoded into an exact 5-qubit unitary dilation:
    .. math::
        U_{\text{coll}} = \begin{pmatrix} \tilde{\mathbf{C}}_{\text{lin}} & \mathbf{D} \\ \mathbf{D} & -\tilde{\mathbf{C}}_{\text{lin}} \end{pmatrix}
        \quad \text{where } \mathbf{D} = \sqrt{\mathbf{I} - \tilde{\mathbf{C}}_{\text{lin}}^2}

    Parameters
    ----------
    tau : float, optional
        BGK relaxation time :math:`\tau > 0.5`, default 0.8.
    name : str, optional
        Operator identifier, default "linearized_quantum_collision".
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        tau: float = 0.8,
        name: str = "linearized_quantum_collision",
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(name=name, logger=logger)
        if tau <= 0.5:
            raise ValueError(f"Relaxation time must satisfy tau > 0.5, got {tau}.")
        self.tau = float(tau)

        # Precompute operator matrices
        self._W = np.diag(np.sqrt(WEIGHTS))
        self._W_inv = np.diag(1.0 / np.sqrt(WEIGHTS))
        self._E = self.compute_projection_matrix_E()
        self._Clin = self.compute_linearized_collision_matrix(self.tau)
        self._Clin_tilde = self._W_inv @ self._Clin @ self._W
        self._U32 = self.build_block_encoding_unitary(self.tau)

        # Build Qiskit circuit
        self._circuit: Optional[QuantumCircuit] = None
        self.circuit = self.create_circuit()

    # -----------------------------------------------------------------
    # Matrix Construction Routines
    # -----------------------------------------------------------------

    @staticmethod
    def compute_projection_matrix_E() -> np.ndarray:
        r"""Construct the 9x9 orthogonal projection matrix onto the conserved moments.

        .. math::
            E_{ij} = w_i \left( 1 + \frac{\mathbf{c}_i \cdot \mathbf{c}_j}{c_s^2} \right)
        """
        E = np.zeros((9, 9), dtype=np.float64)
        for i in range(9):
            for j in range(9):
                cu = C[i, 0] * C[j, 0] + C[i, 1] * C[j, 1]
                E[i, j] = WEIGHTS[i] * (1.0 + cu / CS_SQ)
        return E

    def compute_linearized_collision_matrix(
        self, tau: Optional[float] = None
    ) -> np.ndarray:
        r"""Construct the 9x9 linearized BGK collision matrix :math:`\mathbf{C}_{\text{lin}}`."""
        t = self.tau if tau is None else float(tau)
        omega = 1.0 / t
        return (1.0 - omega) * np.eye(9, dtype=np.float64) + omega * self._E

    def build_block_encoding_unitary(
        self, tau: Optional[float] = None
    ) -> np.ndarray:
        r"""Construct the 32x32 5-qubit unitary block encoding of the collision operator.

        Qubit register allocation (5 qubits total):
        - Qubit 4 (MSB): Ancilla qubit :math:`|a\rangle` (index 16..31 when 1, 0..15 when 0).
        - Qubits 0..3: Discrete velocity register :math:`|v\rangle \in \{0, \dots, 8\}`.
        """
        Clin = self.compute_linearized_collision_matrix(tau)
        Clin_tilde = self._W_inv @ Clin @ self._W

        # Spectral decomposition of symmetric Clin_tilde: Clin_tilde = V Lambda V^T
        vals, vecs = np.linalg.eigh(Clin_tilde)
        # D_vals = sqrt(max(0, 1 - vals^2))
        D_vals = np.sqrt(np.maximum(0.0, 1.0 - vals**2))
        D_mat = vecs @ np.diag(D_vals) @ vecs.T

        # 32x32 unitary embedding
        U32 = np.eye(32, dtype=np.float64)
        U32[:9, :9] = Clin_tilde
        U32[:9, 16:25] = D_mat
        U32[16:25, :9] = D_mat
        U32[16:25, 16:25] = -Clin_tilde

        return U32

    # -----------------------------------------------------------------
    # QuantumCircuit Implementation
    # -----------------------------------------------------------------

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Create the underlying QuantumCircuit for QuantumComponent base."""
        return self.build_circuit()

    @override
    def build_circuit(
        self,
        lattice: Any = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> QuantumCircuit:
        """Construct the 5-qubit Qiskit quantum circuit implementing the collision operator."""
        if self._circuit is None:
            qc = QuantumCircuit(5, name="linearized_quantum_collision")
            gate = UnitaryGate(self._U32, label="U_coll")
            qc.append(gate, range(5))
            self._circuit = qc
            self.circuit = qc
        return self._circuit

    @override
    def resource_estimate(
        self,
        lattice: Any = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Return resource requirements for the block-encoded collision circuit."""
        return {
            "total_qubits": 5,
            "ancilla_qubits": 1,
            "velocity_qubits": 4,
            "circuit_depth": 1,
            "gate_count": 1,
            "two_qubit_gates": 0,
            "matrix_dimension": 32,
        }

    # -----------------------------------------------------------------
    # Execution & Simulation Routines
    # -----------------------------------------------------------------

    def theoretical_error_bound(
        self, u_max: float, tau: Optional[float] = None
    ) -> float:
        r"""Compute the analytical upper bound on collision error versus non-linear BGK.

        By Taylor expansion of :math:`f_i^{\text{eq}}`:
        .. math::
            \|\mathbf{f}_{\text{lin}}^* - \mathbf{f}_{\text{bgk}}^*\|_{\infty} \le \frac{5}{6\tau} u_{\text{max}}^2 = O(\text{Ma}^2)
        """
        t = self.tau if tau is None else float(tau)
        return float((5.0 / (6.0 * t)) * (u_max**2))

    def collide_distribution_array(
        self,
        f: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
        tau: Optional[float] = None,
    ) -> np.ndarray:
        r"""Execute linearized collision across all spatial grid nodes.

        Parameters
        ----------
        f : np.ndarray
            Distribution array of shape `(9, ny, nx)`.
        active_mask : np.ndarray of bool, optional
            Boolean mask of active fluid nodes where collision occurs.
        tau : float, optional
            Relaxation time.

        Returns
        -------
        np.ndarray
            Post-collision distribution array `f_star` of shape `(9, ny, nx)`.
        """
        Clin = self._Clin if tau is None else self.compute_linearized_collision_matrix(tau)

        # Vectorized linear transformation: f_star_i = sum_j Clin_ij * f_j
        f_star = np.einsum("ij,jyx->iyx", Clin, f)

        if active_mask is not None:
            # Preserve unmodified populations on non-active cells
            f_out = np.copy(f)
            f_out[:, active_mask] = f_star[:, active_mask]
            return f_out

        return f_star

    @property
    def projection_matrix(self) -> np.ndarray:
        """The 9x9 projection matrix E onto conserved moments."""
        return self._E

    @property
    def collision_matrix(self) -> np.ndarray:
        """The 9x9 linearized collision matrix C_lin."""
        return self._Clin

    @property
    def unitary_matrix(self) -> np.ndarray:
        """The 32x32 unitary dilation matrix U_coll."""
        return self._U32
