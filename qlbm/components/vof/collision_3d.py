"""Linearized Quantum D3Q19 Collision & Gravitational Forcing Operator with Ancilla Block-Encoding.

This module implements the quantum representation of the 3D D3Q19 hydrodynamic
collision and gravity body-forcing operator for low-Mach flows:
1. Exact projection onto 4 conserved hydrodynamic moments: E = M^-1 M_cons M.
2. Low-Mach BGK linearization: C_lin = (1 - omega) * I + omega * E.
3. Linearized Guo gravitational forcing matrix: G = F_guo / rho * E_mass.
4. Combined collision + gravity operator: M_cg = C_lin + G.
5. Hermite/weight-basis scaling: M_tilde = W^-1 * M_cg * W.
6. Unitary dilation / block encoding into a 6-qubit quantum circuit (1 ancilla + 5 velocity qubits).
7. Machine-precision conservation of mass and exact momentum injection.
"""

from __future__ import annotations

from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import sqrtm
from typing_extensions import override

from qlbm.classical.d3q19 import C, CS_SQ, WEIGHTS
from qlbm.components.vof.base import QuantumCollisionOperator, QuantumVOFComponent


class LinearizedQuantum3DCollisionOperator(QuantumVOFComponent, QuantumCollisionOperator):
    r"""Linearized Quantum D3Q19 Collision & Gravitational Body-Forcing Operator.

    In the low-Mach limit (:math:`\|\mathbf{u}\| \ll c_s`), the D3Q19 BGK equilibrium
    distribution :math:`f_i^{\text{eq}}(\rho, \mathbf{u})` linearized around rest
    (:math:`\rho_0 = 1.0, \mathbf{u}_0 = \mathbf{0}`) is:
    .. math::
        f_i^{\text{eq, lin}} = w_i \sum_{j=0}^{18} f_j + \frac{w_i}{c_s^2} \mathbf{c}_i \cdot \sum_{j=0}^{18} f_j \mathbf{c}_j
        = \sum_{j=0}^{18} E_{ij} f_j

    where :math:`\mathbf{E} \in \mathbb{R}^{19 \times 19}` is the orthogonal projector onto
    the 4 conserved moments :math:`\{\rho, j_x, j_y, j_z\}`.

    The linearized Guo second-order gravitational forcing term is:
    .. math::
        F_i = w_i \left( 1 - \frac{\Delta t}{2\tau} \right) \frac{\mathbf{c}_i \cdot \mathbf{g}}{c_s^2} \rho \Delta t
        = \sum_{j=0}^{18} G_{ij} f_j

    The combined collision + gravity operation is:
    .. math::
        \mathbf{f}^* = \mathbf{M}_{\text{cg}} \mathbf{f} = \left[ (1 - \omega)\mathbf{I} + \omega \mathbf{E} + \mathbf{G} \right] \mathbf{f}

    In the Hermite-weighted basis :math:`\tilde{\mathbf{f}} = \mathbf{W}^{-1} \mathbf{f}`,
    the scaled operator :math:`\tilde{\mathbf{M}}_{\text{cg}} = \mathbf{W}^{-1} \mathbf{M}_{\text{cg}} \mathbf{W}`
    is embedded into a 5-qubit velocity space (:math:`2^5 = 32 \ge 19`), normalized by
    :math:`\alpha = \max(1.0, \|\tilde{\mathbf{M}}\|_2)`, and block-dilated with 1 ancilla qubit
    into an exact 6-qubit (:math:`64 \times 64`) unitary matrix:
    .. math::
        U_{\text{cg}} = \begin{pmatrix} \mathbf{A} & \mathbf{D}_1 \\ \mathbf{D}_2 & -\mathbf{A}^\dagger \end{pmatrix}

    Parameters
    ----------
    tau : float, optional
        BGK relaxation time :math:`\tau > 0.5`, default 0.8.
    g : tuple or np.ndarray, optional
        Gravitational acceleration vector in lattice units, default (0.0, 0.0, -0.01).
    dt : float, optional
        Lattice time step, default 1.0.
    name : str, optional
        Operator identifier, default "linearized_quantum_3d_collision".
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        tau: float = 0.8,
        g: Union[Tuple[float, float, float], np.ndarray] = (0.0, 0.0, -0.01),
        dt: float = 1.0,
        name: str = "linearized_quantum_3d_collision",
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(name=name, logger=logger)
        if tau <= 0.5:
            raise ValueError(f"Relaxation time must satisfy tau > 0.5, got {tau}.")
        self.tau = float(tau)
        self.g = np.asarray(g, dtype=np.float64)
        if self.g.shape != (3,):
            raise ValueError(f"Gravity vector must have shape (3,), got {self.g.shape}")
        self.dt = float(dt)

        # Precompute operator matrices
        self._W = np.diag(np.sqrt(WEIGHTS))
        self._W_inv = np.diag(1.0 / np.sqrt(WEIGHTS))
        self._E = self.compute_projection_matrix_E()
        self._Clin = self.compute_linearized_collision_matrix(self.tau)
        self._G = self.compute_gravity_forcing_matrix(self.g, self.tau, self.dt)
        self._Mcg = self._Clin + self._G

        self._alpha: float = 1.0
        self._U64 = self.build_block_encoding_unitary()

        # Build Qiskit circuit
        self._circuit: Optional[QuantumCircuit] = None
        self.circuit = self.create_circuit()

    # -----------------------------------------------------------------
    # Matrix Construction Routines
    # -----------------------------------------------------------------

    @staticmethod
    def compute_projection_matrix_E() -> np.ndarray:
        r"""Construct the 19x19 orthogonal projection matrix onto the conserved moments.

        .. math::
            E_{ij} = w_i \left( 1 + \frac{\mathbf{c}_i \cdot \mathbf{c}_j}{c_s^2} \right)
        """
        E = np.zeros((19, 19), dtype=np.float64)
        for i in range(19):
            for j in range(19):
                cu = C[i, 0] * C[j, 0] + C[i, 1] * C[j, 1] + C[i, 2] * C[j, 2]
                E[i, j] = WEIGHTS[i] * (1.0 + cu / CS_SQ)
        return E

    def compute_linearized_collision_matrix(
        self, tau: Optional[float] = None
    ) -> np.ndarray:
        r"""Construct the 19x19 linearized BGK collision matrix :math:`\mathbf{C}_{\text{lin}}`."""
        t = self.tau if tau is None else float(tau)
        omega = 1.0 / t
        return (1.0 - omega) * np.eye(19, dtype=np.float64) + omega * self._E

    @staticmethod
    def compute_gravity_forcing_matrix(
        g: np.ndarray,
        tau: float,
        dt: float = 1.0,
    ) -> np.ndarray:
        r"""Construct the 19x19 linearized Guo gravitational forcing matrix :math:`\mathbf{G}`.

        .. math::
            G_{ij} = w_i \left( 1 - \frac{\Delta t}{2\tau} \right) \frac{\mathbf{c}_i \cdot \mathbf{g}}{c_s^2} \Delta t
        """
        prefactor = (1.0 - 0.5 * dt / tau) * dt
        G = np.zeros((19, 19), dtype=np.float64)
        for i in range(19):
            cg = C[i, 0] * g[0] + C[i, 1] * g[1] + C[i, 2] * g[2]
            G[i, :] = WEIGHTS[i] * prefactor * (cg / CS_SQ)
        return G

    def build_block_encoding_unitary(self) -> np.ndarray:
        r"""Construct the 64x64 6-qubit unitary block encoding of the collision + gravity operator.

        Qubit register allocation (6 qubits total):
        - Qubit 5 (MSB): Ancilla qubit :math:`|a\rangle` (0: success subspace, 1: orthogonal dilation).
        - Qubits 0..4: Discrete velocity register :math:`|v\rangle \in \{0, \dots, 18\}` padded to 32.
        """
        Mcg_tilde = self._W_inv @ self._Mcg @ self._W

        # Pad to 32x32 for 5 velocity qubits
        M32 = np.eye(32, dtype=np.float64)
        M32[:19, :19] = Mcg_tilde

        # Contractive normalization
        s_max = float(np.max(np.linalg.svd(M32, compute_uv=False)))
        self._alpha = max(1.0, s_max)
        A = M32 / self._alpha

        # Exact Halmos / Sz.-Nagy unitary dilation using SVD-aligned singular vectors
        # for machine-precision off-diagonal commutativity: A D2 - D1 A = 0
        u, s, vh = np.linalg.svd(A)
        s = np.clip(s, 0.0, 1.0)
        s_d = np.sqrt(np.maximum(0.0, 1.0 - s**2))

        D1 = u @ np.diag(s_d) @ u.T
        D2 = vh.T @ np.diag(s_d) @ vh

        # 64x64 block matrix: [[A, D1], [D2, -A.T]]
        U64 = np.block([[A, D1], [D2, -A.T]])
        return U64

    # -----------------------------------------------------------------
    # Properties & Getters
    # -----------------------------------------------------------------

    @property
    def projection_matrix(self) -> np.ndarray:
        """The 19x19 orthogonal projection matrix E."""
        return self._E

    @property
    def collision_matrix(self) -> np.ndarray:
        """The 19x19 linearized BGK collision matrix C_lin."""
        return self._Clin

    @property
    def gravity_matrix(self) -> np.ndarray:
        """The 19x19 linearized gravitational forcing matrix G."""
        return self._G

    @property
    def combined_matrix(self) -> np.ndarray:
        """The combined 19x19 collision + gravity matrix M_cg = C_lin + G."""
        return self._Mcg

    @property
    def unitary_matrix(self) -> np.ndarray:
        """The 64x64 unitary dilation matrix U_cg."""
        return self._U64

    @property
    def normalization_factor(self) -> float:
        """Block-encoding normalization factor alpha >= ||M_tilde||_2."""
        return self._alpha

    @property
    def success_probability(self) -> float:
        """Expected ancilla success probability |<0| U |0>|^2 = 1 / alpha^2."""
        return 1.0 / (self._alpha**2)

    # -----------------------------------------------------------------
    # Quantum Circuit Generation
    # -----------------------------------------------------------------

    @override
    def create_circuit(self) -> QuantumCircuit:
        """Build the 6-qubit Qiskit quantum circuit for block-encoded collision + gravity."""
        qc = QuantumCircuit(6, name="d3q19_coll_grav")
        gate = UnitaryGate(self._U64, label="CollGrav3D")
        qc.append(gate, range(6))
        return qc

    @override
    def build_circuit(
        self,
        lattice: Any = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> QuantumCircuit:
        """Implement the abstract QuantumCollisionOperator interface."""
        return self.circuit

    @override
    def resource_estimate(
        self,
        lattice: Any = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Implement resource estimation."""
        return {
            "num_qubits": 6,
            "total_qubits": 6,
            "ancilla_qubits": 1,
            "velocity_qubits": 5,
            "depth": self.circuit.depth(),
            "size": self.circuit.size(),
            "unitary_dim": 64,
        }

    # -----------------------------------------------------------------
    # Hydrodynamic Application
    # -----------------------------------------------------------------

    def apply(self, f: np.ndarray) -> np.ndarray:
        """Apply the linearized collision + gravity operator to a distribution array.

        Parameters
        ----------
        f : np.ndarray
            Distribution array of shape `(19, ...)` or `(19,)`.

        Returns
        -------
        np.ndarray
            Post-collision and post-forcing distribution array of same shape.
        """
        f = np.asarray(f, dtype=np.float64)
        orig_shape = f.shape
        if orig_shape[0] != 19:
            raise ValueError(f"First axis must have length 19, got {orig_shape[0]}")

        f_2d = f.reshape(19, -1)
        f_post_2d = self._Mcg @ f_2d
        return f_post_2d.reshape(orig_shape)
