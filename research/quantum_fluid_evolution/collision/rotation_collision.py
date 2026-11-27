"""Quantum Rotation-Based Collision Operator for D3Q19 Lattices.

Inspired by 2026 rotation collision QLBM research (arXiv:2609.16721).

In this formulation, hydrodynamic collision is implemented as a sequence of
exact unitary SO(2) Givens rotations across conjugate opposite velocity pairs:
    (c_i, c_opposite[i])
For D3Q19, the 9 conjugate pairs are:
    (1, 2): +/- x
    (3, 4): +/- y
    (5, 6): +/- z
    (7, 8): (+x+y, -x-y)
    (9, 10): (+x-y, -x+y)
    (11, 12): (+x+z, -x-z)
    (13, 14): (+x-z, -x+z)
    (15, 16): (+y+z, -y-z)
    (17, 18): (+y-z, -y+z)
with rest particle 0 invariant.

Key Advantages:
1. Strictly unitary on the 5-qubit velocity register (NO ancilla qubits required).
2. Ancilla success probability is identically 100% (P_succ == 1.0).
3. Zero postselection or amplitude amplification overhead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from qlbm.classical.d3q19 import OPPOSITE


class RotationCollision3D:
    r"""Unitary Rotation-Based Collision Operator for D3Q19.

    For each conjugate velocity pair :math:`(i, \bar{i})`, an :math:`\text{SO}(2)`
    rotation is applied:
    .. math::
        \begin{pmatrix} f_i^* \\ f_{\bar{i}}^* \end{pmatrix}
        = \begin{pmatrix} \cos \theta_k & -\sin \theta_k \\ \sin \theta_k & \cos \theta_k \end{pmatrix}
        \begin{pmatrix} f_i \\ f_{\bar{i}} \end{pmatrix}

    Parameters
    ----------
    theta : float, optional
        Base rotation angle in radians, default 0.05.
    angle_scaling : list or np.ndarray, optional
        Optional per-pair angle multipliers of length 9.
    """

    # 9 conjugate velocity pairs for D3Q19
    PAIRS: List[Tuple[int, int]] = [
        (1, 2),    # +/- x
        (3, 4),    # +/- y
        (5, 6),    # +/- z
        (7, 8),    # (+x+y, -x-y)
        (9, 10),   # (+x-y, -x+y)
        (11, 12),  # (+x+z, -x-z)
        (13, 14),  # (+x-z, -x+z)
        (15, 16),  # (+y+z, -y-z)
        (17, 18),  # (+y-z, -y+z)
    ]

    def __init__(
        self,
        theta: float = 0.05,
        angle_scaling: Optional[List[float]] = None,
    ) -> None:
        self.theta = float(theta)
        if angle_scaling is None:
            self.angles = [self.theta] * len(self.PAIRS)
        else:
            if len(angle_scaling) != len(self.PAIRS):
                raise ValueError(f"Expected {len(self.PAIRS)} angle values, got {len(angle_scaling)}")
            self.angles = [float(a) * self.theta for a in angle_scaling]

        # 32x32 unitary matrix on 5 velocity qubits (no ancilla needed!)
        self._U32 = self._build_rotation_unitary()

    def _build_rotation_unitary(self) -> np.ndarray:
        """Construct the 32x32 unitary rotation matrix on 5 velocity qubits."""
        U = np.eye(32, dtype=np.float64)

        for (a, b), ang in zip(self.PAIRS, self.angles):
            c = np.cos(ang)
            s = np.sin(ang)
            U[a, a] = c
            U[a, b] = -s
            U[b, a] = s
            U[b, b] = c

        return U

    @property
    def unitary_matrix(self) -> np.ndarray:
        """The 32x32 unitary rotation matrix."""
        return self._U32

    def build_circuit(self) -> QuantumCircuit:
        """Create a 5-qubit Qiskit quantum circuit for the rotation collision operator."""
        qc = QuantumCircuit(5, name="rotation_coll_3d")
        gate = UnitaryGate(self._U32, label="RotColl3D")
        qc.append(gate, range(5))
        return qc

    def apply(self, f: np.ndarray) -> np.ndarray:
        """Apply rotation collision to distribution array f."""
        f = np.asarray(f, dtype=np.float64)
        orig_shape = f.shape
        f_2d = f.reshape(19, -1)

        f_pad = np.zeros((32, f_2d.shape[1]), dtype=np.float64)
        f_pad[:19, :] = f_2d

        f_rot = self._U32 @ f_pad
        f_post = f_rot[:19, :].reshape(orig_shape)
        return f_post

    def resource_metrics(self) -> Dict[str, Any]:
        """Compute circuit and operator resource metrics."""
        return {
            "operator_type": "Unitary SO(2) Rotations (arXiv:2609.16721)",
            "num_qubits": 5,
            "ancilla_qubits": 0,
            "velocity_qubits": 5,
            "unitary_dim": 32,
            "success_probability": 1.0,
            "postselection_required": False,
            "num_rotation_pairs": len(self.PAIRS),
            "base_theta_rad": self.theta,
        }


class RotationCollisionD3Q7:
    r"""Unitary Rotation-Based Collision Operator for D3Q7 Phase-Field Lattices.

    For each conjugate opposite velocity pair :math:`(i, \bar{i})`, an :math:`\text{SO}(2)`
    Givens rotation is applied:
    .. math::
        \begin{pmatrix} g_i^* \\ g_{\bar{i}}^* \end{pmatrix}
        = \begin{pmatrix} \cos \theta_k & -\sin \theta_k \\ \sin \theta_k & \cos \theta_k \end{pmatrix}
        \begin{pmatrix} g_i \\ g_{\bar{i}} \end{pmatrix}

    For D3Q7, the 3 conjugate pairs are:
        (1, 2): +/- x
        (3, 4): +/- y
        (5, 6): +/- z
    with rest particle 0 invariant.

    Parameters
    ----------
    theta : float, optional
        Base rotation angle in radians, default 0.05.
    angle_scaling : list or np.ndarray, optional
        Optional per-pair angle multipliers of length 3.
    """

    PAIRS: List[Tuple[int, int]] = [
        (1, 2),  # +/- x
        (3, 4),  # +/- y
        (5, 6),  # +/- z
    ]

    def __init__(
        self,
        theta: float = 0.05,
        angle_scaling: Optional[List[float]] = None,
    ) -> None:
        self.theta = float(theta)
        if angle_scaling is None:
            self.angles = [self.theta] * len(self.PAIRS)
        else:
            if len(angle_scaling) != len(self.PAIRS):
                raise ValueError(f"Expected {len(self.PAIRS)} angle values, got {len(angle_scaling)}")
            self.angles = [float(a) * self.theta for a in angle_scaling]

        # 8x8 unitary matrix on 3 velocity qubits (zero ancilla!)
        self._U8 = self._build_rotation_unitary()

    def _build_rotation_unitary(self) -> np.ndarray:
        """Construct the 8x8 unitary rotation matrix on 3 velocity qubits."""
        U = np.eye(8, dtype=np.float64)

        for (a, b), ang in zip(self.PAIRS, self.angles):
            c = np.cos(ang)
            s = np.sin(ang)
            U[a, a] = c
            U[a, b] = -s
            U[b, a] = s
            U[b, b] = c

        return U

    @property
    def unitary_matrix(self) -> np.ndarray:
        """The 8x8 unitary rotation matrix."""
        return self._U8

    def build_circuit(self) -> QuantumCircuit:
        """Create a 3-qubit Qiskit quantum circuit for the D3Q7 rotation collision operator."""
        qc = QuantumCircuit(3, name="rotation_coll_d3q7")
        gate = UnitaryGate(self._U8, label="RotCollD3Q7")
        qc.append(gate, range(3))
        return qc

    def apply(self, g: np.ndarray) -> np.ndarray:
        """Apply rotation collision to D3Q7 distribution array g."""
        g = np.asarray(g, dtype=np.float64)
        orig_shape = g.shape
        g_2d = g.reshape(7, -1)

        g_pad = np.zeros((8, g_2d.shape[1]), dtype=np.float64)
        g_pad[:7, :] = g_2d

        g_rot = self._U8 @ g_pad
        g_post = g_rot[:7, :].reshape(orig_shape)
        return g_post

    def resource_metrics(self) -> Dict[str, Any]:
        """Compute circuit and operator resource metrics for D3Q7."""
        return {
            "operator_type": "Unitary SO(2) Rotations D3Q7 (arXiv:2609.16721)",
            "num_qubits": 3,
            "ancilla_qubits": 0,
            "velocity_qubits": 3,
            "unitary_dim": 8,
            "success_probability": 1.0,
            "postselection_required": False,
            "num_rotation_pairs": len(self.PAIRS),
            "base_theta_rad": self.theta,
        }

