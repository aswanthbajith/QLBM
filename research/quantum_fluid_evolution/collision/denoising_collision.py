"""Quantum Denoising Collision Operator for D3Q19 Lattices.

Inspired by 2026 quantum fluid denoising research (arXiv:2604.09997,
MyEntangled/denoise_qlbm).

In this formulation, hydrodynamic collision is treated not as a dissipative
point-wise relaxation, but as a quantum denoising filter that projects
non-equilibrium kinetic perturbations ("noise") onto the smooth hydrodynamic
equilibrium manifold while preserving conserved physical invariants:
1. Hydrodynamic Manifold Projector P_eq: Orthogonal projection onto {rho, jx, jy, jz}.
2. Quantum Reflection Diffuser R_eq = 2 * P_tilde - I (Grover-style diffusion operator).
3. Tunable Filter D_lambda = (1 - lambda) * I + lambda * P_tilde, where lambda = dt / tau.
4. Unitary Block-Encoding via 1-ancilla dilation or LCU (Linear Combination of Unitaries).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from qlbm.classical.d3q19 import C, CS_SQ, WEIGHTS


class DenoisingCollision3D:
    r"""D3Q19 Quantum Denoising Collision Operator.

    The non-equilibrium distributions :math:`f_i` are decomposed into:
    .. math::
        \mathbf{f} = \mathbf{f}^{\text{eq}} + \mathbf{f}^{\text{neq}}

    where :math:`\mathbf{f}^{\text{eq}} = \mathbf{P}_{\text{eq}} \mathbf{f}` lies in the
    conserved hydrodynamic subspace and :math:`\mathbf{f}^{\text{neq}}` contains high-frequency
    viscous stresses and non-conserved kinetic modes ("noise").

    The denoising operation applies a tunable attenuation filter:
    .. math::
        \mathbf{f}^* = \mathbf{f} - \lambda \mathbf{f}^{\text{neq}} = (1 - \lambda)\mathbf{f} + \lambda \mathbf{P}_{\text{eq}}\mathbf{f}
        \equiv \mathbf{D}_{\lambda} \mathbf{f}

    where :math:`\lambda = \frac{\Delta t}{\tau} \in (0, 2)`.

    In the Hermite basis :math:`\tilde{\mathbf{f}} = \mathbf{W}^{-1} \mathbf{f}`,
    the operator :math:`\tilde{\mathbf{D}}_{\lambda}` is strictly symmetric and contractive.

    Parameters
    ----------
    tau : float, optional
        Relaxation time tau > 0.5, default 0.8.
    dt : float, optional
        Lattice time step, default 1.0.
    """

    def __init__(self, tau: float = 0.8, dt: float = 1.0) -> None:
        if tau <= 0.5:
            raise ValueError(f"Relaxation time must satisfy tau > 0.5, got {tau}")
        self.tau = float(tau)
        self.dt = float(dt)
        self.lam = float(self.dt / self.tau)

        # 1. Hermite weights
        self._W = np.diag(np.sqrt(WEIGHTS))
        self._W_inv = np.diag(1.0 / np.sqrt(WEIGHTS))

        # 2. Projector onto conserved moments (19x19)
        self._P_eq = self._compute_projector()
        self._P_tilde = self._W_inv @ self._P_eq @ self._W

        # 3. Denoising filter matrix: D_lambda = (1 - lambda) * I + lambda * P_tilde
        self._D_lambda = (1.0 - self.lam) * np.eye(19, dtype=np.float64) + self.lam * self._P_tilde

        # 4. Grover reflection diffuser: R_eq = 2 * P_tilde - I
        self._R_eq = 2.0 * self._P_tilde - np.eye(19, dtype=np.float64)

        # 5. 6-qubit unitary block dilation
        self._U64 = self._build_unitary_dilation()

    def _compute_projector(self) -> np.ndarray:
        """Construct the 19x19 orthogonal projector onto {rho, jx, jy, jz}."""
        P = np.zeros((19, 19), dtype=np.float64)
        for i in range(19):
            for j in range(19):
                cu = C[i, 0] * C[j, 0] + C[i, 1] * C[j, 1] + C[i, 2] * C[j, 2]
                P[i, j] = WEIGHTS[i] * (1.0 + cu / CS_SQ)
        return P

    def _build_unitary_dilation(self) -> np.ndarray:
        """Embed D_lambda into a 64x64 unitary dilation (1 ancilla + 5 velocity qubits)."""
        # Pad to 32x32
        D32 = np.eye(32, dtype=np.float64)
        D32[:19, :19] = self._D_lambda

        # SVD-aligned unitary dilation
        u, s, vh = np.linalg.svd(D32)
        s = np.clip(s, 0.0, 1.0)
        s_d = np.sqrt(np.maximum(0.0, 1.0 - s**2))

        D1 = u @ np.diag(s_d) @ u.T
        D2 = vh.T @ np.diag(s_d) @ vh

        U64 = np.block([[D32, D1], [D2, -D32.T]])
        return U64

    @property
    def projector(self) -> np.ndarray:
        """The 19x19 projection matrix onto conserved moments."""
        return self._P_eq

    @property
    def filter_matrix(self) -> np.ndarray:
        """The 19x19 denoising filter matrix."""
        return self._D_lambda

    @property
    def reflection_diffuser(self) -> np.ndarray:
        """The 19x19 Grover reflection diffuser R_eq = 2*P - I."""
        return self._R_eq

    @property
    def unitary_matrix(self) -> np.ndarray:
        """The 64x64 block-encoding unitary matrix."""
        return self._U64

    def build_circuit(self) -> QuantumCircuit:
        """Create a 6-qubit Qiskit quantum circuit for the denoising collision operator."""
        qc = QuantumCircuit(6, name="denoise_coll_3d")
        gate = UnitaryGate(self._U64, label="DenoiseColl")
        qc.append(gate, range(6))
        return qc

    def apply(self, f: np.ndarray) -> np.ndarray:
        """Apply denoising collision filter to distribution array f."""
        f = np.asarray(f, dtype=np.float64)
        orig_shape = f.shape
        f_2d = f.reshape(19, -1)

        # Transform to Hermite basis: f_tilde = W^-1 f
        f_tilde = self._W_inv @ f_2d
        # Apply denoising filter: f_tilde* = D_lambda f_tilde
        f_tilde_post = self._D_lambda @ f_tilde
        # Transform back: f* = W f_tilde*
        f_post = self._W @ f_tilde_post
        return f_post.reshape(orig_shape)

    def resource_metrics(self) -> Dict[str, Any]:
        """Compute circuit and operator resource metrics."""
        return {
            "operator_type": "Denoising Projector (arXiv:2604.09997)",
            "num_qubits": 6,
            "ancilla_qubits": 1,
            "velocity_qubits": 5,
            "unitary_dim": 64,
            "relaxation_lambda": self.lam,
            "exact_mass_conserved": True,
            "exact_momentum_conserved": True,
        }
