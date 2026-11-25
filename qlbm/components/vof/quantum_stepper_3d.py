"""End-to-End Multi-Step Quantum 3D Fluid Stepper for D3Q19 Lattices.

Executes continuous multi-step quantum fluid evolution integrating:
1. Linearized D3Q19 BGK collision & gravitational body forcing via 6-qubit block-encoding.
2. Spatial coordinate streaming along 19 discrete velocities via 3D Draper QFT adders.
3. Solid-wall no-slip bounce-back boundary conditions via spatial obstacle oracle.

The quantum fluid statevector |Psi_t> is retained continuously in the quantum register
across multiple time steps without full classical reconstruction and reinitialization
of the fluid distribution field between steps.
"""

from __future__ import annotations

import time
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlbm.classical.d3q19 import C, CS_SQ, WEIGHTS
from qlbm.components.vof.collision_3d import LinearizedQuantum3DCollisionOperator
from qlbm.components.vof.quantum_boundary_3d import QuantumD3Q19SpatialBoundaryOperator
from qlbm.components.vof.streaming_3d import QuantumD3Q19StreamingOperator
from qlbm.tools.utils import is_two_pow


class QuantumFluid3DStepper:
    """End-to-end multi-step quantum fluid stepper for 3D D3Q19 lattices.

    Parameters
    ----------
    nx : int
        Grid nodes in x-direction (must be a power of two).
    ny : int
        Grid nodes in y-direction (must be a power of two).
    nz : int
        Grid nodes in z-direction (must be a power of two).
    tau : float, optional
        BGK relaxation time tau > 0.5, default 0.8.
    g : tuple or np.ndarray, optional
        Gravitational acceleration vector in lattice units, default (0.0, 0.0, -0.01).
    dt : float, optional
        Lattice time step, default 1.0.
    solid_mask : np.ndarray, optional
        Boolean array of shape `(nz, ny, nx)` marking solid nodes.
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        tau: float = 0.8,
        g: Union[Tuple[float, float, float], np.ndarray] = (0.0, 0.0, -0.01),
        dt: float = 1.0,
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
        self.tau = float(tau)
        self.g = np.asarray(g, dtype=np.float64)
        self.dt = float(dt)
        self.logger = logger

        self.num_x_qubits = int(np.log2(self.nx))
        self.num_y_qubits = int(np.log2(self.ny))
        self.num_z_qubits = int(np.log2(self.nz))
        self.num_spatial_qubits = (
            self.num_x_qubits + self.num_y_qubits + self.num_z_qubits
        )
        self.num_vel_qubits = 5  # ceil(log2(19)) = 5
        self.num_coll_ancilla = 1
        self.num_wall_ancilla = 1

        self.total_qubits = (
            self.num_spatial_qubits
            + self.num_vel_qubits
            + self.num_coll_ancilla
            + self.num_wall_ancilla
        )

        if solid_mask is None:
            self.solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        else:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
            if self.solid_mask.shape != (self.nz, self.ny, self.nx):
                raise ValueError(
                    f"solid_mask shape {self.solid_mask.shape} does not match ({self.nz}, {self.ny}, {self.nx})"
                )

        # Precompute Hermite weight scaling matrices
        self._W = np.sqrt(WEIGHTS)[:, None, None, None]
        self._W_inv = (1.0 / np.sqrt(WEIGHTS))[:, None, None, None]

        # Precompute flat basis indices for spatial + velocity registers
        # Register layout:
        # q_0 .. q_(nx-1): x register
        # q_nx .. q_(nx+ny-1): y register
        # q_(nx+ny) .. q_(n_spatial-1): z register
        # q_n_spatial .. q_(n_spatial+4): velocity register
        # q_(n_spatial+5): collision ancilla (0 = success)
        # q_(n_spatial+6): wall boundary ancilla (0 = uncomputed)
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        self._basis_indices = np.empty((19, self.nz, self.ny, self.nx), dtype=np.int64)
        for i in range(19):
            self._basis_indices[i] = (
                x_grid
                | (y_grid << self.num_x_qubits)
                | (z_grid << (self.num_x_qubits + self.num_y_qubits))
                | (i << self.num_spatial_qubits)
            )

        # Boolean mask of success subspace (coll_anc == 0 and wall_anc == 0)
        self._success_mask = np.zeros(2**self.total_qubits, dtype=bool)
        self._success_mask[self._basis_indices.ravel()] = True

        # Initialize constituent quantum operators
        self.collision_op = LinearizedQuantum3DCollisionOperator(
            tau=self.tau, g=self.g, dt=self.dt, logger=self.logger
        )
        self.stream_op = QuantumD3Q19StreamingOperator(
            nx=self.nx, ny=self.ny, nz=self.nz, logger=self.logger
        )
        self.boundary_op = QuantumD3Q19SpatialBoundaryOperator(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            solid_mask=self.solid_mask,
            logger=self.logger,
        )

        # Assemble the composite single-step quantum circuit
        self.step_circuit = self._build_step_circuit()

    def _build_step_circuit(self) -> QuantumCircuit:
        """Construct the composite single-step quantum circuit."""
        reg_spatial = list(range(self.num_spatial_qubits))
        reg_v = list(
            range(self.num_spatial_qubits, self.num_spatial_qubits + self.num_vel_qubits)
        )
        q_coll = self.num_spatial_qubits + self.num_vel_qubits
        q_wall = q_coll + 1

        qc = QuantumCircuit(self.total_qubits, name="quantum_d3q19_step")

        # 1. Linearized collision & gravity block-encoding on (reg_v, q_coll)
        qc.compose(self.collision_op.circuit, qubits=reg_v + [q_coll], inplace=True)

        # 2. Draper QFT streaming on (reg_spatial, reg_v)
        qc.compose(self.stream_op.circuit, qubits=reg_spatial + reg_v, inplace=True)

        # 3. Spatial wall oracle + controlled bounce-back + uncompute on (reg_spatial, reg_v, q_wall)
        qc.compose(
            self.boundary_op.circuit,
            qubits=reg_spatial + reg_v + [q_wall],
            inplace=True,
        )

        return qc

    # -----------------------------------------------------------------
    # State Preparation & Readout
    # -----------------------------------------------------------------

    def prepare_initial_statevector(
        self, f: np.ndarray
    ) -> Tuple[Statevector, float]:
        """Encode initial classical distributions into a normalized quantum statevector.

        Uses direct amplitude encoding in the Hermite-weighted basis:
        .. math::
            |\\Psi_0\\rangle = \\frac{1}{\\mathcal{N}} \\sum_{\\mathbf{x}, i} \\frac{f_i(\\mathbf{x})}{\\sqrt{w_i}} |\\mathbf{x}\\rangle |i\\rangle |0\\rangle_{\\text{coll}} |0\\rangle_{\\text{wall}}

        Parameters
        ----------
        f : np.ndarray
            Initial distribution array of shape `(19, nz, ny, nx)`.

        Returns
        -------
        Tuple[Statevector, float]
            The normalized Statevector and the initial physical norm scalar.
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape != (19, self.nz, self.ny, self.nx):
            raise ValueError(
                f"Expected shape (19, {self.nz}, {self.ny}, {self.nx}), got {f.shape}"
            )

        f_tilde = f * self._W_inv
        initial_scale = float(np.linalg.norm(f_tilde))
        if initial_scale < 1e-15:
            initial_scale = 1.0

        amp = np.zeros(2**self.total_qubits, dtype=np.complex128)
        amp[self._basis_indices] = f_tilde / initial_scale
        psi = Statevector(amp)
        return psi, initial_scale

    def readout_distributions(
        self, psi: Statevector, scale: float
    ) -> np.ndarray:
        """Decode physical distribution array from the quantum statevector.

        .. math::
            f_i(\\mathbf{x}) = \\text{Re}(\\langle \\mathbf{x}, i, 0, 0 | \\Psi \\rangle) \\cdot \\text{scale} \\cdot \\sqrt{w_i}

        Parameters
        ----------
        psi : Statevector
            The current quantum statevector.
        scale : float
            Current physical scaling factor.

        Returns
        -------
        np.ndarray
            Physical distributions of shape `(19, nz, ny, nx)`.
        """
        raw_amps = psi.data[self._basis_indices]
        f_phys = np.real(raw_amps) * scale * self._W
        return f_phys

    # -----------------------------------------------------------------
    # Multi-Step Quantum Execution
    # -----------------------------------------------------------------

    def step_quantum(
        self,
        psi: Statevector,
        scale: float,
    ) -> Tuple[Statevector, float, float]:
        """Execute a single quantum time step without classical wavefunction reconstruction.

        The quantum state evolves under the composite circuit. Ancilla postselection
        projects onto the physical success subspace (|coll_anc=0, wall_anc=0>),
        re-normalizing the statevector while updating the physical scale factor.

        Parameters
        ----------
        psi : Statevector
            Input quantum statevector.
        scale : float
            Current physical scale factor.

        Returns
        -------
        Tuple[Statevector, float, float]
            Updated (psi_next, new_scale, p_succ).
        """
        # 1. Circuit evolution
        psi_evolved = psi.evolve(self.step_circuit)

        # 2. Ancilla success probability on success subspace
        probs = psi_evolved.probabilities()
        p_succ = float(np.sum(probs[self._success_mask]))

        # 3. Project onto physical subspace and re-normalize quantum state
        projected_data = np.zeros_like(psi_evolved.data)
        projected_data[self._success_mask] = psi_evolved.data[self._success_mask]
        norm_succ = float(np.linalg.norm(projected_data))

        if norm_succ > 1e-15:
            projected_data /= norm_succ
        psi_next = Statevector(projected_data)

        # 4. Scale factor update reflecting block-encoding factor alpha
        alpha = self.collision_op.normalization_factor
        new_scale = scale * alpha * norm_succ

        return psi_next, new_scale, p_succ

    def run_quantum_evolution(
        self,
        f_initial: np.ndarray,
        num_steps: int = 5,
    ) -> Dict[str, Any]:
        """Execute multiple consecutive quantum fluid-evolution steps.

        Maintains the quantum statevector continuously across all steps
        without intermediate classical decoding and re-initialization.

        Parameters
        ----------
        f_initial : np.ndarray
            Initial distribution array of shape `(19, nz, ny, nx)`.
        num_steps : int, optional
            Number of consecutive quantum steps to execute.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing final distributions, history of observables,
            success probabilities, and timing metrics.
        """
        t0 = time.perf_counter()
        psi, scale = self.prepare_initial_statevector(f_initial)

        history: List[Dict[str, Any]] = []
        cumulative_p_succ = 1.0

        for s in range(1, num_steps + 1):
            t_step_start = time.perf_counter()
            psi, scale, p_succ = self.step_quantum(psi, scale)
            t_step = time.perf_counter() - t_step_start

            cumulative_p_succ *= p_succ

            # Non-destructive observable evaluation from quantum statevector
            f_curr = self.readout_distributions(psi, scale)
            rho = np.sum(f_curr, axis=0)
            total_mass = float(np.sum(rho))
            jx = float(np.sum(f_curr * C[:, 0, None, None, None]))
            jy = float(np.sum(f_curr * C[:, 1, None, None, None]))
            jz = float(np.sum(f_curr * C[:, 2, None, None, None]))
            kin_energy = 0.5 * (jx**2 + jy**2 + jz**2) / (total_mass + 1e-12)

            history.append({
                "step": s,
                "p_succ": p_succ,
                "cumulative_p_succ": cumulative_p_succ,
                "total_mass": total_mass,
                "jx": jx,
                "jy": jy,
                "jz": jz,
                "kinetic_energy": kin_energy,
                "step_time_sec": t_step,
            })

        t_total = time.perf_counter() - t0
        f_final = self.readout_distributions(psi, scale)

        return {
            "num_steps": num_steps,
            "f_final": f_final,
            "final_statevector": psi,
            "final_scale": scale,
            "history": history,
            "total_time_sec": t_total,
        }

    # -----------------------------------------------------------------
    # Classical Reference Evolution
    # -----------------------------------------------------------------

    def run_classical_evolution(
        self,
        f_initial: np.ndarray,
        num_steps: int = 5,
    ) -> Dict[str, Any]:
        """Execute identical evolution using the classical D3Q19 solver.

        Used for exact step-by-step physical and numerical benchmarking.

        Parameters
        ----------
        f_initial : np.ndarray
            Initial distribution array of shape `(19, nz, ny, nx)`.
        num_steps : int, optional
            Number of steps.

        Returns
        -------
        Dict[str, Any]
            Classical trajectory, final distributions, and observable history.
        """
        t0 = time.perf_counter()
        f = f_initial.copy()
        history: List[Dict[str, Any]] = []

        for s in range(1, num_steps + 1):
            t_step_start = time.perf_counter()

            # 1. Collision + gravity
            f_coll = self.collision_op.apply(f)
            # 2. Streaming
            f_stream = self.stream_op.stream_distribution_array(f_coll)
            # 3. Solid bounce-back
            f = self.boundary_op.apply_to_distribution_array(f_stream)

            t_step = time.perf_counter() - t_step_start

            rho = np.sum(f, axis=0)
            total_mass = float(np.sum(rho))
            jx = float(np.sum(f * C[:, 0, None, None, None]))
            jy = float(np.sum(f * C[:, 1, None, None, None]))
            jz = float(np.sum(f * C[:, 2, None, None, None]))
            kin_energy = 0.5 * (jx**2 + jy**2 + jz**2) / (total_mass + 1e-12)

            history.append({
                "step": s,
                "total_mass": total_mass,
                "jx": jx,
                "jy": jy,
                "jz": jz,
                "kinetic_energy": kin_energy,
                "step_time_sec": t_step,
            })

        t_total = time.perf_counter() - t0
        return {
            "num_steps": num_steps,
            "f_final": f,
            "history": history,
            "total_time_sec": t_total,
        }

    # -----------------------------------------------------------------
    # Resource Metrics
    # -----------------------------------------------------------------

    def resource_summary(self) -> Dict[str, Any]:
        """Return comprehensive quantum resource metrics for the stepper."""
        ops = self.step_circuit.count_ops()
        two_qubit_gates = sum(
            count
            for gate, count in ops.items()
            if gate in ("cx", "cz", "cp", "crz", "swap", "mcx")
        )
        return {
            "total_qubits": self.total_qubits,
            "spatial_qubits": self.num_spatial_qubits,
            "velocity_qubits": self.num_vel_qubits,
            "ancilla_qubits": self.num_coll_ancilla + self.num_wall_ancilla,
            "step_circuit_depth": self.step_circuit.depth(),
            "step_circuit_size": self.step_circuit.size(),
            "two_qubit_gates": two_qubit_gates,
            "statevector_dim": 2**self.total_qubits,
            "statevector_ram_bytes": 16 * (2**self.total_qubits),
            "gate_breakdown": dict(ops),
        }
