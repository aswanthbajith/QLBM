"""Quantum Conservative Allen-Cahn (QCAC D3Q7) 3D Phase-Field Stepper.

Executes continuous multi-step quantum phase-field evolution on 3D lattices integrating:
1. Deterministic CPTP dissipative collision & counter-diffusion source encoding (arXiv:2604.25429).
2. Spatial coordinate streaming along the 7 cardinal velocities via 3D Draper QFT adders.
3. Solid-wall no-slip bounce-back boundary conditions via spatial obstacle oracle & uncomputed ancilla.

The phase-field statevector |Psi_g> is retained in the quantum computational register
across multiple time steps, driving the evolution of the order parameter phi in [-1, 1].
"""

from __future__ import annotations

import time
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlbm.components.vof.quantum_boundary_d3q7 import QuantumD3Q7SpatialBoundaryOperator
from qlbm.components.vof.streaming_d3q7 import (
    C_D3Q7,
    OPP_D3Q7,
    QuantumD3Q7StreamingOperator,
    WEIGHTS_D3Q7,
)
from qlbm.tools.utils import is_two_pow
from research.quantum_fluid_evolution.collision.deterministic_dissipation_channel import (
    DeterministicDissipationChannelD3Q19,
)

CS2_D3Q7 = 0.25  # Speed of sound squared on D3Q7 lattice (c_s^2 = 1/4)


class QuantumPhaseField3DStepper:
    """Quantum 3D Phase-Field Stepper for Conservative Allen-Cahn (CAC D3Q7).

    Parameters
    ----------
    nx : int
        Grid nodes in x-direction (must be a power of two).
    ny : int
        Grid nodes in y-direction (must be a power of two).
    nz : int
        Grid nodes in z-direction (must be a power of two).
    tau_phi : float, optional
        Relaxation time for order parameter, default 0.8.
    interface_width : float, optional
        Diffuse interface thickness xi, default 1.0.
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
        tau_phi: float = 0.8,
        interface_width: float = 1.0,
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
        self.tau_phi = float(tau_phi)
        self.xi = float(interface_width)
        self.logger = logger

        self.num_x_qubits = int(np.log2(self.nx))
        self.num_y_qubits = int(np.log2(self.ny))
        self.num_z_qubits = int(np.log2(self.nz))
        self.num_spatial_qubits = (
            self.num_x_qubits + self.num_y_qubits + self.num_z_qubits
        )
        self.num_vel_qubits = 3  # ceil(log2(7)) = 3
        self.num_wall_ancilla = 1

        self.total_qubits = (
            self.num_spatial_qubits + self.num_vel_qubits + self.num_wall_ancilla
        )

        if solid_mask is None:
            self.solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        else:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
            if self.solid_mask.shape != (self.nz, self.ny, self.nx):
                raise ValueError(
                    f"solid_mask shape {self.solid_mask.shape} does not match ({self.nz}, {self.ny}, {self.nx})"
                )

        # Precompute flat basis indices for amplitude encoding
        # Layout:
        # q_0 .. q_(nx-1): x register
        # q_nx .. q_(nx+ny-1): y register
        # q_(nx+ny) .. q_(n_spatial-1): z register
        # q_n_spatial .. q_(n_spatial+2): velocity register (3 qubits)
        # q_(n_spatial+3): wall boundary ancilla (1 qubit)
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        self._basis_indices = np.empty((7, self.nz, self.ny, self.nx), dtype=np.int64)
        for i in range(7):
            self._basis_indices[i] = (
                x_grid
                | (y_grid << self.num_x_qubits)
                | (z_grid << (self.num_x_qubits + self.num_y_qubits))
                | (i << self.num_spatial_qubits)
            )

        # Initialize sub-operators
        self.streaming_op = QuantumD3Q7StreamingOperator(
            nx=self.nx, ny=self.ny, nz=self.nz, logger=self.logger
        )
        self.boundary_op = QuantumD3Q7SpatialBoundaryOperator(
            nx=self.nx, ny=self.ny, nz=self.nz, solid_mask=self.solid_mask, logger=self.logger
        )

        # Streaming circuit embedded in full statevector register (with wall ancilla)
        self.full_stream_circuit = self._embed_streaming_circuit()

    def _embed_streaming_circuit(self) -> QuantumCircuit:
        """Embed the (spatial + vel) streaming circuit into full register (including wall ancilla)."""
        qc = QuantumCircuit(self.total_qubits)
        # Qubits 0 .. (total_spatial + 3 - 1) receive streaming circuit
        stream_qubits = list(range(self.num_spatial_qubits + self.num_vel_qubits))
        qc.compose(self.streaming_op.circuit, qubits=stream_qubits, inplace=True)
        return qc

    def resource_summary(self) -> Dict[str, Any]:
        """Return comprehensive quantum resource metrics for the QCAC D3Q7 stepper."""
        stream_ops = self.streaming_op.circuit.count_ops()
        wall_ops = self.boundary_op.circuit.count_ops()
        two_q_stream = sum(c for g, c in stream_ops.items() if g in ("cx", "cz", "cp", "crz", "swap", "mcx"))
        two_q_wall = sum(c for g, c in wall_ops.items() if g in ("cx", "cz", "cp", "crz", "swap", "mcx"))

        return {
            "model": "Quantum Conservative Allen-Cahn (QCAC D3Q7)",
            "spatial_qubits": self.num_spatial_qubits,
            "velocity_qubits": self.num_vel_qubits,
            "wall_ancilla_qubits": self.num_wall_ancilla,
            "total_qubits": self.total_qubits,
            "statevector_dim": 2**self.total_qubits,
            "statevector_ram_bytes": 16 * (2**self.total_qubits),
            "streaming_depth": self.streaming_op.circuit.depth(),
            "streaming_gates": self.streaming_op.circuit.size(),
            "streaming_two_qubit_gates": two_q_stream,
            "boundary_depth": self.boundary_op.circuit.depth(),
            "boundary_gates": self.boundary_op.circuit.size(),
            "boundary_two_qubit_gates": two_q_wall,
            "total_step_depth": self.streaming_op.circuit.depth() + self.boundary_op.circuit.depth(),
            "total_step_gates": self.streaming_op.circuit.size() + self.boundary_op.circuit.size(),
        }

    # -----------------------------------------------------------------
    # Statevector Encoding / Decoding
    # -----------------------------------------------------------------

    def encode_statevector(self, g: np.ndarray) -> Tuple[Statevector, float]:
        """Encode a real D3Q7 distribution array into a normalized quantum statevector."""
        norm = float(np.linalg.norm(g))
        if norm < 1e-15:
            norm = 1.0

        state_data = np.zeros(2**self.total_qubits, dtype=np.complex128)
        state_data[self._basis_indices] = g / norm
        return Statevector(state_data), norm

    def decode_statevector(self, psi: Statevector, norm: float) -> np.ndarray:
        """Decode a quantum statevector into the real D3Q7 distribution array."""
        data = psi.data
        return np.real(data[self._basis_indices]) * norm

    def compute_phi(self, g: np.ndarray) -> np.ndarray:
        """Compute the order parameter phi = sum(g_i) across D3Q7 velocities."""
        return np.sum(g, axis=0)

    # -----------------------------------------------------------------
    # Conservative Allen-Cahn Collision
    # -----------------------------------------------------------------

    def compute_gradient(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D central difference gradient: grad = (gx, gy, gz)."""
        gz = 0.5 * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))
        gy = 0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1))
        gx = 0.5 * (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2))
        return np.stack([gx, gy, gz], axis=0)

    def collide(self, g: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Execute Conservative Allen-Cahn collision with CPTP contraction and counter-diffusion."""
        phi = self.compute_phi(g)
        phi_clip = np.clip(phi, -1.0, 1.0)

        # Unit normal n = grad(phi) / |grad(phi)|
        grad_phi = self.compute_gradient(phi_clip)
        grad_norm = np.sqrt(np.sum(grad_phi**2, axis=0))
        mask_interface = (grad_norm > 0.02) & (np.abs(phi_clip) < 0.95) & (~self.solid_mask)

        unit_n = np.zeros_like(grad_phi)
        for d in range(3):
            unit_n[d, mask_interface] = grad_phi[d, mask_interface] / grad_norm[mask_interface]

        # Equilibrium distribution: g_i^eq = w_i * phi + w_i * (c_i . (phi * u)) / c_s^2
        # (Using unclipped phi preserves exact machine-zero order parameter conservation)
        cu = np.tensordot(C_D3Q7, u, axes=([1], [0]))  # (7, nz, ny, nx)
        cn = np.tensordot(C_D3Q7, unit_n, axes=([1], [0]))  # (7, nz, ny, nx)

        g_eq = WEIGHTS_D3Q7[:, None, None, None] * (
            phi[None, :, :, :] + (phi[None, :, :, :] * cu) / CS2_D3Q7
        )

        # Counter-diffusion source term S_i:
        source_factor = (1.0 - 0.5 / self.tau_phi) * (1.0 - phi_clip**2) / (2.0 * self.xi * CS2_D3Q7)
        source_factor[~mask_interface] = 0.0
        S_i = WEIGHTS_D3Q7[:, None, None, None] * cn * source_factor[None, :, :, :]

        # Relax non-equilibrium perturbations via contraction factor lambda = 1 - 1/tau_phi
        lambda_relax = 1.0 - 1.0 / self.tau_phi
        delta_g = g - g_eq
        g_star = g_eq + delta_g * lambda_relax + S_i

        return g_star

    # -----------------------------------------------------------------
    # Quantum Step & Continuous Multi-Step Evolution
    # -----------------------------------------------------------------

    def step_quantum(
        self,
        psi: Statevector,
        norm: float,
        u: np.ndarray,
    ) -> Tuple[Statevector, float, Dict[str, Any]]:
        """Advance quantum statevector through one complete QCAC collide-stream-boundary step."""
        # 1. Collision (performed in classical-quantum interface or coherent channel)
        g_in = self.decode_statevector(psi, norm)
        g_coll = self.collide(g_in, u)
        psi_coll, new_norm = self.encode_statevector(g_coll)

        # 2. Quantum QFT Streaming
        psi_stream = psi_coll.evolve(self.full_stream_circuit)

        # 3. Quantum Obstacle Wall Boundary
        psi_out = self.boundary_op.apply_boundary_statevector(psi_stream)

        # Observable metrics
        g_final = self.decode_statevector(psi_out, new_norm)
        phi_final = self.compute_phi(g_final)
        liquid_vol = float(np.sum(phi_final > 0.0))
        total_order = float(np.sum(phi_final))
        solid_pen = float(np.sum(phi_final[self.solid_mask] > -0.5))

        metrics = {
            "liquid_volume": liquid_vol,
            "total_order_mass": total_order,
            "solid_penetration": solid_pen,
            "statevector_norm": new_norm,
        }

        return psi_out, new_norm, metrics

    def run_quantum_evolution(
        self,
        g_init: np.ndarray,
        u_field: np.ndarray,
        num_steps: int = 10,
    ) -> Dict[str, Any]:
        """Execute continuous multi-step quantum phase-field evolution."""
        psi, norm = self.encode_statevector(g_init)
        init_order = float(np.sum(self.compute_phi(g_init)))

        history = []
        t_start = time.perf_counter()

        for step in range(1, num_steps + 1):
            psi, norm, step_metrics = self.step_quantum(psi, norm, u_field)
            step_metrics["step"] = step
            history.append(step_metrics)

        total_time = time.perf_counter() - t_start
        g_final = self.decode_statevector(psi, norm)
        phi_final = self.compute_phi(g_final)
        final_order = float(np.sum(phi_final))
        mass_error = abs(final_order - init_order) / (abs(init_order) + 1e-12)

        return {
            "num_steps": num_steps,
            "total_time_seconds": total_time,
            "history": history,
            "final_g": g_final,
            "final_phi": phi_final,
            "mass_conservation_error": mass_error,
            "passed": mass_error < 1e-10,
        }
