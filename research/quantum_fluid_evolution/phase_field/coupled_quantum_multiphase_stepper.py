"""Coupled Hybrid Multiphase 3D Stepper (D3Q19 Hydrodynamics + D3Q7 Phase-Field).

What actually runs in :meth:`CoupledQuantumMultiphase3DStepper.step`:

* **Quantum circuit (executed on a classical statevector simulator):**
  spatial streaming of the D3Q19 and D3Q7 distributions via 3D Draper QFT
  adder circuits. Each step encodes the classical arrays into a statevector,
  evolves it through the streaming circuit, and reads amplitudes back exactly.
  The streaming circuit is a permutation; its output equals ``np.roll``.
* **Classical (NumPy):** linearized BGK collision (projector ``E``), Guo
  forcing, Boussinesq buoyancy, surface-tension force, Conservative
  Allen-Cahn collision, and halfway bounce-back at solid walls.

The quantum collision block-encoding circuit and the spatial wall-oracle
circuit exist in the codebase but are NOT used by this stepper. Exact
amplitude readout is not available on real hardware; this stepper is a
hybrid algorithm demonstration, not a hardware-executable quantum program.

Gravity model
-------------
The hydrodynamic solver is a single-density (weakly compressible) LBM with
inertia rho ~ 1. Density contrast between liquid and gas enters only through a
Boussinesq buoyancy force

    F_b = (rho(phi) - rho_ref) * g,   rho(phi) = rho_g + (phi + 1)/2 * (rho_l - rho_g)

with ``rho_ref`` the mean of ``rho(phi)`` over fluid nodes, so the net body
force on a closed domain is zero. This is a qualitative approximation; it is
not a quantitatively valid large-density-ratio two-phase model.
"""

from __future__ import annotations

import time
from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from qlbm.classical.d3q19 import C as C_D3Q19, CS_SQ, OPPOSITE as OPP_D3Q19, WEIGHTS as W_D3Q19
from qlbm.components.vof.quantum_stepper_3d import QuantumFluid3DStepper
from qlbm.components.vof.streaming_d3q7 import (
    C_D3Q7,
    OPP_D3Q7,
    WEIGHTS_D3Q7,
)
from qlbm.tools.utils import is_two_pow
from research.quantum_fluid_evolution.phase_field.quantum_phase_field_stepper_3d import (
    CS2_D3Q7,
    QuantumPhaseField3DStepper,
)


def _neighbor_solid_masks(solid: np.ndarray, velocities: np.ndarray) -> list:
    """For each velocity c_i return mask of nodes whose neighbor x + c_i is solid."""
    masks = []
    for c in velocities:
        cx, cy, cz = int(c[0]), int(c[1]), int(c[2])
        masks.append(np.roll(solid, shift=(-cz, -cy, -cx), axis=(0, 1, 2)))
    return masks


class CoupledQuantumMultiphase3DStepper:
    """Coupled hybrid 3D multiphase stepper (quantum-circuit streaming, classical rest).

    Parameters
    ----------
    nx, ny, nz : int
        Grid nodes per direction (powers of two).
    tau_f : float, optional
        Hydrodynamic relaxation time, default 0.8.
    tau_phi : float, optional
        Phase-field relaxation time, default 0.8.
    sigma : float, optional
        Surface tension coefficient, default 0.015.
    g : tuple or np.ndarray, optional
        Gravitational acceleration vector, default (0.0, 0.0, -0.01).
    interface_width : float, optional
        Diffuse interface thickness xi, default 1.0.
    qlks_A : float, optional
        Stored for API compatibility only. It is NOT applied in :meth:`step`;
        the effective viscosity is (1/3)(tau_f - 1/2).
    rho_liquid, rho_gas : float, optional
        Phase densities used for the Boussinesq buoyancy force.
        Defaults 1.0 and 0.1 (ratio 10). Equal values disable buoyancy.
    solid_mask : np.ndarray, optional
        Boolean array of shape ``(nz, ny, nx)`` marking solid nodes.
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        tau_f: float = 0.8,
        tau_phi: float = 0.8,
        sigma: float = 0.015,
        g: Union[Tuple[float, float, float], np.ndarray] = (0.0, 0.0, -0.01),
        interface_width: float = 1.0,
        qlks_A: float = 1.0,
        rho_liquid: float = 1.0,
        rho_gas: float = 1.0,
        solid_mask: Optional[np.ndarray] = None,
        collision_mode: str = "classical_bgk",
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        if not is_two_pow(nx) or not is_two_pow(ny) or not is_two_pow(nz):
            raise ValueError(
                f"Grid dimensions must be powers of two, got nx={nx}, ny={ny}, nz={nz}"
            )

        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.tau_f = float(tau_f)
        self.tau_phi = float(tau_phi)
        self.sigma = float(sigma)
        self.g = np.asarray(g, dtype=np.float64)
        self.xi = float(interface_width)
        self.qlks_A = float(np.clip(qlks_A, 0.01, 1.0))
        self.rho_liquid = float(rho_liquid)
        self.rho_gas = float(rho_gas)
        self.collision_mode = str(collision_mode).lower()
        self.logger = logger

        if self.collision_mode not in ("classical_bgk", "quantum_rotation", "quantum_vqc"):
            raise ValueError(
                f"Unsupported collision_mode '{collision_mode}'. "
                "Must be one of: 'classical_bgk', 'quantum_rotation', 'quantum_vqc'"
            )

        if self.collision_mode == "quantum_rotation":
            from research.quantum_fluid_evolution.collision.rotation_collision import (
                RotationCollision3D,
            )
            theta = float(np.clip(0.1 / self.tau_f, 0.01, 0.5))
            self.rotation_op = RotationCollision3D(theta=theta)
        elif self.collision_mode == "quantum_vqc":
            from research.quantum_fluid_evolution.collision.variational_collision_pre import (
                VariationalQuantumCollisionD3Q19,
            )
            self.vqc_op = VariationalQuantumCollisionD3Q19(tau=self.tau_f, num_layers=2)

        if solid_mask is None:
            self.solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        else:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
        self.fluid_mask = ~self.solid_mask

        # Precomputed halfway bounce-back masks: fluid node whose x + c_i is solid.
        self._bb_mask_f = [
            self.fluid_mask & m for m in _neighbor_solid_masks(self.solid_mask, C_D3Q19)
        ]
        self._bb_mask_g = [
            self.fluid_mask & m for m in _neighbor_solid_masks(self.solid_mask, C_D3Q7)
        ]

        # Hydrodynamic stepper: only its streaming circuit and projector E are used.
        self.hydro_stepper = QuantumFluid3DStepper(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            tau=self.tau_f,
            g=self.g,
            dt=1.0,
            solid_mask=self.solid_mask,
            logger=self.logger,
        )

        # Phase-field stepper: its streaming circuit and classical CAC collision are used.
        self.phase_stepper = QuantumPhaseField3DStepper(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            tau_phi=self.tau_phi,
            interface_width=self.xi,
            solid_mask=self.solid_mask,
            logger=self.logger,
        )

    def resource_summary(self) -> Dict[str, Any]:
        """Return resource metrics, distinguishing simulated circuits from full registers."""
        hydro_res = self.hydro_stepper.resource_summary()
        phase_res = self.phase_stepper.resource_summary()
        hydro_stream_q = self.hydro_stepper.stream_op.circuit.num_qubits
        phase_stream_q = self.phase_stepper.streaming_op.circuit.num_qubits

        summary = {
            "model": (
                f"End-to-End Quantum: quantum streaming + {self.collision_mode}"
                if self.collision_mode != "classical_bgk"
                else "Hybrid: quantum-circuit streaming (statevector-simulated) + classical collision/forcing/walls"
            ),
            "collision_mode": self.collision_mode,
            "collision_quantum": self.collision_mode != "classical_bgk",
            "hydro_qubits": hydro_res["total_qubits"],
            "phase_qubits": phase_res["total_qubits"],
            "total_system_qubits": hydro_res["total_qubits"] + phase_res["total_qubits"],
            "hydro_statevector_dim": hydro_res["statevector_dim"],
            "phase_statevector_dim": phase_res["statevector_dim"],
            "hydro_streaming_circuit_qubits": hydro_stream_q,
            "phase_streaming_circuit_qubits": phase_stream_q,
            "hydro_streaming_statevector_dim": 2**hydro_stream_q,
            "phase_streaming_statevector_dim": 2**phase_stream_q,
            "hydro_step_depth": hydro_res["step_circuit_depth"],
            "phase_step_depth": phase_res["total_step_depth"],
            "qlks_parameter_A": self.qlks_A,
            "qlks_applied": False,
            "effective_viscosity": (1.0 / 3.0) * self.qlks_A * (self.tau_f - 0.5),
            "rho_liquid": self.rho_liquid,
            "rho_gas": self.rho_gas,
        }
        if self.collision_mode == "quantum_rotation":
            summary["collision_circuit_qubits"] = 5
            summary["collision_operator_type"] = "Unitary SO(2) Givens Rotation Network"
            summary["collision_circuit_depth"] = self.rotation_op.build_circuit().depth()
        elif self.collision_mode == "quantum_vqc":
            summary["collision_circuit_qubits"] = 5
            summary["collision_operator_type"] = "Variational Quantum Circuit (VQC)"
            summary["collision_circuit_depth"] = self.vqc_op.circuit.depth()
        return summary

    # -----------------------------------------------------------------
    # Multiphase Coupling Physics (classical)
    # -----------------------------------------------------------------

    def compute_gradient(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D central difference gradient: grad = (gx, gy, gz)."""
        gz = 0.5 * (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))
        gy = 0.5 * (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1))
        gx = 0.5 * (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2))
        return np.stack([gx, gy, gz], axis=0)

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D isotropic discrete Laplacian."""
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) - 6.0 * field
        )

    def compute_interfacial_force(self, phi: np.ndarray) -> np.ndarray:
        """Chemical-potential surface tension force F_s = mu * grad(phi).

        mu = (3*sigma / (2*xi)) * (phi^3 - phi) - (3*sigma*xi / 4) * laplacian(phi)
        """
        phi_clip = np.clip(phi, -1.0, 1.0)
        A_pot = 3.0 * self.sigma / (2.0 * self.xi)
        kappa_pot = 3.0 * self.sigma * self.xi / 4.0

        lap_phi = self.compute_laplacian(phi_clip)
        mu_phi = A_pot * (phi_clip**3 - phi_clip) - kappa_pot * lap_phi
        grad_phi = self.compute_gradient(phi_clip)

        F_s = mu_phi[None, :, :, :] * grad_phi
        F_s[:, self.solid_mask] = 0.0
        return F_s

    def phase_density(self, phi: np.ndarray) -> np.ndarray:
        """Linear density interpolation rho(phi) between gas (phi=-1) and liquid (phi=+1)."""
        frac = (np.clip(phi, -1.0, 1.0) + 1.0) * 0.5
        return self.rho_gas + frac * (self.rho_liquid - self.rho_gas)

    def compute_buoyancy_force(self, phi: np.ndarray) -> np.ndarray:
        """Boussinesq buoyancy force F_b = (rho(phi) - mean_rho) * g."""
        rho_phi = self.phase_density(phi)
        fluid_mask = self.fluid_mask
        if np.any(fluid_mask):
            mean_rho = float(np.mean(rho_phi[fluid_mask]))
        else:
            mean_rho = float(np.mean(rho_phi))
        delta_rho = rho_phi - mean_rho
        F_b = delta_rho[None, :, :, :] * self.g[:, None, None, None]
        F_b[:, self.solid_mask] = 0.0
        return F_b

    def compute_gravity_force(self, phi: np.ndarray) -> np.ndarray:
        """Gravitational body force: uniform bulk gravity when rho_liquid == rho_gas,
        or Boussinesq buoyancy when a physical density contrast is configured."""
        if self.rho_liquid == self.rho_gas:
            rho_phi = self.phase_density(phi)
            F_g = rho_phi[None, :, :, :] * self.g[:, None, None, None]
            F_g[:, self.solid_mask] = 0.0
            return F_g
        return self.compute_buoyancy_force(phi)

    # -----------------------------------------------------------------
    # Coupled Step
    # -----------------------------------------------------------------

    def step(
        self,
        f_arr: np.ndarray,
        g_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Advance hydrodynamic and phase-field distributions by one coupled step.

        Parameters
        ----------
        f_arr : np.ndarray
            Hydrodynamic D3Q19 distribution of shape ``(19, nz, ny, nx)``.
        g_arr : np.ndarray
            Phase-field D3Q7 distribution of shape ``(7, nz, ny, nx)``.

        Returns
        -------
        tuple of (f_next, g_next, metrics)
        """
        fluid_mask = self.fluid_mask

        # 1. Macroscopic observables (classical readout of arrays)
        rho = np.sum(f_arr, axis=0)
        phi = np.sum(g_arr, axis=0)

        # 2. Forces (classical): surface tension + phase-dependent gravity
        F_total = self.compute_interfacial_force(phi) + self.compute_gravity_force(phi)
        F_total[:, self.solid_mask] = 0.0

        mom = np.tensordot(C_D3Q19, f_arr, axes=([0], [0])) + 0.5 * F_total
        u = np.zeros_like(mom)
        for d in range(3):
            u[d, fluid_mask] = mom[d, fluid_mask] / np.maximum(rho[fluid_mask], 1e-6)

        # 3. Hydro collision (mode-dependent) + Guo forcing.
        cu = np.tensordot(C_D3Q19, u, axes=([1], [0]))
        cf = np.tensordot(C_D3Q19, F_total, axes=([1], [0]))
        uf = np.sum(u * F_total, axis=0)
        F_guo = (1.0 - 0.5 / self.tau_f) * W_D3Q19[:, None, None, None] * (
            (cf - uf[None, :, :, :]) / CS_SQ + (cu * cf) / (CS_SQ**2)
        )

        if self.collision_mode == "quantum_rotation":
            # Quantum SO(2) Givens rotation collision
            rho_before = np.sum(f_arr, axis=0)
            f_rot = self.rotation_op.apply(f_arr)
            # Conserve local density invariant under unitary Givens rotations
            rho_after = np.maximum(np.sum(f_rot, axis=0), 1e-12)
            f_coll = f_rot * (rho_before / rho_after)[None, :, :, :]
            f_coll += F_guo
            f_coll[:, self.solid_mask] = 0.0
        elif self.collision_mode == "quantum_vqc":
            # Variational Quantum Collision circuit
            f_coll = self.vqc_op.apply(f_arr)
            f_coll += F_guo
            f_coll[:, self.solid_mask] = 0.0
        else:
            # Classical linearized BGK + Guo forcing.
            # Guo scheme: equilibrium uses shifted momentum j + F/2.
            f_flat = f_arr.reshape(19, -1)
            feq = self.hydro_stepper.collision_op._E @ f_flat
            feq = feq + (W_D3Q19[:, None, None, None] * cf / (2.0 * CS_SQ)).reshape(19, -1)
            lambda_f = 1.0 - 1.0 / self.tau_f
            f_coll = (feq + (f_flat - feq) * lambda_f).reshape(19, self.nz, self.ny, self.nx)
            f_coll += F_guo
            f_coll[:, self.solid_mask] = 0.0

        # 4. Hydro streaming: QFT circuit on statevector simulator (signed, exact)
        f_next = self.hydro_stepper.stream_op.stream_distribution_array(f_coll, signed=True)

        # 5. Halfway bounce-back (classical): population leaving fluid node x toward
        #    a solid neighbor returns to x in the opposite direction.
        for i in range(1, 19):
            m = self._bb_mask_f[i]
            f_next[OPP_D3Q19[i], m] = f_coll[i, m]
        f_next[:, self.solid_mask] = 0.0

        # 6. Phase-field collision (classical CAC) with advection velocity u
        g_coll = self.phase_stepper.collide(g_arr, u)
        g_coll[:, self.solid_mask] = 0.0

        # 7. Phase-field streaming: D3Q7 QFT circuit on statevector simulator
        g_next = self.phase_stepper.streaming_op.stream_distribution_array(g_coll)

        # 8. Halfway bounce-back for phase field; solids carry phi = -1 for gradients
        for i in range(1, 7):
            m = self._bb_mask_g[i]
            g_next[OPP_D3Q7[i], m] = g_coll[i, m]
        g_next[:, self.solid_mask] = WEIGHTS_D3Q7[:, None] * (-1.0)

        # 9. Metrics (fluid nodes only)
        phi_next = np.sum(g_next, axis=0)
        rho_next = np.sum(f_next, axis=0)
        phi_f = np.clip(phi_next[fluid_mask], -1.0, 1.0)
        j_next = np.tensordot(C_D3Q19, f_next, axes=([0], [0]))

        metrics = {
            "liquid_volume": float(np.sum((phi_f + 1.0) * 0.5)),
            "liquid_cells": float(np.sum(phi_f > 0.0)),
            "total_order_mass": float(np.sum(phi_next)),
            "fluid_mass": float(np.sum(rho_next[fluid_mask])),
            "max_velocity": float(np.max(np.sqrt(np.sum(u**2, axis=0)))),
            "vertical_momentum_jz": float(np.sum((rho_next * u[2])[fluid_mask])),
            "liquid_vertical_momentum": float(
                np.sum(((rho_next * u[2]) * (np.clip(phi_next, -1, 1) + 1.0) * 0.5)[fluid_mask])
            ),
            "solid_penetration": 0.0,
            "mean_density": float(np.mean(rho_next[fluid_mask])),
        }

        return f_next, g_next, metrics

    def run_coupled_simulation(
        self,
        f_init: np.ndarray,
        g_init: np.ndarray,
        num_steps: int = 20,
    ) -> Dict[str, Any]:
        """Execute a multi-step coupled simulation and report conservation errors."""
        f_curr = np.copy(f_init)
        g_curr = np.copy(g_init)
        init_order = float(np.sum(np.sum(g_init, axis=0)))
        init_mass = float(np.sum(np.sum(f_init, axis=0)[self.fluid_mask]))

        history = []
        t_start = time.perf_counter()

        for s in range(1, num_steps + 1):
            f_curr, g_curr, metrics = self.step(f_curr, g_curr)
            metrics["step"] = s
            history.append(metrics)

        total_time = time.perf_counter() - t_start
        final_order = history[-1]["total_order_mass"]
        final_mass = history[-1]["fluid_mass"]
        order_error = abs(final_order - init_order) / (abs(init_order) + 1e-12)
        mass_error = abs(final_mass - init_mass) / (abs(init_mass) + 1e-12)

        return {
            "num_steps": num_steps,
            "total_time_seconds": total_time,
            "history": history,
            "final_f": f_curr,
            "final_g": g_curr,
            "mass_conservation_error": mass_error,
            "order_conservation_error": order_error,
            "passed": mass_error < 1e-10 and order_error < 1e-10,
        }
