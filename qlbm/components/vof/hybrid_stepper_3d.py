"""Modular Hybrid Classical-Quantum Stepper for 3D Free-Surface VOF-LBM."""

from __future__ import annotations

import time
from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np

from qlbm.classical.d3q19 import (
    C,
    CS_SQ,
    OPPOSITE,
    bgk_collision,
    compute_equilibrium,
    compute_macroscopic,
    stream_and_bounce,
)
from qlbm.classical.hybrid_vof_3d import VOF3DHydroSolver
from qlbm.classical.vof_3d import (
    CellType,
    compute_directional_mass_fluxes_3d,
    compute_normals_3d,
    reconstruct_gas_boundaries_3d,
    update_cell_states_3d,
)
from qlbm.components.vof.boundary_3d import QuantumD3Q19BounceBackOperator
from qlbm.components.vof.streaming_3d import QuantumD3Q19StreamingOperator
from qlbm.tools.utils import is_two_pow


class HybridVOF3DStep:
    """Coordinates a 3D hybrid classical-quantum step of the VOF-LBM algorithm.

    In this architecture:
    - Geometric 3D VOF state, cell state classification, 3D interface reconstruction,
      and free-surface boundary pressure conditions are computed classically.
    - 3D Discrete distribution streaming is delegated to quantum circuit simulation
      (:class:`.QuantumD3Q19StreamingOperator`).
    - Collision dynamics defaults to classical 3D BGK + Guo gravity body forcing.

    Parameters
    ----------
    solver : VOF3DHydroSolver
        The underlying 3D VOF hydro solver containing hydrodynamic and geometric fields.
    use_quantum_streaming : bool, optional
        Whether to execute streaming via quantum circuit simulation. Default True.
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        solver: VOF3DHydroSolver,
        use_quantum_streaming: bool = True,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        self.solver = solver
        self.use_quantum_streaming = use_quantum_streaming
        self.logger = logger

        self._quantum_streamer: Optional[QuantumD3Q19StreamingOperator] = None
        self._quantum_bounce: Optional[QuantumD3Q19BounceBackOperator] = None

        nx, ny, nz = self.solver.nx, self.solver.ny, self.solver.nz
        nx_is_pow2 = is_two_pow(nx)
        ny_is_pow2 = is_two_pow(ny)
        nz_is_pow2 = is_two_pow(nz)

        if self.use_quantum_streaming:
            if nx_is_pow2 and ny_is_pow2 and nz_is_pow2:
                self._quantum_streamer = QuantumD3Q19StreamingOperator(
                    nx=nx, ny=ny, nz=nz, logger=self.logger
                )
                self._quantum_bounce = QuantumD3Q19BounceBackOperator(logger=self.logger)
            else:
                self.logger.warning(
                    f"Grid dimensions ({nx}, {ny}, {nz}) are not powers of two. "
                    "Falling back to classical streaming in HybridVOF3DStep."
                )
                self.use_quantum_streaming = False

        # Phase profiling records
        self.step_count: int = 0
        self.last_step_profile: Dict[str, float] = {}
        self.timing_totals: Dict[str, float] = {
            "gravity_and_collision": 0.0,
            "quantum_state_prep": 0.0,
            "quantum_circuit_evolution": 0.0,
            "quantum_readout_decoding": 0.0,
            "boundary_handling": 0.0,
            "mass_flux_advection": 0.0,
            "free_surface_reconstruction": 0.0,
            "state_transitions": 0.0,
            "macroscopic_moments": 0.0,
            "total_step": 0.0,
        }

    def step(self) -> None:
        """Execute one complete time step of the 3D hybrid algorithm."""
        s = self.solver
        active = s.fluid_mask

        t_step_start = time.perf_counter()

        # -------------------------------------------------------------
        # Substep 1: Classical 3D BGK Collision + Guo Gravity Body Forcing
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        feq = compute_equilibrium(s.rho, s.u)
        f_post = np.copy(s.f)

        if np.any(active):
            body_force = s.rho * s.gravity[:, None, None, None]
            f_post[:, active] = bgk_collision(
                s.f[:, active],
                feq[:, active],
                s.tau,
                force=body_force[:, active],
                u=s.u[:, active],
                dt=s.dt,
            )

        f_post[:, ~active] = 0.0
        # Positivity filter for quantum amplitude encoding
        f_post = np.maximum(0.0, f_post)
        t_collision = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 2: Quantum 3D Streaming & Boundary Handling
        # -------------------------------------------------------------
        t_prep = 0.0
        t_evolve = 0.0
        t_read = 0.0

        if self.use_quantum_streaming and self._quantum_streamer is not None:
            f_streamed_raw, q_timing = self._quantum_streamer.stream_distribution_array(
                f_post, return_timing=True
            )
            t_prep = q_timing["t_state_prep"]
            t_evolve = q_timing["t_circuit_evolution"]
            t_read = q_timing["t_readout_decoding"]

            # Solid bounce-back reflection
            t0 = time.perf_counter()
            f_streamed = np.copy(f_streamed_raw)
            if np.any(s.solid_mask):
                for i in range(1, 19):
                    opp = OPPOSITE[i]
                    cx, cy, cz = C[i]
                    solid_nbr = np.roll(
                        np.roll(
                            np.roll(s.solid_mask, shift=-cz, axis=0),
                            shift=-cy,
                            axis=1,
                        ),
                        shift=-cx,
                        axis=2,
                    )
                    bounce_mask = (~s.solid_mask) & solid_nbr
                    f_streamed[opp, bounce_mask] = f_post[i, bounce_mask]
            t_boundary = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            f_streamed = stream_and_bounce(f_post, solid_mask=s.solid_mask)
            t_boundary = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 3: 3D Directional mass fluxes across interface links
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        delta_m = compute_directional_mass_fluxes_3d(
            f_post, s.fill_level, s.cell_types, solid_mask=s.solid_mask, dt=s.dt
        )
        t_mass_flux = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 4: 3D Free-surface gas boundary reconstruction
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        f_reconstructed = reconstruct_gas_boundaries_3d(
            f_streamed,
            f_post,
            s.cell_types,
            rho_gas=s.p_gas,
            u=s.u,
        )
        t_gas_recon = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 5: 3D Cell state transitions and volume fraction update
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        old_types = np.copy(s.cell_types)
        new_mass, new_fill, new_types = update_cell_states_3d(
            s.mass,
            s.rho,
            s.cell_types,
            s.solid_mask,
            delta_m,
            conversion_hysteresis=s.conversion_hysteresis,
        )

        # Initialize newly converted interface cells from gas with neighbor equilibrium
        new_iface = (old_types == CellType.GAS) & (new_types == CellType.INTERFACE)
        if np.any(new_iface):
            for z, y, x in zip(*np.where(new_iface)):
                u_sum = np.zeros(3, dtype=np.float64)
                cnt = 0
                for ci in range(1, 19):
                    cx, cy, cz = C[ci]
                    nz_c, ny_c, nx_c = z + cz, y + cy, x + cx
                    if 0 <= nz_c < s.nz and 0 <= ny_c < s.ny and 0 <= nx_c < s.nx:
                        if old_types[nz_c, ny_c, nx_c] in (CellType.LIQUID, CellType.INTERFACE):
                            u_sum += s.u[:, nz_c, ny_c, nx_c]
                            cnt += 1
                u_avg = u_sum / max(cnt, 1)
                u_mag = float(np.linalg.norm(u_avg))
                if u_mag > 0.15:
                    u_avg = u_avg * (0.15 / u_mag)
                f_reconstructed[:, z, y, x] = compute_equilibrium(1.0, u_avg)

        s.mass = new_mass
        s.fill_level = new_fill
        s.cell_types = new_types
        t_state_trans = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 6: Update distribution functions & macroscopic moments
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        s.f = f_reconstructed
        s.f[:, s.cell_types == CellType.GAS] = 0.0
        s.f[:, s.solid_mask] = 0.0

        body_force = s.rho * s.gravity[:, None, None, None]
        new_rho, new_u = compute_macroscopic(s.f, force=body_force, dt=s.dt)

        s.rho = new_rho
        s.u = new_u
        s.u[:, s.solid_mask] = 0.0
        s.u[:, s.cell_types == CellType.GAS] = 0.0

        s.normals, _ = compute_normals_3d(s.fill_level, s.solid_mask)
        t_moments = time.perf_counter() - t0

        s.time_step += 1
        t_step_total = time.perf_counter() - t_step_start

        # Record profile
        self.step_count += 1
        self.last_step_profile = {
            "gravity_and_collision": t_collision,
            "quantum_state_prep": t_prep,
            "quantum_circuit_evolution": t_evolve,
            "quantum_readout_decoding": t_read,
            "boundary_handling": t_boundary,
            "mass_flux_advection": t_mass_flux,
            "free_surface_reconstruction": t_gas_recon,
            "state_transitions": t_state_trans,
            "macroscopic_moments": t_moments,
            "total_step": t_step_total,
        }
        for k, v in self.last_step_profile.items():
            self.timing_totals[k] += v

    def report_profiling(self) -> str:
        """Generate a formatted profiling summary of cumulative time spent."""
        t_tot = self.timing_totals.get("total_step", 1e-9)
        lines = [
            f"=== 3D Hybrid VOF-LBM Execution Profile ({self.step_count} steps) ===",
            f"Total Step Time: {t_tot:.4f} s",
        ]
        for key, val in self.timing_totals.items():
            if key == "total_step":
                continue
            pct = (val / t_tot * 100.0) if t_tot > 0 else 0.0
            label = "[QUANTUM]" if "quantum" in key else "[CLASSICAL]"
            lines.append(f"  {label:<12} {key:<30}: {val:9.4f} s ({pct:5.1f}%)")
        return "\n".join(lines)
