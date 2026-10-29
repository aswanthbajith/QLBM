"""Modular Hybrid Classical-Quantum Stepper for Free-Surface VOF-LBM."""

from __future__ import annotations

import time
from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np

from qlbm.classical.d2q9 import (
    C,
    CS_SQ,
    OPPOSITE,
    bgk_collision,
    compute_equilibrium,
    compute_macroscopic,
)
from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver
from qlbm.classical.vof import CellType
from qlbm.components.vof.base import QuantumCollisionOperator
from qlbm.components.vof.boundary import QuantumD2Q9BounceBackOperator
from qlbm.components.vof.streaming import QuantumD2Q9StreamingOperator


class HybridVOFStep:
    """Coordinates a hybrid classical-quantum step of the VOF-LBM algorithm (Option A).

    In this architecture:
    - Geometric VOF state, cell state classification, PLIC interface reconstruction,
      and free-surface boundary pressure conditions are computed classically.
    - Discrete distribution streaming is delegated to quantum circuit simulation
      (:class:`.QuantumD2Q9StreamingOperator`).
    - Collision dynamics defaults to classical BGK + Guo forcing, with an abstract
      hook for research quantum collision approximations.

    Parameters
    ----------
    solver : VOFHydroSolver
        The underlying VOFHydroSolver containing hydrodynamic and geometric fields.
    use_quantum_streaming : bool, optional
        Whether to execute streaming via quantum circuit simulation. Default True.
    quantum_collision_op : QuantumCollisionOperator, optional
        Optional candidate quantum collision operator.
    logger : Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        solver: VOFHydroSolver,
        use_quantum_streaming: bool = True,
        quantum_collision_op: Optional[QuantumCollisionOperator] = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        self.solver = solver
        self.use_quantum_streaming = use_quantum_streaming
        self.quantum_collision_op = quantum_collision_op
        self.logger = logger

        # Lazy initialization of quantum streaming operator if dimensions are powers of 2
        self._quantum_streamer: Optional[QuantumD2Q9StreamingOperator] = None
        self._quantum_bounce: Optional[QuantumD2Q9BounceBackOperator] = None

        nx, ny = self.solver.nx, self.solver.ny
        nx_is_pow2 = (nx > 0) and ((nx & (nx - 1)) == 0)
        ny_is_pow2 = (ny > 0) and ((ny & (ny - 1)) == 0)

        if self.use_quantum_streaming:
            if nx_is_pow2 and ny_is_pow2:
                self._quantum_streamer = QuantumD2Q9StreamingOperator(
                    nx=nx, ny=ny, logger=self.logger
                )
                self._quantum_bounce = QuantumD2Q9BounceBackOperator(logger=self.logger)
            else:
                self.logger.warning(
                    f"Grid dimensions ({nx}, {ny}) are not powers of two. "
                    "Falling back to classical streaming in HybridVOFStep."
                )
                self.use_quantum_streaming = False

        # Phase profiling records
        self.step_count: int = 0
        self.last_step_profile: Dict[str, float] = {}
        self.timing_totals: Dict[str, float] = {
            "gravity": 0.0,
            "collision": 0.0,
            "quantum_state_prep": 0.0,
            "quantum_circuit_evolution": 0.0,
            "quantum_readout_decoding": 0.0,
            "boundary_handling": 0.0,
            "free_surface_reconstruction": 0.0,
            "mass_flux_advection": 0.0,
            "state_transitions": 0.0,
            "macroscopic_moments": 0.0,
            "geometry_reconstruction": 0.0,
            "total_step": 0.0,
        }

    def step(self) -> None:
        """Execute one complete time step of the hybrid algorithm."""
        s = self.solver
        active = s.fluid_mask

        t_step_start = time.perf_counter()

        # -------------------------------------------------------------
        # Substep 1: Gravity body force
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        s._compute_body_force(active)
        t_gravity = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 2: Collision step on active cells
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if self.quantum_collision_op is not None and hasattr(
            self.quantum_collision_op, "collide_distribution_array"
        ):
            f_post = self.quantum_collision_op.collide_distribution_array(
                s.f, active_mask=active, tau=s.tau
            )
            if s.gravity != (0.0, 0.0):
                from qlbm.classical.d2q9 import compute_guo_force

                F_i = compute_guo_force(s.u, s.force, s.tau, dt=s.dt)
                f_post[:, active] += s.dt * F_i[:, active]
        else:
            f_eq = compute_equilibrium(s.rho, s.u)
            f_post = bgk_collision(
                s.f,
                f_eq,
                s.tau,
                force=s.force,
                u=s.u,
                dt=s.dt,
            )
        f_post[:, ~active] = 0.0
        t_collision = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 3: Quantum Streaming & Boundary Handling
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

            # Apply solid bounce-back:
            t0 = time.perf_counter()
            f_streamed = np.copy(f_streamed_raw)
            if np.any(s.solid_mask):
                for i in range(1, 9):
                    opp = OPPOSITE[i]
                    cx, cy = C[i]
                    solid_nbr = np.roll(
                        np.roll(s.solid_mask, shift=-cy, axis=0),
                        shift=-cx,
                        axis=1,
                    )
                    bounce_mask = (~s.solid_mask) & solid_nbr
                    f_streamed[opp, bounce_mask] = f_post[i, bounce_mask]
            t_boundary = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            from qlbm.classical.d2q9 import stream_and_bounce
            f_streamed = stream_and_bounce(f_post, solid_mask=s.solid_mask)
            t_boundary = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 4: Free-surface boundary reconstruction (Körner et al.)
        # Substeps 4-7 delegate to the solver so that the classical and
        # hybrid paths share one implementation of the free-surface scheme.
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        s._reconstruct_gas_populations(f_streamed, f_post)
        t_gas_recon = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 5: Pairwise Anti-Symmetric Mass Exchange
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        s._exchange_mass(f_post)
        t_mass_flux = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 6: State transitions and initialization
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        s._handle_state_transitions(f_streamed)
        t_state_trans = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 7: Macroscopic moments update
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        s._update_macroscopic()
        t_moments = time.perf_counter() - t0

        # -------------------------------------------------------------
        # Substep 8: Reconstruct geometry
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        s.reconstruct_geometry()
        t_geometry = time.perf_counter() - t0

        s.time_step += 1
        t_step_total = time.perf_counter() - t_step_start

        # Record profile
        self.step_count += 1
        self.last_step_profile = {
            "gravity": t_gravity,
            "collision": t_collision,
            "quantum_state_prep": t_prep,
            "quantum_circuit_evolution": t_evolve,
            "quantum_readout_decoding": t_read,
            "boundary_handling": t_boundary,
            "free_surface_reconstruction": t_gas_recon,
            "mass_flux_advection": t_mass_flux,
            "state_transitions": t_state_trans,
            "macroscopic_moments": t_moments,
            "geometry_reconstruction": t_geometry,
            "total_step": t_step_total,
        }
        for k, v in self.last_step_profile.items():
            self.timing_totals[k] += v

    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Return timing metrics aggregated over all completed steps."""
        if self.step_count == 0:
            return {}
        total = self.timing_totals["total_step"]
        summary = {}
        for phase, total_time in self.timing_totals.items():
            summary[phase] = {
                "total_seconds": total_time,
                "mean_seconds_per_step": total_time / self.step_count,
                "percentage": (total_time / max(total, 1e-12)) * 100.0,
            }
        return summary
