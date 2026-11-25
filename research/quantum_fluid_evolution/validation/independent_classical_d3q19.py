"""Standalone Independent Classical D3Q19 Lattice Boltzmann Solver.

Completely decoupled from quantum components, classes, or wrappers.
Used for independent scientific verification of quantum fluid evolution.

Implements:
1. Pure classical spatial streaming via standard 3D array index rolling.
2. Exact solid-wall bounce-back reflection.
3. Dual collision models:
   - Linearized low-Mach BGK collision
   - Full non-linear Maxwell-Boltzmann BGK collision
4. Second-order Guo gravitational body forcing.
5. Macroscopic moment and momentum diagnostics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qlbm.classical.d3q19 import C, CS_SQ, OPPOSITE, WEIGHTS, compute_equilibrium, compute_guo_force, compute_macroscopic


class IndependentClassicalD3Q19Solver:
    """Standalone independent classical D3Q19 hydrodynamic solver.

    Parameters
    ----------
    nx : int
        Grid nodes in x.
    ny : int
        Grid nodes in y.
    nz : int
        Grid nodes in z.
    tau : float, optional
        BGK relaxation time, default 0.8.
    g : tuple or np.ndarray, optional
        Gravity vector in lattice units, default (0, 0, -0.01).
    dt : float, optional
        Time step, default 1.0.
    solid_mask : np.ndarray, optional
        Boolean obstacle array of shape `(nz, ny, nx)`.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        tau: float = 0.8,
        g: Tuple[float, float, float] = (0.0, 0.0, -0.01),
        dt: float = 1.0,
        solid_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.tau = float(tau)
        self.g = np.asarray(g, dtype=np.float64)
        self.dt = float(dt)
        self.omega = self.dt / self.tau

        if solid_mask is None:
            self.solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        else:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)
            if self.solid_mask.shape != (self.nz, self.ny, self.nx):
                raise ValueError(f"solid_mask shape {self.solid_mask.shape} != ({self.nz}, {self.ny}, {self.nx})")

        # Precompute 19x19 moment projector matrix E for linearized BGK
        self._E = np.zeros((19, 19), dtype=np.float64)
        for i in range(19):
            for j in range(19):
                cu = C[i, 0] * C[j, 0] + C[i, 1] * C[j, 1] + C[i, 2] * C[j, 2]
                self._E[i, j] = WEIGHTS[i] * (1.0 + cu / CS_SQ)

    def stream(self, f: np.ndarray) -> np.ndarray:
        """Stream populations along 19 discrete velocities via pure array shifts."""
        f_streamed = np.empty_like(f)
        for i in range(19):
            cx, cy, cz = C[i]
            # Spatial axis order: (z, y, x) -> axes (0, 1, 2)
            f_streamed[i] = np.roll(f[i], shift=(cz, cy, cx), axis=(0, 1, 2))
        return f_streamed

    def apply_boundaries(self, f: np.ndarray) -> np.ndarray:
        """Apply no-slip bounce-back reflection on solid nodes."""
        f_out = f.copy()
        if np.any(self.solid_mask):
            for i in range(19):
                f_out[OPPOSITE[i], self.solid_mask] = f[i, self.solid_mask]
        return f_out

    def collide_linearized(self, f: np.ndarray) -> np.ndarray:
        """Execute linearized BGK collision with Guo gravity body forcing."""
        orig_shape = f.shape
        f_2d = f.reshape(19, -1)

        # Linearized collision
        Clin = (1.0 - self.omega) * np.eye(19, dtype=np.float64) + self.omega * self._E

        # Guo gravity forcing matrix
        prefactor = (1.0 - 0.5 * self.dt / self.tau) * self.dt
        G = np.zeros((19, 19), dtype=np.float64)
        for i in range(19):
            cg = C[i, 0] * self.g[0] + C[i, 1] * self.g[1] + C[i, 2] * self.g[2]
            G[i, :] = WEIGHTS[i] * prefactor * (cg / CS_SQ)

        M_cg = Clin + G
        f_post_2d = M_cg @ f_2d
        return f_post_2d.reshape(orig_shape)

    def collide_nonlinear(self, f: np.ndarray) -> np.ndarray:
        """Execute full non-linear Maxwell-Boltzmann BGK collision with Guo forcing."""
        # Compute macroscopic density and velocity
        rho, u = compute_macroscopic(f, force=self.g, dt=self.dt)

        # Full non-linear equilibrium
        feq = compute_equilibrium(rho, u)

        # BGK relaxation
        f_post = (1.0 - self.omega) * f + self.omega * feq

        # Second-order Guo forcing
        if np.any(self.g != 0.0):
            Fi = compute_guo_force(u, self.g, self.tau, dt=self.dt)
            f_post += self.dt * Fi

        return f_post

    def step(self, f: np.ndarray, linearized: bool = True) -> np.ndarray:
        """Execute a single classical LBM step: Collide -> Stream -> Bounce."""
        if linearized:
            f_coll = self.collide_linearized(f)
        else:
            f_coll = self.collide_nonlinear(f)

        f_stream = self.stream(f_coll)
        f_next = self.apply_boundaries(f_stream)
        return f_next

    def run(
        self,
        f_initial: np.ndarray,
        num_steps: int = 4,
        linearized: bool = True,
    ) -> Dict[str, Any]:
        """Run multiple classical simulation steps."""
        f = f_initial.copy()
        history = []
        t0 = time.perf_counter()

        for s in range(1, num_steps + 1):
            t_start = time.perf_counter()
            f = self.step(f, linearized=linearized)
            t_step = time.perf_counter() - t_start

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
            "linearized": linearized,
        }
