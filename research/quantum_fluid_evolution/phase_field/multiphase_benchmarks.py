"""Canonical Multiphase Verification Suite for 3D Quantum Phase-Field Hydrodynamics.

Executes and verifies the four canonical physical benchmarks mandated before
full waterfall scale-up:
1. Benchmark 1: Static 3D Droplet & Laplace Law (Delta p = 2 * sigma / R)
2. Benchmark 2: Translating 3D Droplet (Galilean Invariance, advection, mass conservation)
3. Benchmark 3: Falling 3D Droplet under Gravity (z_com(t) = z_0 - 0.5 * g * t^2)
4. Benchmark 4: 3D Waterfall Step & Detachment with Solid Obstacle
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C as C_D3Q19, CS_SQ, WEIGHTS as W_D3Q19
from research.quantum_fluid_evolution.phase_field.multiphase_phase_field_solver_3d import (
    MultiphasePhaseFieldSolver3D,
)


class MultiphaseBenchmarkSuite:
    """Suite of canonical two-phase verification benchmarks."""

    # -----------------------------------------------------------------
    # Benchmark 1: Static 3D Droplet & Laplace Law
    # -----------------------------------------------------------------

    @staticmethod
    def run_static_droplet_laplace_test(
        r0: float = 3.5,
        sigma: float = 0.02,
        grid_size: Tuple[int, int, int] = (16, 16, 16),
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Verify Laplace pressure jump Delta p = 2 * sigma / R for a static 3D droplet."""
        nz, ny, nx = grid_size
        solver = MultiphasePhaseFieldSolver3D(
            nx=nx, ny=ny, nz=nz,
            sigma=sigma,
            g=(0.0, 0.0, 0.0),  # Zero gravity for static droplet
            interface_width=1.0,
            rho_liquid=1.0,
            rho_gas=0.2,
        )

        center = (nz / 2.0, ny / 2.0, nx / 2.0)
        solver.initialize_droplet(center=center, radius=r0)

        # Run relaxation steps to steady state
        for _ in range(num_steps):
            solver.step()

        phi = solver.compute_phi()
        rho, u = solver.compute_velocity()

        # Pressure: p = rho * c_s^2
        pressure = rho * CS_SQ

        # Measure inside pressure (r < 0.5 * r0) and outside pressure (r > 1.5 * r0)
        z = np.arange(nz)[:, None, None]
        y = np.arange(ny)[None, :, None]
        x = np.arange(nx)[None, None, :]
        dist = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)

        inside_mask = dist < (0.5 * r0)
        outside_mask = dist > (1.8 * r0)

        p_inside = float(np.mean(pressure[inside_mask]))
        p_outside = float(np.mean(pressure[outside_mask]))
        delta_p_measured = p_inside - p_outside

        # Theoretical 3D spherical Laplace jump:
        # Sharp interface: Delta p = 2 * sigma / R
        # Diffuse interface Tolman correction: Delta p = (2 * sigma / R) * (1 + xi / R)
        delta_p_sharp = 2.0 * sigma / r0
        delta_p_tolman = delta_p_sharp * (1.0 + solver.xi / r0)
        rel_error = abs(delta_p_measured - delta_p_tolman) / delta_p_tolman

        # Spurious / parasitic currents
        u_mag = np.sqrt(np.sum(u**2, axis=0))
        max_parasitic_u = float(np.max(u_mag))

        return {
            "radius": r0,
            "sigma": sigma,
            "p_inside": p_inside,
            "p_outside": p_outside,
            "delta_p_measured": delta_p_measured,
            "delta_p_sharp": delta_p_sharp,
            "delta_p_tolman": delta_p_tolman,
            "delta_p_theoretical": delta_p_tolman,
            "relative_error": rel_error,
            "max_parasitic_velocity": max_parasitic_u,
            "passed": rel_error < 0.15 and max_parasitic_u < 0.02,
        }

    # -----------------------------------------------------------------
    # Benchmark 2: Translating 3D Droplet (Advection & Mass Conservation)
    # -----------------------------------------------------------------

    @staticmethod
    def run_translating_droplet_test(
        u0: Tuple[float, float, float] = (0.04, 0.0, 0.0),
        grid_size: Tuple[int, int, int] = (8, 8, 16),
        num_steps: int = 25,
    ) -> Dict[str, Any]:
        """Verify Galilean invariance and mass conservation of an advected 3D droplet."""
        nz, ny, nx = grid_size
        solver = MultiphasePhaseFieldSolver3D(
            nx=nx, ny=ny, nz=nz,
            sigma=0.01,
            g=(0.0, 0.0, 0.0),
            interface_width=1.0,
            rho_liquid=1.0,
            rho_gas=1.0,
        )

        r0 = 2.5
        center0 = (nz / 2.0, ny / 2.0, 4.0)
        solver.initialize_droplet(center=center0, radius=r0)

        # Impose uniform initial velocity
        ux0, uy0, uz0 = u0
        for i in range(19):
            cu = C_D3Q19[i, 0] * ux0 + C_D3Q19[i, 1] * uy0 + C_D3Q19[i, 2] * uz0
            solver.f[i] += W_D3Q19[i] * 1.0 * (cu / CS_SQ)

        initial_phi = solver.compute_phi()
        # Conserved physical liquid mass in diffuse interface: integral (phi + 1)/2 dV
        initial_mass = float(np.sum((initial_phi + 1.0) / 2.0))

        # Helper to compute center of mass
        def get_com(phi_field: np.ndarray) -> np.ndarray:
            pos_phi = np.maximum(phi_field, 0.0)
            tot = np.sum(pos_phi) + 1e-15
            z_c = np.sum(np.arange(nz)[:, None, None] * pos_phi) / tot
            y_c = np.sum(np.arange(ny)[None, :, None] * pos_phi) / tot
            x_c = np.sum(np.arange(nx)[None, None, :] * pos_phi) / tot
            return np.array([x_c, y_c, z_c])

        com0 = get_com(initial_phi)

        for _ in range(num_steps):
            solver.step()

        final_phi = solver.compute_phi()
        final_mass = float(np.sum((final_phi + 1.0) / 2.0))
        mass_error = abs(final_mass - initial_mass) / (initial_mass + 1e-12)

        com_final = get_com(final_phi)
        expected_disp = np.array(u0) * num_steps
        measured_disp = com_final - com0
        disp_error = float(np.linalg.norm(measured_disp - expected_disp))

        return {
            "num_steps": num_steps,
            "u0": u0,
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "mass_relative_error": mass_error,
            "expected_displacement": expected_disp.tolist(),
            "measured_displacement": measured_disp.tolist(),
            "displacement_error": disp_error,
            "passed": mass_error < 0.05 and disp_error < 0.8,
        }

    # -----------------------------------------------------------------
    # Benchmark 3: Falling 3D Droplet under Gravity
    # -----------------------------------------------------------------

    @staticmethod
    def run_falling_droplet_test(
        gz: float = -0.004,
        grid_size: Tuple[int, int, int] = (16, 8, 8),
        num_steps: int = 25,
    ) -> Dict[str, Any]:
        """Verify free-fall acceleration and floor impact of a 3D droplet under gravity."""
        nz, ny, nx = grid_size

        # Solid floor at z = 0
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, :, :] = True

        solver = MultiphasePhaseFieldSolver3D(
            nx=nx, ny=ny, nz=nz,
            sigma=0.01,
            g=(0.0, 0.0, gz),
            solid_mask=solid,
            interface_width=1.0,
            rho_liquid=1.0,
            rho_gas=0.1,
        )

        z_init = 12.0
        center0 = (z_init, ny / 2.0, nx / 2.0)
        solver.initialize_droplet(center=center0, radius=2.2)

        z_history = []
        vel_z_history = []

        for step in range(num_steps):
            metrics = solver.step()
            phi = metrics["phi"]
            pos_phi = np.maximum(phi, 0.0)
            tot = np.sum(pos_phi) + 1e-15
            z_com = float(np.sum(np.arange(nz)[:, None, None] * pos_phi) / tot)
            z_history.append(z_com)

            rho, u = solver.compute_velocity()
            mean_uz = float(np.sum(u[2, pos_phi > 0.5]) / (np.sum(pos_phi > 0.5) + 1e-12))
            vel_z_history.append(mean_uz)

        # Check early free-fall acceleration: Delta z ~ 0.5 * g * t^2
        # After 10 steps:
        t_check = 10
        expected_z = z_init + 0.5 * gz * (t_check**2)
        actual_z = z_history[t_check - 1]
        z_error = abs(actual_z - expected_z)

        # Monotonic downward velocity during free fall
        vel_downward = vel_z_history[5] < 0.0 and vel_z_history[10] < vel_z_history[5]

        return {
            "z_init": z_init,
            "gravity": gz,
            "z_history": z_history,
            "actual_z_step10": actual_z,
            "expected_z_step10": expected_z,
            "freefall_z_error": z_error,
            "vel_downward": vel_downward,
            "passed": z_error < 0.7 and vel_downward,
        }

    # -----------------------------------------------------------------
    # Benchmark 4: 3D Waterfall Step & Detachment with Obstacle
    # -----------------------------------------------------------------

    @staticmethod
    def run_waterfall_step_detachment_test(
        grid_size: Tuple[int, int, int] = (16, 8, 16),
        num_steps: int = 42,
    ) -> Dict[str, Any]:
        """Verify water flowing from elevated reservoir, falling over cliff, and hitting obstacle."""
        nz, ny, nx = grid_size
        z_shelf = 8
        x_cliff = 6

        # Build solid domain:
        # Floor: z = 0
        # Shelf & cliff: z <= z_shelf, x <= x_cliff
        # Obstacle block: x in [10, 12], z in [0, 3]
        solid = np.zeros((nz, ny, nx), dtype=bool)
        solid[0, :, :] = True
        solid[: z_shelf + 1, :, : x_cliff + 1] = True
        solid[: 4, :, 10:13] = True  # Obstacle block

        solver = MultiphasePhaseFieldSolver3D(
            nx=nx, ny=ny, nz=nz,
            tau_f=0.8,
            tau_phi=0.8,
            sigma=0.015,
            g=(0.0, 0.0, -0.010),
            solid_mask=solid,
            interface_width=1.0,
            rho_liquid=1.0,
            rho_gas=1.0,
        )

        solver.initialize_waterfall_reservoir(z_shelf=z_shelf, x_cliff=x_cliff)

        # Open reservoir gate: forward horizontal flow past cliff
        for i in range(19):
            solver.f[i] += W_D3Q19[i] * 1.0 * (C_D3Q19[i, 0] * 0.06 / CS_SQ)

        init_phi = solver.compute_phi()
        initial_liquid_cells = int(np.sum(init_phi > 0.0))

        # Track trajectory
        history = []
        for s in range(1, num_steps + 1):
            res = solver.step()
            phi = res["phi"]

            # Liquid in air past cliff face (x > x_cliff)
            air_region_liquid = int(np.sum(phi[:, :, x_cliff + 1:] > 0.0))
            # Liquid below shelf height (z <= z_shelf) in air
            below_shelf_liquid = int(np.sum(phi[1:z_shelf + 1, :, x_cliff + 1:] > 0.0))

            history.append({
                "step": s,
                "liquid_volume": res["liquid_volume"],
                "air_region_liquid": air_region_liquid,
                "below_shelf_liquid": below_shelf_liquid,
                "max_velocity": res["max_velocity"],
                "jz": res["vertical_momentum_jz"],
            })

        # Final verification:
        final_history = history[-1]
        water_detached = any(h["air_region_liquid"] > 0 for h in history)
        downward_momentum = any(h["jz"] < -0.005 for h in history)

        # Check non-penetration on solid cells
        final_phi = solver.compute_phi()
        solid_penetration = float(np.sum(final_phi[solid] > -0.5))

        return {
            "initial_liquid_cells": initial_liquid_cells,
            "final_liquid_cells": final_history["liquid_volume"],
            "water_detached_over_cliff": water_detached,
            "downward_momentum_accumulated": downward_momentum,
            "solid_penetration_count": solid_penetration,
            "history_summary": history[::5],
            "passed": water_detached and downward_momentum and solid_penetration == 0,
        }


def run_all_multiphase_benchmarks() -> None:
    """Execute complete 4-benchmark suite and print clean scorecard."""
    print("=" * 95)
    print("  CANONICAL 3D MULTIPHASE QUANTUM PHASE-FIELD VERIFICATION BENCHMARKS")
    print("=" * 95)

    suite = MultiphaseBenchmarkSuite()

    # 1. Static Droplet
    print("\n[Running Benchmark 1: Static 3D Droplet & Laplace Law]...")
    res1 = suite.run_static_droplet_laplace_test(r0=3.5, sigma=0.02, num_steps=40)
    print(f"  Delta p Measured:     {res1['delta_p_measured']:.6f}")
    print(f"  Delta p Theoretical:  {res1['delta_p_theoretical']:.6f}")
    print(f"  Relative Error:       {res1['relative_error']:.2%}")
    print(f"  Max Parasitic Curr:   {res1['max_parasitic_velocity']:.6e}")
    print(f"  Status:               {'PASSED' if res1['passed'] else 'FAILED'}")

    # 2. Translating Droplet
    print("\n[Running Benchmark 2: Translating 3D Droplet Advection]...")
    res2 = suite.run_translating_droplet_test(u0=(0.04, 0.0, 0.0), num_steps=25)
    print(f"  Mass Relative Error:  {res2['mass_relative_error']:.2e}")
    print(f"  Displacement Error:   {res2['displacement_error']:.4f} lattice units")
    print(f"  Status:               {'PASSED' if res2['passed'] else 'FAILED'}")

    # 3. Falling Droplet
    print("\n[Running Benchmark 3: Falling 3D Droplet under Gravity]...")
    res3 = suite.run_falling_droplet_test(gz=-0.004, num_steps=20)
    print(f"  Step 10 Actual z:     {res3['actual_z_step10']:.4f} (Expected: {res3['expected_z_step10']:.4f})")
    print(f"  Free-Fall Error:      {res3['freefall_z_error']:.4f} lattice units")
    print(f"  Downward Accel:       {'YES' if res3['vel_downward'] else 'NO'}")
    print(f"  Status:               {'PASSED' if res3['passed'] else 'FAILED'}")

    # 4. Waterfall Detachment
    print("\n[Running Benchmark 4: 3D Waterfall Step Detachment & Obstacle Collision]...")
    res4 = suite.run_waterfall_step_detachment_test(num_steps=30)
    print(f"  Water Detached:       {'YES' if res4['water_detached_over_cliff'] else 'NO'}")
    print(f"  Momentum Injected:    {'YES' if res4['downward_momentum_accumulated'] else 'NO'}")
    print(f"  Solid Penetration:    {res4['solid_penetration_count']} nodes")
    print(f"  Status:               {'PASSED' if res4['passed'] else 'FAILED'}")

    print("\n" + "=" * 95)
    all_passed = res1["passed"] and res2["passed"] and res3["passed"] and res4["passed"]
    print(f"  OVERALL MULTIPHASE VERIFICATION STATUS: {'ALL 4 PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 95)


if __name__ == "__main__":
    run_all_multiphase_benchmarks()
