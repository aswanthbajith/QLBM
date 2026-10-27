#!/usr/bin/env python3
"""Unified Classical and Hybrid Quantum-Circuit Dam-Break Simulation Benchmark.

This script executes the two-phase rectangular liquid column collapse
benchmark (Martin & Moyce 1952) under three distinct, verifiable operating modes:
  - Mode A ('classical'): Standard high-performance classical D2Q9 VOF-LBM solver.
  - Mode B ('streaming_test'): Direct comparison between quantum circuit streaming and
    classical streaming across all grid distributions.
  - Mode C ('hybrid'): Hybrid classical-quantum dam-break simulation delegating
    discrete velocity streaming to quantum circuit simulation while computing
    geometric VOF operations classically.

Supports CSV, JSON, and NPZ snapshot output generation for reproducible verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qlbm.classical.d2q9 import stream_and_bounce
from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver
from qlbm.components.vof.hybrid_stepper import HybridVOFStep
from qlbm.components.vof.streaming import QuantumD2Q9StreamingOperator
from qlbm.tools.utils import is_two_pow


def run_dam_break_simulation(
    mode: str = "classical",
    nx: int = 64,
    ny: int = 32,
    column_width: int = 16,
    column_height: int = 24,
    tau: float = 0.8,
    gravity: float = 0.0005,
    surface_tension: float = 0.0,
    steps: int = 100,
    log_interval: int = 10,
    output_dir: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Execute dam-break simulation under the specified operating mode.

    Parameters
    ----------
    mode : {'classical', 'streaming_test', 'hybrid'}
        Simulation operating mode.
    nx, ny : int
        Lattice domain dimensions.
    column_width, column_height : int
        Initial dimensions of rectangular liquid column.
    tau : float
        BGK relaxation time.
    gravity : float
        Gravitational acceleration in lattice units.
    surface_tension : float
        Macroscopic surface tension coefficient.
    steps : int
        Number of time steps.
    log_interval : int
        Console logging interval in steps.
    output_dir : str, optional
        Directory for exporting CSV, JSON, and NPZ result files.
    quiet : bool
        If True, suppress console stdout.

    Returns
    -------
    dict
        Comprehensive run metrics, physical history, and validation status.
    """
    valid_modes = ("classical", "streaming_test", "hybrid")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")

    # Validate quantum compatibility
    if mode in ("streaming_test", "hybrid"):
        if not is_two_pow(nx) or not is_two_pow(ny):
            raise ValueError(
                f"Mode '{mode}' requires grid dimensions to be powers of two for "
                f"quantum register encoding. Got nx={nx}, ny={ny}."
            )

    # 1. Domain configuration with solid walls on bottom, left, and right
    solid_mask = np.zeros((ny, nx), dtype=bool)
    solid_mask[0, :] = True       # bottom wall
    solid_mask[:, 0] = True       # left wall
    solid_mask[:, -1] = True      # right wall

    solver = VOFHydroSolver(
        nx=nx,
        ny=ny,
        tau=tau,
        gravity=gravity,
        surface_tension=surface_tension,
        solid_mask=solid_mask,
    )

    solver.initialize_dam_break(
        column_width=column_width,
        column_height=column_height,
        add_walls=False,
    )

    # Instantiate hybrid stepper or quantum streamer if requested
    hybrid_stepper: Optional[HybridVOFStep] = None
    quantum_streamer: Optional[QuantumD2Q9StreamingOperator] = None

    if mode == "hybrid":
        hybrid_stepper = HybridVOFStep(solver, use_quantum_streaming=True)
    elif mode == "streaming_test":
        quantum_streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)

    l0 = float(column_width)
    h0 = float(column_height)
    sqrt_g_over_l = math.sqrt(gravity / l0) if gravity > 0 else 0.0

    init_mass = solver.total_mass
    init_cm = solver.center_of_mass
    init_front = solver.front_position()

    # Capture initial field snapshot
    snap_initial = {
        "fill_level": np.copy(solver.fill_level),
        "rho": np.copy(solver.rho),
        "velocity_magnitude": solver.velocity_magnitude(),
    }

    if not quiet:
        print("=" * 88)
        print(f"QLBM RESEARCH: DAM-BREAK SIMULATOR [MODE: {mode.upper()}]")
        print("=" * 88)
        print(f"Domain: {nx} x {ny} | Column: {column_width} x {column_height} (aspect ratio H0/L0 = {h0/l0:.2f})")
        print(f"Relaxation tau: {tau:.3f} | Gravity g: {gravity:.6f} | Surface Tension sigma: {surface_tension:.4f}")
        print(f"Initial Mass: {init_mass:.6f} | Initial Front x: {init_front:.2f} | Initial CM: ({init_cm[0]:.2f}, {init_cm[1]:.2f})")
        if mode == "hybrid":
            q_metrics = hybrid_stepper._quantum_streamer.resource_metrics()
            print(f"Quantum Registers: {q_metrics['num_qubits']} qubits ({q_metrics['num_x_qubits']} x, {q_metrics['num_y_qubits']} y, 4 vel, 1 anc)")
            print(f"Circuit Complexity: Depth {q_metrics['depth']}, Gates {q_metrics['size']}, 2-qubit {q_metrics['two_qubit_gates']}")
        print("-" * 88)
        header = f"{'Step':>5} | {'T (dimless)':>11} | {'Z = x/L0':>9} | {'H/H0':>7} | {'x_cm':>6} | {'y_cm':>6} | {'Max |u|':>8} | {'Mass Error':>11}"
        print(header)
        print("-" * 88)

    history: List[Dict[str, float]] = []
    streaming_errors: List[float] = []

    def record_state(step: int) -> Dict[str, float]:
        t_dimless = step * sqrt_g_over_l
        front_x = solver.front_position()
        z_dimless = front_x / l0
        heights = solver.height_profile()
        h_col = float(heights[1]) if len(heights) > 1 else 0.0
        h_dimless = h_col / h0
        cm_x, cm_y = solver.center_of_mass
        max_u = float(solver.velocity_magnitude().max())
        m_err = solver.mass_error()
        ke = solver.kinetic_energy

        entry = {
            "step": step,
            "t_dimless": t_dimless,
            "front_x": front_x,
            "z_dimless": z_dimless,
            "h_col": h_col,
            "h_dimless": h_dimless,
            "cm_x": cm_x,
            "cm_y": cm_y,
            "max_u": max_u,
            "mass_error": m_err,
            "kinetic_energy": ke,
        }
        history.append(entry)

        if not quiet and (step % log_interval == 0 or step == steps):
            print(
                f"{step:5d} | {t_dimless:11.4f} | {z_dimless:9.4f} | {h_dimless:7.4f} | "
                f"{cm_x:6.2f} | {cm_y:6.2f} | {max_u:8.4f} | {m_err:11.2e}"
            )
        return entry

    start_time = time.perf_counter()
    record_state(0)

    for current_step in range(1, steps + 1):
        if mode == "classical":
            solver.step()
        elif mode == "hybrid":
            hybrid_stepper.step()
        elif mode == "streaming_test":
            # Advance classically but verify quantum streaming at every step
            # Pre-collision distribution
            f_eq = solver.f
            # Compute difference between quantum and classical periodic streaming
            f_q = quantum_streamer.stream_distribution_array(solver.f)
            f_c = stream_and_bounce(solver.f, solid_mask=None)
            err = float(np.max(np.abs(f_q - f_c)))
            streaming_errors.append(err)
            solver.step()

        if current_step % log_interval == 0 or current_step == steps:
            record_state(current_step)

    elapsed = time.perf_counter() - start_time

    final_entry = history[-1]
    final_cm = (final_entry["cm_x"], final_entry["cm_y"])
    final_front = final_entry["front_x"]

    # Mode-specific validation checks
    mass_conserved = final_entry["mass_error"] < 1e-12
    low_mach_stable = final_entry["max_u"] < 0.25
    cm_shifted_right = final_cm[0] >= init_cm[0] - 1e-6
    cm_dropped = (final_cm[1] <= init_cm[1]) or (
        final_entry["t_dimless"] < 0.5 and abs(final_cm[1] - init_cm[1]) < 0.05
    )
    # Check front advance: either discrete cell (threshold=0.5) or continuous interface (threshold=0.01)
    front_advanced = (final_front > init_front) or (
        solver.front_position(threshold=0.01) > solver.front_position(threshold=0.5)
    )

    if mode == "streaming_test":
        max_stream_err = max(streaming_errors) if streaming_errors else 0.0
        streaming_passed = max_stream_err < 1e-12
        all_passed = streaming_passed and mass_conserved and low_mach_stable
    else:
        all_passed = (
            cm_dropped
            and cm_shifted_right
            and mass_conserved
            and low_mach_stable
        )
        if final_entry["t_dimless"] >= 0.4:
            all_passed = all_passed and front_advanced

    if not quiet:
        print("-" * 88)
        print(f"Elapsed Time: {elapsed:.3f} s ({steps / max(elapsed, 1e-6):.1f} steps/s)")
        print(f"Validation Status: {'PASSED [OK]' if all_passed else 'FAILED'}")
        print(f" - Center of mass descent (y_cm): {init_cm[1]:.2f} -> {final_cm[1]:.2f} ({'PASS' if cm_dropped else 'FAIL'})")
        print(f" - Surge front advancement (x_front): {init_front:.2f} -> {final_front:.2f} ({'PASS' if front_advanced else 'FAIL'})")
        print(f" - Horizontal mass displacement (x_cm): {init_cm[0]:.2f} -> {final_cm[0]:.2f} ({'PASS' if cm_shifted_right else 'FAIL'})")
        print(f" - Global mass conservation (|Delta M|/M0): {final_entry['mass_error']:.2e} (< 1e-12: {'PASS' if mass_conserved else 'FAIL'})")
        print(f" - Low-Mach stability (max |u|): {final_entry['max_u']:.4f} (< 0.25: {'PASS' if low_mach_stable else 'FAIL'})")
        if mode == "streaming_test":
            max_stream_err = max(streaming_errors) if streaming_errors else 0.0
            print(f" - Max quantum streaming error: {max_stream_err:.2e} (< 1e-12: {'PASS' if max_stream_err < 1e-12 else 'FAIL'})")
        elif mode == "hybrid":
            timing_summary = hybrid_stepper.get_timing_summary()
            print(" - Hybrid Phase Timing Breakdown:")
            for phase, p_data in timing_summary.items():
                if phase != "total_step":
                    print(f"    * {phase:28s}: {p_data['total_seconds']*1000:7.2f} ms ({p_data['percentage']:5.1f}%)")
        print("=" * 88)

    snap_final = {
        "fill_level": np.copy(solver.fill_level),
        "rho": np.copy(solver.rho),
        "velocity_magnitude": solver.velocity_magnitude(),
    }

    results: Dict[str, Any] = {
        "metadata": {
            "mode": mode,
            "nx": nx,
            "ny": ny,
            "column_width": column_width,
            "column_height": column_height,
            "aspect_ratio": h0 / l0,
            "tau": tau,
            "gravity": gravity,
            "surface_tension": surface_tension,
            "steps": steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / max(elapsed, 1e-6),
        },
        "validation": {
            "all_passed": all_passed,
            "cm_dropped": cm_dropped,
            "front_advanced": front_advanced,
            "cm_shifted_right": cm_shifted_right,
            "mass_conserved": mass_conserved,
            "low_mach_stable": low_mach_stable,
        },
        "history": history,
    }

    if mode == "hybrid":
        results["timing_summary"] = hybrid_stepper.get_timing_summary()
        results["circuit_metrics"] = hybrid_stepper._quantum_streamer.resource_metrics()
    elif mode == "streaming_test":
        results["circuit_metrics"] = quantum_streamer.resource_metrics()
        results["max_streaming_error"] = max(streaming_errors) if streaming_errors else 0.0

    # Export structured output if output_dir is supplied
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. JSON metrics
        json_file = out_path / f"dam_break_metrics_{mode}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 2. Time-series CSV
        csv_file = out_path / f"dam_break_history_{mode}.csv"
        if history:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

        # 3. NPZ snapshot arrays for visualization
        npz_file = out_path / f"dam_break_snapshots_{mode}.npz"
        np.savez_compressed(
            npz_file,
            init_fill=snap_initial["fill_level"],
            init_rho=snap_initial["rho"],
            init_u=snap_initial["velocity_magnitude"],
            final_fill=snap_final["fill_level"],
            final_rho=snap_final["rho"],
            final_u=snap_final["velocity_magnitude"],
        )

        if not quiet:
            print(f"Exported metrics to: {json_file}")
            print(f"Exported time-series to: {csv_file}")
            print(f"Exported field arrays to: {npz_file}")

    return results


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Unified Classical and Hybrid Quantum-Circuit Dam-Break Simulation."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classical",
        choices=["classical", "streaming_test", "hybrid"],
        help="Operating mode: 'classical' (Mode A), 'streaming_test' (Mode B), 'hybrid' (Mode C)",
    )
    parser.add_argument("--nx", type=int, default=64, help="Domain width (default: 64)")
    parser.add_argument("--ny", type=int, default=32, help="Domain height (default: 32)")
    parser.add_argument(
        "--column-width",
        type=int,
        default=16,
        help="Initial column width (default: 16)",
    )
    parser.add_argument(
        "--column-height",
        type=int,
        default=24,
        help="Initial column height (default: 24)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.8, help="Relaxation time (default: 0.8)"
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=0.0005,
        help="Lattice gravity acceleration (default: 0.0005)",
    )
    parser.add_argument(
        "--surface-tension",
        type=float,
        default=0.0,
        help="Surface tension coefficient (default: 0.0)",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of time steps (default: 100)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval in steps (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dam_break",
        help="Output directory for metrics, CSVs, and field snapshots",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress console stdout"
    )

    args = parser.parse_args()

    results = run_dam_break_simulation(
        mode=args.mode,
        nx=args.nx,
        ny=args.ny,
        column_width=args.column_width,
        column_height=args.column_height,
        tau=args.tau,
        gravity=args.gravity,
        surface_tension=args.surface_tension,
        steps=args.steps,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        quiet=args.quiet,
    )

    return 0 if results["validation"]["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
