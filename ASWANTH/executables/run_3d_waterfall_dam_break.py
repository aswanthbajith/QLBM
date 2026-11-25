#!/usr/bin/env python3
"""3D Gravity-Driven Waterfall and Dam-Break Simulation with Quantum Circuit Streaming.

Executes a 3D gravity-driven waterfall simulation:
  1. Water is held behind an openable gate on an elevated shelf.
  2. The gate opens; water flows across the shelf edge.
  3. Water detaches and plunges vertically under gravity (g = (0, 0, -9.81) m/s^2).
  4. Water impacts the lower basin, forms splash waves and pools.
  5. Hydrodynamic discrete distribution streaming is executed via 3D Draper QFT
     phase adders on spatial coordinate registers (Qiskit Aer statevector simulation).

Operating Modes:
  - 'classical': 100% classical 3D D3Q19 VOF-LBM solver.
  - 'streaming_test': Exact basis-state and field equivalence verification between
    quantum D3Q19 streaming circuit and classical 3D periodic streaming.
  - 'hybrid': Full hybrid simulation delegating 3D streaming to quantum circuits.
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

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from qlbm.classical.d3q19 import CS_SQ, stream_and_bounce
from qlbm.classical.hybrid_vof_3d import VOF3DHydroSolver
from qlbm.classical.vof_3d import CellType
from qlbm.components.vof.hybrid_stepper_3d import HybridVOF3DStep
from qlbm.components.vof.streaming_3d import QuantumD3Q19StreamingOperator
from qlbm.tools.utils import is_two_pow
from ASWANTH.visualization.visualize_3d_waterfall import export_vti, plot_waterfall_slice


def run_waterfall_simulation(
    mode: str = "classical",
    nx: int = 16,
    ny: int = 4,
    nz: int = 16,
    shelf_height: int = 7,
    shelf_length: int = 6,
    gate_x: int = 6,
    water_height: int = 4,
    water_width: Optional[int] = None,
    water_depth: Optional[int] = None,
    tau: float = 0.7,
    gravity_lattice: float = 0.025,
    steps: int = 100,
    gate_open_step: int = 2,
    log_interval: int = 5,
    snapshot_interval: int = 2,
    output_dir: Optional[str] = None,
    export_vtk: bool = True,
    generate_gif: bool = True,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Execute the 3D waterfall and elevated dam-break simulation.

    Parameters
    ----------
    mode : {'classical', 'streaming_test', 'hybrid'}
        Simulation operating mode.
    nx, ny, nz : int
        Grid dimensions (must be powers of 2 for quantum modes).
    shelf_height : int
        Elevation of the shelf floor in lattice nodes.
    shelf_length : int
        Horizontal length of the shelf.
    gate_x : int
        Position of the openable vertical gate.
    water_height : int
        Height of the liquid column.
    tau : float
        BGK relaxation time.
    gravity_lattice : float
        Gravitational acceleration in lattice units (g_z = -gravity_lattice).
    steps : int
        Total time steps to simulate.
    gate_open_step : int
        Time step at which the gate is opened.
    log_interval : int
        Console logging interval in steps.
    snapshot_interval : int
        Snapshot interval for NPZ and VTK output.
    output_dir : str, optional
        Output folder for metrics, snapshots, and plots.
    export_vtk : bool, optional
        Whether to export .vti files for ParaView.
    quiet : bool, optional
        Suppress console output if True.

    Returns
    -------
    dict
        Summary dictionary of results and verification metrics.
    """
    t_start_total = time.perf_counter()

    # Prepare output directories
    out_path = Path(output_dir) if output_dir else Path("ASWANTH/results/waterfall_3d")
    snap_dir = out_path / "snapshots"
    out_path.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize 3D Hydro Solver
    gravity_vector = (0.0, 0.0, -float(gravity_lattice))
    solver = VOF3DHydroSolver(
        nx=nx,
        ny=ny,
        nz=nz,
        tau=tau,
        gravity=gravity_vector,
        dt=1.0,
    )

    solver.initialize_waterfall_dam_break(
        shelf_height=shelf_height,
        shelf_length=shelf_length,
        gate_x=gate_x,
        water_height=water_height,
        water_width=water_width,
        water_depth=water_depth,
        rho_liquid=1.0,
    )

    initial_mass = solver.total_mass
    solver.initial_mass = initial_mass

    # 2. Setup Mode Operations
    hybrid_stepper: Optional[HybridVOF3DStep] = None
    streamer_3d: Optional[QuantumD3Q19StreamingOperator] = None
    circuit_metrics: Dict[str, Any] = {}

    if mode in ("streaming_test", "hybrid"):
        if not (is_two_pow(nx) and is_two_pow(ny) and is_two_pow(nz)):
            raise ValueError(
                f"Grid ({nx}, {ny}, {nz}) must have power-of-two dimensions for quantum mode '{mode}'"
            )
        streamer_3d = QuantumD3Q19StreamingOperator(nx=nx, ny=ny, nz=nz)
        circuit_metrics = streamer_3d.resource_metrics()

    if mode == "hybrid":
        hybrid_stepper = HybridVOF3DStep(
            solver=solver, use_quantum_streaming=True
        )

    # Streaming test mode: verify equivalence and return immediately
    if mode == "streaming_test":
        if not quiet:
            print("=" * 80)
            print("[QUANTUM CIRCUIT] D3Q19 QUANTUM STREAMING TEST")
            print(f"Grid: {nx}x{ny}x{nz} | Total Qubits: {circuit_metrics['num_qubits']} | Depth: {circuit_metrics['depth']}")
            print("=" * 80)

        rng = np.random.default_rng(42)
        f_test = rng.uniform(0.01, 0.1, size=(19, nz, ny, nx))
        f_q, q_time = streamer_3d.stream_distribution_array(f_test, return_timing=True)
        f_c = stream_and_bounce(f_test, solid_mask=None)

        max_err = float(np.max(np.abs(f_q - f_c)))
        if not quiet:
            print(f"Maximum absolute difference E_max = {max_err:.4e}")
            print(f"State preparation: {q_time['t_state_prep']*1000:.2f} ms")
            print(f"Circuit evolution: {q_time['t_circuit_evolution']*1000:.2f} ms")
            print(f"Readout decoding:  {q_time['t_readout_decoding']*1000:.2f} ms")
            print("Verification Result: " + ("PASS (E_max < 1e-12)" if max_err < 1e-12 else "FAIL"))
            print("=" * 80)

        return {
            "mode": mode,
            "max_streaming_error": max_err,
            "circuit_metrics": circuit_metrics,
            "timing": q_time,
            "success": max_err < 1e-12,
        }

    # 3. Banner
    mode_label = (
        "[HYBRID QUANTUM-CLASSICAL]" if mode == "hybrid" else "[CLASSICAL]"
    )
    if not quiet:
        print("=" * 95)
        print(f"{mode_label} 3D WATERFALL & DAM-BREAK SIMULATOR")
        print(f"Grid: {nx} x {ny} x {nz} nodes | Shelf Floor: z={shelf_height} | Gate: x={gate_x}")
        print(f"Gravity: g_z = -{gravity_lattice:.4f} (lattice) | Initial Water Mass: {initial_mass:.4f}")
        if mode == "hybrid":
            print(f"Quantum Circuit: {circuit_metrics['num_qubits']} qubits ({circuit_metrics['num_x_qubits']}x + {circuit_metrics['num_y_qubits']}y + {circuit_metrics['num_z_qubits']}z + {circuit_metrics['num_vel_qubits']}v) | Depth: {circuit_metrics['depth']}")
        print("=" * 95)
        header = (
            f"{'Step':>4} | {'Gate':<6} | {'Mass':>10} | {'dMass':>10} | "
            f"{'CM (x, y, z)':^18} | {'Front':>5} | {'Max |u_z|':>9} | {'Max Speed':>9} | {'Phase/Event':<15}"
        )
        print(header)
        print("-" * 95)

    # 4. Simulation Loop
    metrics_records: List[Dict[str, Any]] = []

    for step in range(steps):
        # Check gate opening trigger
        if step == gate_open_step:
            solver.open_gate()

        gate_status = "OPEN" if solver.is_gate_open else "CLOSED"

        # Advance one step
        if mode == "hybrid":
            hybrid_stepper.step()
        else:
            solver.step()

        # Compute diagnostics
        current_mass = solver.total_mass
        mass_error = abs(current_mass - initial_mass)
        x_cm, y_cm, z_cm = solver.center_of_mass()
        x_front = solver.front_position()
        uz_min = float(np.min(solver.u[2, solver.fluid_mask])) if np.any(solver.fluid_mask) else 0.0
        max_downward_speed = abs(min(0.0, uz_min))
        max_speed = float(np.max(solver.speed[solver.fluid_mask])) if np.any(solver.fluid_mask) else 0.0

        # Physical regime identification
        active_z = np.where(solver.fluid_mask)[0] if np.any(solver.fluid_mask) else np.array([shelf_height + 1])
        min_z = int(np.min(active_z))
        if not solver.is_gate_open:
            event_phase = "Enclosed Rest"
        elif min_z <= 2:
            event_phase = "Basin Pooling" if z_cm < (shelf_height + 0.5) else "Basin Impact"
        elif x_front >= shelf_length:
            event_phase = "Vertical Plunge"
        else:
            event_phase = "Surge on Shelf"

        record = {
            "step": solver.time_step,
            "gate_status": gate_status,
            "mass": current_mass,
            "mass_error": mass_error,
            "x_cm": x_cm,
            "y_cm": y_cm,
            "z_cm": z_cm,
            "x_front": x_front,
            "max_downward_uz": max_downward_speed,
            "max_speed": max_speed,
            "event_phase": event_phase,
        }
        metrics_records.append(record)

        if not quiet and (step % log_interval == 0 or step == steps - 1):
            cm_str = f"({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f})"
            line = (
                f"{solver.time_step:4d} | {gate_status:<6} | {current_mass:10.4f} | "
                f"{mass_error:10.2e} | {cm_str:^18} | {x_front:5.1f} | "
                f"{max_downward_speed:9.4f} | {max_speed:9.4f} | {event_phase:<15}"
            )
            print(line)

        # Snapshot export
        if step % snapshot_interval == 0 or step == steps - 1:
            step_tag = f"step_{solver.time_step:04d}"
            npz_file = snap_dir / f"{step_tag}.npz"
            np.savez_compressed(
                npz_file,
                step=solver.time_step,
                fill_level=solver.fill_level,
                rho=solver.rho,
                u=solver.u,
                solid_mask=solver.solid_mask,
                cell_types=solver.cell_types,
            )

            if export_vtk:
                vti_file = snap_dir / f"{step_tag}.vti"
                export_vti(
                    filepath=vti_file,
                    fill_level=solver.fill_level,
                    rho=solver.rho,
                    u=solver.u,
                    solid_mask=solver.solid_mask,
                    cell_types=solver.cell_types,
                )
                plot_waterfall_slice(
                    npz_file,
                    save_path=snap_dir / f"{step_tag}_slice.png",
                )

    t_total = time.perf_counter() - t_start_total

    # 5. Generate Animated GIF
    gif_file = out_path / "waterfall_3d_animation.gif"
    gif_generated = False
    if generate_gif and export_vtk:
        try:
            from ASWANTH.visualization.visualize_3d_waterfall import generate_waterfall_gif
            generate_waterfall_gif(snap_dir, gif_file, duration_ms=120)
            gif_generated = gif_file.exists()
        except Exception as e:
            if not quiet:
                print(f"Warning: GIF generation skipped ({e})")

    # 6. Export Metrics CSV & Metadata JSON
    csv_file = out_path / "waterfall_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(metrics_records[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_records)

    metadata = {
        "mode": mode,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "geometry": {
            "shelf_height": shelf_height,
            "shelf_length": shelf_length,
            "gate_x": gate_x,
            "water_height": water_height,
        },
        "physics": {
            "tau": tau,
            "gravity_lattice": gravity_lattice,
            "gravity_vector": list(gravity_vector),
        },
        "simulation": {
            "steps": steps,
            "gate_open_step": gate_open_step,
            "initial_mass": initial_mass,
            "final_mass": metrics_records[-1]["mass"],
            "max_mass_error": max(r["mass_error"] for r in metrics_records),
            "total_runtime_seconds": t_total,
            "animation_gif": str(gif_file) if gif_generated else None,
        },
        "quantum_circuit": circuit_metrics,
    }

    json_file = out_path / "waterfall_metadata.json"
    json_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not quiet:
        print("=" * 95)
        print(f"Simulation completed in {t_total:.2f} s ({t_total/steps:.4f} s/step)")
        print(f"Data saved to: {out_path.resolve()}")
        print(f"  - CSV Time Series: {csv_file.name}")
        print(f"  - Metadata JSON:   {json_file.name}")
        print(f"  - Snapshots:       {len(list(snap_dir.glob('*.npz')))} files (.npz, .vti, .png)")
        if gif_generated:
            print(f"  - Animated GIF:    {gif_file.name} ({gif_file.stat().st_size / 1024:.1f} KB)")
        if hybrid_stepper is not None:
            print(hybrid_stepper.report_profiling())
        print("=" * 95)

    return metadata


def main() -> None:
    """CLI entrypoint for running 3D waterfall simulations."""
    parser = argparse.ArgumentParser(
        description="Run 3D gravity-driven waterfall & dam-break simulation with quantum streaming."
    )
    parser.add_argument(
        "--mode",
        choices=["classical", "streaming_test", "hybrid"],
        default="classical",
        help="Operating mode (default: classical)",
    )
    parser.add_argument("--nx", type=int, default=16, help="Grid size X (power of 2, default: 16)")
    parser.add_argument("--ny", type=int, default=4, help="Grid size Y (power of 2, default: 4)")
    parser.add_argument("--nz", type=int, default=16, help="Grid size Z (power of 2, default: 16)")
    parser.add_argument("--shelf-height", type=int, default=7, help="Shelf floor height (default: 7)")
    parser.add_argument("--shelf-length", type=int, default=6, help="Shelf length (default: 6)")
    parser.add_argument("--gate-x", type=int, default=6, help="Gate x-coordinate (default: 6)")
    parser.add_argument("--water-height", type=int, default=4, help="Water column height (default: 4)")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps (default: 100)")
    parser.add_argument("--gate-open-step", type=int, default=2, help="Step to open gate (default: 2)")
    parser.add_argument("--gravity", type=float, default=0.025, help="Lattice gravity g_z (default: 0.025)")
    parser.add_argument("--tau", type=float, default=0.7, help="Relaxation time tau (default: 0.7)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ASWANTH/results/waterfall_3d",
        help="Output directory",
    )
    parser.add_argument("--no-vtk", action="store_true", help="Disable VTK/PNG export")
    parser.add_argument("--no-gif", action="store_true", help="Disable animated GIF generation")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")

    args = parser.parse_args()

    run_waterfall_simulation(
        mode=args.mode,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        shelf_height=args.shelf_height,
        shelf_length=args.shelf_length,
        gate_x=args.gate_x,
        water_height=args.water_height,
        steps=args.steps,
        gate_open_step=args.gate_open_step,
        gravity_lattice=args.gravity,
        tau=args.tau,
        output_dir=args.output_dir,
        export_vtk=not args.no_vtk,
        generate_gif=not args.no_gif and not args.no_vtk,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
