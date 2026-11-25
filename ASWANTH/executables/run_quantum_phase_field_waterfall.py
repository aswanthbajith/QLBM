"""Production Executable: 3D Multiphase Quantum Phase-Field Waterfall Simulation.

Simulates 3D gravity-driven, two-phase fluid dynamics with:
1. Elevated reservoir holding liquid behind a release gate on a high shelf.
2. Gate opening releasing forward momentum over a cliff edge.
3. Free-fall acceleration under vertical gravity g = (0, 0, gz).
4. Jet detachment, impact on a solid floor/basin obstacle, and splash pooling.
5. Conservative Allen-Cahn (CAC D3Q7) phase-field tracking coupled to D3Q19 hydrodynamics.
6. Execution engines:
   - 'classical': Optimized NumPy reference engine on arbitrary grid resolutions.
   - 'quantum': Coupled Quantum Multiphase 3D Stepper (D3Q19 hydrodynamics + D3Q7 phase-field)
     driving both fluid and order parameter statevectors on QFT streaming and oracle circuits.

Outputs:
- Real-time ASCII progress and physics scorecard.
- NPZ/JSON simulation state data and metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C, CS_SQ, WEIGHTS
from qlbm.components.vof.streaming_d3q7 import WEIGHTS_D3Q7
from research.quantum_fluid_evolution.collision.deterministic_dissipation_channel import (
    DeterministicDissipationChannelD3Q19,
)
from research.quantum_fluid_evolution.phase_field.coupled_quantum_multiphase_stepper import (
    CoupledQuantumMultiphase3DStepper,
)
from research.quantum_fluid_evolution.phase_field.multiphase_phase_field_solver_3d import (
    MultiphasePhaseFieldSolver3D,
)


def run_waterfall_simulation(
    grid_size: Tuple[int, int, int] = (16, 8, 16),
    num_steps: int = 40,
    gz: float = -0.010,
    sigma: float = 0.015,
    tau: float = 0.8,
    gate_velocity: float = 0.06,
    engine: str = "classical",
    qlks_a: float = 1.0,
    z_shelf: Optional[int] = None,
    x_cliff: Optional[int] = None,
    save_npz: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete 3D multiphase waterfall simulation."""
    nz, ny, nx = grid_size
    if z_shelf is None:
        z_shelf = nz // 2
    if x_cliff is None:
        if engine == "quantum" and nx <= 8:
            x_cliff = max(2, nx // 4)
        else:
            x_cliff = nx * 3 // 8

    # Auto-tune gate velocity for small quantum lattice prototypes if using default
    if engine == "quantum" and nx <= 8 and gate_velocity == 0.06:
        gate_velocity = 0.10

    print("=" * 105)
    print(f"  3D MULTIPHASE WATERFALL ENGINE [{engine.upper()} EXECUTION]")
    print(f"  Grid: {nx} x {ny} x {nz} | Steps: {num_steps} | gz: {gz} | sigma: {sigma} | tau: {tau}")
    print(f"  Shelf: z <= {z_shelf}, x <= {x_cliff} | Lower Basin: z < {z_shelf}, x > {x_cliff}")
    print("=" * 105)

    # 1. Geometry Setup
    solid = np.zeros((nz, ny, nx), dtype=bool)
    solid[0, :, :] = True  # Lower basin floor
    solid[: z_shelf + 1, :, : x_cliff + 1] = True  # Shelf and cliff face
    solid[: max(2, nz // 4), :, (x_cliff + 3):(x_cliff + 6)] = True  # Impact obstacle in basin

    solid_nodes = int(np.sum(solid))
    fluid_nodes = int(np.sum(~solid))

    # Initial distributions
    phi_init = np.full((nz, ny, nx), -1.0, dtype=np.float64)
    x_reservoir_end = (x_cliff + 1) if (engine == "quantum" and nx <= 8) else x_cliff
    for z_i in range(z_shelf + 1, nz - 1):
        for y_i in range(1, ny - 1):
            for x_i in range(1, x_reservoir_end):
                if not solid[z_i, y_i, x_i]:
                    phi_init[z_i, y_i, x_i] = 1.0

    init_liquid_vol = int(np.sum(phi_init > 0.0))
    init_total_order = float(np.sum(phi_init))

    print("\n--- Domain & Initial State ---")
    print(f"  Total Domain Nodes:  {nx * ny * nz}")
    print(f"  Fluid Cells:         {fluid_nodes}")
    print(f"  Solid Obstacle Cells:{solid_nodes}")
    print(f"  Reservoir Liquid Vol:{init_liquid_vol} cells")
    print(f"  Initial Total Order: {init_total_order:.6f}")

    if engine == "quantum":
        quantum_stepper = CoupledQuantumMultiphase3DStepper(
            nx=nx, ny=ny, nz=nz,
            tau_f=tau, tau_phi=tau,
            sigma=sigma, g=(0.0, 0.0, gz),
            interface_width=1.0,
            qlks_A=qlks_a,
            solid_mask=solid,
        )
        res_info = quantum_stepper.resource_summary()
        print("\n--- Quantum Circuit Architecture ---")
        print(f"  Hydrodynamic Qubits: {res_info['hydro_qubits']} (D3Q19 register, dim {res_info['hydro_statevector_dim']:,})")
        print(f"  Phase-Field Qubits:  {res_info['phase_qubits']} (D3Q7 register, dim {res_info['phase_statevector_dim']:,})")
        print(f"  Total System Qubits: {res_info['total_system_qubits']}")
        print(f"  Step Circuit Depth:  Hydro {res_info['hydro_step_depth']} | Phase {res_info['phase_step_depth']}")
        print(f"  QLKS Parameter A:    {res_info['qlks_parameter_A']} (nu_eff = {res_info['effective_viscosity']:.4f})")

        f_curr = np.zeros((19, nz, ny, nx), dtype=np.float64)
        for i in range(19):
            cu_gate = C[i, 0] * gate_velocity / CS_SQ
            f_curr[i] = WEIGHTS[i] * 1.0 * (1.0 + cu_gate)

        g_curr = np.zeros((7, nz, ny, nx), dtype=np.float64)
        for i in range(7):
            g_curr[i] = WEIGHTS_D3Q7[i] * phi_init

    else:
        solver = MultiphasePhaseFieldSolver3D(
            nx=nx, ny=ny, nz=nz,
            tau_f=tau, tau_phi=tau,
            sigma=sigma, g=(0.0, 0.0, gz),
            solid_mask=solid,
            interface_width=1.0,
            rho_liquid=1.0,
            rho_gas=1.0,
        )
        solver.initialize_waterfall_reservoir(z_shelf=z_shelf, x_cliff=x_cliff)
        for i in range(19):
            solver.f[i] += WEIGHTS[i] * 1.0 * (C[i, 0] * gate_velocity / CS_SQ)

    print("\n--- Time Evolution Scorecard ---")
    header = (
        f"{'Step':^6} | {'Time (s)':^9} | {'Liquid Vol':^10} | {'Air Vol (>Cliff)':^16} | "
        f"{'Down Mom Jz':^12} | {'Max Vel':^9} | {'Solid Penetration':^17}"
    )
    print(header)
    print("-" * 105)

    history = []
    t_start = time.perf_counter()

    for s in range(1, num_steps + 1):
        if engine == "quantum":
            f_curr, g_curr, step_res = quantum_stepper.step(f_curr, g_curr)
            phi = np.sum(g_curr, axis=0)
            rho = np.sum(f_curr, axis=0)
            u = None
        else:
            step_res = solver.step()
            phi = step_res["phi"]
            rho = step_res["rho"]
            u = step_res["u"]

        air_liquid = int(np.sum(phi[:, :, x_cliff + 1:] > 0.0))
        penetration = int(np.sum(phi[solid] > -0.5))
        elapsed = time.perf_counter() - t_start

        record = {
            "step": s,
            "elapsed_s": elapsed,
            "liquid_volume": step_res["liquid_volume"],
            "air_falling_volume": air_liquid,
            "jz": step_res["vertical_momentum_jz"],
            "max_u": step_res["max_velocity"],
            "solid_penetration": penetration,
            "total_order_mass": step_res["total_order_mass"],
        }
        history.append(record)

        if s == 1 or s % 5 == 0 or s == num_steps:
            row = (
                f"{s:^6d} | "
                f"{elapsed:^9.3f} | "
                f"{int(step_res['liquid_volume']):^10d} | "
                f"{air_liquid:^16d} | "
                f"{step_res['vertical_momentum_jz']:^12.5f} | "
                f"{step_res['max_velocity']:^9.4f} | "
                f"{penetration:^17d}"
            )
            print(row)

    total_time = time.perf_counter() - t_start
    final_record = history[-1]

    detached = any(h["air_falling_volume"] > 0 for h in history)
    downward_mom = any(h["jz"] < -0.005 for h in history)
    zero_penetration = all(h["solid_penetration"] == 0 for h in history)
    mass_conservation_err = abs(final_record["total_order_mass"] - init_total_order) / (abs(init_total_order) + 1e-12)

    print("-" * 105)
    print("\n--- Physical Validation Summary ---")
    print(f"  Simulation Wall Time:   {total_time:.2f} s ({total_time / num_steps * 1000:.1f} ms/step)")
    print(f"  Water Jet Detachment:   {'VERIFIED (Free Fall Past Cliff)' if detached else 'FAILED'}")
    print(f"  Downward Momentum:      {'VERIFIED (Negative Gravity Jz)' if downward_mom else 'FAILED'}")
    print(f"  Solid Wall Penetration: {'VERIFIED (Strict Zero Penetration)' if zero_penetration else 'FAILED'}")
    print(f"  Order Parameter Mass Err:{mass_conservation_err:.2e} (CAC Conservation)")

    overall_passed = detached and downward_mom and zero_penetration and (mass_conservation_err < 0.10)
    print(f"\n  OVERALL VALIDATION STATUS: {'SUCCESS - PHYSICALLY VALID' if overall_passed else 'CHECK FAILED'}")
    print("=" * 105)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        metrics_file = out_path / f"waterfall_{engine}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "engine": engine,
                "grid_size": list(grid_size),
                "num_steps": num_steps,
                "gravity": gz,
                "sigma": sigma,
                "tau": tau,
                "qlks_a": qlks_a,
                "total_time_seconds": total_time,
                "validation_passed": overall_passed,
                "mass_conservation_relative_error": mass_conservation_err,
                "history": history,
            }, f, indent=2)
        print(f"\n[Saved Metrics JSON] -> {metrics_file}")

        if save_npz:
            npz_file = out_path / f"waterfall_{engine}_final_state.npz"
            if engine == "classical":
                final_rho, final_u = solver.compute_velocity()
                final_phi = solver.compute_phi()
            else:
                final_rho = np.sum(f_curr, axis=0)
                final_phi = np.sum(g_curr, axis=0)
                final_u = np.zeros((3, nz, ny, nx))
            np.savez_compressed(
                npz_file,
                phi=final_phi,
                rho=final_rho,
                u=final_u,
                solid=solid,
            )
            print(f"[Saved State NPZ]    -> {npz_file}")

    return {
        "passed": overall_passed,
        "total_time": total_time,
        "history": history,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="3D Multiphase Quantum Phase-Field Waterfall Simulation"
    )
    parser.add_argument("--nx", type=int, default=16, help="Lattice width (X)")
    parser.add_argument("--ny", type=int, default=8, help="Lattice depth (Y)")
    parser.add_argument("--nz", type=int, default=16, help="Lattice height (Z)")
    parser.add_argument("--steps", type=int, default=35, help="Number of simulation steps")
    parser.add_argument("--gz", type=float, default=-0.010, help="Vertical gravity acceleration")
    parser.add_argument("--sigma", type=float, default=0.015, help="Surface tension coefficient")
    parser.add_argument("--tau", type=float, default=0.8, help="LBM relaxation time")
    parser.add_argument("--gate-u", type=float, default=0.06, help="Initial gate release velocity")
    parser.add_argument("--z-shelf", type=int, default=None, help="Height of reservoir shelf")
    parser.add_argument("--x-cliff", type=int, default=None, help="Position of cliff edge")
    parser.add_argument("--engine", choices=["classical", "quantum"], default="classical",
                        help="Execution engine ('classical' NumPy or 'quantum' Qiskit circuit stepper)")
    parser.add_argument("--qlks-a", type=float, default=1.0, help="QLKS viscosity parameter A in (0, 1]")
    parser.add_argument("--output-dir", type=str, default="data/waterfall_phase_field_output",
                        help="Directory to save JSON metrics and NPZ fields")
    parser.add_argument("--no-npz", action="store_true", help="Disable NPZ field save")

    args = parser.parse_args()

    run_waterfall_simulation(
        grid_size=(args.nz, args.ny, args.nx),
        num_steps=args.steps,
        gz=args.gz,
        sigma=args.sigma,
        tau=args.tau,
        gate_velocity=args.gate_u,
        engine=args.engine,
        qlks_a=args.qlks_a,
        z_shelf=args.z_shelf,
        x_cliff=args.x_cliff,
        save_npz=not args.no_npz,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
