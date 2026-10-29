#!/usr/bin/env python3
"""Record and animate the VOF-LBM dam-break simulation across classical & quantum modes.

Outputs (in --output-dir):
* <prefix>_<mode>.gif      two-panel animation (C and |u|);
* <prefix>_<mode>_diagnostics.png front position, mass error, Mach, and CM;
* <prefix>_<mode>_frames.npz      raw frames for reproducibility;
* <prefix>_<mode>_diagnostics.json diagnostic time series and run metadata;
* dam_break_4modes_comparison.gif 4-panel synchronized comparison across all 4 modes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from qlbm.classical.d2q9 import stream_and_bounce  # noqa: E402
from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver  # noqa: E402
from qlbm.components.vof.collision import LinearizedQuantumCollisionOperator  # noqa: E402
from qlbm.components.vof.hybrid_stepper import HybridVOFStep  # noqa: E402
from qlbm.components.vof.streaming import QuantumD2Q9StreamingOperator  # noqa: E402
from qlbm.tools.utils import is_two_pow  # noqa: E402

MM_T_MM = np.array(
    [0.0, 0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98, 2.20, 2.32, 2.51, 2.65, 2.83, 2.98, 3.11, 3.33]
)
MM_Z = np.array(
    [1.0, 1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67, 3.89, 4.11]
)


def build_solver(nx, ny, tau, gravity, column_width, column_height, scheme="korner"):
    """Replicate the scenario of vof_dam_break.run_dam_break_simulation."""
    solid_mask = np.zeros((ny, nx), dtype=bool)
    solid_mask[0, :] = True
    solid_mask[:, 0] = True
    solid_mask[:, -1] = True
    solver = VOFHydroSolver(
        nx=nx,
        ny=ny,
        tau=tau,
        gravity=gravity,
        surface_tension=0.0,
        solid_mask=solid_mask,
        free_surface_scheme=scheme,
    )
    solver.initialize_dam_break(
        column_width=column_width, column_height=column_height, add_walls=False
    )
    return solver, solid_mask


def run(args, mode=None):
    current_mode = mode if mode is not None else getattr(args, "mode", "classical")
    if current_mode in ("streaming_test", "hybrid", "full_quantum"):
        if not is_two_pow(args.nx) or not is_two_pow(args.ny):
            raise ValueError(
                f"Mode '{current_mode}' requires grid dimensions to be powers of two for "
                f"quantum register encoding. Got nx={args.nx}, ny={args.ny}."
            )

    solver, solid = build_solver(
        args.nx,
        args.ny,
        args.tau,
        args.gravity,
        args.column_width,
        args.column_height,
        scheme=args.scheme,
    )
    l0 = float(args.column_width)
    t_scale = math.sqrt(args.gravity / l0) if args.gravity > 0 else 0.0
    nu = (args.tau - 0.5) / 3.0
    u_ref = math.sqrt(args.gravity * args.column_height)
    reynolds = u_ref * args.column_height / nu

    hybrid_stepper = None
    quantum_streamer = None
    if current_mode == "hybrid":
        hybrid_stepper = HybridVOFStep(solver, use_quantum_streaming=True)
    elif current_mode == "full_quantum":
        collision_op = LinearizedQuantumCollisionOperator(tau=args.tau)
        hybrid_stepper = HybridVOFStep(
            solver, use_quantum_streaming=True, quantum_collision_op=collision_op
        )
    elif current_mode == "streaming_test":
        quantum_streamer = QuantumD2Q9StreamingOperator(nx=args.nx, ny=args.ny)

    fills, speeds, diag = [], [], []

    def record(step):
        fills.append(solver.fill_level.copy())
        speed = solver.velocity_magnitude().copy()
        speed[~solver.fluid_mask] = np.nan
        speeds.append(speed)
        x_cm, y_cm = solver.center_of_mass
        front = solver.front_position(threshold=0.5)
        diag.append(
            {
                "step": step,
                "t_dimless": step * t_scale,
                "front_x": front,
                "z_dimless": front / l0,
                "cm_x": x_cm,
                "cm_y": y_cm,
                "max_u": float(np.nanmax(speed)) if np.any(np.isfinite(speed)) else 0.0,
                "mass_error": solver.mass_error(),
                "volume_error": solver.volume_error(),
                "kinetic_energy": solver.kinetic_energy,
            }
        )

    t0 = time.perf_counter()
    record(0)
    for step in range(1, args.steps + 1):
        if current_mode in ("hybrid", "full_quantum"):
            hybrid_stepper.step()
        elif current_mode == "streaming_test":
            f_q = quantum_streamer.stream_distribution_array(solver.f)
            f_c = stream_and_bounce(solver.f, solid_mask=None)
            err = float(np.max(np.abs(f_q - f_c)))
            solver.step()
        else:
            solver.step()

        if step % args.frame_interval == 0 or step == args.steps:
            record(step)
            if not np.isfinite(solver.rho).all():
                print(f"Non-finite density at step {step}; stopping.")
                break
    elapsed = time.perf_counter() - t0
    print(f"[{current_mode.upper()}] Simulated {diag[-1]['step']} steps in {elapsed:.1f} s ({len(fills)} frames)")

    meta = {
        "mode": current_mode,
        "nx": args.nx,
        "ny": args.ny,
        "tau": args.tau,
        "nu_lattice": nu,
        "gravity": args.gravity,
        "column_width": args.column_width,
        "column_height": args.column_height,
        "aspect_ratio": args.column_height / args.column_width,
        "reynolds_sqrt_gH": reynolds,
        "mach_ref": u_ref / math.sqrt(1.0 / 3.0),
        "steps": diag[-1]["step"],
        "frame_interval": args.frame_interval,
        "scheme": args.scheme,
        "elapsed_seconds": elapsed,
    }
    return np.array(fills), np.array(speeds), diag, solid, meta


def make_gif(fills, speeds, diag, solid, meta, path, fps):
    vmax_u = max(float(np.nanmax(speeds)), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    wall = np.ma.masked_where(~solid, np.ones_like(solid, dtype=float))

    im_c = axes[0].imshow(fills[0], origin="lower", cmap="Blues", vmin=0, vmax=1)
    im_u = axes[1].imshow(speeds[0], origin="lower", cmap="inferno", vmin=0, vmax=vmax_u)
    for ax in axes:
        ax.imshow(wall, origin="lower", cmap="Greys", vmin=0, vmax=1.4)
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
    axes[0].set_title("Liquid fill level C")
    axes[1].set_title("Velocity magnitude |u| (liquid cells)")
    fig.colorbar(im_c, ax=axes[0], shrink=0.85, label="C")
    fig.colorbar(im_u, ax=axes[1], shrink=0.85, label="|u|")
    title = fig.suptitle("")

    mode_name = meta.get("mode", "classical").replace("_", " ").title()

    def update(i):
        d = diag[i]
        im_c.set_data(fills[i])
        im_u.set_data(speeds[i])
        title.set_text(
            f"VOF-LBM Dam Break ({mode_name})  {meta['nx']}x{meta['ny']}, "
            f"col {meta['column_width']}x{meta['column_height']}  |  "
            f"step {d['step']}  T*={d['t_dimless']:.2f}  Z={d['z_dimless']:.2f}  "
            f"mass err={d['mass_error']:.1e}"
        )
        return im_c, im_u, title

    anim = FuncAnimation(fig, update, frames=len(fills), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_4panel_comparison_gif(all_results, solid, path, fps):
    """Generate a synchronized 2x2 comparison GIF of all 4 simulation modes."""
    modes = ["classical", "streaming_test", "hybrid", "full_quantum"]
    mode_titles = {
        "classical": "Mode A: Classical (BGK + Classical Streaming)",
        "streaming_test": "Mode B: Streaming Test (Verified Draper QFT)",
        "hybrid": "Mode C: Hybrid QLBM (Draper QFT Streaming)",
        "full_quantum": "Mode D: Full Quantum (QFT + 5-Qubit Block Coll)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    wall = np.ma.masked_where(~solid, np.ones_like(solid, dtype=float))

    ims = []
    subtitles = []
    n_frames = len(all_results["classical"][0])

    for ax, m in zip(axes_flat, modes):
        fills, speeds, diag, _, meta = all_results[m]
        im = ax.imshow(fills[0], origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax.imshow(wall, origin="lower", cmap="Greys", vmin=0, vmax=1.4)
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
        title_text = f"{mode_titles[m]}\nZ={diag[0]['z_dimless']:.2f} | mass err={diag[0]['mass_error']:.1e}"
        st = ax.set_title(title_text, fontsize=10)
        ims.append(im)
        subtitles.append(st)

    fig.colorbar(ims[0], ax=axes, shrink=0.6, label="Fill Level C", location="right")
    sup = fig.suptitle("QLBM Multi-Mode Dam-Break Comparison (Step 0)", fontsize=13, fontweight="bold")

    def update(i):
        for idx, m in enumerate(modes):
            fills, _, diag, _, _ = all_results[m]
            ims[idx].set_data(fills[i])
            d = diag[i]
            txt = f"{mode_titles[m]}\nStep {d['step']} (T*={d['t_dimless']:.2f}) | Z={d['z_dimless']:.2f} | mass err={d['mass_error']:.1e}"
            subtitles[idx].set_text(txt)
        d0 = all_results["classical"][2][i]
        sup.set_text(
            f"QLBM Dam-Break Multi-Mode Evolution  |  Step {d0['step']} (T*={d0['t_dimless']:.2f})"
        )
        return ims + subtitles + [sup]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_diagnostics_plot(diag, meta, path):
    t = np.array([d["t_dimless"] for d in diag])
    z = np.array([d["z_dimless"] for d in diag])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    mode_name = meta.get("mode", "classical").upper()
    fig.suptitle(f"Dam-Break Hydrodynamic Diagnostics [{mode_name}]", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(t, z, "b-", label=f"QLBM [{mode_name}] (H0/L0={meta['aspect_ratio']:.2f})")
    ax.plot(MM_T_MM / math.sqrt(2.0), MM_Z, "ko", mfc="none", label="Martin & Moyce 1952")
    ax.set_xlabel(r"$T^* = t\sqrt{g/L_0}$")
    ax.set_ylabel(r"$Z = x_{front}/L_0$")
    ax.set_title("Surge-front position")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    err = np.array([max(d["mass_error"], 1e-18) for d in diag])
    ax.semilogy(t, err, "g-")
    ax.set_xlabel(r"$T^*$")
    ax.set_ylabel(r"$|M(t)-M_0|/M_0$")
    ax.set_title("Relative mass error")
    ax.grid(alpha=0.4, which="both")

    ax = axes[1, 0]
    mach = np.array([d["max_u"] for d in diag]) / math.sqrt(1.0 / 3.0)
    ax.plot(t, mach, "r-")
    ax.axhline(0.3, color="k", ls="--", lw=1, label="Ma = 0.3 limit")
    ax.set_xlabel(r"$T^*$")
    ax.set_ylabel(r"max $|u|/c_s$")
    ax.set_title("Peak Mach number")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, [d["cm_x"] / meta["column_width"] for d in diag], label=r"$x_{cm}/L_0$")
    ax.plot(t, [d["cm_y"] / meta["column_height"] for d in diag], label=r"$y_{cm}/H_0$")
    ax.set_xlabel(r"$T^*$")
    ax.set_title("Liquid centre of mass")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)

    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        default="all",
        choices=["classical", "streaming_test", "hybrid", "full_quantum", "all"],
        help="Operating mode: classical, streaming_test, hybrid, full_quantum, or all",
    )
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--gravity", type=float, default=0.002)
    p.add_argument("--column-width", type=int, default=4)
    p.add_argument("--column-height", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frame-interval", type=int, default=1)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output-dir", default="results/dam_break_final")
    p.add_argument("--prefix", default="dam_break")
    p.add_argument("--scheme", default="korner", choices=["korner", "legacy"])
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "all":
        modes = ["classical", "streaming_test", "hybrid", "full_quantum"]
        all_results = {}
        for m in modes:
            print("\n" + "=" * 70)
            print(f"Running simulation in mode: {m.upper()}")
            print("=" * 70)
            m_args = copy.deepcopy(args)
            m_args.mode = m
            m_prefix = f"dam_break_{m}"
            fills, speeds, diag, solid, meta = run(m_args, mode=m)
            all_results[m] = (fills, speeds, diag, solid, meta)

            np.savez_compressed(out / f"{m_prefix}_frames.npz", fill=fills, speed=speeds, solid=solid)
            with open(out / f"{m_prefix}_diagnostics.json", "w") as fh:
                json.dump({"metadata": meta, "history": diag}, fh, indent=1)
            make_diagnostics_plot(diag, meta, out / f"{m_prefix}_diagnostics.png")
            make_gif(fills, speeds, diag, solid, meta, out / f"{m_prefix}.gif", args.fps)
            print(f"Generated {out / f'{m_prefix}.gif'}")

        print("\n" + "=" * 70)
        print("Generating 4-panel comparison GIF across all modes...")
        comp_gif = out / "dam_break_4modes_comparison.gif"
        make_4panel_comparison_gif(all_results, solid, comp_gif, args.fps)
        print(f"Generated comparison GIF: {comp_gif}")
    else:
        fills, speeds, diag, solid, meta = run(args)
        m_prefix = f"{args.prefix}_{args.mode}" if args.prefix == "dam_break" else args.prefix
        np.savez_compressed(out / f"{m_prefix}_frames.npz", fill=fills, speed=speeds, solid=solid)
        with open(out / f"{m_prefix}_diagnostics.json", "w") as fh:
            json.dump({"metadata": meta, "history": diag}, fh, indent=1)
        make_diagnostics_plot(diag, meta, out / f"{m_prefix}_diagnostics.png")
        make_gif(fills, speeds, diag, solid, meta, out / f"{m_prefix}.gif", args.fps)
        print(f"Wrote outputs with prefix '{m_prefix}' to {out}")


if __name__ == "__main__":
    main()
