#!/usr/bin/env python3
"""Record and animate the classical VOF-LBM dam-break simulation.

``vof_dam_break.py`` only stores the initial and final fields, so it cannot
be used to make an animation. This script rebuilds the same scenario (solid
bottom/left/right walls, ``VOFHydroSolver``, rectangular liquid column in the
lower-left corner), advances it step by step, and records:

* frames of the fill level ``C`` and velocity magnitude ``|u|`` every
  ``--frame-interval`` steps;
* per-frame diagnostics (front position, centre of mass, mass error, max |u|).

Outputs (in ``--output-dir``):

* ``<prefix>_evolution.gif``   two-panel animation (C and |u|);
* ``<prefix>_diagnostics.png`` front position vs Martin & Moyce (1952),
  mass error, max |u| (Mach) and centre-of-mass history;
* ``<prefix>_frames.npz``      raw frames for reproducibility;
* ``<prefix>_diagnostics.json`` the diagnostic time series and run metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver  # noqa: E402

# Martin & Moyce (1952), Phil. Trans. R. Soc. A 244, 312-324, column with
# n^2 = H0/L0 = 2, as commonly tabulated in VOF/SPH validation papers.
# Their time scale is T_MM = t * sqrt(2 g / L0), i.e. T_MM = sqrt(2) * T*.
# Values are digitised and should be checked against the original table
# before being quoted in a publication.
MM_T_MM = np.array(
    [0.0, 0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98, 2.20, 2.32, 2.51, 2.65, 2.83, 2.98, 3.11, 3.33]
)
MM_Z = np.array(
    [1.0, 1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67, 3.89, 4.11]
)


def build_solver(nx, ny, tau, gravity, column_width, column_height, scheme="korner"):
    """Replicate the scenario of ``vof_dam_break.run_dam_break_simulation``."""
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


def run(args):
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
    t_scale = math.sqrt(args.gravity / l0)
    nu = (args.tau - 0.5) / 3.0
    u_ref = math.sqrt(args.gravity * args.column_height)
    reynolds = u_ref * args.column_height / nu

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
                # front_position returns the cell index; the wall occupies x = 0,
                # so the index equals the distance from the left wall face + 0.5.
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
        solver.step()
        if step % args.frame_interval == 0 or step == args.steps:
            record(step)
            if not np.isfinite(solver.rho).all():
                print(f"Non-finite density at step {step}; stopping.")
                break
    elapsed = time.perf_counter() - t0
    print(f"Simulated {diag[-1]['step']} steps in {elapsed:.1f} s ({len(fills)} frames)")

    meta = {
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

    def update(i):
        d = diag[i]
        im_c.set_data(fills[i])
        im_u.set_data(speeds[i])
        title.set_text(
            f"Classical VOF-LBM dam break  {meta['nx']}x{meta['ny']}, "
            f"column {meta['column_width']}x{meta['column_height']}  |  "
            f"step {d['step']}  T*={d['t_dimless']:.2f}  Z={d['z_dimless']:.2f}  "
            f"mass err={d['mass_error']:.1e}"
        )
        return im_c, im_u, title

    anim = FuncAnimation(fig, update, frames=len(fills), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_diagnostics_plot(diag, meta, path):
    t = np.array([d["t_dimless"] for d in diag])
    z = np.array([d["z_dimless"] for d in diag])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t, z, "b-", label=f"VOF-LBM (H0/L0={meta['aspect_ratio']:.2f}, Re~{meta['reynolds_sqrt_gH']:.0f})")
    ax.plot(MM_T_MM / math.sqrt(2.0), MM_Z, "ko", mfc="none", label="Martin & Moyce 1952 (H0/L0=2)")
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
    ax.axhline(0.3, color="k", ls="--", lw=1, label="Ma = 0.3 (incompressibility limit)")
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
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--gravity", type=float, default=0.0005)
    p.add_argument("--column-width", type=int, default=16)
    p.add_argument("--column-height", type=int, default=24)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--frame-interval", type=int, default=10)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output-dir", default="results/dam_break_final")
    p.add_argument("--prefix", default="qlbm")
    p.add_argument("--scheme", default="korner", choices=["korner", "legacy"])
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fills, speeds, diag, solid, meta = run(args)

    np.savez_compressed(out / f"{args.prefix}_frames.npz", fill=fills, speed=speeds, solid=solid)
    with open(out / f"{args.prefix}_diagnostics.json", "w") as fh:
        json.dump({"metadata": meta, "history": diag}, fh, indent=1)
    make_diagnostics_plot(diag, meta, out / f"{args.prefix}_diagnostics.png")
    make_gif(fills, speeds, diag, solid, meta, out / f"{args.prefix}_evolution.gif", args.fps)
    print(f"Wrote outputs with prefix '{args.prefix}' to {out}")


if __name__ == "__main__":
    main()
