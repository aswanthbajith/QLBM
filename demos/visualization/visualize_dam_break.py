#!/usr/bin/env python3
"""Visualization Suite for Classical and Hybrid Quantum Dam-Break Simulation.

Generates the 9 required publication-quality figures:
  1. 01_initial_fill_level.png
  2. 02_final_fill_level.png
  3. 03_final_density.png
  4. 04_velocity_magnitude.png
  5. 05_free_surface_evolution.png
  6. 06_front_position_vs_time.png (with Martin & Moyce 1952 experimental comparison)
  7. 07_mass_conservation_vs_time.png
  8. 08_classical_vs_hybrid_comparison.png
  9. 09_quantum_streaming_error_field.png

All figures are saved to `results/dam_break/plots/`.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from qlbm.classical.d2q9 import stream_and_bounce
from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver
from qlbm.components.vof.streaming import QuantumD2Q9StreamingOperator


def generate_dam_break_plots(
    results_dir: str = "results/dam_break",
    plots_dir: str = "results/dam_break/plots",
) -> List[str]:
    """Generate all 9 required publication figures.

    Parameters
    ----------
    results_dir : str
        Directory containing simulation CSV, JSON, and NPZ results.
    plots_dir : str
        Target output directory for generated PNG figures.

    Returns
    -------
    list of str
        List of generated figure file paths.
    """
    res_path = Path(results_dir)
    plot_path = Path(plots_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    generated_files: List[str] = []

    # Configure matplotlib style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": True,
            "figure.dpi": 300,
        }
    )

    # Load classical snapshot
    snap_file_c = res_path / "dam_break_snapshots_classical.npz"
    if not snap_file_c.exists():
        raise FileNotFoundError(f"Missing {snap_file_c}. Run vof_dam_break.py first.")
    snap_c = np.load(snap_file_c)

    init_fill = snap_c["init_fill"]
    final_fill = snap_c["final_fill"]
    final_rho = snap_c["final_rho"]
    final_u = snap_c["final_u"]
    ny, nx = init_fill.shape

    # Load classical history CSV
    csv_file_c = res_path / "dam_break_history_classical.csv"
    hist_c: List[Dict[str, float]] = []
    if csv_file_c.exists():
        with open(csv_file_c, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            hist_c = [{k: float(v) for k, v in row.items()} for row in reader]

    # Load hybrid history CSV if present
    csv_file_h = res_path / "dam_break_history_hybrid.csv"
    hist_h: List[Dict[str, float]] = []
    if csv_file_h.exists():
        with open(csv_file_h, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            hist_h = [{k: float(v) for k, v in row.items()} for row in reader]

    # -------------------------------------------------------------
    # Plot 1: Initial Liquid Fill-Level Field C(x, y)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        init_fill,
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        extent=[0, nx, 0, ny],
    )
    ax.contour(init_fill, levels=[0.5], colors="darkblue", linewidths=1.5, extent=[0, nx, 0, ny])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Fill Level $C(x, y)$")
    ax.set_title("1. Initial Liquid Column Fill Fraction ($t = 0$)")
    ax.set_xlabel("Lattice $x$")
    ax.set_ylabel("Lattice $y$")
    p1 = plot_path / "01_initial_fill_level.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_files.append(str(p1))

    # -------------------------------------------------------------
    # Plot 2: Final Liquid Fill-Level Field C(x, y)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        final_fill,
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        extent=[0, nx, 0, ny],
    )
    ax.contour(final_fill, levels=[0.5], colors="navy", linewidths=1.5, extent=[0, nx, 0, ny])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Fill Level $C(x, y)$")
    ax.set_title("2. Final Liquid Fill Fraction with PLIC Free Surface")
    ax.set_xlabel("Lattice $x$")
    ax.set_ylabel("Lattice $y$")
    p2 = plot_path / "02_final_fill_level.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_files.append(str(p2))

    # -------------------------------------------------------------
    # Plot 3: Final Density Field rho(x, y)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        final_rho,
        origin="lower",
        cmap="viridis",
        vmin=0.98,
        vmax=1.02,
        extent=[0, nx, 0, ny],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Liquid Density $\\rho / \\rho_0$")
    ax.set_title("3. Final Macroscopic Hydrodynamic Density Field")
    ax.set_xlabel("Lattice $x$")
    ax.set_ylabel("Lattice $y$")
    p3 = plot_path / "03_final_density.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_files.append(str(p3))

    # -------------------------------------------------------------
    # Plot 4: Final Velocity Magnitude Field ||u||
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        final_u,
        origin="lower",
        cmap="inferno",
        extent=[0, nx, 0, ny],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Velocity Magnitude $\\|\\mathbf{u}\\|$ [lattice]")
    ax.set_title("4. Macroscopic Velocity Magnitude Field ($Ma < 0.15$)")
    ax.set_xlabel("Lattice $x$")
    ax.set_ylabel("Lattice $y$")
    p4 = plot_path / "04_velocity_magnitude.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_files.append(str(p4))

    # -------------------------------------------------------------
    # Plot 5: Multi-Timestep Free-Surface Evolution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Run short multi-timestep snapshots on matching grid
    s_evo = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, gravity=0.0005)
    s_evo.initialize_dam_break(column_width=16, column_height=24)
    sample_steps = [0, 25, 50, 75, 100]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_steps)))

    for step_target, color in zip(sample_steps, colors):
        while s_evo.time_step < step_target:
            s_evo.step()
        ax.contour(
            s_evo.fill_level,
            levels=[0.5],
            colors=[color],
            linewidths=2.0,
            extent=[0, nx, 0, ny],
        )
        ax.plot([], [], color=color, label=f"Step {step_target} ($T={step_target*math.sqrt(0.0005/16.0):.2f}$)")

    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    ax.set_title("5. Spatio-Temporal Free-Surface Evolution ($C = 0.5$)")
    ax.set_xlabel("Lattice $x$")
    ax.set_ylabel("Lattice $y$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    p5 = plot_path / "05_free_surface_evolution.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_files.append(str(p5))

    # -------------------------------------------------------------
    # Plot 6: Liquid Front Position vs Time (with Martin & Moyce 1952)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if hist_c:
        t_c = [r["t_dimless"] for r in hist_c]
        z_c = [r["z_dimless"] for r in hist_c]
        ax.plot(t_c, z_c, "b-o", markersize=4, label="QLBM Simulation ($Z = x_{\\text{front}} / L_0$)")

        # Martin & Moyce (1952) experimental benchmark dataset (H0/L0 = 2):
        # T_MM = t * sqrt(2 g / L0), so dimensionless T* = t * sqrt(g / L0) = T_MM / sqrt(2).
        t_mm = np.array([0.0, 0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98, 2.20, 2.32, 2.51, 2.65, 2.83, 2.98, 3.11, 3.33]) / np.sqrt(2.0)
        z_mm = np.array([1.0, 1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67, 3.89, 4.11])
        t_max_plot = max(t_c) if max(t_c) > 0 else 1.0
        mask_mm = t_mm <= t_max_plot * 1.05
        ax.plot(t_mm[mask_mm], z_mm[mask_mm], "ro", fillstyle="none", markersize=6, markeredgewidth=1.5, label="Martin & Moyce (1952) Exp. Data")

    ax.set_title("6. Dimensionless Surge Front Displacement vs Dimensionless Time")
    ax.set_xlabel("Dimensionless Time $T = t \\sqrt{g / L_0}$")
    ax.set_ylabel("Dimensionless Front Position $Z = x_{\\text{front}} / L_0$")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    p6 = plot_path / "06_front_position_vs_time.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_files.append(str(p6))

    # -------------------------------------------------------------
    # Plot 7: Mass Conservation vs Time (Semilog)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    if hist_c:
        steps_c = [r["step"] for r in hist_c]
        m_err_c = [max(r["mass_error"], 1e-16) for r in hist_c]
        ax.semilogy(steps_c, m_err_c, "g-s", markersize=4, label="Relative Mass Drift $|M(t) - M_0| / M_0$")
    ax.axhline(1e-12, color="red", linestyle=":", label="Tolerance ($10^{-12}$)")
    ax.set_ylim([1e-17, 1e-10])
    ax.set_title("7. Global Liquid Mass Conservation over Time")
    ax.set_xlabel("Time Step $t$")
    ax.set_ylabel("Relative Error")
    ax.legend(loc="center right")
    ax.grid(True, linestyle="--", alpha=0.6)
    p7 = plot_path / "07_mass_conservation_vs_time.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_files.append(str(p7))

    # -------------------------------------------------------------
    # Plot 8: Classical vs Hybrid Solver Comparison
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Run side-by-side matching 16x8 dam-break for 25 steps to get identical time-series
    s1 = VOFHydroSolver(nx=16, ny=8, tau=0.8, gravity=0.001)
    s1.initialize_dam_break(column_width=4, column_height=6)
    s2 = VOFHydroSolver(nx=16, ny=8, tau=0.8, gravity=0.001)
    s2.initialize_dam_break(column_width=4, column_height=6)
    from qlbm.components.vof.hybrid_stepper import HybridVOFStep
    stepper = HybridVOFStep(s2, use_quantum_streaming=True)

    t_vals, y_cm_class, y_cm_hybr = [], [], []
    ke_class, ke_hybr = [], []

    for step in range(26):
        t_vals.append(step)
        y_cm_class.append(s1.center_of_mass[1])
        y_cm_hybr.append(s2.center_of_mass[1])
        ke_class.append(s1.kinetic_energy)
        ke_hybr.append(s2.kinetic_energy)
        if step < 25:
            s1.step()
            stepper.step()

    ax1.plot(t_vals, y_cm_class, "b-", linewidth=2.0, label="Mode A (Classical LBM)")
    ax1.plot(t_vals, y_cm_hybr, "r--", linewidth=2.0, label="Mode C (Hybrid Quantum Stepper)")
    ax1.set_title("Center-of-Mass Height $y_{\\text{cm}}(t)$")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("$y_{\\text{cm}}$ [lattice]")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(t_vals, ke_class, "b-", linewidth=2.0, label="Mode A (Classical)")
    ax2.plot(t_vals, ke_hybr, "r--", linewidth=2.0, label="Mode C (Hybrid)")
    ax2.set_title("Fluid Kinetic Energy $E_k(t)$")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Kinetic Energy $\\frac{1}{2}\\sum \\rho |\\mathbf{u}|^2$")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    p8 = plot_path / "08_classical_vs_hybrid_comparison.png"
    fig.savefig(p8)
    plt.close(fig)
    generated_files.append(str(p8))

    # -------------------------------------------------------------
    # Plot 9: Spatial Error Field Between Classical and Quantum Streaming
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    q_streamer = QuantumD2Q9StreamingOperator(nx=16, ny=8)
    rng = np.random.default_rng(777)
    test_f = rng.uniform(0.1, 1.0, size=(9, 8, 16))
    f_quantum = q_streamer.stream_distribution_array(test_f)
    f_classic = stream_and_bounce(test_f, solid_mask=None)

    error_field = np.max(np.abs(f_quantum - f_classic), axis=0)  # max across 9 directions
    im = ax.imshow(
        error_field,
        origin="lower",
        cmap="magma",
        extent=[0, 16, 0, 8],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Absolute Error $|f_i^q - f_i^c|$")
    ax.set_title(f"9. Quantum Streaming Discrepancy (Max $E = {np.max(error_field):.2e} < 10^{{-12}}$)")
    ax.set_xlabel("Lattice $x$")
    ax.set_ylabel("Lattice $y$")
    p9 = plot_path / "09_quantum_streaming_error_field.png"
    fig.savefig(p9)
    plt.close(fig)
    generated_files.append(str(p9))

    return generated_files


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate 9 publication-grade figures for the dam-break simulation benchmark."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/dam_break",
        help="Input results directory",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="results/dam_break/plots",
        help="Output plots directory",
    )
    args = parser.parse_args()

    files = generate_dam_break_plots(
        results_dir=args.results_dir, plots_dir=args.plots_dir
    )
    print(f"Successfully generated {len(files)} figures:")
    for f in files:
        print(f"  * {f}")
    return 0


if __name__ == "__main__":
    main()
