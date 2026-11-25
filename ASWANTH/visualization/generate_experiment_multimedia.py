"""Production Multimedia Generator for 3D Multiphase Waterfall Experiment.

Generates:
1. Multi-Angle Perspectives GIF ('waterfall_multi_angle_experiment.gif'):
   - Side elevation (X-Z)
   - Top-down plan view (X-Y)
   - 3D Isometric perspective view
2. Multi-Property Physical Field GIF ('waterfall_physical_properties.gif'):
   - Interface Phase-Field phi
   - Velocity magnitude ||u|| and quiver vectors
   - Vertical plunge momentum Jz = rho * uz
   - Dynamic pressure P = cs^2 * rho
3. High-resolution keyframe snapshots across all simulation phases.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C, CS_SQ, WEIGHTS
from research.quantum_fluid_evolution.phase_field.multiphase_phase_field_solver_3d import (
    MultiphasePhaseFieldSolver3D,
)


def run_and_generate_multimedia(
    output_dir: str = "/home/aswa/.gemini/antigravity/brain/f58571c4-4fa6-45df-b710-0b6735ca7509",
    nx: int = 32,
    ny: int = 16,
    nz: int = 28,
    num_steps: int = 125,
    sample_stride: int = 3,
) -> None:
    """Execute complete 3D waterfall simulation and render multi-angle & multi-property GIFs."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    temp_frames_dir = out_path / "temp_frames"
    temp_frames_dir.mkdir(parents=True, exist_ok=True)

    z_shelf = nz // 2  # 14
    x_cliff = 12
    y_mid = ny // 2    # 8

    # 1. Geometry Setup
    solid = np.zeros((nz, ny, nx), dtype=bool)
    solid[0, :, :] = True  # Lower basin floor
    solid[: z_shelf + 1, :, : x_cliff + 1] = True  # Shelf and cliff face
    solid[: 5, :, (x_cliff + 5):(x_cliff + 10)] = True  # Basin obstacle

    solver = MultiphasePhaseFieldSolver3D(
        nx=nx, ny=ny, nz=nz,
        tau_f=0.8, tau_phi=0.8,
        sigma=0.015, g=(0.0, 0.0, -0.015),
        solid_mask=solid,
        interface_width=1.0,
    )
    solver.initialize_waterfall_reservoir(z_shelf=z_shelf, x_cliff=x_cliff)

    # Initial forward gate release velocity
    gate_u = 0.08
    for i in range(19):
        solver.f[i] += WEIGHTS[i] * 1.0 * (C[i, 0] * gate_u / CS_SQ)

    print("=" * 80)
    print("  3D WATERFALL MULTIMEDIA GENERATION")
    print(f"  Grid: {nx}x{ny}x{nz} | Steps: {num_steps} | Stride: {sample_stride}")
    print(f"  Shelf: z <= {z_shelf}, x <= {x_cliff} | Floor: z = 0 | Obstacle: x in [{x_cliff+5}, {x_cliff+10}]")
    print("=" * 80)

    angle_frames = []
    prop_frames = []

    t_start = time.perf_counter()

    for s in range(0, num_steps + 1):
        if s > 0:
            step_res = solver.step()
            phi = step_res["phi"]
            rho = step_res["rho"]
            u = step_res["u"]
        else:
            phi = solver.compute_phi()
            rho, u = solver.compute_velocity()

        if s % sample_stride == 0 or s == num_steps:
            # Metrics
            air_liquid = int(np.sum(phi[:, :, x_cliff + 1:] > 0.0))
            max_spd = float(np.max(np.sqrt(np.sum(u**2, axis=0))))
            min_uz = float(np.min(u[2]))

            # -----------------------------------------------------------------
            # 1. RENDER MULTI-ANGLE FRAME
            # -----------------------------------------------------------------
            fig1 = plt.figure(figsize=(14, 9), dpi=120)
            fig1.patch.set_facecolor("#0b0f19")

            # Panel A: Side Elevation (X-Z Slice at y=mid)
            ax1 = fig1.add_subplot(2, 2, 1)
            ax1.set_facecolor("#0f172a")
            phi_xz = phi[:, y_mid, :]
            ux_xz = u[0, :, y_mid, :]
            uz_xz = u[2, :, y_mid, :]
            spd_xz = np.sqrt(ux_xz**2 + uz_xz**2)
            solid_xz = solid[:, y_mid, :]

            # Background water fill contour
            im1 = ax1.imshow(
                phi_xz, origin="lower", cmap="Blues_r", vmin=-1.0, vmax=1.0,
                extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.85
            )
            # Solid walls
            ax1.imshow(
                np.ma.masked_where(~solid_xz, solid_xz), origin="lower",
                cmap="gray_r", vmin=0, vmax=1,
                extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.95
            )
            # Velocity quivers on liquid
            fluid_mask_xz = (phi_xz > -0.2) & (~solid_xz)
            if np.any(fluid_mask_xz):
                z_g, x_g = np.mgrid[0:nz, 0:nx]
                ax1.quiver(
                    x_g[fluid_mask_xz], z_g[fluid_mask_xz],
                    ux_xz[fluid_mask_xz], uz_xz[fluid_mask_xz],
                    color="#f59e0b", scale=max(0.2, max_spd) * 12.0,
                    width=0.006, headwidth=3.5, alpha=0.9
                )
            ax1.set_title("Side Elevation View (X-Z Center Slice)", color="white", fontsize=11, fontweight="bold")
            ax1.set_xlabel("X (Flow Distance)", color="#94a3b8", fontsize=9)
            ax1.set_ylabel("Z (Vertical Height)", color="#94a3b8", fontsize=9)
            ax1.tick_params(colors="#94a3b8", labelsize=8)
            ax1.grid(color="#334155", linestyle=":", alpha=0.5)

            # Panel B: Top-Down Plan View (X-Y Slice at mid basin z=4)
            ax2 = fig1.add_subplot(2, 2, 2)
            ax2.set_facecolor("#0f172a")
            # Projected depth of water: sum of liquid cells along vertical Z
            liquid_depth = np.sum(phi > 0.0, axis=0)  # (ny, nx)
            solid_xy = solid[1, :, :]  # Floor level solids

            im2 = ax2.imshow(
                liquid_depth, origin="lower", cmap="YlGnBu",
                extent=(-0.5, nx - 0.5, -0.5, ny - 0.5), alpha=0.9
            )
            ax2.set_title("Top-Down Plan View (X-Y Liquid Column Depth)", color="white", fontsize=11, fontweight="bold")
            ax2.set_xlabel("X (Flow Distance)", color="#94a3b8", fontsize=9)
            ax2.set_ylabel("Y (Transverse Width)", color="#94a3b8", fontsize=9)
            ax2.tick_params(colors="#94a3b8", labelsize=8)
            ax2.axvline(x=x_cliff, color="#ef4444", linestyle="--", linewidth=1.5, label="Cliff Edge")
            ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8)

            # Panel C: 3D Perspective Isometric Surface View
            ax3 = fig1.add_subplot(2, 1, 2, projection="3d")
            ax3.set_facecolor("#0b0f19")
            ax3.xaxis.pane.set_facecolor("#0f172a")
            ax3.yaxis.pane.set_facecolor("#0f172a")
            ax3.zaxis.pane.set_facecolor("#0b0f19")

            # Sample liquid nodes for 3D visualization
            liquid_pts = np.where((phi > 0.0) & (~solid))
            if len(liquid_pts[0]) > 0:
                p_z, p_y, p_x = liquid_pts
                sc = ax3.scatter(
                    p_x, p_y, p_z, c=p_z, cmap="cool", s=22, alpha=0.75, edgecolors="none"
                )

            # Render solid shelf as wireframe outline
            shelf_x = np.array([0, x_cliff, x_cliff, 0, 0])
            shelf_y = np.array([0, 0, ny-1, ny-1, 0])
            ax3.plot(shelf_x, shelf_y, z_shelf, color="#64748b", linewidth=1.8, linestyle="--")

            # Render obstacle as box outline
            obs_x0, obs_x1 = x_cliff + 5, x_cliff + 10
            ax3.plot([obs_x0, obs_x1, obs_x1, obs_x0, obs_x0], [0, 0, ny-1, ny-1, 0], 5, color="#f43f5e", linewidth=1.5)

            ax3.set_xlim(0, nx)
            ax3.set_ylim(0, ny)
            ax3.set_zlim(0, nz)
            ax3.view_init(elev=24, azim=-55)
            ax3.set_title("3D Isometric Perspective View", color="white", fontsize=11, fontweight="bold")
            ax3.set_xlabel("X", color="#94a3b8", fontsize=9)
            ax3.set_ylabel("Y", color="#94a3b8", fontsize=9)
            ax3.set_zlabel("Z", color="#94a3b8", fontsize=9)
            ax3.tick_params(colors="#94a3b8", labelsize=8)

            # HUD Badge
            hud = (
                f"Step: {s:03d}/{num_steps} | Air Falling Vol: {air_liquid} cells | "
                f"Max Speed: {max_spd:.3f} | Downward uz: {min_uz:.3f}"
            )
            fig1.suptitle(f"3D Multiphase Waterfall Experiment [Multi-Angle Evolution]\n{hud}",
                          color="#38bdf8", fontsize=12, fontweight="bold")

            plt.tight_layout()
            frame_angle_path = temp_frames_dir / f"angle_frame_{s:04d}.png"
            fig1.savefig(frame_angle_path, facecolor=fig1.get_facecolor(), edgecolor="none")
            plt.close(fig1)
            angle_frames.append(frame_angle_path)

            # -----------------------------------------------------------------
            # 2. RENDER MULTI-PROPERTY FRAME
            # -----------------------------------------------------------------
            fig2, axs = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
            fig2.patch.set_facecolor("#0b0f19")

            # Property 1: Interface Order Parameter phi
            ax_p1 = axs[0, 0]
            ax_p1.set_facecolor("#0f172a")
            im_p1 = ax_p1.imshow(phi_xz, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                                 extent=(-0.5, nx - 0.5, -0.5, nz - 0.5))
            ax_p1.imshow(np.ma.masked_where(~solid_xz, solid_xz), origin="lower", cmap="gray_r",
                         vmin=0, vmax=1, extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.95)
            ax_p1.contour(phi_xz, levels=[0.0], colors=["#f8fafc"], linewidths=1.8, origin="lower",
                          extent=(-0.5, nx - 0.5, -0.5, nz - 0.5))
            ax_p1.set_title("1. Water-Air Interface (Phase-Field phi)", color="white", fontsize=11, fontweight="bold")
            fig2.colorbar(im_p1, ax=ax_p1, shrink=0.7, pad=0.02).set_label("phi (-1: Air, +1: Water)", color="white", fontsize=8)

            # Property 2: Velocity Speed Magnitude ||u||
            ax_p2 = axs[0, 1]
            ax_p2.set_facecolor("#0f172a")
            im_p2 = ax_p2.imshow(spd_xz, origin="lower", cmap="inferno", vmin=0.0, vmax=0.35,
                                 extent=(-0.5, nx - 0.5, -0.5, nz - 0.5))
            ax_p2.imshow(np.ma.masked_where(~solid_xz, solid_xz), origin="lower", cmap="gray_r",
                         vmin=0, vmax=1, extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.95)
            if np.any(fluid_mask_xz):
                ax_p2.quiver(
                    x_g[fluid_mask_xz], z_g[fluid_mask_xz],
                    ux_xz[fluid_mask_xz], uz_xz[fluid_mask_xz],
                    color="#38bdf8", scale=max(0.2, max_spd) * 12.0,
                    width=0.006, headwidth=3.5, alpha=0.9
                )
            ax_p2.set_title("2. Flow Speed Magnitude ||u|| & Quivers", color="white", fontsize=11, fontweight="bold")
            fig2.colorbar(im_p2, ax=ax_p2, shrink=0.7, pad=0.02).set_label("Speed (lattice units)", color="white", fontsize=8)

            # Property 3: Vertical Plunge Momentum Jz = rho * uz
            ax_p3 = axs[1, 0]
            ax_p3.set_facecolor("#0f172a")
            jz_xz = (rho * u[2])[:, y_mid, :]
            im_p3 = ax_p3.imshow(jz_xz, origin="lower", cmap="coolwarm", vmin=-0.30, vmax=0.10,
                                 extent=(-0.5, nx - 0.5, -0.5, nz - 0.5))
            ax_p3.imshow(np.ma.masked_where(~solid_xz, solid_xz), origin="lower", cmap="gray_r",
                         vmin=0, vmax=1, extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.95)
            ax_p3.set_title("3. Vertical Momentum Jz = rho * uz (Gravitational Plunge)", color="white", fontsize=11, fontweight="bold")
            fig2.colorbar(im_p3, ax=ax_p3, shrink=0.7, pad=0.02).set_label("Jz (Blue: Downward)", color="white", fontsize=8)

            # Property 4: Hydrodynamic Pressure P = cs^2 * rho
            ax_p4 = axs[1, 1]
            ax_p4.set_facecolor("#0f172a")
            press_xz = (CS_SQ * rho)[:, y_mid, :]
            im_p4 = ax_p4.imshow(press_xz, origin="lower", cmap="plasma", vmin=0.30, vmax=0.38,
                                 extent=(-0.5, nx - 0.5, -0.5, nz - 0.5))
            ax_p4.imshow(np.ma.masked_where(~solid_xz, solid_xz), origin="lower", cmap="gray_r",
                         vmin=0, vmax=1, extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.95)
            ax_p4.set_title("4. Hydrodynamic Pressure P = cs^2 * rho", color="white", fontsize=11, fontweight="bold")
            fig2.colorbar(im_p4, ax=ax_p4, shrink=0.7, pad=0.02).set_label("Pressure", color="white", fontsize=8)

            for ax in axs.flat:
                ax.set_xlabel("X", color="#94a3b8", fontsize=8)
                ax.set_ylabel("Z", color="#94a3b8", fontsize=8)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                ax.grid(color="#334155", linestyle=":", alpha=0.4)

            fig2.suptitle(f"3D Multiphase Waterfall Physics [Multi-Property Fields] | Step {s:03d}/{num_steps}",
                          color="#38bdf8", fontsize=12, fontweight="bold")

            plt.tight_layout()
            frame_prop_path = temp_frames_dir / f"prop_frame_{s:04d}.png"
            fig2.savefig(frame_prop_path, facecolor=fig2.get_facecolor(), edgecolor="none")
            plt.close(fig2)
            prop_frames.append(frame_prop_path)

            # Save keyframe PNGs to main artifacts directory
            if s in (0, 30, 60, 90, 120):
                keyframe_name = out_path / f"waterfall_keyframe_step_{s:04d}.png"
                fig_key, ax_key = plt.subplots(figsize=(9, 6), dpi=140)
                fig_key.patch.set_facecolor("#0b0f19")
                ax_key.set_facecolor("#0f172a")
                ax_key.imshow(phi_xz, origin="lower", cmap="Blues_r", vmin=-1.0, vmax=1.0,
                              extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.85)
                ax_key.imshow(np.ma.masked_where(~solid_xz, solid_xz), origin="lower", cmap="gray_r",
                              vmin=0, vmax=1, extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), alpha=0.95)
                if np.any(fluid_mask_xz):
                    ax_key.quiver(
                        x_g[fluid_mask_xz], z_g[fluid_mask_xz],
                        ux_xz[fluid_mask_xz], uz_xz[fluid_mask_xz],
                        color="#f59e0b", scale=max(0.2, max_spd) * 10.0,
                        width=0.007, headwidth=4, alpha=0.9
                    )
                ax_key.set_title(f"3D Waterfall Keyframe - Step {s:03d} (Physical Flow Evolution)",
                                 color="white", fontsize=12, fontweight="bold")
                ax_key.set_xlabel("X (Flow Distance)", color="#94a3b8", fontsize=10)
                ax_key.set_ylabel("Z (Vertical Height)", color="#94a3b8", fontsize=10)
                ax_key.tick_params(colors="#94a3b8", labelsize=9)
                plt.tight_layout()
                fig_key.savefig(keyframe_name, facecolor=fig_key.get_facecolor(), edgecolor="none")
                plt.close(fig_key)

            print(f"  Rendered Frame Step {s:03d}/{num_steps} -> Air Liquid: {air_liquid} cells, uz: {min_uz:.3f}")

    print("\n--- Compiling Animated GIFs ---")
    gif_duration = 180  # ms per frame

    # Compile GIF 1: Multi-Angle Perspectives
    gif1_path = out_path / "waterfall_multi_angle_experiment.gif"
    frames1 = [Image.open(f) for f in angle_frames]
    frames1[0].save(
        gif1_path, save_all=True, append_images=frames1[1:],
        duration=gif_duration, loop=0, optimize=True
    )
    print(f"  [Created GIF 1] -> {gif1_path} ({len(frames1)} frames)")

    # Compile GIF 2: Multi-Physical Properties
    gif2_path = out_path / "waterfall_physical_properties.gif"
    frames2 = [Image.open(f) for f in prop_frames]
    frames2[0].save(
        gif2_path, save_all=True, append_images=frames2[1:],
        duration=gif_duration, loop=0, optimize=True
    )
    print(f"  [Created GIF 2] -> {gif2_path} ({len(frames2)} frames)")

    total_time = time.perf_counter() - t_start
    print(f"\nAll multimedia artifacts created successfully in {total_time:.2f} s!")


if __name__ == "__main__":
    run_and_generate_multimedia()
