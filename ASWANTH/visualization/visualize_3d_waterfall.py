"""Visualization and VTK Export Utilities for 3D Waterfall & Dam-Break Simulations.

Exports simulation snapshots to structured VTK ImageData (.vti) files for ParaView
and generates publication-quality 2D/3D slice visualizations with velocity vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qlbm.classical.d3q19 import CS_SQ
from qlbm.classical.vof_3d import CellType


def export_vti(
    filepath: Union[str, Path],
    fill_level: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    solid_mask: np.ndarray,
    cell_types: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    r"""Export 3D flow fields to a standard VTK XML ImageData (.vti) file.

    Fields exported:
      - FillFraction (Float32): VOF volume fraction :math:`C \in [0, 1]`
      - Density (Float32): Macroscopic fluid density :math:`\rho`
      - Pressure (Float32): Hydrodynamic pressure :math:`P = c_s^2 \rho`
      - Speed (Float32): Velocity magnitude :math:`\|\mathbf{u}\|`
      - Solid (Int32): Solid obstacle mask (1 for solid, 0 for fluid/gas)
      - Velocity (Float32, 3-component): Velocity vector :math:`(u_x, u_y, u_z)`

    Parameters
    ----------
    filepath : str or Path
        Destination .vti file path.
    fill_level : np.ndarray
        Volume fraction array of shape `(nz, ny, nx)`.
    rho : np.ndarray
        Density field of shape `(nz, ny, nx)`.
    u : np.ndarray
        Velocity vector field of shape `(3, nz, ny, nx)`.
    solid_mask : np.ndarray
        Solid obstacle mask of shape `(nz, ny, nx)`.
    cell_types : np.ndarray, optional
        Discrete cell type classification array of shape `(nz, ny, nx)`.
    spacing : tuple of 3 floats, optional
        Voxel spacing along (x, y, z). Default is (1.0, 1.0, 1.0).
    origin : tuple of 3 floats, optional
        Grid origin coordinate. Default is (0.0, 0.0, 0.0).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = fill_level.shape
    pressure = CS_SQ * rho
    speed = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    # In VTK ImageData, memory indexing is x fastest, then y, then z.
    # Arrays with shape (nz, ny, nx) flattened in C-order naturally map to (z, y, x).
    flat_fill = fill_level.flatten()
    flat_rho = rho.flatten()
    flat_press = pressure.flatten()
    flat_speed = speed.flatten()
    flat_solid = solid_mask.astype(np.int32).flatten()
    flat_u = np.stack(
        [u[0].flatten(), u[1].flatten(), u[2].flatten()], axis=1
    ).flatten()

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">',
        f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" Origin="{origin[0]} {origin[1]} {origin[2]}" Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">',
        f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">',
        '      <PointData Scalars="FillFraction" Vectors="Velocity">',
        '        <DataArray type="Float32" Name="FillFraction" NumberOfComponents="1" format="ascii">',
        " " + " ".join(f"{v:.4f}" for v in flat_fill),
        "        </DataArray>",
        '        <DataArray type="Float32" Name="Density" NumberOfComponents="1" format="ascii">',
        " " + " ".join(f"{v:.4f}" for v in flat_rho),
        "        </DataArray>",
        '        <DataArray type="Float32" Name="Pressure" NumberOfComponents="1" format="ascii">',
        " " + " ".join(f"{v:.4f}" for v in flat_press),
        "        </DataArray>",
        '        <DataArray type="Float32" Name="Speed" NumberOfComponents="1" format="ascii">',
        " " + " ".join(f"{v:.4f}" for v in flat_speed),
        "        </DataArray>",
        '        <DataArray type="Int32" Name="Solid" NumberOfComponents="1" format="ascii">',
        " " + " ".join(str(v) for v in flat_solid),
        "        </DataArray>",
    ]

    if cell_types is not None:
        flat_types = cell_types.astype(np.int32).flatten()
        lines.extend([
            '        <DataArray type="Int32" Name="CellType" NumberOfComponents="1" format="ascii">',
            " " + " ".join(str(v) for v in flat_types),
            "        </DataArray>",
        ])

    lines.extend([
        '        <DataArray type="Float32" Name="Velocity" NumberOfComponents="3" format="ascii">',
        " " + " ".join(f"{v:.4f}" for v in flat_u),
        "        </DataArray>",
        "      </PointData>",
        "    </Piece>",
        "  </ImageData>",
        "</VTKFile>",
    ])

    filepath.write_text("\n".join(lines), encoding="utf-8")


def plot_waterfall_slice(
    npz_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    y_slice: Optional[int] = None,
    title: Optional[str] = None,
) -> Path:
    """Generate a publication-grade vertical (x-z) slice plot of the waterfall simulation.

    Parameters
    ----------
    npz_path : str or Path
        Path to input NPZ snapshot.
    save_path : str or Path, optional
        Destination image path (.png).
    y_slice : int, optional
        Y-plane index (defaults to domain center ny // 2).
    title : str, optional
        Custom figure title.

    Returns
    -------
    Path
        Saved image path.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    fill_level = data["fill_level"]
    u = data["u"]
    solid_mask = data["solid_mask"]
    step = int(data.get("step", 0))

    nz, ny, nx = fill_level.shape
    if y_slice is None:
        y_slice = ny // 2

    # Extract 2D slice along y
    c_slice = fill_level[:, y_slice, :]
    ux_slice = u[0, :, y_slice, :]
    uz_slice = u[2, :, y_slice, :]
    solid_slice = solid_mask[:, y_slice, :]
    speed_slice = np.sqrt(ux_slice**2 + uz_slice**2)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # 1. Background fill fraction contour
    im = ax.imshow(
        c_slice,
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        extent=(-0.5, nx - 0.5, -0.5, nz - 0.5),
        alpha=0.85,
    )

    # 2. Overlay solid obstacles (shelf, cliff, floor, walls)
    solid_display = np.ma.masked_where(~solid_slice, solid_slice)
    ax.imshow(
        solid_display,
        origin="lower",
        cmap="gray_r",
        vmin=0,
        vmax=1,
        extent=(-0.5, nx - 0.5, -0.5, nz - 0.5),
        alpha=0.9,
    )

    # 3. Free surface interface contour (C = 0.5)
    if np.any((c_slice > 0.1) & (~solid_slice)):
        try:
            ax.contour(
                c_slice,
                levels=[0.5],
                colors=["#002b49"],
                linewidths=2.0,
                origin="lower",
                extent=(-0.5, nx - 0.5, -0.5, nz - 0.5),
            )
        except Exception:
            pass

    # 4. Velocity quiver vectors on active fluid nodes
    fluid_mask = (c_slice > 0.05) & (~solid_slice)
    if np.any(fluid_mask):
        z_grid, x_grid = np.mgrid[0:nz, 0:nx]
        q_scale = max(0.1, np.max(speed_slice)) * 8.0
        ax.quiver(
            x_grid[fluid_mask],
            z_grid[fluid_mask],
            ux_slice[fluid_mask],
            uz_slice[fluid_mask],
            speed_slice[fluid_mask],
            cmap="autumn",
            scale=q_scale,
            width=0.007,
            headwidth=4,
        )

    # 5. Diagnostic telemetry badge
    max_downward = abs(min(0.0, float(np.min(uz_slice)))) if np.any(fluid_mask) else 0.0
    max_spd = float(np.max(speed_slice[fluid_mask])) if np.any(fluid_mask) else 0.0
    info_text = (
        f"Time Step: {step}\n"
        f"Max |u_z|:  {max_downward:.4f}\n"
        f"Max Speed: {max_spd:.4f}"
    )
    ax.text(
        0.03,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#aaaaaa"),
    )

    ax.set_xlabel("X (Lattice Nodes)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Z (Vertical Height)", fontsize=11, fontweight="bold")
    plot_title = (
        title
        or f"3D Gravity-Driven Waterfall (Y-Slice={y_slice}, Step {step}) - Plunging Flow"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Water Fill Fraction $C(x, z)$", fontsize=10)

    if save_path is None:
        save_path = npz_path.parent / f"waterfall_slice_step_{step:04d}.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def generate_waterfall_gif(
    images_or_dir: Union[str, Path, list],
    output_gif_path: Union[str, Path],
    duration_ms: int = 250,
    loop: int = 0,
) -> Path:
    """Generate an animated GIF from a sequence of PNG slice plots or snapshot directory.

    Parameters
    ----------
    images_or_dir : str, Path, or list of paths
        Directory containing PNG slice files, or list of image paths.
    output_gif_path : str or Path
        Destination .gif file path.
    duration_ms : int, optional
        Frame duration in milliseconds (default 250 ms = 4 fps).
    loop : int, optional
        Number of loops (0 for infinite looping).

    Returns
    -------
    Path
        Path to the created animated GIF.
    """
    from PIL import Image

    output_gif_path = Path(output_gif_path)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(images_or_dir, (str, Path)):
        p = Path(images_or_dir)
        if p.is_dir():
            png_files = sorted(p.glob("*_slice.png"))
            if not png_files:
                png_files = sorted(p.glob("*.png"))
        else:
            png_files = [p]
    else:
        png_files = [Path(img) for img in images_or_dir]

    if not png_files:
        raise ValueError(f"No PNG images found to compile into GIF at {images_or_dir}")

    frames = [Image.open(f) for f in png_files]
    first_frame = frames[0]

    first_frame.save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )

    return output_gif_path


def main() -> None:
    """CLI entrypoint for converting NPZ snapshots to VTK, images, and animated GIF."""
    parser = argparse.ArgumentParser(
        description="Convert 3D waterfall NPZ snapshots to VTK (.vti), PNG slice plots, and animated GIF."
    )
    parser.add_argument("input_path", type=str, help="Path to NPZ file or directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for exported files",
    )
    parser.add_argument(
        "--y-slice",
        type=int,
        default=None,
        help="Y-slice index for 2D plots (default: center plane)",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Generate animated GIF from slice plots",
    )
    parser.add_argument(
        "--gif-out",
        type=str,
        default=None,
        help="Custom output path for generated GIF",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second for animated GIF (default: 4.0)",
    )
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent

    files = [in_path] if in_path.is_file() else sorted(in_path.glob("*.npz"))
    if not files:
        print(f"No .npz files found at {in_path}")
        sys.exit(1)

    print(f"Processing {len(files)} snapshot(s)...")
    png_paths = []
    for f in files:
        data = np.load(f)
        stem = f.stem
        vti_dest = out_dir / f"{stem}.vti"
        export_vti(
            filepath=vti_dest,
            fill_level=data["fill_level"],
            rho=data["rho"],
            u=data["u"],
            solid_mask=data["solid_mask"],
            cell_types=data.get("cell_types"),
        )
        png_dest = out_dir / f"{stem}_slice.png"
        plot_waterfall_slice(f, save_path=png_dest, y_slice=args.y_slice)
        png_paths.append(png_dest)
        print(f"  -> Exported: {vti_dest.name} & {png_dest.name}")

    if args.gif and png_paths:
        duration_ms = int(1000.0 / max(0.1, args.fps))
        gif_dest = Path(args.gif_out) if args.gif_out else out_dir / "waterfall_3d_animation.gif"
        generate_waterfall_gif(png_paths, gif_dest, duration_ms=duration_ms)
        print(f"  -> Animated GIF generated: {gif_dest}")


if __name__ == "__main__":
    main()
