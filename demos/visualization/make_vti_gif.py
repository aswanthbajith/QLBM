#!/usr/bin/env python3

import sys
from pathlib import Path

import vtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_vti(filename):
    """Read a VTI file and return its first point-data array as a 2D field."""

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(filename))
    reader.Update()

    data = reader.GetOutput()

    if data.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Could not read VTI file: {filename}")

    point_data = data.GetPointData()

    if point_data.GetNumberOfArrays() == 0:
        raise RuntimeError(f"No point-data arrays found: {filename}")

    # Use the first point-data array automatically
    array = point_data.GetArray(0)

    if array.GetNumberOfComponents() != 1:
        raise RuntimeError(
            f"First array is not scalar: "
            f"{array.GetName()} has "
            f"{array.GetNumberOfComponents()} components"
        )

    values = np.array([
        array.GetTuple1(i)
        for i in range(array.GetNumberOfTuples())
    ])

    nx, ny, nz = data.GetDimensions()

    if nz != 1:
        raise RuntimeError(
            f"This script expects a 2D VTI file, got dimensions "
            f"{data.GetDimensions()}"
        )

    return values.reshape((ny, nx)), array.GetName()


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print()
        print("  python make_vti_gif.py <result_folder>")
        print()
        print("Example:")
        print(
            "  python make_vti_gif.py "
            "demos/simulation/qlbm-output/"
            "ms-4x4-16x16-1-obstacle-qiskit/paraview"
        )
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser().resolve()

    if not folder.is_dir():
        print(f"ERROR: Folder does not exist:")
        print(folder)
        sys.exit(1)

    # Find all VTI files
    files = sorted(folder.glob("step_*.vti"))

    if not files:
        print("ERROR: No step_*.vti files found in:")
        print(folder)
        sys.exit(1)

    print(f"\nFound {len(files)} VTI files")
    print(f"Folder: {folder}\n")

    # Read all fields
    fields = []
    step_numbers = []

    array_name = None

    for filename in files:

        print(f"Reading {filename.name}")

        field, current_array_name = read_vti(filename)

        fields.append(field)

        if array_name is None:
            array_name = current_array_name

        # Extract timestep number
        try:
            step = int(filename.stem.split("_")[1])
        except (IndexError, ValueError):
            step = len(fields) - 1

        step_numbers.append(step)

    fields = np.array(fields)

    print("\nData shape:", fields.shape)
    print("Grid size:", fields.shape[2], "x", fields.shape[1])
    print("Scalar:", array_name)

    # Global colour scale
    vmin = np.nanmin(fields)
    vmax = np.nanmax(fields)

    print("Minimum:", vmin)
    print("Maximum:", vmax)

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        fields[0],
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(array_name)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = ax.set_title(
        f"{folder.name} — Step {step_numbers[0]}"
    )

    # ---------------------------------------------------------
    # Animation
    # ---------------------------------------------------------

    def update(frame):

        image.set_data(fields[frame])

        title.set_text(
            f"{folder.name} — Step {step_numbers[frame]}"
        )

        return image, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(fields),
        interval=300,
        repeat=True,
    )

    # Save GIF in the same folder as the VTI files
    output_file = folder / "qlbm_evolution.gif"

    print("\nCreating GIF...")
    print(output_file)

    animation.save(
        output_file,
        writer="pillow",
        fps=4,
    )

    print("\nDONE!")
    print("GIF:", output_file)

    plt.close(fig)


if __name__ == "__main__":
    main()
