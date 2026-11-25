"""Run ten collisionless D2Q9 dam releases and export their animations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from qiskit_aer import AerSimulator
from vtk import vtkXMLImageDataReader
from vtk.util.numpy_support import vtk_to_numpy

from qlbm.components import (
    ABDiscreteUniformInitialConditions,
    ABGridMeasurement,
    CQLBM,
    EmptyPrimitive,
)
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import ABLattice

SIZE = 16
SHOTS = 4096
STEPS = 8
EAST = (1, 5, 8)
NORTH = (2, 5, 6)
SOUTH = (4, 7, 8)

# Each reservoir is a rectangle with power-of-two side lengths and aligned bounds.
# Walls are inclusive (x_min, x_max, y_min, y_max) rectangles.
CASES = {
    "01-narrow-gate": (
        "Narrow central gate", (4, 0, 4, 16), EAST,
        [(9, 10, 0, 6), (9, 10, 9, 15)],
    ),
    "02-full-collapse": (
        "Full dam collapse across the channel", (4, 4, 4, 4), EAST,
        [],
    ),
    "03-side-breach": (
        "Breach at one end", (4, 0, 4, 16), EAST,
        [(9, 10, 4, 15)],
    ),
    "04-twin-breaches": (
        "Twin breaching jets", (4, 0, 4, 16), EAST,
        [(9, 10, 0, 1), (9, 10, 5, 10), (9, 10, 14, 15)],
    ),
    "05-northbound-release": (
        "Northbound reservoir release", (0, 4, 16, 4), NORTH,
        [(0, 4, 9, 10), (11, 15, 9, 10)],
    ),
    "06-southbound-release": (
        "Southbound release through an offset breach", (8, 8, 8, 4), SOUTH,
        [(0, 8, 5, 6), (13, 15, 5, 6)],
    ),
    "07-staggered-dams": (
        "Staggered dams", (4, 0, 4, 16), EAST,
        [(8, 9, 0, 3), (8, 9, 10, 15), (12, 13, 0, 5), (12, 13, 12, 15)],
    ),
    "08-island-split": (
        "Breach flow splits around an island", (4, 0, 4, 16), EAST,
        [(8, 9, 0, 3), (8, 9, 12, 15), (12, 13, 6, 9)],
    ),
    "09-narrow-canal": (
        "Dam release into a straight canal", (4, 4, 4, 4), EAST,
        [(8, 9, 0, 3), (8, 9, 9, 15), (10, 15, 2, 3), (10, 15, 9, 10)],
    ),
    "10-offset-outlet": (
        "Release into a basin with an offset outlet", (4, 8, 4, 4), EAST,
        [(8, 9, 0, 6), (8, 9, 13, 15), (12, 13, 5, 13)],
    ),
}


def simulate(output, reservoir, velocities, walls):
    x, y, width, height = reservoir
    lattice = ABLattice({
        "lattice": {"dim": {"x": SIZE, "y": SIZE}, "velocities": "d2q9"},
        "geometry": [
            {"shape": "cuboid", "x": [x0, x1], "y": [y0, y1], "boundary": "bounceback"}
            for x0, x1, y0, y1 in walls
        ],
    })
    initial = ABDiscreteUniformInitialConditions(
        lattice, list(velocities),
        (list(range(width.bit_length() - 1)), list(range(height.bit_length() - 1))),
    ).circuit
    for axis, origin in enumerate((x, y)):
        for bit, qubit in enumerate(lattice.grid_index(axis)):
            if origin & (1 << bit):
                initial.x(qubit)

    config = SimulationConfig(
        initial_conditions=initial,
        algorithm=CQLBM(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=ABGridMeasurement(lattice),
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        statevector_sampling=True,
        execution_backend=AerSimulator(method="statevector", max_parallel_threads=2),
        sampling_backend=AerSimulator(
            method="statevector", max_parallel_threads=2, seed_simulator=42,
        ),
    )
    config.prepare_for_simulation()
    QiskitRunner(config, lattice).run(
        STEPS, SHOTS, str(output), statevector_snapshots=True,
    )


def animate(output, title, walls):
    frames = []
    for step in range(STEPS + 1):
        reader = vtkXMLImageDataReader()
        reader.SetFileName(str(output / "paraview" / f"step_{step:03d}.vti"))
        reader.Update()
        counts = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
        frames.append(counts.reshape(SIZE, SIZE) / SHOTS)

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(
        frames[0], origin="lower", cmap="Blues", interpolation="nearest",
        vmin=0, vmax=max(frame.max() for frame in frames),
    )
    for x0, x1, y0, y1 in walls:
        ax.add_patch(Rectangle(
            (x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
            facecolor="#4b5563", edgecolor="#1f2937",
        ))
    ax.set(xlabel="x", ylabel="y", xticks=range(0, SIZE, 3), yticks=range(0, SIZE, 3))
    fig.colorbar(image, ax=ax, label="Measured position probability")

    def draw(step):
        image.set_data(frames[step])
        ax.set_title(f"{title} · step {step:02d}")

    animation = FuncAnimation(fig, draw, frames=len(frames))
    animation.save(output / "flow.gif", writer=PillowWriter(fps=2), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", metavar="CASE", help="Case IDs. Default: all.")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "qlbm-output" / "dam-break-cases",
    )
    args = parser.parse_args()
    for name in args.cases or CASES:
        if name not in CASES:
            parser.error(f"Unknown case {name}. Choose from {', '.join(CASES)}.")
        title, reservoir, velocities, walls = CASES[name]
        output = args.output / name
        print(f"Running {name}", flush=True)
        simulate(output, reservoir, velocities, walls)
        animate(output, title, walls)


if __name__ == "__main__":
    main()
