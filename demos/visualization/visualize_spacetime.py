import vtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BASE = (
    "demos/simulation/qlbm-output/"
    "spacetime-d2q4-8x8-qiskit/paraview"
)

NUM_STEPS = 11


def read_vti(step):

    filename = f"{BASE}/step_{step:03d}.vti"

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    data = reader.GetOutput()

    array = data.GetPointData().GetArray("Scalars_")

    values = np.array([
        array.GetTuple1(i)
        for i in range(array.GetNumberOfTuples())
    ])

    nx, ny, nz = data.GetDimensions()

    return values.reshape((ny, nx))


# ------------------------------------------------
# Load all timesteps
# ------------------------------------------------

fields = []

for step in range(NUM_STEPS):

    print(f"Loading step {step:03d}")

    fields.append(read_vti(step))

fields = np.array(fields)

print("\nLoaded data:", fields.shape)

# Same colour scale for every timestep
vmin = fields.min()
vmax = fields.max()


# ------------------------------------------------
# Create animation
# ------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    fields[0],
    origin="lower",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("QLBM Scalar")

ax.set_xlabel("x")
ax.set_ylabel("y")

title = ax.set_title(
    "Space-Time QLBM — Step 0"
)


def update(frame):

    image.set_data(fields[frame])

    title.set_text(
        f"Space-Time QLBM — Step {frame}"
    )

    return image, title


animation = FuncAnimation(
    fig,
    update,
    frames=NUM_STEPS,
    interval=400,
    repeat=True,
)
animation = FuncAnimation(
    fig,
    update,
    frames=len(fields),
    interval=300,
    blit=False,
    repeat=True,
)

animation.save(
    "qlbm_spactime.gif",
    writer="pillow",
    fps=4,
)

plt.tight_layout()
plt.show()
plt.tight_layout()

plt.show()
