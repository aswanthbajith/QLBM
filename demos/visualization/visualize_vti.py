import vtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BASE = (
    "demos/simulation/qlbm-output/"
    "ms-4x4-16x16-1-obstacle-qiskit/paraview"
)

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


# Load all 21 timesteps
fields = []

for step in range(21):
    print(f"Loading step {step:03d}")
    fields.append(read_vti(step))

fields = np.array(fields)

print("Loaded:", fields.shape)

# Keep same scale for every frame
vmin = fields.min()
vmax = fields.max()

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(
    fields[0],
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("QLBM Scalar")

ax.set_xlabel("x")
ax.set_ylabel("y")

title = ax.set_title("QLBM Multi-Speed — Step 0")


def update(frame):
    image.set_data(fields[frame])
    title.set_text(f"QLBM Multi-Speed — Step {frame}")
    return image, title


animation = FuncAnimation(

    fig,
    update,
    frames=len(fields),
    interval=300,
    blit=False,
    repeat=True,
)

animation.save(
    "qlbm_ms_evolution.gif",
    writer="pillow",
    fps=4,
)

plt.tight_layout()
plt.show()
