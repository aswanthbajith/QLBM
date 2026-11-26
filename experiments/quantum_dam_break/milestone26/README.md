# Milestone 26 — scientific visualization and wall-pressure diagnostics

Objective: two-phase water–air dam-break on a quantum computer. This milestone is an output/data-capture improvement, **not** a new hydrodynamic or quantum algorithm. It re-runs the exact M25 `slow212_gate` case: D3Q19/D3Q7, Re=212.119, density ratio=831.67, 32×4×16 (X,Y,Z), 1280 timesteps. M24 spanwise-periodic, span-uniform mirror-symmetric thin slab; no genuine spanwise instability or 3D turbulent flow is demonstrated. Gravity is −Z. All lattice units, not calibrated water impact loads. Gate-simulated dissipation and streaming; physical equilibrium, forcing, limiter and state preparation/readout remain classical.

## Results and use

- `results/*_multiangle.gif`: 21 snapshots, every 64 solver steps, fixed per-field scale. Matching PNGs at 0, 640, 1280. Side X–Z at y=2 and plan X–Y at z=8 use linear interpolation between the two central cell planes; near-floor X–Y at z=0.5 is an actual cell-centred slice. Black contour is alpha=0.5; for phase, blue 3D mesh is the zero-level interface; other fields color the same mesh by interpolating the scalar at triangle centroids. The translucent 3D cut planes show center span, mid-height, and downstream wall-adjacent cells. Mesh via marching cubes of actual 3D samples; not extrapolated beyond cell centres.
- `water_body_3d.gif`: two isometric perspectives of actual occupied liquid cells (alpha>=0.5); no smoothing or invented spanwise structure.
- `wall_pressure.png` and `.gif`: gauge pressure in cells adjacent to downstream wall (x=31.5; wall at x=32), with probes at z=0.5,1.5,3.5,7.5. Wall map and pressure probes, alpha=0.5 contour. `wall_probes.csv` and full wall plane pressure/alpha in NPZ have **every** timestep, including initialization. Maximum cell pressure peaks before threshold wetting; do not interpret it as the peak liquid impact stress. Wall pressure sum is a unit-area cell-pressure sum proxy, not measured traction or calibrated impact force.
- `fields.npz`: 21 full-volume captures, arrays `[time,z,y,x]`, `velocity` `[time,component,z,y,x]` in XYZ component order. Also `wall_pressure`, `wall_alpha`: `[1281,z,y]`. `dam_break.pvd` opens the full time series in ParaView (relative VTK files in `vtk/`; time is dimensionless `t*`, not seconds). Cell-data arrays and coordinates preserve full precision.

| Field | Definition |
|---|---|
| `phi` | 2 alpha − 1; water +1, air −1, diffuse interface 0 |
| `speed` | Euclidean norm of force-corrected decoded velocity |
| `jz` | material density × vertical velocity; negative downward, positive upward |
| `pressure` | solver gauge pressure; not absolute pressure or dynamic pressure |
| `rho` | normalized material density (liquid reference 1) |
| `mu` | dynamic viscosity in normalized lattice units |
| `dynamic_pressure` | 1/2 rho |u|²; distinct from wall gauge pressure |

Pressure data are exploratory. No calibrated pressure sensor, wall traction, or experimental benchmark. The model is still over-viscous relative to physical water at the illustrative scale; no claim of a physically validated full dam break or quantum hardware run. Image playback is not physical real time.

## Reproduce

From `experiments/quantum_dam_break`, use the existing dependencies plus `scikit-image`, `scipy`, `matplotlib`, `pillow`. Run `bash milestone26/reproduce.sh` (optional `--full-regression`). The M25 NPZ baseline is required for the exact replay comparison. `capture.py` fails if final populations or saved phase slices differ by more than 1e-11. The executed replay matched all three bit-for-bit. Renderer operates only on captured data; no CFD code altered.

Next physical work: validate wall loads independently (traction and pressure-probe convergence), reduce viscosity error and expand to genuinely 3D span-varying flow. Future quantum work: implement and validate the remaining classical operators with compatible encoding.
