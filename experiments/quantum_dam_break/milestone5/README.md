# Milestone 5 — two-dimensional reversible shallow-water dynamics

## Status and scope

This is a new research prototype extending Milestone 4's reversible finite-volume approach. It is NOT a D2Q9 lattice Boltzmann solver. Earlier QLBM encoding, streaming, and collision experiments remain unchanged. We use a synthetic 16 × 16 top-down domain (not a reconstruction of the original STL obstacle geometry), flat bed, g=1, dx=dy=1, dt=1/8, and initially 1.5 dimensionless depth in columns 0–4. Obstacles occupy rows 5–10, columns 8–9, and rows 1–3, column 12 (zero-based, inclusive). Outer walls and obstacles are impermeable, free-slip reflecting boundaries. No physical viscosity, drag, turbulence, or air–water interface is implemented.

## Physical equations and algorithm

For U=(h,qx,qy), q=h*u, the flat-bed shallow-water equations are U_t + F_x(U)_x + F_y(U)_y = 0. The x flux is (qx, qx²/h + gh²/2, qx*qy/h); the y flux is (qy, qx*qy/h, qy²/h + gh²/2). Dry states use zero flux. Gravity enters through the hydrostatic pressure flux, not a separate flat-bed source.

We use a Rusanov split with fixed wave bound a=3. For each outward direction n, the per-cell transfer is D_n=(dt/(2dx))(aU+F_n(U)), where F_n is the signed normal flux (dy=dx). Subtract D_n from its source and add the same value to the neighboring cell, or reflect normal momentum into the source at an impermeable wall. This is algebraically equivalent to the ordinary Rusanov face flux before quantization. Fixed-point mass is rounded down, signed momentum is rounded to nearest (ties away from zero), and momentum transfer is suppressed when the quantized mass transfer is zero. This changes the scheme near shallow fronts and can pin wetting. The four nonnegative mass transfers sum to at most 2*a*dt/dx*h=0.75h before flooring, so depth cannot be negative. Mass transfers cancel exactly on the integer lattice. Walls may change total momentum by applying an impulse; no momentum conservation claim is made for a closed domain with walls. Numerical diffusion is not calibrated water viscosity.

The supported input range is 0<=h<2, |u_x|,|u_y|<=1.5, with additional signed word limits. Every next state is checked for admissibility; unsupported states are rejected rather than clipped or wrapped. This is not a general wetting–drying method with a proven convergence rate.

## Reversible circuit and scalability

At scale 32, the input cell has 6 unsigned depth bits plus two 7-bit signed discharge words (20 bits); output accumulators are widened by four bits per field. A shared 24-bit signed transfer scratch register is reused. For each face direction, an offline compiled single-cell table specifies a pattern-controlled-X XOR oracle. The gate stream then subtracts the transfer from the source accumulator, adds it (or its reflected version) at the destination, and applies the oracle again to clear the scratch. Inputs remain unchanged. The signed adders and gate primitives are imported from `milestone4/solver.py`, not duplicated. A step is an exact reversible permutation on *all* basis states, with an arbitrary zero extension of the oracle for invalid local codes. `encode` and `decode` enforce the physical input convention at the boundary of the simulation.

Single-cell splitting reduces oracle controls from a hypothetical 40-bit pair of 20-bit cells to 20 bits. This is a structural improvement, not a demonstrated speedup: the table still grows exponentially with precision. The 16×16 example is a conceptual 13,336-logical-qubit, 10,926,586,632 pattern-controlled-X circuit per step before decomposition. These are counts, NOT gates physically executed on hardware. The indexed simulator skips gates whose immutable controls do not match, and executes the remaining basis gates; it does not evaluate classical CFD flux formulas online. For small tests `iter_gates()` explicitly enumerates the complete gate stream. `qiskit()` exports a circuit only below a safety cap. No Qiskit installation was available; the optional native test was skipped. Sparse two-branch permutation checks do not constitute a full coherent statevector run. Classical basis-state extraction and re-preparation are used between timesteps. No quantum advantage is demonstrated.

## Reproduction

From `experiments/quantum_dam_break` in the extracted repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pytest matplotlib pillow
python milestone5/run_2d.py
python milestone5/plot_results.py
python -m pytest -q tests milestone2/tests milestone3/tests milestone4/tests milestone5/tests
```

Optional: install compatible Qiskit in a separate environment, then run the optional tests. Do not attempt full-state simulation of the large 16×16 circuit. `05_reversible_2d_dam_break.ipynb` is a companion notebook (not executed here; its underlying scripts were executed).

## Executed results

- 32 steps; 33 ASCII VTK ImageData files with cell data and named fields. Mass (integrated nondimensional depth) 120 at every step. Wet cells 80 → 183 out of 241 fluid cells. No obstacle mass. Max transverse discharge 0.0625; final depth RMSE versus floating reference 0.0362129, over nonsolid cells.
- All 32 gate-driven updates exactly match a separate scalar quantized reference. Every step passes scratch cleanup and inverse round-trip.
- A separate 32-cell dry-bed Ritter dam-break benchmark (dx=0.5, dt=0.0625, t=1, h0=1.5) compares point-sampled analytical depths, not analytical cell averages. Errors versus floating Rusanov for depth quanta 1/8, 1/16 and 1/32: 0.091584, 0.062187 and 0.023585. Errors versus Ritter: 0.136659, 0.085535 and 0.094759; nonmonotonic, so do not claim convergence to the exact PDE. Floating reference versus Ritter RMSE 0.088713. This benchmark varies register precision, not spatial resolution. No mesh convergence study yet.
- Test suite: 55 passed, 5 optional Qiskit modules skipped. See `results/test_report.txt` and `report.json`.

The VTK file includes `depth_nd`, `discharge_nd`, `velocity_nd`, `speed_nd`, `hydrostatic_flux_nd`, `wet_mask`, `solid_mask`, `reference_depth_nd`, `depth_error_nd`. Velocity and pressure flux are postprocessed from the decoded physical state. Reference fields come only from the independent reference; they do not drive the gate simulator. Open `results/step_000.vti` as a file series in ParaView, select cell data `depth_nd`, and fix the color range at 0–1.5. `dam_break_2d.gif` is generated from the actual history.

## Next research priorities

Replace lookup oracles with scalable reversible add/multiply/divide and controlled rounding; establish precision/mesh convergence; test low-depth momentum stabilization and physical dissipation; run native small quantum circuits; decide whether to retain this finite-volume architecture as a comparison track or return to a physical QLBM collision formulation. A genuine D2Q9 water solver remains a separate unsolved objective. No claim of a completed quantum CFD product or hardware readiness is made.
