# Milestone 7 — compact reversible arithmetic and linear drag

This extension preserves all previous milestones and does not alter upstream files. It is a quantum-compatible, reversible shallow-water finite-volume prototype, **not a D2Q9 quantum LBM solver**.

## Circuit and physics

`optimization.py` extracts the dependency cone of each of the mass, normal momentum and transverse momentum outputs of the M6 compute-copy-uncompute oracle. Each cone is computed, copied, then uncomputed before reusing the same work wires. The source gates are retained (with remapped wires); common subexpressions can be recomputed, so this is a space/time tradeoff, not a general optimal reversible compiler. Unused gates are pruned. The compact circuit is equivalent on clean work registers; it need not be equivalent on dirty work registers.

`drag.py` implements the signed reversible map |q>|y>|0> -> |q>|y XOR round(beta*q)>|0> using X, CX and CCX. The old q is retained, output is signed two's-complement, and rounding is to nearest with ties away from zero. The drag source is dq/dt = -k q, with beta = exp(-k dt) = 7/8 in the demonstration. This is an illustrative linear bed-resistance coefficient, not measured physical water viscosity or calibrated Manning friction. The operator is unitary on the full retained-register space; damping of the selected physical output does not imply a nonunitary closed-system gate. Re-preparation between timesteps discards history. At small momentum magnitudes, rounding pins values (up to four ticks for beta=7/8); higher precision lowers this floor.

`solver.py` executes the compact M6 transport and then writes damped momenta into extra registers, retaining pre-drag values and reusing the clean arithmetic workspace. The entire logical gate stream is available; cached local gate results accelerate the computational-basis simulation. `reference.py` provides independent quantized and floating comparisons only. The full grid state is not simulated as a statevector. No Qiskit or hardware run was performed. Local and small composite gate circuits can be exported if Qiskit is installed. Logical counts are untranspiled and do not establish quantum advantage.

## SI interpretation

Chosen scales: H0=1 m, L0=1 m, g=9.81 m/s², U0=sqrt(g H0), T0=L0/U0. Each cell has area 1 m² and the simulated time step is T0/8. Constant illustrative water density is 998.2 kg/m³ (approximately 20°C); it only enters postprocessing, not the shallow-water evolution. Discharge = nondimensional discharge times H0 U0. Hydrostatic bed gauge pressure = rho g h. `water_density_kg_m3` is zero outside wet cells as a visualization convention, not an air-density field. The geometry remains the synthetic M5 mask; no original STL reconstruction. No vertical splashing, air entrainment or surface tension.

## Reproduction

From `experiments/quantum_dam_break` with numpy, pytest, matplotlib and pillow:

```sh
python milestone7/run.py
python milestone7/plot.py
python -m pytest -q tests milestone2/tests milestone3/tests milestone4/tests milestone5/tests milestone6/tests milestone7/tests
```

The 16x16 test uses S=32 and 16 steps with drag, compared to a separate quantized solver and a floating solver with the same drag. Reversal, mass conservation, clean work registers and obstacle exclusion are checked at every step. `spatial_reference.csv` reports **classical-only** floating planar Ritter refinement on a fixed domain and final time, with no drag. It is not a convergence proof of the fixed-point quantum solver. The M6 dry-front zero-mass suppression and numerical diffusion remain unresolved. `drag_decay.csv` measures the isolated source circuit against its analytic exponential. All VTK fields in `results/step_*.vti` are ASCII and use SI labels; see source for scale definitions.
