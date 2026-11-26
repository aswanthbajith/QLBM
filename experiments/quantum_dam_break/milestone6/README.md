# Milestone 6 — arithmetic rather than lookup

This experiment extends M5's flat-bed, nondimensional shallow-water finite-volume solver. It is NOT a D2Q9 LBM solver. No classical CFD routine advances the target state; the independent M5 reference is called only for comparison. All earlier milestones are retained.

## Algorithm

For depth ticks H, normal and transverse discharge ticks N,T (normal sign applied first), scale S, fixed dt/dx = dt/dy = 1/8, wave-speed bound a=3 and gravity g=1, the M5 outgoing directional contribution is:

- M = floor((3H+N)/16)
- P = sign * round((2S*N*(3H+N)+H^3)/(32S*max(H,1)))
- R = round(T*(3H+N)/(16*max(H,1)))

Round nearest with ties away from zero. Invalid states (H=0, 2|N|>3H, 2|T|>3H) and zero-mass transfers return zero. The zero-mass suppression is the prior model's approximation; front pinning and artificial dissipation remain possible. Out-of-range physical outputs are rejected instead of clipped. Closed boundaries reflect normal momentum. All three components are accumulated conservatively with the previous M5 controlled-increment gates; walls exchange momentum with the fluid.

`arithmetic.py` compiles signed and unsigned binary operations using reversible X/CNOT/Toffoli gates with fresh Boolean work wires. It implements ripple-carry addition, two's-complement negation, schoolbook multiplication, unsigned restoring division, signed nearest rounding, validity masking, and compute-copy-uncompute. No enumerated flux table is built by the new solver. Widths scale linearly with precision and the arithmetic gate/workspace counts scale polynomially (quadratically for these fixed-width schoolbook operations); the implementation is not ancilla-optimal.

`solver.py` reuses the M5 layout and global accumulation. Each local oracle is applied by exact basis gate simulation; repeated inputs use memoized outputs of that gate simulation. Clean workspace is required; the fully expanded gate stream is also available, and tested on a small grid. The accelerator XORs the already gate-evaluated output into the flux register. This is classical simulation of reversible quantum circuits, not a full statevector simulation, hardware execution, or evidence of quantum advantage. The circuit is a basis permutation and uses no interference to accelerate computation. Input states are retained for reversibility and re-prepared between timesteps. The local oracle can be exported through `FluxOracle.qiskit()` if Qiskit is installed; this has not been executed here. Global accumulation still uses multi-controlled X gates; all counts are untranspiled logical gates, not comparable hardware costs or runtime measurements.

## Reproduce

From `experiments/quantum_dam_break` with Python and `numpy`, `pytest`, `matplotlib` installed:

```sh
python milestone6/run.py
python milestone6/precision.py
python milestone6/plot.py
python -m pytest -q tests milestone2/tests milestone3/tests milestone4/tests milestone5/tests milestone6/tests
```

`run.py` reruns the M5 16x16, 32-step synthetic-obstacle experiment at S=32 and checks exact agreement with the archived M5 trajectory and independent quantized reference. `precision.py` evaluates higher precision at S=32,64,128 on a 32-cell planar Ritter benchmark (16 steps, dx=0.5, dt=0.0625, t=1). Physical validation is limited to the existing numerical/analytical comparisons. No new physical viscosity, bed slope integration, turbulence, surface tension, two-phase interface or original STL reconstruction is claimed. Higher precision is separate from higher spatial resolution; there is no demonstrated grid-convergence order.

Results: `report.json`, `diagnostics.csv`, `arithmetic_scaling.csv`, `precision.csv`, `history.npz`, `precision.npz`, `step_000.vti` through `step_032.vti`, and `benchmark.png`. Quantities are nondimensional; the VTK output retains M5 field names. `test_report.txt` records the full suite. Resource counts in `report.json` are for a single timestep with reusable oracle workspace. Native Qiskit tests in earlier milestones may be skipped.
