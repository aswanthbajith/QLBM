# Milestone 4 — integrated reversible 1D shallow-water step

## Scope and provenance
This is a NEW finite-volume quantum arithmetic experiment, not an extension of the D2Q9 collision circuit itself. Milestones 1–3 remain intact. Milestone 3's EOS test found that its occupancy equilibrium did not reproduce the shallow-water pressure law. This milestone carries forward the fixed-point reversible arithmetic principle of milestone 3, but compiles a new joint nonlinear interface oracle; it does not call the old standalone pressure or gravity circuits. It is not a completed quantum LBM, 2D dam break, real-hardware run or quantum advantage demonstration.

## Equations and discrete model
For nondimensional flat-bed 1D shallow water, let U=(h,q), q=h*u, with flux F=(q,q²/h+g*h²/2), and g=1. At dry cells set F=0; the input domain enforces q=0 if h=0. Use a Rusanov interface flux G=(F_L+F_R)/2-a*(U_R-U_L)/2 with a=2.5, dx=1, dt=0.25. Update U_j'=U_j+(dt/dx)*(G_left-G_right). Reflecting ends use mirrored discharge and equal depth, so wall mass flux vanishes, but wall pressure can transfer momentum. Total momentum is not required to be constant with walls.

The input uses 3 unsigned depth bits, 4 signed discharge bits, both with resolution 0.25. Valid inputs obey 0<=h<=1.75, |q|<=h, hence maximum signal speed <=1+sqrt(1.75)<2.5. The fixed-point compiler rounds each *scaled interface flux* to a multiple of 0.25, ties away from zero. It caps mass export at floor(depth_word/2) for each face, guaranteeing nonnegative depth when the two outgoing fluxes are added. A shared face increment is subtracted from one cell and added to the other, preserving the discrete total exactly. This limiter changes the original Rusanov scheme and can stall very shallow fronts. It is NOT proof of entropy stability, physical accuracy, well-balanced topographic behaviour or convergent wet/dry dynamics. Fluxes on invalid input patterns are set to zero as a reversible extension; the driver rejects invalid initial and subsequent states. No silent clipping of momentum or wrapping into the narrow input registers.

## Quantum mechanism and actual execution
The circuit maps |U>|0>|0> to |U>|U_new>|0>. Copy input bits to wider signed accumulators using CNOT, then for each face:
1. Compute the quantized mass and momentum increments into a reusable 12-bit scratch register via pattern-controlled X (MCX with open/closed controls), with controls on the immutable input cells.
2. Use controlled binary increments to add or subtract each signed flux in neighbouring output registers; subtract by reversing the gate sequence.
3. Uncompute scratch by repeating the XOR oracle.

The output has 5 signed depth and 6 signed discharge bits per cell (checked for admissibility before next step). For valid inputs the accumulators are wide enough for the intermediate increments. The complete circuit is a permutation even for nonzero output/scratch inputs; only zero initialization implements the stated fluid update. For six cells there are 42 input + 66 output + 12 scratch = 120 qubits and 148864 logical pattern-controlled X gates, *before* decomposition, hardware routing or state preparation. The nonlinear pressure and convective terms are calculated in exact rational arithmetic **offline** for every possible local input pattern, not calculated by a classical solver at runtime. This is exponentially scaling truth-table compilation, not scalable coherent arithmetic. The simulator indexes mutually exclusive oracle controls to skip gates that cannot fire, executes matching controlled-X gates and the adders, and checks reversal. A separate test compares the indexed executor to the entire literal gate stream. It is an exact computational-basis permutation simulator, not a dense statevector simulation of 120 qubits. A sparse two-branch amplitude check demonstrates the abstract reversible map preserves amplitudes and can be inverted; no entanglement or interference advantage is claimed.

Between timesteps the driver reads the deterministic basis output, checks the supported domain, and prepares the next basis input. No classical fluid solver supplies an evolution update to the quantum gate model. This is not a fully coherent multistep circuit; discarding old states or measuring them is not unitary. Native Qiskit export exists (potentially expensive), but Qiskit is unavailable here, so it has NOT been executed or transpiled. The separate reference is used only for validation. VTK depth/discharge are decoded outputs; velocity and hydrostatic flux are derived in post-processing, not measured separately.

## Reproduce
From experiments/quantum_dam_break (Python 3 with numpy, pytest, matplotlib, pillow):
```bash
python milestone4/run_step.py
python milestone4/plot_results.py
python -m pytest -q tests milestone2/tests milestone3/tests milestone4/tests
```
The notebook 04_reversible_fluid_step.ipynb provides the same example. Output VTI files are cell data in a 6x1 strip, not position counts or SI quantities. The GIF plots actual computed depth and the independent floating-point reference. All outputs are under results/.

## Results and limitations
Initial 3 wet cells at h=1.5 and 3 dry cells; zero discharge. Eight steps to nondimensional t=2. Initial and final integrated depth = 4.5, exact discrete conservation. All six cells become wet. Each update agrees with a separately implemented quantized finite-volume reference and is reversible with clean scratch. Final depth RMSE vs the *unquantized coarse-grid* floating Rusanov reference is 0.15550547; this is not an error vs the analytic dam-break solution or OpenLB. The wetting is partly numerical diffusion from the Rusanov flux, not a physical interface proof. At step 1 the initial pressure imbalance generates nonzero discharge. No physical water viscosity, vertical gravity source (flat bed), surface tension, two-phase physics, or obstacle geometry is implemented here. No native QPU, no full statevector, no quantum advantage. The results only validate implementation of this selected discretization, not the continuum model.

## Next development gates
1. Replace table compilation with reusable reversible add/multiply/divide, rounding, signed comparison and overflow flag arithmetic. Evaluate transpiled resource costs on small native circuits.
2. Increase depth/discharge precision, perform grid/time refinement against an analytic shallow-water dam-break case; assess limiter and dry-front error. Use wider registers and stable admissibility handling rather than silently clamp states.
3. Extend to 2D h, qx, qy with directional face fluxes and reflective obstacles. Investigate well-balanced gravity/bed source and physical friction/dissipation separately.
4. Investigate a coherent history-aware timestep strategy, and decide whether the target remains a quantum shallow-water finite-volume solver or a genuine QLBM with a suitable equilibrium and collision. These are distinct research tracks.
