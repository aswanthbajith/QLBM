# Milestone 8 — Reversible D2Q9 shallow-water lattice Boltzmann collision

## Scope and result

This is a separate D2Q9 collide-stream implementation, not an extension of the M7 finite-volume evolution. It uses nine unsigned fixed-point population registers per cell and reuses the M1 D2Q9 velocity convention, bounce-back link permutation, and M6 arithmetic builder. It does not reuse the upstream ABQLBM position/velocity amplitude circuit directly: the encoding is different. All earlier files are unchanged.

The equilibrium is the shallow-water BGK equilibrium described in the 2D submodel of *A 1D–2D Coupled Lattice Boltzmann Model for Shallow Water Flows in Large Scale River-Lake Systems*, Applied Sciences 10(1), 108 (2020), https://doi.org/10.3390/app10010108. For unit lattice speed, f0=h−5gh²/6−2h|u|²/3; fi=wi h(3gh/2+3ci·u+9(ci·u)²/2−3|u|²/2) for i>0, with wi=1/9 axial and 1/36 diagonal. The first three moments are h, hu, and h u uᵀ+gh²I/2. The chosen g_lu=1/8 is a research parameter, not SI gravity; dx=dt=1 in lattice units. Flat bed; no explicit bed slope, drag, or free-surface multiphase physics. The formal BGK viscosity is (1/omega−1/2)/3 in lattice units.

## Gate compilation and evolution

Integer ticks H=sum(fi_ticks), X=sum(cx fi_ticks), Y=sum(cy fi_ticks), D=ci·(X,Y), scale S. For the six free populations (3..8) compute round[(H³+8S(2DH+3D²−X²−Y²))/(48SH)] for axial directions, with denominator multiplied by 4 for diagonals. Dry-cell division uses a safe denominator. Apply omega in {1/2,1,3/2} to the rounded equilibrium difference, rounding ties away from zero. Reconstruct +x,+y and rest from exact mass and momentum residuals. If any candidate is negative or does not fit its register, retain the original populations and write a rejection flag. This limiter is conservative but is not a validated wetting/drying algorithm, and asymmetric residual assignment can bias rounding. Even with no rejected collision, quantization changes BGK slightly.

`collision.py` compiles the above with X, CX, CCX and compute-copy-uncompute. The map XORs the candidate and rejection flag into output registers, retaining input. Indexed gates are executed on batches of independent computational-basis inputs packed across uint64 bit lanes using Numba; this executes all local logical gates, not a floating BGK function. Parallel lanes are a classical simulation acceleration, not a quantum speedup. `solver.py` applies exact wordwise streaming permutation equivalent to SWAP gates, reusing M1 link logic. Reversal of local oracle and inverse transport are tested; previous timesteps are read out and re-prepared, so this is not a coherent full-history quantum simulation. Local and capped global Qiskit export methods are included, but Qiskit is not installed here and neither export nor hardware execution is verified. The global resource count assumes one shared clean local workspace, separate input and output population registers, and a rejection flag for each active cell. Work count is large due to retaining intermediate values during reversible compilation.

## Benchmarks

- Wet-bed 16x16, 15 synthetic solid cells, h_left=1.5, h_right=1.25, S=4096, omega=1, 32 steps. 0 rejected collisions of 7712; exact volume and local collision momentum conservation, no solid penetration, all steps match independent integer reference. Floating LBM comparison only (not external physical validation). Final fluid-cell depth RMSE 0.00078317 lu.
- Dry-bed 16x16 stress, same geometry, h_right=0, 24 steps: 80 to 235 wet cells and exact depth sum 120, but 2304/5784 collisions rejected; unconstrained floating LBM produces negative populations at step 2 and is stopped at step 7 for negative depth, nonfinite values or excessive populations (see run.py). The limiter can leave a ballistic front. This experiment fails physical validation and must not be presented as a successful water dam break. VTK reference depth is NaN after the reference stops; diagnostics omit RMSE then.
- Pressure: 33 resting states with depth 0.5..1.5, relative L2 error 0.402% vs gh²/2. This is a local moment check, not full continuum accuracy.
- Viscosity: periodic 4x32 shear wave, amplitude 0.01, S=16384, 64 steps, fit steps 8..48. Fitted gate viscosity 0.366528, 0.164320, 0.028074 for omega 0.5,1,1.5 vs formal 0.5,1/6,1/18. Floating fits are closer. Fixed-point rounding causes material calibration errors for some relaxation rates. Molecular viscosity of water not established; SI scaling and Reynolds-number matching remain open.

Results are in lattice units. `f0_lu`..`f8_lu` are decoded populations; depth=sum populations, velocity=momentum/depth. `hydrostatic_flux_lu` is a derived equilibrium target, not a measured local pressure tensor. VTK dry stress reference arrays contain NaN after reference failure. No claim of hardware execution or advantage.

## Run

From `experiments/quantum_dam_break` (Python with numpy, numba, pytest, matplotlib, pillow):

```sh
python milestone8/run.py
python milestone8/plot.py
python -m pytest -q tests milestone2/tests milestone3/tests milestone4/tests milestone5/tests milestone6/tests milestone7/tests milestone8/tests
```

Next: design a positivity-preserving well-balanced collision/limiter with verified dry-front accuracy, improve unbiased rounding and workspace, compare against analytical and independent classical LBM/CFD, then native circuit tests at manageable sizes. OpenLB validation is not performed.
