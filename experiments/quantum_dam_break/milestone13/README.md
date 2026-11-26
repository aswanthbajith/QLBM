# Milestone 13 — Conservative pressure-moment repair

## Scope
This extension retains M12's D2Q25 nonnegative fixed-point populations, flat-bed shallow-water model, partial BGK relaxation and reversible multispeed transport. It adds pressure-targeted marginal correction and a less dissipative conditional reflection projection. The prior milestone is preserved and is used for matched comparisons. All quantities below are lattice units (lu). This is a research prototype, not a validated SI water solver.

## Run
From `experiments/quantum_dam_break`:

```bash
python -m pip install numpy numba pytest matplotlib pillow
python milestone13/run.py
python milestone13/plot.py
python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12,13}/tests
```

`collision.py` compiles the new calculation into X, CNOT and Toffoli gates and inherits M12's bit-sliced computational-basis execution and compute/copy/uncompute method. `reference.py` is an independent integer implementation of the new correction, sharing the earlier equilibrium construction. `run.py` also runs a separate floating-point projected kinetic comparator. `13_pressure_rounding.ipynb` is an executable demonstration (not a saved executed notebook). No classical reference advances the gate-executed state. Between steps, basis populations are reconstructed and re-prepared; no full-grid coherent circuit, native Qiskit run, or QPU run is claimed.

## Mathematical change
For a marginal population vector p indexed by k=-2,-1,0,1,2, H=sum(p), J=sum(k*p), and integer depth scale S, the required integer second moment P=sum(k^2*p) targets J^2/H + H^2/(4*S) when H>0. This follows from c=k/2, g=1/8. The pressure error numerator is 4*S*J^2+H^3-4*S*H*P. A transfer (a,b,c)->(a+n,b-2n,c+n) on three adjacent velocities preserves H and J and changes P by 2*n. Its inverse lowers P by 2*n. The circuit rounds the magnitude of the pressure error to the nearest number of such transfers and caps transfers by available populations. It uses a finite greedy sequence, not an optimal projection; at zero mean only the central triple is used. It preserves sign-reflection of the marginal for nonzero J. On valid input the marginal correction cannot increase the absolute pressure error, though the final 2D coupling and symmetry correction can affect the moments.

For an axis on which the input is already reflection symmetric, M13 replaces mirrored output populations a,b by floor((a+b)/2) each and moves the odd residual (a+b)%2 to the zero-normal-speed population. M12 instead moved the entire absolute imbalance to the central channel. M13 preserves mass, tangential momentum and zero normal momentum and loses at most 25 integer second-moment units per projection across five lines. This is not general reflection equivariance for arbitrary inputs. The partial BGK residual redistribution of M12 is unchanged.

## Executed evidence
- Matched local test at S=256, omega=1: 1024 random local inputs, 812 pressure-feasible. Diagonal pressure-tensor RMSE: M12 0.0051047423, M13 0.0005584032 (89.06% lower). Cross component RMSE: 0.00181116 -> 0.00171581. Exact mass and both momentum conservation; independent integer reference agreement. Arbitrary x-reflection mismatches increased 28 -> 53 of 1024; general equivariance remains unresolved.
- Matched 16x16 obstacle case, S=256, omega=0.5, 16 steps: 80 initially wet cells, 234 finally wet cells, 105 of them pressure-infeasible (M12: 219 wet, 99 infeasible). Integer depth sum 30720 ticks, water volume 30 lu^3, zero mass loss, zero solid penetration, exact local collision momentum, inverse and scratch checks, 16/16 independent integer-reference agreement. Final depth RMSE vs same-initial-state floating comparator 0.0294595 -> 0.0252675. 17 VTK snapshots. Geometry is synthetic.
- Matched 256x1 periodic strip, S=1024, dx=0.5, 24 steps: central |x|<28 Ritter RMSE at omega=1: 0.0840360 -> 0.0803301, omega=0.5: 0.1074878 -> 0.1060149. Artificial transverse velocity exactly zero in both models. At omega=1 discrepancy against the float reference over the whole grid actually increases from 0.00359485 to 0.00554402. This is not spatial convergence; pressure realizability still fails in some cells.
- Shear-wave benchmark: 16 matched gate runs (M12 and M13, four omega settings, two wavelengths 16 and 32 lu), S=1024, 40 steps each, fit over positive amplitudes at steps 10..40. M13 viscosities at omega=.25,.5,.75,1: wavelength 16 = .0844304,.0936506,.0654729,.0480698; wavelength 32 = .1486055,.1228987,.0736903,.0514159. Longer wavelength restores ordering but large differences from the formal long-wave estimate remain, especially at .25. This is not molecular viscosity calibration. Floating reference also deviates; finite wavelength, projected equilibrium and fixed-point coupling contribute.
- All 153 executable regression tests passed, 5 optional Qiskit test modules skipped. Fourteen new tests include exhaustive independent marginal checks at three small register scales, randomized gate/reference agreement, positivity, mass and momentum, symmetry, overflow flag, output XOR, reversal and grid transport.

## Resource trade-off
At identical S=256, omega=0.5, 16x16 configuration, local collision gates increase from 421905 to 499497, local scratch 128812 to 152692; conceptual global qubits 257053 to 280933 and untranspiled gates per step 101832615 to 120532287. These are not hardware measurements. More accurate local moments currently cost more resources. The finite velocity set remains structurally incapable of representing some dry-front pressure states. The next research priority is an equivariant population coupling with pressure-aware residual relaxation rounding, followed by higher precision/wavelength calibration and small native circuit execution. Independent OpenLB or experimental validation remains unperformed. No quantum advantage is established.

## Files
`results/report.json` contains metrics and resources; `local_moments.csv`, `viscosity.csv`, `shear.csv` and per-experiment diagnostics retain measured values. `obstacle_M13/step_*.vti` contains 25 populations, reconstructed depth, discharge, velocity, target hydrostatic flux, actual kinetic moments, masks, and independent float reference depth. The GIF and plots are generated from executed data, not illustrations.
