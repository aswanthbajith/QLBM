# M24 — Dam-release collapse and runout

## Purpose
The M23 figure showed almost no spreading after 96 steps. Its front advanced from cell index 10 to 11. The narrow four-cell span was closed by no-slip bounce-back walls, and the observed duration was short on the gravitational scale. This milestone changes the boundary geometry and simulation horizon, **not** the physical collision model. It addresses the user's concern about the water column failing to collapse and flow out into the downstream region.

## Method
- Retain M23 D3Q19/D3Q7 pressure-based hybrid solver, density ratio 998/1.2, dynamic viscosity ratio 0.001/0.000018, `nu_liquid=.02`, `sigma=1e-5`, `xi=3`, bounded classical FCT, fixed hydrostatic reference, and local dual-rail quantum dissipation.
- Use a 32 x 4 x 16 lattice with gravity along -z. Replace front/back (y) no-slip walls by periodic links to model a span-uniform planar release. Floor, ceiling, left and right remain closed stationary halfway bounce-back. There is no outlet; the dam is already removed in the initial condition. This is not a general 3D periodic interface solver: the planar/mirror-symmetry guard rejects unsupported fields, because the inherited FCT and force stencils do not implement arbitrary periodic y fluxes.
- Use a bijective amplitude-label permutation `SpanTransport`, compiled to fully controlled X gate pairs, and a Numba executor which performs the ordered swaps on the dense complex statevector. Independently verify with slice-based push and y-roll. The Numba executor is gate-by-gate, not a permutation-index shortcut.
- Run a matched 640-step gravity comparison at g=0.00025, including a closed-sidewall control and a zero-gravity control. The fast reference shares the vector FCT but uses population-space collision algebra and independent streaming. A separate small-grid trajectory uses the scalar FCT reference for independent limiter verification.
- Initial nominal H=10.4 cells, width=10.56 cells. Report the gravity time scale t*=t sqrt(g/H), not a fictitious SI time. M23's 96 steps at g=.0005 correspond to t*=0.666; M24's 640 steps at g=.00025 correspond to t*=3.138. The gravity scan is classical and records speed-guard termination, not proven instability.
- Quantify floor front (largest x index with alpha>0.5 at z=0), left-column height (number of cells up to the highest alpha>0.5 cell at x=0), liquid center of mass using cell centers, phase bounds, and mass. All figures and GIF frames originate directly from computed fields; no animation of imposed geometry.

## Interpretation
See `results/report.json`, `trajectory.csv`, `screen.json`, `history.npz`. The old run used index coordinates for COM, whereas M24 uses cell-center coordinates (+0.5); compare displacements rather than absolute COM across the two milestones. The new result has visually clear collapse and runout, but still has a low nominal Reynolds number sqrt(g H)*H/nu_liquid ≈26.5. It is not a quantitatively validated water-air dam-break experiment. The model's material ratios alone do not establish correct physical viscosity, surface tension, momentum flux, or wetting. The reduced viscosity/length/time mapping must be chosen to match experimental dimensionless groups, and then verified at multiple resolutions. The periodic span removes sidewall drag intentionally for a quasi-2D benchmark; for a real finite-width tank retain correctly resolved walls. A speed guard >0.1 lu is a low-speed validity guard, not evidence of divergence by itself.

## Quantum scope
The local dual-rail dissipation and D3Q19/D3Q7 streaming are simulated quantum gate subroutines. Classical state preparation, exact expectation readout, nonlinear equilibrium, forces, phase limiter and population reconstruction remain. No hardware run, quantum speedup, or complete coherent evolution is demonstrated. New transport gate counts are logical MCX before hardware decomposition. Original M23 files and results are preserved.

## Reproduce
From `experiments/quantum_dam_break` use the tested environment in `milestone14/requirements-tested.txt`, then `bash milestone24/reproduce.sh` (add `--full-regression` for all earlier tests).
