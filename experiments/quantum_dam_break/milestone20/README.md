# M20 — Bound-preserving phase transport in the focused two-phase dam-break solver

## Objective and acceptance criteria
The sole project target is water–air dam-break simulation on a quantum computer. M19 established a hybrid D3Q19/D3Q7 small-density-contrast dam-release experiment with quantum circuit-simulated dissipation and transport, but failed the phase bound. This milestone retains those exact circuit mechanisms and repairs the phase transport locally. This is **not** a high-density-ratio model or a quantum implementation of the new limiter. No claim of physical water–air validation, quantum advantage, real hardware, or fully coherent nonlinear two-phase evolution.

Acceptance: preserve phase sum and local collision moments, stay within [-1,1] (1e-10 tolerance), verify the streamed phase against a face-flux balance, independently reproduce population trajectories, and retain the gravity response and no-gravity control. Run matched 96-step cases, including ideal and sampled collision readout. A failure of any bound check aborts the bounded run.

## Algorithm and proof
Let signed D3Q7 populations g sum to phi, and a=(1+phi)/2. For each interior link from cell L to R in positive coordinate direction i, the candidate phase-fraction flux is H=(g_i(L)-g_opp(R))/2; the constant equilibrium background cancels. Closed halfway bounce-back gives zero external flux. Define a low-order flux B=d(a_L-a_R), d=1/8; its update is a convex combination of a cell and its up-to-six neighbours because 6d<=1. Thus 0<=a_low<=1 for admissible inputs. Set antidiffusive flux D=H-B. Accumulate positive and negative contributions P+, P- to each cell from all adjacent faces, and compute R+=min(1,(1-a_low)/P+), R-=min(1,a_low/P-) with ratio 1 for zero denominators. Capacity numerators are floored at zero against roundoff. On a face with D>=0 choose theta=min(R-_L,R+_R); for D<0 choose theta=min(R+_L,R-_R). The new flux is B+theta D. The sums of positive and negative corrected contributions cannot exceed the cell capacities; consequently the updated a is bounded in exact arithmetic and the paired face updates conserve sum(a). The global fast path returns the original populations unchanged if the unlimited candidate is already bounded. This is a Zalesak-inspired flux-corrected transport (FCT) adaptation, not a published CAC discretization reproduction or a proof of convergence to the desired continuum PDE.

To implement the limited flux **before** quantum streaming, let delta=(B+theta D)-H. Add delta to g_i(L), subtract it from g_opp(R), and compensate the two local rest populations by -delta at L and +delta at R. This preserves each cell's pre-stream phase sum and changes the fraction flux by exactly delta. The pre-existing quantum transport then advances the corrected population field. No cellwise phase clipping or global redistribution is used. M19's force evaluation still contains its original safety clipping, which is inactive apart from roundoff when the phase is bounded. Individual signed populations are not required to be nonnegative. The guard can alter nonconserved moments and interface diffusion; it is not assumed to be a physically exact correction.

The vectorized implementation is checked against a separately coded scalar graph limiter, independently assembled population-space MRT algebra, and classical slice streaming. Both reference and target share M19 physical equilibrium/source formulas, so the match validates numerical execution, not continuum physics. The input is a closed rectangular tank, no interior obstacle in this benchmark; do not apply the limiter across arbitrary solid masks without extending the edge graph.

## Reproduce
From `experiments/quantum_dam_break` in the complete archive:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r milestone14/requirements-tested.txt
bash milestone20/reproduce.sh
# optional full historical suite
bash milestone20/reproduce.sh --full-regression
```
`run.py` reruns 6 cases on the same 16(z) x 4(y) x 32(x) grid, 96 steps, with the same M19 tau, sigma, xi and reduced gravity. `plot.py` generates plots, animation and an illustrative material interpolation diagnostic. `results/report.json`, `diagnostics.csv`, `history.npz` and seven VTI snapshots contain the actual executed fields; the results store phase, LB density and velocity, not a calibrated physical density field. `results/test_report.txt` records the focused M18–M20 regression. Exact ideal rail readout and classical preparation/reconstruction at every timestep remain necessary. Finite-shot samples are from ideal quantum circuit probabilities; no QPU. Qiskit export remains inherited and optional.

## Results (executed)
- M19 reproduced with independent classical transport: final invalid cells 482; 52,116 invalid cell-step occurrences; range -1.0130143 to 1.1267054.
- M20 ideal hybrid: zero invalid cells at all timesteps within 1e-10; signed phase drift 2.2055e-11; LB mass drift 5.7753e-10; max local force-momentum error 1.0882e-16; wall impulse balance 2.5249e-12. Max flux balance error 5.5511e-16. Final phase RMSE 6.5432e-14 against the independent bounded numerical implementation; max population difference 4.9914e-13 over all steps.
- Gravity run: phase-weighted COM x 4.958 -> 5.328, z 4.878 -> 4.356; zero-gravity z 4.850. These are cell-index coordinates with unchanged cell-centre offset, not dimensional trajectories. M20 vs M19 final phase RMSE 0.0367601; the limiter changes the physical trajectory. 62/96 limiter-active steps and 115,112 changed face updates (threshold 1e-14).
- Finite-shot 256 and 4096 shots per dissipative mode: final phase RMSE 0.00850256 and 0.00386106 respectively vs bounded ideal numerical reference, both zero bound violations. One seeded trajectory at each budget, not a statistical convergence study; errors are larger than M19's corresponding seeded errors. All runs use unchanged 21 dissipative modes/cell, hence naive 4096-shot budget remains 16,911,433,728 shots for collision alone.
- Material admissibility postprocessing: representative endpoints 998 and 1.2 kg/m^3 (ratio 831.67); rho=1.2+996.8*a. The M19 phase would map to min -5.286 and max 1061.150 kg/m^3, including 7,816 negative-density cell-step occurrences. M20 maps to the chosen material interval to roundoff. **This diagnostic does not couple density to momentum or pressure.** M20 remains a small-contrast Boussinesq model, with LB rho and rho/3 pressure proxy.
- Focused regression: 48 passed, 2 optional skipped, including 14 new tests. Results include independent adversarial random fluxes, saturated states, reflection, local conservation, no-op cases, and integrated ideal/sampled tests.

## Next direct development
Use a published pressure-based high-density-ratio phase-field LBM (Fakhari et al. 2017) as the matched classical numerical baseline. Validate equilibrium hydrostatic pressure, stationary droplet pressure jump vs 2sigma/R at several radii, low spurious currents, correct density/momentum consistency and viscosity contrast; then integrate the same moment dissipation and conservative streaming circuits. Evaluate the numerical diffusion of FCT (grid and interface-width convergence) and whether it remains appropriate for that model. Do not claim that density interpolation alone gives physical water–air simulation. After physical validation, address quantum implementation of equilibrium/forcing/limiting and readout overhead. Preserve all M1–M19 files.

- Complete historical regression: 235 passed, 11 skipped, zero failures (40.36 seconds).
