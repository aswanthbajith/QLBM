# Milestone 22 — Explicit pressure gauge and hydrostatic residual subtraction

**Objective:** improve the integrated two-phase water–air dam-break hybrid QLBM, preserving the M21 D3Q19/D3Q7 solver, M20 limiter and M18–21 quantum subroutines. This is not a replacement solver, a fully quantum evolution algorithm, or an experimental dam-break validation.

## Method

M21 encodes `P = p/(rho cs²)`. At variable density, an arbitrary constant shift in physical pressure changes P nonuniformly. M22 explicitly takes a pressure-reference scalar and stores gauge pressure `p_g = p_abs - p_ref` before encoding. `decode()` returns gauge pressure; `absolute_pressure()` adds the reference. `initialize_gauge()` accepts gauge pressure directly to avoid subtractive cancellation. This is invariance under a **joint** shift of supplied absolute pressure and reference, not invariance to changing encoded gauge pressure while holding the reference fixed. It does not correct the underlying variable-density continuum equations.

A fixed horizontally uniform hydrostatic background is constructed from a supplied phase profile, with top face gauge pressure zero and cell-centred integrated density. Let `F_H` denote its populations (including the half-force shift so decoded velocity vanishes) and `H_H` the frozen phase populations. `C` is the hydrodynamic collision including force, `S` the M18 closed-tank link transport. At each actual step:

```
F_next = F_H + S(C(F,H) - C_0(F_H,H_H))
```

`C_0` is the M21 collision with capillarity disabled only in the fixed reference. Its output is cached once independently for the gate and reference backends. Actual capillary force and phase transport remain active. The residual is independent of the evolving state. It has zero global pressure-population sum to floating-point accuracy because the collision and streaming conserve that sum. The correction does **not** establish physical momentum conservation or local well-balance for arbitrary moving interfaces. Subtracting a fixed residual leaves the frozen-interface Jacobian unchanged at a given state. For the dynamic experiment the reference uses the initial horizontally averaged liquid fraction; this choice is experimental and is not uniquely physically mandated. The default `BalancedDamBreak` without a background only applies explicit gauge handling.

The hydrodynamic difference populations are signed amplitudes. A nonzero difference is normalized and streamed by the existing compiled fully-controlled-X network, then reconstructed and the background added classically. A zero vector has no valid normalized quantum encoding and bypasses transport. Phase collision/limiting/streaming are unchanged. The ideal dual-rail dissipative circuits continue to execute batched complex gate actions; pressure preparation, subtraction, reference caching, nonlinear equilibrium, limiter, forcing, reconstruction and diagnostics remain classical. No measurement-based state tomography, native Qiskit execution, or QPU execution is demonstrated.

## Reproduce

From `experiments/quantum_dam_break` with the previous Python environment:

```
bash milestone22/reproduce.sh --full-regression
```

The source also runs with `python milestone22/run.py`, `python milestone22/stability.py`, `python milestone22/plot.py`. Results: CSV, JSON, NPZ, plots, five VTI files (gauge pressure). Tested NumPy/Python versions in `report.json`. No stochastic draws. Tests share physical equations with M21, but use its separate population-space collision and independent link streaming/limiter reference.

## Results and limitations

- Matched 96-step 32×4×16 high-ratio dam release (831.6667:1, dynamic viscosity ratio 55.5556:1, g=0.0005 lu): bounded phase, phase drift 8.64e-12, material-mass drift 8.58e-12, phase RMSE gate/reference 3.66e-16, max final population discrepancy 9.01e-15. Fixed-background correction changes final phase relative to uncorrected baseline by RMSE 6.59e-4; no external reference establishes which is physically superior.
- Explicit gauge offset 1e-5 (no background) and 1e-3 (with background) leaves matched trajectories unchanged to roundoff when the same offset is supplied as reference. This does not establish general invariance to arbitrary numerical gauge choice.
- Frozen hydrostatic reference at density ratio 831.67:1: exact rest preserved over 64 steps for widths 0.1, 0.5, 1.0 lu; at zero perturbation the construction cancels the fixed residual. This is a targeted equilibrium result, not a claim of stability. A 1e-7 velocity perturbation fails the 0.1 speed guard at step 4 (width .1) and step 54 (width .5). Width 1.0 decays to 1.65e-9 by step 64. Phase is frozen in all these tests; capillarity disabled.
- Finite-difference central Jacobian of the actual frozen-interface reference map (608 population variables, eps=1e-8 and 1e-9) confirms spectral radii 281.7015, 1.30518 and approximately 1 for the three widths. Counts of modes with |lambda|>1+1e-5: 17, 2, 0. This is a local linear diagnostic, not a proof of nonlinear stability. Equal-density control has no unstable modes by this threshold. Results at both eps values agree closely.
- Strong gravity g=0.003 lu fails after step 3 even with the correction. No improvement to the acceptable high-gravity regime established. Keep optional background correction experimental until independent physical tests justify it.
- Full regression: 256 passed, 11 skipped. Focused regression: 69 passed, 2 skipped. No historical code modified. All units are lattice/nondimensional except representative material ratios; no SI calibration, physical Laplace law validation or experimental dam-break comparison yet.

## Next focused step

Use the instability spectrum as a regression target when replacing the pressure/force discretization with a genuinely well-balanced high-density-ratio formulation. Validate perturbed hydrostatic columns (not only exact rest), two-layer viscosity flow, and Laplace law before stronger gravity and full dam-collapse tests. Preserve the quantum transport and dissipative circuit interfaces. Only after a physically reliable solver is established should more of the classical coupling be compiled into quantum operations.
