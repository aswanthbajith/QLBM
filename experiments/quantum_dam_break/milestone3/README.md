# Milestone 3 — pressure, forcing and viscosity acceptance gates

**Result: the existing Boolean lattice gas is not a calibrated shallow-water solver.** The tested equilibrium fails the hydrostatic pressure closure. We preserve it as a kinetic quantum-channel experiment, and add two reversible fixed-point arithmetic primitives as a separate candidate architecture. All results here are in lattice units, not SI water properties. No complete dam-break simulation or hardware execution is claimed.

## Reproduce

From `experiments/quantum_dam_break`:

```bash
python -m pip install numpy pytest
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests milestone2/tests milestone3/tests
OPENBLAS_NUM_THREADS=1 python milestone3/run_physics.py
```

Open `03_pressure_gravity_viscosity.ipynb` with its working directory set to `milestone3`. `run_physics.py` regenerates the CSV and JSON results; optional `plot_results.py` requires Matplotlib and generates the included diagnostics figure. Optional Qiskit checks require a compatible installation and were skipped here.

## 1. Exact equation of state audit

Reuse the M2 velocity and mass convention and the same conservative Fourier-block collision. For basis configuration b, `m(b)=2*b0+sum(b1..b8)`, `p(b)=sum(bi*ci)`. The grand-canonical diagonal state is `w(b)=exp(lambda*m(b)+beta·p(b))/Z`. Its weights are constant within collision classes, so it is stationary under M2's dephased channel. At rest beta=0, the equilibrium is a product of independent Bernoulli occupancies: each moving bit has probability `z/(1+z)`, the rest bit has probability `z²/(1+z²)` where `z=exp(lambda)`. Therefore the mean mass is `m=8*z/(1+z)+2*z²/(1+z²)` and `Pi_xx=Pi_yy=6*z/(1+z)`; cross stress vanishes. The derivative `dPi_xx/dm` is computed by covariance over the 512 basis configurations, and checked by finite differences. At small m, the kinetic pressure tends to 0.75*m, whereas shallow-water hydrostatic momentum flux is proportional to h². A constant scaling of mass and stress cannot fix this low-density power-law mismatch.

We tested 39 means between 0.25 and 9.75 and fit a single coefficient g to `g*m²/2` by unweighted least squares. Relative L2 error uses the kinetic-pressure vector as the denominator. This is a failed physical acceptance gate (27.2% relative error), not evidence that no possible correlated occupancy equilibrium could ever be engineered. No pressure-wave speed has been independently measured; the derivative is an EOS diagnostic.

## 2. Directional quantum force channel

For each pair of opposite components in the chosen axis (for x: E/W, NE/NW, SE/SW), if exactly one wire is occupied, replace the pair with a one-excitation reservoir with probability `rate`. Its positive-axis probability is `(1+bias)/2`. The optional Qiskit circuit computes parity, controls Fredkin swaps with a prepared reservoir, resets five ancillas, and dephases the data. The exact diagonal channel and Monte Carlo simulator implement this map without computing a separate classical fluid solution. Mass and transverse momentum are invariant on every basis transition. The mean axis impulse for a pair is `rate*(bias*(b_plus XOR b_minus) - (b_plus-b_minus))`. Consequently effective acceleration depends on occupancy and includes relaxation of pre-existing momentum. For fixed bias and rate it is not a uniform physical gravity term. Seven density points are tested with 100,000 samples each; the exact response is available separately. No real gravity constant is assigned.

## 3. Shear-wave transport

A periodic 8×16 lattice is initialized from product equilibria with a small sinusoidal transverse drift, beta_x(y)=0.15*sin(2*pi*y/16), lambda=-0.6. We simulate 2,048 diagonal quantum-channel trajectories for 30 steps using the unchanged M2 collision/streaming. For the ensemble transverse Fourier mode A(t), estimate effective nu from the log-linear decay `A(t)~exp(-nu*k²*t)`, k=2*pi/16. The predeclared fit uses steps 4–20 to omit the first few transient steps; the sample standard errors describe cross-trajectory variation, not independent uncertainty across time. The fitted nu is approximately 0.885 lattice units²/step, R²≈0.991; all trajectories conserve mass and periodic total momentum. This is one finite-size, finite-wave-number kinetic transport estimate, not a convergence study or a physical water viscosity calibration.

## 4. Alternative architecture: reversible fixed-point quantum arithmetic

A binary register stores a discretized physical variable directly, rather than identifying fluid depth with a single Boolean occupancy. For an unsigned three-qubit depth word H, `h=H/4`. A six-qubit output word P represents `pressure=P/16`. The pressure oracle applies

`|H,P> -> |H,P XOR round(16*g*(H/4)^2/2)>`.

We choose dimensionless g=1 and ties rounded upward. The compiler enumerates only eight depth words and emits a controlled-X gate for each output bit that should flip. These are reversible gates, including on a superposition; no classical fluid timestep is executed at runtime. Nine data qubits and 13 pattern-controlled X gates suffice for this tiny example. Maximum rounding error is half an output LSB (0.03125). The oracle is self-inverse, and applying it twice restores arbitrary superpositions. This is a table-compiled proof of concept, not efficient quantum arithmetic at realistic precision: gate count scales exponentially with input bit width and MCX decomposition costs are not included.

For a prescribed bed slope or body acceleration, momentum q is stored as a signed five-qubit two's-complement word Q with scale 1/16. The bed source circuit implements `|H,Q> -> |H,Q + round(16*h*a*dt) mod 32>`. For a*dt=-1/4, the increment is exactly -H in register units. The gate compiler uses controlled modular increments with carry conditions. All 256 input words were checked, including negative momenta and reverse-gate uncomputation. The example starts at q=0 and has zero quantization error. The modular circuit is reversible; overflow of physical signed values must be prevented by the eventual solver, and no saturation or clipping is silently performed. Eight data qubits, 69 pattern-controlled X gates. A negative source is not a complete gravity treatment: shallow-water pressure includes gravity even on a flat bed, while the bed-slope source must be balanced against flux gradients for lake-at-rest states.

## 5. Next acceptance gates

The fixed-point route is a new candidate architecture, not drop-in compatible with M1 or M2 encodings. Reuse position, reflection, measurement, and reversible arithmetic mechanisms where compatible. Develop a conservative interface flux oracle for `(h,qx,qy)` with explicit precision and overflow rules. For a finite-volume update, adjacent cells must receive equal and opposite interface transfers; store old registers or other reversible history and uncompute temporary work. A generic dissipative update need not be invertible on the physical registers alone. Verify positivity, dry cells, CFL stability, lake at rest, volume conservation, dry-front speed, and independent reference comparison before calling it a quantum dam break. A classical reference is for validation only. Actual circuit execution and resource estimates, including history and ancillas, remain open. No claim of speedup.

## Output files

- `pressure_eos.csv`: exact equilibrium pressure, fit, derivative, stationarity residual.
- `force_response.csv`: exact and sampled impulse and mass invariant.
- `shear_decay.csv`: measured Fourier amplitude, standard error, mass/momentum error.
- `pressure_oracle.csv`, `bed_source_oracle.csv`: fixed-point basis outputs and errors.
- `report.json`, `run_log.txt`, `test_report.txt`, `benchmark_diagnostics.png`.

There are deliberately no new water-depth VTK or dam-break GIF files for this milestone: doing so would imply a physical evolution that has not yet been implemented. Prior outputs are preserved.
