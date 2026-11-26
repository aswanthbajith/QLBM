# Milestone 15 — Coherent nonlinear collision–streaming with retained history

## Research question
Can the M14 D2Q25 symmetry-covariant collision dilation be used for *two full timesteps* without measurement, reset, or classical reconstruction between steps? Can its orientation registers be cleaned after copying the final physical output? This is an addition-only experiment, not a new physical model. The original 16×16 dry-front pressure feasibility problem remains unresolved.

## Model and circuit
The fixed-point nonnegative D2Q25 population words of M13/M14 are retained. A closed 2×1 cell strip (dx=0.5, dt=1, g=1/8 in lattice units), full relaxation omega=1, and two input basis configurations are used. A is a resting reservoir gradient h=(1.5,0.5); B is an asymmetric seeded kinetic configuration with the same total depth but nonzero momentum. These two states are coherently prepared with relative phase i. B is a diagnostic state, not a validated physical initial reservoir. Preparation is represented by the specified sparse input amplitudes; a general state-preparation circuit is not included in resource counts.

The word registers F0,F1,F2 remain distinct; each collision writes into a zero destination via reversible XOR and cleans its arithmetic scratch. Each cell and timestep has fresh three-bit orientation labels and one overflow flag. Every local gate from M14 is explicitly remapped into persistent global wire addresses, then physical streaming is implemented as a word-SWAP circuit (three CX per bit). The same scratch is reused across cells and steps; neither population nor selector registers are measured or reset during execution. The H gates on all 12 selectors commute to the start because later selectors are untouched until used. Their support is expanded exactly and the indexed X/CX/CCX gate stream is executed on every basis label in a bit-sliced representation. Complex amplitudes are retained in parallel, so this is an exact sparse-state simulation for this gate set, not a dense statevector simulation or a QPU run. No classical fluid solver or lookup is used to advance the state. An independent integer reference explicitly computes local collisions and closed-wall destination links, separately from gate execution. It verifies local mass/momentum, every stored population and flags, not hydrodynamic convergence.

For two timesteps, the output has the form

    sum_f a_f / 64 * sum_(g0,g1) |F0=f>|F1>|F2>|flags>|g0,g1>|0_scratch>

where g0 and g1 each contain independent orientation labels for both cells (64 possibilities per step). The 4096 label combinations per input are all represented; there are 8192 nonzero terms for the two-input superposition. The two-step history is a reversible dilation of the stochastic population map, not a unitary map on only the final fluid population. All gates are inverted on the same global basis wires and then the inverse 12 Hadamards are executed numerically on the complex amplitudes. Reversibility is verified for the complete prepared state. Qiskit construction is provided in `Circuit.qiskit()`; optional test skips without Qiskit. The arbitrary input superposition itself is supplied, not synthesized as a state-preparation circuit. No hardware transpilation or real quantum processor execution is claimed.

## History cleanup limitation
The complete forward circuit is invertible when the final result is not retained. If F2 is copied with CNOT into a new computational-basis register before reversing, the inverse permutation returns all old population/flag registers to their original values, but the copied output may remain entangled with the orientation labels. For a fixed input and uniformly weighted labels, the probability that all selector qubits end at zero after inverse H is sum_y p(y|f)^2, where p is the final-output distribution over labels. For the input superposition this is averaged with the input basis probabilities. The formula is evaluated from the executed histories and independently checked by an explicit Hadamard transform of the output-conditioned amplitude arrays in the tests. This is an exact calculation of the copied-output experiment, not an additional full gate-stream execution with an allocated copy register. Probability <1 means unconditional cleanup is not achieved. It is not a universal lower bound on all possible channel implementations. It demonstrates the limitation of the simple copy/reverse method here. A deterministic collision branch can be uncomputed with a retained basis-word result, but a branch-dependent output generally leaves environment correlations. Do not reset a selector in a purported coherent trajectory without modeling the resulting channel.

## Reproduce
From `experiments/quantum_dam_break`:

```bash
python -m pip install -r milestone14/requirements-tested.txt
bash milestone15/reproduce.sh
```

The script reruns two precision settings and all previous tests. Qiskit is optional and not installed here. `results/coherent_histories.npz` and `results/precision64/coherent_histories.npz` hold complete basis histories, flags, selector labels, complex amplitudes and input states. `observables.csv` gives the ensemble means and standard deviations of physical depth and discharge. These are derived post-run, not used in the circuit. `report.json` records circuit and validation metrics. No new GIF or VTK sequence is generated: two cells are not an informative dam-break visualization; the prior validated numerical comparisons remain available in M14.

## Executed results
| Metric | S=16 | S=64 |
|---|---:|---:|
| Population bits | 6 | 8 |
| Nonzero sparse-state support | 8192 | 8192 |
| All history values vs independent reference | exact | exact |
| Flag mismatches | 0 | 0 |
| Mass conservation error (ticks) | 0 | 0 |
| Negative populations / overflow flags | 0 / 0 | 0 / 0 |
| Norm | 0.9999999999999999 | 0.9999999999999999 |
| Full inverse amplitude max error | 6.66e-16 | 6.66e-16 |
| Unique final fields for A / B | 1 / 8 | 1 / 6 |
| Copy then inverse: all selector bits zero probability | 0.60345459 | 0.64648437 |
| Logical qubits (incl. reused scratch) | 66104 | 100614 |
| X/CX/CCX gates | 869160 | 1319276 |
| Hadamard gates | 12 | 12 |

The same arithmetic scheme is tested at two precisions, but the seeded multinomial state B changes when its population tick count changes; this is not a matched precision-convergence experiment. The resting initial condition A is pinned at both resolutions in this tiny closed strip. At S=16, B's mean left depth decreases from 1.5 to 1.3276367 after two steps; at S=64 it decreases to 1.43359375. This verifies nontrivial circuit transport, not accurate dam-break propagation. Wall reflection can transfer momentum to the wall; total fluid momentum is not generally conserved across the closed boundary, while local collision momentum is conserved. The fixed-point pressure model still cannot represent all dry-front states. No physical validation, viscosity calibration, two-phase interface model, resource advantage, or real hardware execution follows from these results.

## Next research gate
1. Reduce local scratch allocation with lifetime/reversible pebbling analysis; quantify history and selector scaling beyond two steps. Avoid claiming clean selectors after irreversible output retention.
2. Treat pressure realizability as a separate physical-model gate: compare signed populations and/or a different wetting scheme on a matched domain, resolution and time, with positivity/stability and analytical convergence tests before quantum compilation.
3. For the newer GitHub D3Q19 research, audit actual postselection and measurement semantics and nonlinear phase-field coupling independently. Do not merge this depth-averaged D2Q25 code into the 3D free-surface branch without a verified mapping. See SOURCE_AUDIT.md.

Final regression: 164 passed, 7 skipped, 0 failed (6 new executable tests and one optional Qiskit skip). The initial development run caught a Python method/attribute name collision (`history`); renamed the decoder `unpack_history` and reran both benchmarks and full tests. Qiskit was absent. Gate counts exclude state preparation, final measurement, and optional copied-output experiment. Exact test log and environment versions are provided.
