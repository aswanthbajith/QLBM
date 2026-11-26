# Milestone 10 — multispeed moment feasibility and reversible primitives

## Run
From `experiments/quantum_dam_break`:

```bash
python milestone10/run.py
python milestone10/plot.py
python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10}/tests
```
Dependencies: numpy, matplotlib, pytest; previous milestones additionally use numba and pillow. Optional Qiskit export is not executed in this environment.

## Mathematical finding
For a finite 1D speed set `v`, nonnegative normalized populations with mean `u` have minimum second moment `(a+b)u-ab` where adjacent speeds `a<=u<=b`. The maximum second moment is `(vmin+vmax)u-vmin*vmax`. Equivalently, required variance `g*h/2` must lie between `(u-a)(b-u)` and `(u-vmin)(vmax-u)`. These inequalities are necessary and sufficient for one marginal. A convex combination of adjacent-speed and endpoint distributions achieves any moment between the bounds. For Cartesian tensor-product 2D velocity sets, independent marginals reproduce the required cross moment `h*ux*uy`, so the two marginal constraints are also sufficient for the specified second-moment tensor. This is a realizability construction, not a hydrodynamic convergence proof. No fixed finite set resolves all positive depths approaching zero at arbitrary velocities between speed nodes.

Candidate labels D2Q9/25/81/289/1089 denote Cartesian products of the listed 1D speed sets, not established standard LBM quadratures. Intermediate physical speeds are realized by integer displacements on a common subgrid. D2Q25 uses {-1,-1/2,0,1/2,1}; D2Q81 uses increments 1/4 on [-1,1]; D2Q289 uses increments 1/4 on [-2,2]; D2Q1089 uses increments 1/8 on [-2,2]. Sample fractions are one-axis depth–velocity pairs, not percentages of 2D flow cells; they depend on chosen sample range. Ritter fan coverage sampled uniformly in x at t=1. Product equilibrium tests use random feasible 2D inputs. Higher-order moments, isotropy and stability not established.

## Classical screening (separate from quantum execution)
`kinetic.py` tests the marginal kinetic evolution with omega=1, g=1/8, initial reservoir 1.5, dry downstream, dt=1, dx=1/8, periodic 2048-cell strip, 24 steps. The central region |x|<28 is compared to Ritter before interaction with the distant interface (distance 128). When the pressure moment is impossible the equilibrium clips its second moment to the feasible interval, preserving mass and momentum. Signed D1Q3 uses exact moment algebra without positivity protection. The runs are exploratory; the candidate equilibrium uses endpoints and adjacent speeds, may suffer lattice subgrid decoupling, and has not been derived as a consistent higher-order kinetic scheme. Larger lattices can worsen error. All results are in lattice units. This is **not** a quantum dam-break simulation; no new physical VTK/GIF is produced.

## Actual quantum primitives
`oracle.py` implements a small reversible D1Q5 equilibrium lookup. Inputs: unsigned depth H (3 bits), signed momentum J=sum k_i*f_i (5 bits), k=(-2,-1,0,1,2), physical velocities k/2, population scale S=4. Exact integer enumeration at compile time chooses nonnegative populations of sum H and momentum J minimizing absolute second-moment error relative to J^2/H+H^2/(4*S), with deterministic tie breaking. The flag indicates impossible integer **mass/momentum**, not exact pressure feasibility. Runtime consists solely of pattern-controlled X gates; output is XORed, inputs retained. Not scalable; not a full collision update. All 256 input patterns are tested; a smaller 17-qubit version is actually executed on a complex statevector with a nontrivial relative phase. The Qiskit export remains optional and unverified.

`transport.py` adapts M1's transposition/Gray-path mechanism to a 25-direction lattice on 4x4 cells. Long links encountering any solid cell in the symmetric bounding rectangle reflect at their source. This is a conservative, reversible but over-restrictive corner treatment, not a physically validated wall model. Word SWAP execution and amplitude-basis permutation are separately checked; the amplitude statevector has 9 qubits including padding. Neither is connected to an integrated D2Q25 fluid timestep yet.

## Next decision
D2Q25 is the best candidate in this specific strip screen, not a globally validated model. Before building a scalable D2Q25 quantum collision, investigate better equilibrium selection (higher moments), dry-front consistency, grid/time refinement, and stable adaptive or alternative kinetic treatments. Consider signed distributions only with demonstrated stability. Then implement reversible moment reconstruction, collision, and multispeed transport in a common fixed-point register architecture. No physical viscosity calibration, OpenLB validation, quantum speedup or QPU evidence in M10.
