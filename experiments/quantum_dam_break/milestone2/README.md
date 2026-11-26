# Milestone 2 — momentum-conserving D2Q9 quantum lattice-gas collision

This is a separate experimental branch of the model, not a replacement for the amplitude encoding from Milestone 1. No original QLBM source has been changed. The implemented physics is a Boolean lattice gas, **not a calibrated shallow-water dam break**. The purpose is to establish a conservative, dissipative collision mechanism before attempting the full fluid model.

## Run

From `experiments/quantum_dam_break`:

```bash
python -m pip install numpy pytest
python -m pytest -q tests milestone2/tests
python milestone2/run_collision.py
python milestone2/stress_benchmark.py
```

For notebook use, launch Jupyter from `milestone2` and open `02_quantum_collision.ipynb`. The optional Qiskit tests require compatible Qiskit and the upstream package installed. They were not executed here. `results/step_*.vti` can be opened as a time series in ParaView; `occupancy_evolution.gif` is generated from the actual trajectory ensemble. No result has been converted into SI water-depth units.

## Source reuse and architectural change

The upstream `qlbm/lattice/spacetime/properties_base.py` defines D2Q9 velocities and channel masses `[2,1,1,1,1,1,1,1,1]`. `qlbm/lattice/eqc/eqc_generator.py` partitions the 512 Boolean velocity configurations by mass and momentum. `qlbm/components/common/cbse_collision/cbse_collision.py` builds permutation, redistribution and inverse permutation; `cbse_redistribution.py` uses controlled rotations or truncated QFT. `qlbm/components/lqlga/collision.py` applies the local collision to all cells. `qlbm/components/lqlga/streaming.py` currently explicitly notes that its implementation must be generalized to 2D/3D and operates on axis 0.

We reuse the channel ordering, mass convention and equivalence-class construction. The executed implementation independently constructs Fourier blocks (not a direct run of upstream Qiskit code). `upstream_native()` directly imports the original EQC circuit, and an optional parity test compares its Born transition probabilities against our new circuit. The new 2D streaming operator uses wire SWAPs and link bounce-back. The original amplitude model has 4 direction bits encoding a single channel; the new model has 9 occupancy qubits per site. These cannot be interchanged without an explicit encoding redesign.

## Mathematical model

For a site bitstring b, m = 2 b0 + sum(b1..b8), p = sum(bi ci), and kinetic stress Pi = sum(bi ci ci^T). Each class G fixes (m, px, py). The collision unitary is a direct sum of normalized discrete Fourier transforms of each class. It commutes with mass and both momentum observables. Fourier blocks need not match the upstream unitary phase convention.

A coin ancilla is prepared by Ry(2 asin sqrt(omega)), controls the collision, then is reset. Finally each data qubit is dephased by copying to an ancilla and resetting it. The local channel is D[(1-omega) rho + omega U rho U†]. Explicit dephasing is a design decision to obtain a controlled dissipative kinetic model and to allow exact diagonal-trajectory simulation; it discards phase coherence. On diagonal input the transition matrix is (1-omega) I + omega |U|². This quantum channel does not evaluate BGK or call a classical fluid solver. The NumPy simulator samples that Born kernel at each site and applies a permutation of occupancy wires. Because dephasing keeps the joint state diagonal, this is a trajectory simulation of the specified quantum channel, not an approximation to an undephased coherent global circuit. It is efficiently classically simulable, and no quantum advantage is implied. The optional circuit specifies reset and SWAP operations, with no measurement feedback or re-preparation from a classical fluid solution.

Collision conserves local m and p for each trajectory. Streaming preserves global mass; in a periodic domain it also preserves momentum. Closed walls exchange momentum with the fluid through reflection. For an initially E-W occupied pair, the expected kinetic stress difference decays as 2(1-omega)^t. This is an exact local stress-relaxation benchmark, not a viscosity calibration. The pressure tensor exported is a kinetic second moment, not hydrostatic pressure.

## Actual execution

The default 8×4 simulation has a solid obstacle, closed walls, an initial rest-channel mass of 2 per left-half site and an empty right half. It uses 288 occupancy data qubits in the conceptual register, one reusable ancilla in the optional circuit, 2048 sampled trajectories, 16 steps, omega=0.7. No full 289-qubit statevector or actual hardware run was performed. All local Fourier blocks were numerically constructed and tested. See metadata and diagnostics for exact results. The included VTK fields are lattice-unit mass, mean velocity, kinetic stress components and solid mask. The GIF shows the ensemble average, not one circuit's shot count map.

## Limitations and next gate

- Boolean occupancy caps populations; this is not the previous square-root encoding or a discretization of the shallow-water equations. The equation of state, isotropy, sound speed and transport coefficients are not yet established.
- The pressure tensor is reconstructed but no hydrostatic pressure closure g h²/2 is implemented. Gravity, wet/dry-front physics and calibrated viscosity are open. Mass conservation alone is not validation of liquid flow.
- The native QLBM collision parity and Qiskit gate execution are pending; controlled dense unitary synthesis may be expensive. The 288-qubit scaling is a major tradeoff compared with Milestone 1's compact representation.
- Next: derive or select a physically consistent shallow-water equilibrium and encoding, verify a pressure-wave benchmark and shear-wave viscosity, then introduce bed-slope forcing and test against independent dam-break reference data. If Boolean occupancy cannot realize the target equation of state, use fixed-point population registers or another quantum-compatible approximation rather than labeling this gas as water.
