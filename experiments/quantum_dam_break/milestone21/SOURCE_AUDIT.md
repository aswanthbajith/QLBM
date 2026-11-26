# M21 source audit

Baseline: exact QLBM_Quantum_DamBreak_Milestone20.zip, inspected milestone19/solver.py, collision.py, milestone18/transport.py, channel.py and milestone20/solver.py, limiter.py. All previous code and results retained. No private master instruction/agent skill access established; focused objective from conversation: two-phase water-air dam-break on a quantum computer.

## Scientific source

Fakhari, A., Mitchell, T., Leonardi, C., and Bolster, D. (2017). Improved locality of the phase-field lattice-Boltzmann model for immiscible fluids at high density ratios. Physical Review E 96, 053301. https://doi.org/10.1103/PhysRevE.96.053301 . Accepted manuscript https://link.aps.org/accepted/10.1103/PhysRevE.96.053301 . Verified text of equations 1, 4-7, 10, 13-21, 24-29, 32-35, and choice of zero light-phase fraction. PDF page screenshot retrieval failed (cache miss); text was accessible. This is not a reproduction of their exact D2Q9 model or benchmarks. Our pressure product-rule discretization, 3D lattices, moment basis, finite-difference viscosity correction, D3Q7 phase collision and limiter are explicit independent implementation choices.

Khan, Succi, Yao (2026), Deterministic Realization of Classical Dissipation on Quantum Computers, https://arxiv.org/abs/2604.25429 . Reuses M18-M20 verified dual-rail channel; only the per-cell angle support is new. Prior source audit and circuit tests preserved; no claim of deterministic readout or full nonlinear collision by this channel alone.

## GitHub

Live branch list of aswanthbajith/QLBM checked 2026-11-25: main, research/multiphase-vof, research/quantum-fluid-evolution, research/quantum-fluid-verification, research/quantum-multiphase-benchmarks, research/quantum-phase-field-circuits, research/quantum-waterfall-impact-vqc, research/variational-quantum-multiphase. Head of research/quantum-waterfall-impact-vqc verified at 66a0de43d066db20756872c4f48b79ebd10b9be0 (unchanged since M20). No full remote clone or independent reproduction in M21, no merge/push. The authoritative execution baseline for this milestone is M20 archive, not a claim that every remote branch was integrated.
