# Source audit and provenance

Baseline: `QLBM_Quantum_DamBreak_Milestone25.zip`, specifically `milestone25/run.py`, `milestone24/solver.py`, `milestone23/solver.py`, `milestone22/solver.py`, `milestone21/solver.py`, and `milestone25/results/slow212_gate.npz`. The archived run and replay are compared directly. The M25 archive has not been modified. No scientific algorithm is changed in this output milestone.

GitHub branch `aswanthbajith/QLBM:research/quantum-waterfall-impact-vqc` inspected via GitHub connector: revision `9794b4525eb08f0683b101afcdf30db3fca550a5`, commit title `feat(visualization): generate multi-angle and multi-property experimental GIFs for 3D waterfall`. No merge or remote write; its output is not used to generate this dam-break data. Other private master instruction/skill files remain unavailable as in preceding milestones.

The phase order parameter conversion, velocity norm, momentum density and dynamic-pressure diagnostics are direct definitions on the decoded fields. The pressure field uses the M23 solver's `p = rho * CS2 * P` and M22 explicit reference convention; the displayed downstream wall pressure is the adjacent cell's gauge pressure, not stress calculated at the wall. No new literature-dependent algorithm has been claimed. Prior scientific literature and limitations remain recorded in M25 and earlier source audits.
