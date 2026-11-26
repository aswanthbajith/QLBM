# M22 scientific and repository audit

Execution baseline: QLBM_Quantum_DamBreak_Milestone21.zip, specifically inspected milestone21/solver.py, run.py, tests, report and M18–M20 transport/collision/limiter dependencies. Existing archive contents preserved. The fixed research objective is two-phase water–air dam-break simulation on a quantum computer using QLBM. No unrelated numerical solver introduced.

## Literature verified

- Fakhari, Mitchell, Leonardi & Bolster (2017). Improved locality of the phase-field lattice-Boltzmann model for immiscible fluids at high density ratios. Physical Review E 96, 053301. https://doi.org/10.1103/PhysRevE.96.053301 . Verified publisher abstract: high density and viscosity contrasts, pressure/velocity decoupling, local stress computation. M21 is already an adaptation, not a faithful reproduction.
- Guo, Zheng & Shi (2011). Force imbalance in lattice Boltzmann equation for two-phase flows. Physical Review E 83, 036707. https://doi.org/10.1103/PhysRevE.83.036707 . Publisher abstract verified: discrete force-balance analysis of a flat interface. Motivation for examining equilibrium residuals, not a claim to implement their scheme.
- Guo (2021). Well-balanced lattice Boltzmann model for two-phase systems. Physics of Fluids 33, 031709. Preprint https://arxiv.org/abs/2012.11075 . Verified abstract: discrete force imbalance can prevent exact physical equilibrium. M22 uses a fixed-reference residual subtraction, not Guo's modified equilibrium and force scheme.
- Theoretical and numerical study on the well-balanced regularized lattice Boltzmann model for two-phase flow (2023). Physical Review E 108, 055309. https://doi.org/10.1103/PhysRevE.108.055309 . Verified abstract: well-balanced Hermite expansion and force discretization, stationary droplet validation. Candidate for subsequent implementation.
- A Well-Balanced Lattice Boltzmann Model for Binary Fluids Based on the Incompressible Phase-Field Theory (2025 journal listing; preprint 2023). https://www.global-sci.com/cicp/article/view/17176 and https://arxiv.org/abs/2311.10827 . Verified abstract on discrete force balance, Cahn–Hilliard source formulation. Not adopted in M22.

These papers establish the scientific relevance of discrete force balance. The M22 explicit pressure-gauge API, fixed planar reference, and numerical stability analysis are independent implementation choices; no claim that any publication prescribes this precise correction or that M22 reproduces their benchmarks.

## GitHub and skills

Live branch list for aswanthbajith/QLBM checked; `research/quantum-waterfall-impact-vqc` still at `66a0de43d066db20756872c4f48b79ebd10b9be0`. Other listed branches unchanged from M21 listing; their heads were not all re-fetched. No remote clone, remote modification or merge; local source patch against M21 archive. No dedicated quantum/QLBM/scientific-literature plugin skill present in the skill list. Private master instruction and external Codex/Antigravity skills were not accessible in this session. Prior project instructions followed from conversation. No claim to execute inaccessible skills.
