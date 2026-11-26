# Milestone 17 — reversible two-way phase–shear coupling

**Scientific classification:** matched-density, periodic, reduced Cahn–Hilliard / projected incompressible momentum research demonstrator. It is **not** a D3Q19 or D3Q7 LBM solver, not a water–air waterfall simulation, and not evidence of quantum advantage. The M16 interface-only implementation is retained unchanged. The D3Q19 amplitude-encoded GitHub research remains a separate baseline.

## Why this milestone

M16 executed a nonlinear conservative phase equation, but velocity was identically zero and the phase could not feed back on momentum. Simply appending an arbitrary force to a separate solver is not physically defensible. A compatible advection/force pair should exchange energy rather than create it. We first validate this pair in a tractable reduced model; then a full fluid momentum solver can reuse it. See `SOURCE_AUDIT.md` for the pinned GitHub source and literature evidence.

## Reduced physical model

Unit grid spacing, periodic axes (array order z,y,x), four y rows, constant matched density 1, `A=kappa=M=1`, kinematic viscosity `nu=1/8`, explicit Euler `dt=1/256`. The 3D phase field is unrestricted in principle; no clipping is used. The velocity is constrained to the one-dimensional divergence-free, zero-mean shear subspace `u_x=a*s(y)`, `s=[+1,+1,-1,-1]`, `u_y=u_z=0`. This mode has discrete `lap(s)=-2*s`. It is an illustrative Galerkin projection, not the complete Navier–Stokes equation; convection of the shear by itself is zero and the model cannot describe general velocity, pressure, gravity or capillary waves.

Cell chemical potential `mu=phi^3-phi-lap(phi)` is the discrete variational derivative of the free energy `E_phi=sum[(phi^2-1)^2/4 + sum_axes((phi_right-phi)^2/2)]`. On positive-oriented faces, `J_adv=phi_face*u_face`, `J_diff=mu_left-mu_right`, `phi_face=(phi_left+phi_right)/2`. The cell update is `dphi/dt=-div(J_adv+J_diff)`. Each edge has exactly one shared signed flux; periodic summation telescopes. Capillary force on x-faces is `F_x=phi_face*(mu_left-mu_right)`. The projected mode satisfies

```
da/dt = sum(s * F_x)/N - a/4
```

where N is cell count. The x face velocity is divergence free. Net momentum vanishes by the zero-mean mode constraint, not because arbitrary full-field force sums are proven conservative. With kinetic energy `E_k=N*a^2/2`, summation by parts gives

```
sum(mu*(-div J_adv)) + sum(u_x*F_x) = 0
(d/dt)(E_phi+E_k) = -sum_faces((mu_right-mu_left)^2) - N*a^2/4.
```

This is a **semi-discrete spatial identity**, before time discretization or fixed-point rounding. It does not prove unconditional energy stability of explicit Euler. Tested runs have no energy increases; this is empirical, not a general theorem. The diagnostics evaluate the unrounded spatial RHS at the quantized fields, not actual gate-rounding energy cancellation.

## Reversible fixed-point algorithm

Set `p=S*phi`, `a_tick=S*a`, `m=round(S*mu)`; the M16 chemical-potential oracle produces `S^3*mu`, then the new scaled-chemical oracle rounds by `S^2`. All shifts are dyadic and ties round away from zero. The face update uses one rounded combined flux:

```
J_tick = round((2*S*(m_left-m_right)+(p_left+p_right)*a_tick*s)/(512*S))
F_num  = s*(p_left+p_right)*(m_left-m_right)
T      = sum(F_num)   # gate-executed signed pair reduction
p_next = p + sum(incoming J_tick - outgoing J_tick)
a_next = a_tick + round((T-N*S*a_tick/2)/(512*N*S))
```

Other directions have zero advective velocity. The control removes capillary feedback (`T=0`) but preserves the same phase equation and viscosity. With zero initial velocity it remains at rest. The target solver does not call the reference solver. Host operations in `CoupledStepper` arrange known grid connectivity and pack/read basis words; every arithmetic stage is actually executed as X, CX and CCX gates via M16's bit-sliced simulator. Input/outputs have signed range and the cell/velocity updates flag overflow. Intermediate widths are chosen for the tested scales and grids; references use int64 within their tested domain. Local gate counts include each oracle's internal compute-copy-uncompute, not global routing or history costs.

## Executed experiments

Five matched 256-step trajectories on 4 x 4 x 8 (zyx) periodic cells. Initial field is `0.8*(0.65*sin(2*pi*x/8)+0.35*cos(2*pi*(x/8+y/4))+0.15*cos(2*pi*z/4))`, rounded once to the selected precision. This is a smooth manufactured phase pattern with nontrivial capillary projection; no physical droplet geometry. Identical quantized input is used for each paired integer and float reference.

| scale | feedback | requested a0 | final a | float final a | phase RMSE |
|---|---|---:|---:|---:|---:|
| 256 | yes | 0 | 0 | 0.008887239 | 0.168770099 |
| 4096 | yes | 0 | 0.006103516 | 0.008868666 | 0.009965431 |
| 65536 | yes | 0 | 0.009078979 | 0.008868609 | 0.000115473 |
| 65536 | no | 0 | 0 | 0 | 0.000107263 |
| 65536 | yes | 0.15 | 0.122680664 | 0.122630456 | 0.000113224 |

All 1,280 gate-batched steps match the separately coded integer reference exactly, conserve phase ticks and have zero mean mode momentum. No energy increase events observed. Highest-precision rest run energy goes 35.068177926 -> 28.896750552; maximum velocity error 0.000308667. Final coupled-control phase RMSE 0.000554688. All observed phase values remain in [-1,1], but boundedness is **not guaranteed** by this scheme. At scale 256 the capillary force is nonzero but the velocity is pinned by timestep quantization. Precision comparison is not spatial convergence. Physical water mass, density ratio and dimensional viscosity are not established.

## Global coherent execution (separate experiment)

`coherent.py` composes the same seven arithmetic types on persistent physical registers. A 4 x 4 x 4 grid (64 cells), two timesteps, S=65536 and two distinct supplied computational-basis inputs with amplitudes (1,i)/sqrt(2) are executed on an exact sparse complex support. All local gate calls are remapped to global wires; no intermediate word decoding, reconstruction, resetting or measurement. A signed pair-reduction tree widens the leaves using a dedicated adder. Temporary raw and scaled potentials, edge fluxes, reduction tree, constants and scratch are reversed after the field updates. The final field/mode can be copied in the basis encoding, after which reversal of the complete circuit clears intermediate history while retaining original input and the final copy. This copies basis labels, not arbitrary unknown quantum states.

| result | value |
|---|---:|
| logical qubits | 28,276 |
| forward X/CX/CCX gates | 20,624,658 |
| extra final copy qubits | 1,235 |
| compute-copy-reverse gates | 41,250,551 |
| maximum integer mismatch | 0 ticks |
| initial-to-final a ticks, branch A | 0 -> 17 -> 33 |
| initial-to-final a ticks, branch B | -10000 -> -10007 -> -10013 |
| inverse error | 0 bits |
| relative phase ratio | i |

Only basis-permutation gates act, so distinct input labels stay orthogonal and attached complex coefficients are preserved exactly; no amplitude-interference operation is claimed. Initial state preparation is supplied, not synthesized. No dense statevector of 28,276 qubits, QPU, native Qiskit execution, or speedup. Local oracle Qiskit export is inherited but the optional SDK test skips when unavailable. Counts are untranspiled and ignore hardware routing.

## Reproduction

From `experiments/quantum_dam_break/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r milestone14/requirements-tested.txt
bash milestone17/reproduce.sh
# optional complete historical regression:
bash milestone17/reproduce.sh --full-regression
```

`run.py` saves every phase and velocity history, CSV diagnostics, reports and nine ASCII VTI snapshots for the high-precision coupled rest run. `run_coherent.py` saves global history and coherent circuit report. `plot.py` makes benchmark and central-slice figures. `tests/test_m17.py` includes exact integer-reference trajectories, phase/momentum conservation, reflection, signed rounding ties, dirty output XOR, work identity, local sparse composition and global two-step copy/reverse. `results/environment.json` records runtime versions; `results/test_report.txt` and `results/full_regression.txt` are actual test output. Set `OPENBLAS_NUM_THREADS=1` to reproduce test runtime conditions.

## Next milestone acceptance criteria

1. Develop a multi-mode or full divergence-free velocity projection with staggered, adjoint capillary force and pressure solve, compare against an independent manufactured solution, and check mass/momentum and energy; do not feed classical force updates into a claimed all-quantum solver.
2. Study consistent variable-density NSCH mass/momentum transport before attempting actual water–air density ratio. Test spatial/time refinement and static droplet Laplace pressure, spurious currents and solid-wall contact angle. No water/air VOF claim before boundedness and mass definitions are validated.
3. Audit encoding bridge to the D3Q19 amplitude-based solver, including global heralding and readout overhead; no incompatible branch merge. Optimize signed arithmetic and history storage; execute native small Qiskit circuits when available. OpenLB or another independent physical benchmark remains required.
