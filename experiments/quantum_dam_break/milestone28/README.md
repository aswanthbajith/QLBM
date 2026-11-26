# M28 — Reversible macroscopic moments and material interpolation

Goal: convert easy, currently classical pieces of the existing M27 two-phase dam-break solver, without changing the governing equations. `QuantumMacroDamBreak` is an alternative backend; M27 remains unmodified.

## Circuit

Computational-basis signed fixed-point inputs are 19 pressure-distribution populations f and 7 phase populations h. With S=2^k, round each population to a signed integer tick. Reversible CDKM ripple-carry additions compute alpha=sum(h), P=sum(f), and j_d=sum_i c_id f_i for d=x,y,z. These are raw velocity moments of the pressure-based LBM (NOT physical momentum rho*u, and not yet force-corrected velocity). P is a normalized pressure variable, not physical pressure. Each accumulation uses a reusable sign-extended input word and a clean carry. The exact integer outputs retain the inputs, with no workspace garbage.

Density and dynamic viscosity are computed by reversible controlled shift/add multiplication and round-to-nearest (ties toward +infinity):

    A = alpha*S (integer)
    r0 = round(S*property_air), r1 = round(S*property_liquid)
    result_ticks = (r0*S + A*(r1-r0) + S//2)//S

The material stage directly reads the phase output register inside the combined circuit. It computes, copies, and uncomputes its scratch. The signed input is constrained to a representable phase sum (physical alpha near 0..1). Host range checks are not hidden clipping. The final outputs are decoded by dividing by S on the host. The pressure and viscosity coefficients are set classically once. No intermediate classical arithmetic is used between moment and material circuits.

At k=40: 1,621 logical qubits for ONE cell (1,118 inputs, 326 outputs, 177 shared scratch), 119,174 logical X/CX/CCX gates; 35,648 CCX. This is a reference oracle, not a practical hardware resource claim or amplitude-encoded grid-wide algorithm. Cell preparation and readout remain classical, and full-field reconstruction remains expensive. Basis-permutation circuits are simulated by actual compiled X/CX/CCX gate actions on bit-sliced basis inputs. A separate sparse complex-superposition test executes every gate and reverses the circuit. Optional Qiskit export test requires Qiskit; no QPU run.

## Integrated validation

`run.py` evaluates the initial M27 state and the stored final populations (step 1280) at fractional widths 16,24,32,40 over 8192 cells each; independently computes exact integer references; checks clean scratch, input preservation and inverse. The 40-bit backend then continues BOTH the unchanged M27 gate backend and new hybrid backend for 16 timesteps from the same stored step-1280 checkpoint, including fixed hydrostatic background and all existing dissipation, transport, and limiter operations. This is NOT a full restart of 1280 steps. See `results/report.json` and `history.csv`. No experimental physical calibration or hardware feasibility claimed.

Kinematic viscosity mu/rho, physical pressure rho*cs2*P, force corrections, equilibrium, MRT transform, surface tension, limiter, state preparation/readout remain classical. The cached hydrostatic background uses M27 initialization and reference collision. Lower precision can quantize the gas viscosity badly; a 16-bit demonstration is not adequate for production. The circuit is local, not coherent across timesteps or across the amplitude encoding used for streaming. The new results are a limited continuation; no full-dam trajectory has been rerun with this backend.

## Run

From experiments/quantum_dam_break (existing requirements from prior milestones):

    bash milestone28/reproduce.sh

Optional `--full-regression` also runs the historical suite. View `results/dam_break.pvd` in ParaView for five full-volume snapshots (steps 1280,1284,1288,1292,1296), or the multi-angle GIFs. The wall pressure diagnostic is cell-centred, not validated impact traction.

Next: quantum division for nu and quantum pressure multiplication, then compare resource-efficient arithmetic and investigate an encoding compatible with transport. No remote GitHub modification.
