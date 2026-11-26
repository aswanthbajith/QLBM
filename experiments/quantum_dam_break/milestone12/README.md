# Milestone 12 — D2Q25 reflection preservation and dyadic BGK relaxation

## Scope and reproduction
This extension retains M11's 25 nonnegative population words, projected marginal equilibrium, and exact streaming/reflection permutation. It adds dyadic BGK rates omega = 1/4, 1/2, 3/4, 1 and conditional local reflection preservation. From the `experiments/quantum_dam_break` directory:

```bash
python -m pip install numpy numba pytest matplotlib pillow
python milestone12/run.py
python milestone12/plot.py
python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11,12}/tests
```

`collision.py` compiles the update to X/CNOT/Toffoli gates, using the indexed arithmetic builder inherited from M8 and M11. `reference.py` independently implements the integer update (sharing M11's equilibrium formula), while `run.py` also advances a separate floating-point model. The quantum circuit retains its input, XORs the result and overflow flag into output registers, and uncomputes its scratch. The full simulation uses computational-basis bit-sliced gate execution and reversible word permutations; new basis states are prepared between timesteps. No full-grid statevector, Qiskit run or QPU run is claimed.

## Mathematical update
Let E(F) be the integer projected equilibrium from M11, H=sum F and J=sum F*K with integer velocity indices K=(-2,-1,0,1,2) along each axis. For m in {1,2,3,4}, first floor each component of ((4-m)F+mE)/4. For partial relaxation the missing mass and momentum are exactly those of the fractional remainders; they lie within the convex hull of the velocity set. Reconstruct residual nonnegative integer marginals on adjacent speeds, then greedily couple them into 25 channels. This exactly restores H,Jx,Jy for valid local states. It is a conservative rounding strategy, not an exact real-valued BGK step. Residual greedy allocation is not generally reflection equivariant.

For each reflection axis on which the INPUT population array is exactly symmetric, replace each mirrored output pair (a,b) with (min(a,b),min(a,b)) and add |a-b| to the corresponding zero-normal-speed channel. This preserves mass and tangential momentum; the input normal momentum is zero and the resulting normal momentum is zero. Two projections preserve both symmetries when both were present. This is NOT a proof of general reflection equivariance for arbitrary nonsymmetric inputs. The projection changes normal kinetic pressure and can introduce extra diffusion. It does not fix the finite-velocity moment-feasibility problem.

## Executed experiments
- 16x16 synthetic obstacle case, S=256, omega=0.5, 16 steps; all mass and local collision momentum checks exact, zero solid penetration, every update agrees with integer reference. 80 -> 219 wet cells; 99 final wet cells still pressure-infeasible.
- 256-cell periodic strip, dx=.5, S=1024, 24 steps, common symmetric initial populations. M11 full relaxation has maximum transverse velocity 0.0067114; M12 full and half relaxation have exactly zero. Full-relaxation analytical depth RMSE is unchanged at 0.0840360, half relaxation increases it to 0.1074878 (central |x|<28 region). This is a matched comparison, not a spatial convergence study.
- 32-cell periodic shear wave, depth 1.5, initial x-velocity amplitude about 0.08, dx=.5, S=1024, 40 steps. Fit log fundamental amplitude over steps 5..40 to estimate effective viscosity. Measured values for omega=.25,.5,.75,1 are 0.08059,0.08986,0.05857,0.04745. The formal long-wave reference is (g*h/2)*(1/omega-1/2), not a validated calibration for this projected equilibrium at the chosen wavelength. The floating comparator also deviates at low omega. Rounding, higher moments and wavelength effects remain; the gate fit is nonmonotonic at the two smallest rates. Molecular water viscosity is not established.

All numbers are in lattice units. `results/report.json` has raw diagnostics and circuit resources. VTK fields are inherited from M11 and remain explicitly named physical and kinetic quantities; reference depth is the separate floating model. All generated plots and GIF frames derive from executed arrays, not drawn illustrations.

## Limits and next steps
The 16x16 omega=.5 circuit needs 257,053 conceptual logical qubits and 101,832,615 untranspiled X/CX/CCX-equivalent gates per timestep, so it is not feasible on current small quantum machines. The primary research tasks are improved conservative rounding and moment matching; long-wave viscosity convergence across wavelength and precision; native small Qiskit execution; independent shallow-water validation and a suitable OpenLB reference. Neither true free-surface 3D water-air dynamics nor quantum advantage is established. Earlier milestones remain unchanged.
