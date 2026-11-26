# Milestone 11 — Integrated reversible D2Q25 collision–streaming

## Run
From `experiments/quantum_dam_break` (Python environment with numpy, numba, pytest, matplotlib, pillow):

```bash
python milestone11/run.py
python milestone11/plot.py
python -m pytest -q tests milestone{2,3,4,5,6,7,8,9,10,11}/tests
```

The new model is a Cartesian-product 25-speed shallow-water kinetic experiment. It is not a standard validated D2Q25 quadrature. The previous experiments remain intact. No original QLBM source was changed.

## Physical state and lattice
Each fluid cell contains 25 nonnegative fixed-point population words F_i, with f_i=F_i/S. Speed index k=(kx,ky) has components -2,-1,0,1,2; physical lattice velocity c=k/2. The population order is y-major then x-major. H=sum F_i, J=sum k_i F_i. Depth h=H/S, discharge q=J/(2S), velocity u=q/h (zero at dry cells). Gravity g=1/8 in lattice units. The desired equilibrium tensor is h*u*u^T+(g*h^2/2)I. Flat bed; no bed slope, calibrated viscosity, explicit drag or free surface tracking. Full relaxation omega=1; no claim of standard D2Q25 viscosity formula or higher-order isotropy.

## Reversible collision algorithm
`collision.py` reuses M6's Boolean arithmetic builder and M8's indexed bit-sliced gate executor. It reconstructs H,Jx,Jy with reversible adders. For each axis, it forms exact integer minimum-variance adjacent-speed populations and a high-second-moment endpoint distribution (with one residual interior particle if needed). The target index second moment in ticks is J^2/H+H^2/(4S). It interpolates between those distributions using an integer fraction with w fractional bits, clamps to the attainable endpoints, floors the populations, and allocates the remaining mass and momentum to adjacent speeds. This is a constructive approximate equilibrium, not a globally optimal integer moment projection. The two marginal vectors are coupled via cumulative proportional rounding: each row receives its required mass and the remaining column capacities are updated; the last row gets the remainder. Each row and column sum is exact. The cross moment is only approximate and the ordering can break symmetry. Negative populations are prevented by construction for valid input depth. If H exceeds the w-bit capacity, the circuit returns the original populations and sets an overflow flag. The invalid-input path is reversible as well.

The gate circuit computes into fresh workspace, XORs outputs and flag, and uncomputes all temporary wires. Input words are retained. No classical equilibrium evaluation or lookup is called during gate execution. `CollisionOracle.qiskit()` exports the local gate sequence if Qiskit is installed; export/execution was not tested here. The backend is exact bit-sliced simulation of computational-basis X/CX/CCX gates, NOT coherent full-grid statevector evolution. Global streaming is the equivalent word permutation (SWAPs). Classical basis state preparation occurs between steps, so there is no claim of an end-to-end coherent quantum algorithm or quantum advantage. `reference.py` implements independent integer and floating comparisons; `run.py` verifies forward results and inverse/scratch cleanup for each step.

## Results
- 16x16, dx=0.5, dt=1, S=256, 15 synthetic solid cells, 24 steps. 30 lattice volume units retained exactly (120 is the sum of depths, not the area-weighted volume). 80 to 240 wet cells, no obstacle penetration or overflow rejection, 24/24 agreement with independent integer reference, final depth RMSE vs floating D2Q25 0.055322. 91 of 240 wet cells still fail the continuous equilibrium feasibility condition at the last step. Geometry is not recovered from original STL files; whole-link bounceback uses the conservative M10 bounding-rectangle rule.
- 2048-cell periodic strip, dx=0.125, dt=1, 24 steps, h0=1.5 and dry downstream. Ritter comparison uses |x|<28, away from periodic wrap interactions. S=64,256,1024 gives depth RMSE 0.110901, 0.088459, 0.084231 respectively; the floating D2Q25 reference gives 0.083318. Float-reference RMSE is computed over the full strip (not the Ritter window). Fixed-point error decreases but the model error remains. The strip has width dx in the represented 2D area and area-weighted volume is 24; depth integral along x alone would be 192. Stride=4 uses shifts that are multiples of four cells, creating decoupled subgrids; reducing dx alone with fixed dt does not establish spatial convergence. Small nonzero transverse velocity in this nominally 1D flow is a rounding/coupling artifact (max 0.006667 at S=1024).
- 1024 random local population states at S=256: exact mass and both momentum components. Among 801 pressure-feasible cases, moment tensor component RMSE 0.003844 and cross-moment RMSE 0.001879.
- At S=256, 109247 local qubits (108746 scratch), 355545 local X/CX/CCX gates; 236987 conceptual global qubits and 85839855 untranspiled X/CX/CCX-equivalent gates per step for the obstacle case (each bit SWAP counted as 3 CNOT). These resources are not feasible on current hardware. No QPU or native Qiskit execution.

## Output
`results/obstacle/step_*.vti`: ASCII VTK ImageData with cell-centered depth, discharge, velocity, target hydrostatic flux, actual kinetic moments xx/yy/xy, wet and solid masks, float-reference depth, and f00 through f24. All in lattice units. Target hydrostatic flux and actual kinetic tensor are distinct: the latter is not guaranteed to reproduce the desired physical pressure. `history.npz`, CSV diagnostics, JSON report and plots are derived from executed gate-simulation histories. The GIF is a numerical research visualization, not proof of validated water flow.

## Next research gate
Investigate symmetry-preserving coupling, alternative equilibria or adaptive velocities for unrealizable pressure, and a consistent space/time refinement study before claiming dry-bed accuracy. Add viscosity-dependent partial relaxation and test shear decay/higher-order moments. Optimize arithmetic scratch and test native quantum circuits on small registers. OpenLB comparison and SI water calibration remain pending. Do not interpret agreement with the quantized reference as independent physical validation; Ritter exposes a substantial residual model error.
