# Milestone 9 — bounded conservative collision and dry-bed feasibility audit

## Run
From `experiments/quantum_dam_break`, using Python with numpy, numba, pytest, matplotlib and pillow:

```bash
python milestone9/run.py
python milestone9/plot.py
python -m pytest -q tests milestone{2,3,4,5,6,7,8,9}/tests
```

## Method
Preserves M8's D2Q9 shallow-water equilibrium, integer registers, g=1/8, omega=1 for integrated experiments, scale S=4096. The existing full BGK candidate is calculated by a reversible circuit. If feasible (all words in [0,2S-1]), use it unchanged. Otherwise, start from the original populations and apply bounded corrections in free directions i=3..8. For each such direction with discrete velocity (cx,cy), a change d is compensated by changes -cx*d in population 1, -cy*d in population 2, and (cx+cy-1)*d in population 0. Thus every correction has zero mass and momentum moments. Clamp the magnitude of d to the available nonnegative capacity of each affected register, including overflow limits. The order of corrections is fixed, which may introduce directional bias; this is not an optimal projection or an isotropic collision rule.

The output flag denotes infeasibility of the full BGK candidate; diagnostics distinguish partial updates from blocked collisions. A full BGK update can also be a no-op at equilibrium. Collision compiled from M8 arithmetic (X/CX/CCX); no classical reference used to advance the target trajectory. Inputs preserved, output XORed, scratch uncomputed; tested inverse on actual inputs. M8 streaming/bounce-back permutation reused. Classical independent integer reference and M8 comparison are separate. Bit-sliced computational-basis simulation only, no native Qiskit execution, coherent multistep evolution, quantum speedup, or QPU evidence.

## Structural limitation
With unit D2Q9 velocities and nonnegative f_i, q_alpha = sum c_i_alpha f_i and Pi_alphaalpha = sum c_i_alpha^2 f_i. Because c_i_alpha belongs to {-1,0,1}, |q_alpha| <= Pi_alphaalpha <= h. The target shallow-water momentum flux is h*u_alpha^2 + g*h^2/2. A necessary (not sufficient) realizability condition is |u_alpha| <= u_alpha^2+g*h/2 <= 1 at h>0. Example h=1, u_x=0.5, g=1/8: desired Pi_xx=0.3125 but |q_x|=0.5. No nonnegative D2Q9 distribution can simultaneously satisfy these moments. At saturated front velocity u_x=1, Pi_xx must equal h, while the target includes positive hydrostatic pressure. This is a mathematical obstruction under the current velocity and positivity assumptions, not merely fixed-point error.

## Results and interpretation
- 16x16 obstacle dry bed, 24 steps: exact mass 120 lu conserved, no obstacle penetration, local collision momentum conserved. M8 rejected 2304 updates; M9 executed 3191 full BGK, 1101 partial, 1492 blocked (5784 total). Different trajectories, so counts are not paired comparisons. At final step 141 of 241 fluid cells violate the necessary pressure bound. No physical validation claimed.
- Wet bed, 8 steps: exact reproduction of M8 with no fallback. 
- 2x128 periodic strip, central interface evaluated on |x|<32 before interaction with the other interface: Ritter depth RMSE 0.179501 lu at step 24, identical to M8; observed front (depth >0.01) x=23.5 vs analytical support x=20.7846. The different front definitions make this a descriptive comparison, not a precise shock-speed measurement. No accuracy improvement.
- 104 tests passed, 5 optional modules skipped. Resources are conceptual logical counts before hardware decomposition, not executed on hardware. See report.json.

## Next research decision
Evaluate alternate discrete velocity sets or a different kinetic representation against the necessary moment inequalities and positivity/stability tests before compiling gates. Signed populations or modified equilibrium and wet/dry interface treatments require independent stability and physical validation. The M4–7 finite-volume branch remains available as a separate reference path; do not label it D2Q9. Preserve all earlier work. No physical water viscosity calibration, OpenLB comparison, or native Qiskit execution in M9.

VTK files contain M8-compatible fields; `reference_depth_lu` is M8's integer trajectory, **not** analytical ground truth. See `ritter.csv` for the analytical benchmark. All units are lattice units.
