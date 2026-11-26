# Milestone 27: finite-width 3D two-phase dam release

Goal: extend the existing two-phase quantum-subroutine dam-break model from a span-uniform 32×4×16 lattice to a genuinely nonuniform 32×16×16 (X,Y,Z) tank. All six walls use stationary halfway bounce-back. Initially the liquid reservoir occupies about 8 cells across Y, centered at Y=8. There is no explicitly modeled removal gate. The existing diffuse-interface, high-density-ratio equations and M25 nominal Re=212.1, density ratio 831.67, dynamic viscosity ratio 55.56 and 1280-step horizon remain unchanged. No synthetic geometry extrusion or animation interpolation.

## Run
From `experiments/quantum_dam_break` with the existing Python dependencies:

```bash
bash milestone27/reproduce.sh --full-regression
```

Runs gate-based, independent collision/transport reference, and gravity-disabled reference cases. The latter's own gravity-scaled time is zero; comparison plots use the active case's reference-gravity time for all runs. The 0.1 lattice-speed guard is a chosen numerical validity criterion. Output contains time histories, full 3D volumes, final populations, VTK time series, and GIFs. Use `results/gate/dam_break.pvd` in ParaView; the nine data arrays are cell-centered. Coordinates in arrays are (z,y,x), velocity vector components (x,y,z). Color scales are fixed across frames. Spanwise velocity is also visualized directly from the velocity vector.

## Quantum scope
`solver.py` compiles a closed-boundary permutation into fully controlled logical X gates, executes the corresponding statevector basis swaps, and reuses the actual dual-rail dissipation subroutines of M19/M21. Classical amplitude preparation, readout, equilibrium, forces, hydrostatic correction, material interpolation and FCT are still required; no single coherent full evolution, QPU or acceleration claim. No new physical equations have been validated here. Material mass conservation is inferred from affine density and conserved phase; not proof of consistent physical momentum balance. Cell-centred pressure at the downstream wall is a diagnostic, not a wall-traction measurement.

See `results/report.json` for gate/reference numerical differences and the independent gravity control. Geometry and surface rendering use actual cells or the alpha=0.5 isosurface. See SOURCE_AUDIT.md for literature, known model limitations and remote provenance.
