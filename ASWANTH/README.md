# ASWANTH — Production-Ready Verified Quantum Fluid Dynamics Suite

Welcome to the **`ASWANTH`** production suite within the QLBM research repository.

This directory contains the curated, standalone, production-ready executables, visualization pipelines, and verification test harnesses developed for 2D and 3D quantum lattice Boltzmann hydrodynamics, multiphase Volume-of-Fluid (VOF) modeling, and gravity-driven waterfall simulations.

---

## Directory Architecture

```
ASWANTH/
├── executables/
│   ├── run_3d_waterfall_dam_break.py    # Production 3D Waterfall & Dam-Break simulator
│   └── __init__.py
├── visualization/
│   ├── visualize_3d_waterfall.py        # VTK .vti exporter & publication slice plotter
│   └── __init__.py
├── verification/
│   ├── verify_aswanth_suite.py          # Master verification test harness & scorecard
│   └── __init__.py
├── results/                             # Generated benchmark data, CSVs, and VTK files
│   └── waterfall_3d/
│       ├── waterfall_metrics.csv
│       ├── waterfall_metadata.json
│       └── snapshots/                   # .npz snapshots, .vti (ParaView), .png slices
├── CONCEPTS.md                          # Theoretical foundation and mathematical derivation
├── SKILLS_SHOWCASE.md                   # Engineering skills, methodologies, and competencies
└── README.md                            # This overview document
```

---

## Quickstart Guide

### 1. Run Master Verification Audit
To verify that all 2D and 3D quantum and classical fluid dynamics invariants pass:
```bash
.venv/bin/python ASWANTH/verification/verify_aswanth_suite.py
```
Expected output:
```text
=====================================================================================
           ASWANTH PRODUCTION VERIFICATION AUDIT & SCORECARD
=====================================================================================
[TEST 1/6] D3Q19 Lattice Normalization & Sound Speed...  PASS
[TEST 2/6] 3D Draper QFT Quantum Streaming Equivalence... PASS (E_max < 1e-12)
[TEST 3/6] Quantum D3Q19 Bounce-Back Reflection Unitary... PASS (0 error)
[TEST 4/6] Physical Free-Fall Acceleration Under Gravity... PASS (Error < 1e-16)
[TEST 5/6] 3D Waterfall Elevated Dam-Break Mass Conservation... PASS
[TEST 6/6] Hybrid Quantum-Classical vs Classical Physical Trajectory Match... PASS (< 1e-11)
=====================================================================================
OVERALL SCORECARD: 6/6 AUDIT CHECKS PASSED
STATUS: PRODUCTION-READY AND FULLY VERIFIED
=====================================================================================
```

### 2. Run Quantum D3Q19 Streaming Test
Verify that the 13-qubit Draper QFT streaming circuit matches classical 3D periodic streaming down to $10^{-13}$:
```bash
.venv/bin/python ASWANTH/executables/run_3d_waterfall_dam_break.py --mode streaming_test --nx 8 --ny 4 --nz 8
```

### 3. Run Hybrid 3D Waterfall Simulation
Simulate water flow behind an elevated gate, gate opening, vertical fall down a cliff under gravity, and lower collection basin impact:
```bash
.venv/bin/python ASWANTH/executables/run_3d_waterfall_dam_break.py \
    --mode hybrid \
    --nx 8 --ny 4 --nz 8 \
    --shelf-height 3 \
    --shelf-length 4 \
    --gate-x 4 \
    --water-height 2 \
    --steps 15 \
    --gate-open-step 2 \
    --gravity 0.001 \
    --output-dir ASWANTH/results/waterfall_3d
```

### 4. Visualize in ParaView & Generate 2D Plots
Convert simulation snapshots to structured VTK (`.vti`) files and render cross-sectional figures:
```bash
.venv/bin/python ASWANTH/visualization/visualize_3d_waterfall.py ASWANTH/results/waterfall_3d/snapshots
```
The resulting `.vti` files can be directly opened in ParaView to inspect:
- 3D Volume-of-Fluid free surface interface
- 3D Velocity vectors $(u_x, u_y, u_z)$ plunging into the basin
- Hydrodynamic pressure $P = c_s^2 \rho$
- Solid wall obstacles and cliff geometry

---

## Physical Problem: The 3D Elevated Waterfall

The simulation model consists of:
1. **Elevated Reservoir**: Water held on a shelf of height $z_{\text{shelf}}$ between $x = 1$ and $x_{\text{gate}}$.
2. **Openable Gate**: A vertical solid barrier that opens at step $t_{\text{open}}$.
3. **Vertical Drop (Cliff)**: A steep vertical drop from $z = z_{\text{shelf}}$ down to the lower basin floor at $z = 0$.
4. **Lower Basin**: An impact region where falling water strikes the floor, pools, and dissipates energy.
5. **Real Gravity**: Scaled lattice acceleration corresponding to $g = (0, 0, -9.81)\,\text{m/s}^2$ with low-Mach stability ($\text{Ma} < 0.25$).
