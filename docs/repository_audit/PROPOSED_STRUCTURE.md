# Proposed Clean Repository Structure & Migration Plan

**Date**: 2026-10-29  

> [!IMPORTANT]
> **Zero Breaking Changes Principle**: Public library imports (`qlbm.*`), test discovery paths, and Sphinx documentation paths must remain fully operational. All reorganization moves should be phased and require explicit user approval.

---

## 1. Target Directory Organization

```text
qlbm/                           # Root Python package (UNTOUCHED - 100% API COMPATIBLE)
├── classical/                  # Classical reference solvers (D2Q9, Shan-Chen, VOF)
├── components/                 # Quantum operators (AB, MS, SpaceTime, LQLGA, VOF, Common)
├── infra/                      # Execution infrastructure (Compiler, Runners, Decoders)
├── lattice/                    # Geometry & lattice models (AB, MS, SpaceTime, LGA)
└── tools/                      # Utilities & exceptions

demos/                          # Demonstrations & benchmarks
├── benchmarks/                 # Scalability and profiling notebooks
├── lattices/                   # Pre-configured lattice JSON geometries
├── openqasm/                   # OpenQASM export/import utilities
├── simulation/                 # Simulation demo notebooks & dam-break CLI runner
└── visualization/              # Visualization notebooks, figures & helper scripts
    └── helpers/                # [Proposed Target] make_vti_gif.py, visualize_vti.py

research_experiments/           # Dedicated stand-alone research experiments
├── lqlga/                      # LQLGA D1Q3 standalone experiments
└── spacetime_reinitialization/ # Space-time mid-circuit reset experiments

tests/                          # Unified test suite (Proposed consolidation)
├── framework/                  # [Proposed Target for test/] Unit, Integration, E2E
└── research/                   # Physical validation tests (125 tests, invariant checks)

docs/                           # Documentation
├── research/                   # Research monographs (00-15), dam-break audits, diary
├── repository_audit/           # Dated audit reports & file manifests
└── source/                     # Sphinx HTML API documentation

results/                        # All generated simulation outputs, metrics, and plots
├── dam_break/                  # Dam-break benchmark outputs & publication plots
├── dam_break_final/            # Hi-res frames and multi-mode comparison diagnostics
└── flowfield_media/            # [Proposed Target] Generated GIFs from repo root

trash_review/                   # Dedicated human-review holding area (No files deleted)
```

---

## 2. Proposed File Migration Mapping Table

| Current Path | Proposed Destination | Type | Migration Justification | Compatibility Impact |
|---|---|---|---|---|
| `make_vti_gif.py` | `demos/visualization/make_vti_gif.py` | Root Utility | Cleans repo root; places visualizer alongside visualization demos | None (standalone CLI) |
| `visualize_vti.py` | `demos/visualization/visualize_vti.py` | Root Utility | Consolidates VTI visualizers in demos/visualization | None (standalone CLI) |
| `visualize_spacetime.py` | `demos/visualization/visualize_spacetime.py` | Root Utility | Places space-time visualizer with spacetime demos | None (standalone CLI) |
| `qlbm_*.gif` (5 files in root) | `results/flowfield_media/` | Generated GIF | Cleans repo root; groups flowfield media in results | None (binary outputs) |
| `research_experiments/.../nano decode_real_counts.py` | `trash_review/nano decode_real_counts.py` | Typo Artifact | Isolates accidental duplicate file for user review | Zero impact |
| `test/unit/collision/eqc_permutation_test.py` | `trash_review/eqc_permutation_test.py` | Empty File | Isolates empty 0-byte file awaiting decision | Zero impact |
| `qlbm/*` (All 114 files) | **UNCHANGED** | Core Library | Preserves installed package API, imports, and pip compatibility | 100% Protected |
| `demos/simulation/*` | **UNCHANGED** | Demos/Runners | Retains established demo execution paths | 100% Protected |
| `tests/research/*` | **UNCHANGED** | Research Tests | Retains active 125-test research test suite | 100% Protected |
