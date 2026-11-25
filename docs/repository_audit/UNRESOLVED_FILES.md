# Unresolved Files Requiring Decision

**Date**: 2026-10-29  

This document tracks files whose ownership, operational status, or long-term disposition requires explicit human feedback.

---

## Items Awaiting Decision

### 1. Empty Test File: `test/unit/collision/eqc_permutation_test.py`
- **File Path**: [`test/unit/collision/eqc_permutation_test.py`](file:///home/aswa/Research/qlbm/test/unit/collision/eqc_permutation_test.py)
- **Current Content**: 0 bytes (Empty).
- **Context**: Created during initial framework scaffolding for testing EQC permutations, but test assertions were never added.
- **Decision Needed**: Should this file be implemented with permutation tests for `eqc_permutation.py`, or should it be removed from the framework test suite?

### 2. Standalone Root Visualization Scripts
- **Files**: [`make_vti_gif.py`](file:///home/aswa/Research/qlbm/make_vti_gif.py), [`visualize_vti.py`](file:///home/aswa/Research/qlbm/visualize_vti.py), [`visualize_spacetime.py`](file:///home/aswa/Research/qlbm/visualize_spacetime.py)
- **Context**: Standalone CLI scripts located in the repository root rather than in `demos/visualization/`.
- **Decision Needed**: Do you approve moving these scripts into `demos/visualization/` during Phase 3 reorganization?

### 3. Generated GIFs in Repository Root
- **Files**: `qlbm_evolution.gif`, `qlbm_evolution1.gif`, `qlbm_evolution2.gif`, `qlbm_ms_evolution.gif`, `qlbm_spactime.gif`
- **Context**: Generated flowfield animations saved in repository root (ignored by `.gitignore`).
- **Decision Needed**: Move these files to `results/` or `demos/visualization/sample-flowfield/` to keep the root directory clean?
