# Duplicates, Near-Duplicates, and Obsolete File Analysis

**Date**: 2026-10-29  

---

## 1. Exact Duplicate Files (Identical SHA256 Hash)

| SHA256 Prefix | Duplicate File Instances | Assessment & Recommended Action |
|---|---|---|
| `21541a9427bd...` | [`Docker/build_cpu.Dockerfile.dockerignore`](file:///home/aswa/Research/qlbm/Docker/build_cpu.Dockerfile.dockerignore)<br>[`Docker/build_gpu.Dockerfile.dockerignore`](file:///home/aswa/Research/qlbm/Docker/build_gpu.Dockerfile.dockerignore) | Harmless: Identical exclusion rules for CPU and GPU Docker builds. **KEEP**. |
| `62e59de9223b...` | [`demos/lattices/3d_8x8x16_0_obstacle.json`](file:///home/aswa/Research/qlbm/demos/lattices/3d_8x8x16_0_obstacle.json)<br>[`demos/lattices/3d_8x8x8_0_obstacle.json`](file:///home/aswa/Research/qlbm/demos/lattices/3d_8x8x8_0_obstacle.json) | Two lattice JSON configurations with identical dimensions. **REVIEW**. |
| `51946cfab5df...` | [`results/dam_break_final/modes/dam_break_classical_frames.npz`](file:///home/aswa/Research/qlbm/results/dam_break_final/modes/dam_break_classical_frames.npz)<br>[`results/dam_break_final/modes/dam_break_streaming_test_frames.npz`](file:///home/aswa/Research/qlbm/results/dam_break_final/modes/dam_break_streaming_test_frames.npz) | Bitwise identical verification output confirming Mode A and Mode B equivalence. **KEEP**. |
| `608cd4549ad8...` | [`test/artifacts/collisionless-sampling-0/lattice.json`](file:///home/aswa/Research/qlbm/test/artifacts/collisionless-sampling-0/lattice.json)<br>[`test/artifacts/collisionless-sampling-1/lattice.json`](file:///home/aswa/Research/qlbm/test/artifacts/collisionless-sampling-1/lattice.json) | Identical lattice config for multiple test sample sets. **KEEP**. |

---

## 2. Near-Duplicate Files & Typo Artifacts

### Typo Artifact: `research_experiments/spacetime_reinitialization/nano decode_real_counts.py`
- **Resembles**: `research_experiments/spacetime_reinitialization/decode_real_counts.py`
- **Diff**: Bitwise identical except for a single trailing newline (`\n`).
- **Cause**: Created when running the command `nano decode_real_counts.py` inside an environment where the argument was misinterpreted as a filename with a leading `nano ` string.
- **Risk if removed**: None. `decode_real_counts.py` is the legitimate script.
- **Action**: Marked as **TRASH CANDIDATE — Safe to remove only after approval**.

---

## 3. Superseded / Legacy Files

### Legacy Operator: `qlbm/components/spacetime/collision/d2q4_old.py`
- **Description**: Legacy pairwise $R_y(\theta)$ rotation operator for D2Q4 velocities ($|0101\rangle \leftrightarrow |1010\rangle$) from Mezzacapo et al.
- **Superseded by**: `qlbm/components/spacetime/collision/eqc_collision.py` (generalized equivalence class collision).
- **Status**: Still directly imported by `qlbm/components/spacetime/spacetime.py` and experiment scripts `experiment_collision.py` and `experiment_collision_matrix.py`.
- **Risk if removed**: **HIGH (Breaking change)**. Removing this file immediately breaks spacetime simulation and experiment scripts.
- **Action**: Retain as **KEEP — Historical research**.
