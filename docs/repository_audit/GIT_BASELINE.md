# Git Baseline & Repository Safety Audit

**Audit Timestamp**: 2026-10-29 20:29:51  
**Current Branch**: `research/multiphase-vof`  
**HEAD Commit**: `785775f1a9ec1c73badfdbf4ab31c0f6dc0ad725`  
**Remote Origin**: `https://github.com/QCFD-Lab/qlbm.git`  

---

## 1. Tracking Status Summary

- **Tracked Files in Git Index**: 394
- **Untracked Working Tree Files**: 65
- **Gitignored Working Tree Files**: 34489

### Safety Guarantee
> [!IMPORTANT]
> **Zero Data Loss Policy**: No files will be deleted or moved during Phase 1. All untracked research scripts, test outputs, and uncommitted results are fully cataloged below and must be preserved.

---

## 2. Complete Inventory of Untracked Files

| Relative Path | Size | Type | Likely Purpose | Safety Recommendation |
|---|---|---|---|---|
| `demos/visualization/make_vti_gif.py` | 4.2 KB | `.py` | Standalone visualization or utility script | **PRESERVE** (Do not discard) |
| `demos/visualization/visualize_spacetime.py` | 2.0 KB | `.py` | Standalone visualization or utility script | **PRESERVE** (Do not discard) |
| `demos/visualization/visualize_vti.py` | 1.6 KB | `.py` | Standalone visualization or utility script | **PRESERVE** (Do not discard) |
| `docs/repository_audit/DEPENDENCY_GRAPH.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/DUPLICATES_AND_OBSOLETE.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/EXPERIMENT_INDEX.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/FILE_INVENTORY.csv` | Unknown | `.csv` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/FINAL_REPORT.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/GIT_BASELINE.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/INITIAL_STRUCTURE.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/POTENTIALLY_UNUSED.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/PROPOSED_STRUCTURE.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/SHARED_MODULES.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `docs/repository_audit/UNRESOLVED_FILES.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `research_experiments/lqlga/experiment_d1q3.py` | 2.5 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/decode_real_counts.py` | 2.0 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/experiment_bounceback.py` | 7.3 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/experiment_collision.py` | 6.4 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/experiment_collision_matrix.py` | 5.6 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/experiment_integrated_d2q4.py` | 10.2 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/experiment_multistep_bounceback.py` | 6.1 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/experiment_one_step.py` | 7.8 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `research_experiments/spacetime_reinitialization/experiment_two_steps.py` | 6.7 KB | `.py` | Independent research experiment script | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_history_classical.csv` | 0.8 KB | `.csv` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_history_full_quantum.csv` | 0.3 KB | `.csv` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_history_hybrid.csv` | 0.3 KB | `.csv` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_history_streaming_test.csv` | 0.2 KB | `.csv` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_metrics_classical.json` | 2.4 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_metrics_full_quantum.json` | 1.1 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_metrics_hybrid.json` | 3.4 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_metrics_streaming_test.json` | 1.4 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_snapshots_classical.npz` | 13.8 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_snapshots_full_quantum.npz` | 3.4 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_snapshots_hybrid.npz` | 3.4 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break/dam_break_snapshots_streaming_test.npz` | 2.2 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/dam_break_history_classical.csv` | 2.8 KB | `.csv` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/dam_break_metrics_classical.json` | 7.6 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/dam_break_snapshots_classical.npz` | 19.5 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/hires_korner_diagnostics.json` | 29.5 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/hires_korner_frames.npz` | 1789.3 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/mm_n2_diagnostics.json` | 30.8 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/mm_n2_frames.npz` | 2166.0 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_classical_diagnostics.json` | 13.8 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_classical_frames.npz` | 18.3 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_full_quantum_diagnostics.json` | 14.0 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_full_quantum_frames.npz` | 18.4 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_hybrid_diagnostics.json` | 13.8 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_hybrid_frames.npz` | 18.5 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_streaming_test_diagnostics.json` | 13.8 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/modes/dam_break_streaming_test_frames.npz` | 18.3 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/qlbm_diagnostics.json` | 30.8 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/qlbm_frames.npz` | 1835.0 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/qlbm_korner_diagnostics.json` | 30.7 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/qlbm_korner_frames.npz` | 391.8 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/tau_sweep/tau_0.52_diagnostics.json` | 13.9 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/tau_sweep/tau_0.52_frames.npz` | 721.6 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/tau_sweep/tau_0.55_diagnostics.json` | 13.9 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/tau_sweep/tau_0.55_frames.npz` | 720.0 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/tau_sweep/tau_0.6_diagnostics.json` | 13.9 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/dam_break_final/tau_sweep/tau_0.6_frames.npz` | 719.9 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/test_modes/dam_break_history_hybrid.csv` | 0.5 KB | `.csv` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/test_modes/dam_break_metrics_hybrid.json` | 4.0 KB | `.json` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `results/test_modes/dam_break_snapshots_hybrid.npz` | 2.0 KB | `.npz` | Simulation output data (CSV, JSON, NPZ) | **PRESERVE** (Do not discard) |
| `trash_review/README.md` | Unknown | `.md` | Untracked research file | **PRESERVE** (Do not discard) |
| `trash_review/nano decode_real_counts.py` | Unknown | `.py` | Standalone visualization or utility script | **PRESERVE** (Do not discard) |

---

## 3. Recent Commit History

```text
785775f feat(demos): add multi-mode video generation and 4-panel comparison for dam break
6cb915f research: free-surface accuracy upgrade & Körner mass exchange validation (Phase 16)
6d51d40 research: cross-paradigm synthesis, validation suite & thesis roadmap (Phase 15)
745a1ba research: implement and validate linearized quantum collision operator (Phase 14)
6623fa9 research: transfer OpenLB dam break and implement hybrid quantum VOF architecture (Phase 13)
9c8e90f research: implement and validate hybrid VOF-LBM solver (Phase 12)
b14fc67 research: implement and validate VOF interface reconstruction and PLIC (Phase 11)
79de44f research: validate Young-Laplace law, surface tension, and spurious currents (Phase 10)
```

---

## 4. Working Tree Status (`git status --short`)

```text
R  test/unit/collision/eqc_permutation_test.py -> trash_review/eqc_permutation_test.py
?? demos/visualization/make_vti_gif.py
?? demos/visualization/visualize_spacetime.py
?? demos/visualization/visualize_vti.py
?? docs/repository_audit/
?? research_experiments/
?? results/dam_break/
?? results/dam_break_final/dam_break_history_classical.csv
?? results/dam_break_final/dam_break_metrics_classical.json
?? results/dam_break_final/dam_break_snapshots_classical.npz
?? results/dam_break_final/hires_korner_diagnostics.json
?? results/dam_break_final/hires_korner_frames.npz
?? results/dam_break_final/mm_n2_diagnostics.json
?? results/dam_break_final/mm_n2_frames.npz
?? results/dam_break_final/modes/
?? results/dam_break_final/qlbm_diagnostics.json
?? results/dam_break_final/qlbm_frames.npz
?? results/dam_break_final/qlbm_korner_diagnostics.json
?? results/dam_break_final/qlbm_korner_frames.npz
?? results/dam_break_final/tau_sweep/
?? results/test_modes/
?? trash_review/README.md
?? "trash_review/nano decode_real_counts.py"
```
