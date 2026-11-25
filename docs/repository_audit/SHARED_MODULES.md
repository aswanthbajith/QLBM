# Genuinely Shared Modules & Cross-Experiment Infrastructure

**Date**: 2026-10-29  

A module is defined as **genuinely shared** if it satisfies at least one of the following criteria:
1. It is imported by **two or more distinct experiments** (e.g. both Dam-Break and Multi-Speed).
2. It provides fundamental mathematical abstractions (e.g. quantum adders, comparators, lattice base classes).
3. It provides the universal compiler, simulation backends, or execution configurations.

---

## 1. Cross-Experiment Usage Matrix

| Core Module | Dam-Break (Exp 1) | Shan-Chen (Exp 2) | D2Q9 (Exp 3) | Multi-Speed (Exp 4) | Space-Time (Exp 5) | LQLGA (Exp 6) | AB / Linear (Exp 7) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [`qlbm/tools/utils.py`](file:///home/aswa/Research/qlbm/qlbm/tools/utils.py) | X | X | X | X | X | X | X |
| [`qlbm/tools/exceptions.py`](file:///home/aswa/Research/qlbm/qlbm/tools/exceptions.py) | - | - | - | X | X | X | X |
| [`qlbm/components/base.py`](file:///home/aswa/Research/qlbm/qlbm/components/base.py) | X | - | - | X | X | X | X |
| [`qlbm/components/common/adders.py`](file:///home/aswa/Research/qlbm/qlbm/components/common/adders.py) | X | - | - | X | - | - | X |
| [`qlbm/components/common/comparators.py`](file:///home/aswa/Research/qlbm/qlbm/components/common/comparators.py) | - | - | - | X | X | - | X |
| [`qlbm/components/common/primitives.py`](file:///home/aswa/Research/qlbm/qlbm/components/common/primitives.py) | - | - | - | - | X | X | X |
| [`qlbm/components/cqlbm.py`](file:///home/aswa/Research/qlbm/qlbm/components/cqlbm.py) | - | - | - | - | - | - | X |
| [`qlbm/lattice/lattices/base.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/lattices/base.py) | - | - | - | X | X | X | X |
| [`qlbm/lattice/spacetime/properties_base.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/spacetime/properties_base.py) | X | - | - | - | X | X | X |
| [`qlbm/lattice/geometry/shapes/base.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/geometry/shapes/base.py) | - | - | - | X | X | X | X |
| [`qlbm/lattice/geometry/shapes/block.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/geometry/shapes/block.py) | - | - | - | X | X | - | X |
| [`qlbm/lattice/geometry/shapes/circle.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/geometry/shapes/circle.py) | - | - | - | - | X | - | X |
| [`qlbm/lattice/geometry/shapes/ymonomial.py`](file:///home/aswa/Research/qlbm/qlbm/lattice/geometry/shapes/ymonomial.py) | - | - | - | - | - | - | X |
| [`qlbm/infra/compiler.py`](file:///home/aswa/Research/qlbm/qlbm/infra/compiler.py) | - | - | - | X | X | X | X |
| [`qlbm/infra/runner/base.py`](file:///home/aswa/Research/qlbm/qlbm/infra/runner/base.py) | - | - | - | - | - | - | - |
| [`qlbm/infra/runner/qiskit_runner.py`](file:///home/aswa/Research/qlbm/qlbm/infra/runner/qiskit_runner.py) | - | - | - | - | - | - | - |
| [`qlbm/infra/runner/qulacs_runner.py`](file:///home/aswa/Research/qlbm/qlbm/infra/runner/qulacs_runner.py) | - | - | - | - | - | - | - |
| [`qlbm/infra/runner/simulation_config.py`](file:///home/aswa/Research/qlbm/qlbm/infra/runner/simulation_config.py) | - | - | - | - | - | - | - |
| [`qlbm/classical/d2q9.py`](file:///home/aswa/Research/qlbm/qlbm/classical/d2q9.py) | X | X | X | - | - | - | X |

---

## 2. Public API Surface (`qlbm/__init__.py`)

The primary library initialization [`qlbm/__init__.py`](file:///home/aswa/Research/qlbm/qlbm/__init__.py) and its package `__init__.py` files expose the core components. Retaining these exact module paths guarantees that existing imports remain 100% backward-compatible.
