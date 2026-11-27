"""Scalable Fault-Tolerant & NISQ Quantum Advantage Analyzer for 3D Lattice Boltzmann QCFD.

Performs rigorous asymptotic scaling and quantum advantage analysis comparing:
  1. Classical Lattice Boltzmann Method (LBM) on High-Performance GPUs (NVIDIA H100)
  2. Quantum Lattice Boltzmann Method (QLBM) with Draper Streaming & Givens Collision
  3. Fault-Tolerant Rotated Surface Code Physical Architecture & Magic State Factories

Key Metrics Evaluated:
- Exponential spatial compression: N^3 physical nodes -> 3 log2(N) + 8 logical qubits.
- Classical memory and FLOP complexity vs Quantum gate count and depth.
- Surface code distance d, physical qubit footprint, and logical T-count.
- Quantum Advantage Crossover Grid Resolution N_crossover.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class QuantumFluidResourceScalingEstimator:
    """Asymptotic resource scaling and quantum advantage analyzer for 3D QCFD."""

    def __init__(
        self,
        physical_error_rate: float = 1e-3,
        threshold_error_rate: float = 1e-2,
        cycle_time_us: float = 1.0,
        gpu_glups_h100: float = 8.5,  # Giga-Lattice Updates/sec on NVIDIA H100
    ) -> None:
        self.p_phys = float(physical_error_rate)
        self.p_thresh = float(threshold_error_rate)
        self.tau_cycle_us = float(cycle_time_us)
        self.gpu_glups = float(gpu_glups_h100)

    def analyze_grid_size(
        self,
        N: int,
        num_steps: int = 1000,
        target_failure_prob: float = 1e-3,
    ) -> Dict[str, Any]:
        """Compute exhaustive classical vs quantum resource scaling for grid N x N x N."""
        log2_N = int(math.ceil(math.log2(N)))
        total_nodes = N ** 3

        # -------------------------------------------------------------
        # 1. Classical LBM Complexity (D3Q19 Double Precision)
        # -------------------------------------------------------------
        # 19 velocity populations * 8 bytes/double * total_nodes
        classical_bytes_f = 19 * total_nodes * 8
        classical_ram_mb = classical_bytes_f / (1024.0 ** 2)
        classical_ram_gb = classical_bytes_f / (1024.0 ** 3)

        # ~650 FLOPs per node update in standard BGK + Guo forcing
        flops_per_step = total_nodes * 650.0
        total_classical_flops = flops_per_step * num_steps

        # GPU Runtime on H100 (GLUPS = 1e9 updates/s)
        node_updates_per_s = self.gpu_glups * 1e9
        gpu_time_per_step_s = total_nodes / node_updates_per_s
        total_gpu_time_s = gpu_time_per_step_s * num_steps

        # -------------------------------------------------------------
        # 2. Quantum LBM Complexity (Draper QFT + Givens Collision)
        # -------------------------------------------------------------
        # Spatial register: 3 * log2(N)
        spatial_qubits = 3 * log2_N
        velocity_qubits = 5  # D3Q19 is strictly 5 qubits
        wall_ancilla = 1     # Cleanly recycled wall obstacle ancilla
        qpe_ancilla = 2      # Zero-spectral-leakage QPE velocity readout

        total_logical_qubits = spatial_qubits + velocity_qubits + wall_ancilla + qpe_ancilla
        compression_ratio = float(total_nodes) / float(total_logical_qubits)

        # Quantum Gate Counts per Step
        # Collision: 9 conjugate pairs of Givens rotations = 9 SO(2) gates on 5 qubits
        collision_gates_per_step = 9 * 15  # ~15 native 1Q/2Q gates per Givens rotation
        # Streaming: 3 Cartesian axes. Draper QFT on log2_N qubits:
        # QFT has log2_N * (log2_N + 1) / 2 gates + controlled phase shifts
        qft_gates_per_axis = (log2_N * (log2_N + 1)) // 2 + log2_N * 19
        streaming_gates_per_step = 3 * qft_gates_per_axis
        # Boundary: Oracle + Bounce involution + uncomputation
        boundary_gates_per_step = 2 * (log2_N * 3) + 19

        gates_per_step = (
            collision_gates_per_step
            + streaming_gates_per_step
            + boundary_gates_per_step
        )
        total_logical_gates = gates_per_step * num_steps
        logical_circuit_depth_per_step = int(gates_per_step * 0.45)

        # -------------------------------------------------------------
        # 3. Fault-Tolerant Surface Code Architecture
        # -------------------------------------------------------------
        # Target logical error rate per gate: epsilon_L = p_target / total_gates
        p_target = target_failure_prob
        p_L_per_gate = p_target / max(total_logical_gates, 1)

        # Surface code distance: d = ceil( 2 * log10(1 / p_L) / log10(p_thresh / p_phys) )
        noise_suppression = self.p_thresh / self.p_phys  # 10x per code distance step
        log_suppression = math.log10(noise_suppression) if noise_suppression > 1.0 else 1.0
        d_exact = 2.0 * math.log10(1.0 / max(p_L_per_gate, 1e-18)) / log_suppression
        # Ensure code distance is an odd integer >= 3
        d = int(math.ceil(d_exact))
        if d % 2 == 0:
            d += 1
        d = max(3, d)

        # Rotated surface code requires 2 * d^2 physical qubits per logical qubit
        physical_qubits_per_logical = 2 * (d ** 2)
        total_physical_data_qubits = total_logical_qubits * physical_qubits_per_logical

        # T-Count and Magic State Factory Footprint
        # Each arbitrary rotation decomposed via Ross-Selinger synthesis requires ~28 T-gates
        arbitrary_rotations_per_step = 9 + streaming_gates_per_step // 4
        t_count_per_step = arbitrary_rotations_per_step * 28
        total_t_count = t_count_per_step * num_steps
        # 15-to-1 magic state distillation factory footprint: ~15 * d^2 physical qubits
        factory_physical_qubits = 15 * (d ** 2)
        total_fault_tolerant_physical_qubits = total_physical_data_qubits + factory_physical_qubits

        # Fault-tolerant execution time: depth * d * cycle_time
        logical_step_duration_s = logical_circuit_depth_per_step * d * (self.tau_cycle_us * 1e-6)
        total_quantum_time_s = logical_step_duration_s * num_steps

        # Quantum Advantage Speedup Ratio (GPU time / Quantum time)
        advantage_ratio = total_gpu_time_s / max(total_quantum_time_s, 1e-9)

        return {
            "grid_dimension_N": N,
            "total_lattice_nodes": total_nodes,
            "classical_lbm": {
                "memory_ram_mb": round(classical_ram_mb, 2),
                "memory_ram_gb": round(classical_ram_gb, 4),
                "flops_per_step": flops_per_step,
                "total_flops": total_classical_flops,
                "gpu_time_per_step_s": round(gpu_time_per_step_s, 6),
                "total_gpu_time_s": round(total_gpu_time_s, 3),
            },
            "quantum_lbm": {
                "spatial_qubits": spatial_qubits,
                "velocity_qubits": velocity_qubits,
                "ancilla_qubits": wall_ancilla + qpe_ancilla,
                "total_logical_qubits": total_logical_qubits,
                "spatial_compression_factor": round(compression_ratio, 1),
                "gates_per_step": gates_per_step,
                "circuit_depth_per_step": logical_circuit_depth_per_step,
                "total_logical_gates": total_logical_gates,
                "quantum_time_per_step_s": round(logical_step_duration_s, 6),
                "total_quantum_time_s": round(total_quantum_time_s, 3),
            },
            "fault_tolerant_surface_code": {
                "code_distance_d": d,
                "physical_qubits_per_logical": physical_qubits_per_logical,
                "total_physical_data_qubits": total_physical_data_qubits,
                "magic_state_factory_qubits": factory_physical_qubits,
                "total_system_physical_qubits": total_fault_tolerant_physical_qubits,
                "logical_t_count_per_step": t_count_per_step,
                "total_t_count": total_t_count,
            },
            "quantum_advantage": {
                "speedup_ratio_vs_h100": round(advantage_ratio, 3),
                "achieves_speedup": advantage_ratio > 1.0,
            },
        }

    def generate_scaling_curve(
        self,
        grid_sizes: Optional[List[int]] = None,
        num_steps: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Generate complete scaling curve across grid resolutions."""
        if grid_sizes is None:
            grid_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        return [self.analyze_grid_size(N, num_steps=num_steps) for N in grid_sizes]


def main() -> None:
    """CLI Entrypoint for Quantum Advantage Scaling Analysis."""
    parser = argparse.ArgumentParser(
        description="Scalable Fault-Tolerant & NISQ Quantum Advantage Analyzer for QCFD"
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
        help="Grid resolutions N (where lattice is N x N x N)",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of fluid time steps")
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/systematic_qcfd_output/quantum_advantage_scaling.json",
        help="Path to save output JSON scaling report",
    )
    args = parser.parse_args()

    estimator = QuantumFluidResourceScalingEstimator()
    results = estimator.generate_scaling_curve(grid_sizes=args.grid_sizes, num_steps=args.steps)

    print("=" * 115)
    print("  QUANTUM COMPUTATIONAL FLUID DYNAMICS (QCFD) ASYMPTOTIC ADVANTAGE SCALING")
    print(f"  Time Steps: {args.steps:,} | Physical Error Rate: 1e-3 | Threshold: 1e-2")
    print("=" * 115)

    header = (
        f"{'Grid N^3':^12} | {'Cells':^12} | {'RAM (GB)':^10} | {'Log Qubits':^10} | "
        f"{'Phys Qubits':^12} | {'H100 Time':^11} | {'QLBM Time':^11} | {'Advantage':^10}"
    )
    print(header)
    print("-" * 115)

    crossover_N = None
    for res in results:
        N = res["grid_dimension_N"]
        nodes = res["total_lattice_nodes"]
        ram_gb = res["classical_lbm"]["memory_ram_gb"]
        log_q = res["quantum_lbm"]["total_logical_qubits"]
        phys_q = res["fault_tolerant_surface_code"]["total_system_physical_qubits"]
        gpu_t = res["classical_lbm"]["total_gpu_time_s"]
        q_t = res["quantum_lbm"]["total_quantum_time_s"]
        ratio = res["quantum_advantage"]["speedup_ratio_vs_h100"]

        if ratio >= 1.0 and crossover_N is None:
            crossover_N = N

        adv_str = f"{ratio:.2f}x" if ratio < 1000 else f"{ratio:.1e}x"
        print(
            f"{f'{N}x{N}x{N}':^12} | "
            f"{nodes:^12,d} | "
            f"{ram_gb:^10.2f} | "
            f"{log_q:^10d} | "
            f"{phys_q:^12,d} | "
            f"{f'{gpu_t:.2f} s':^11} | "
            f"{f'{q_t:.2f} s':^11} | "
            f"{adv_str:^10}"
        )

    print("-" * 115)
    if crossover_N:
        print(f"  >>> QUANTUM ADVANTAGE CROSSOVER RESOLUTION: N = {crossover_N} ({crossover_N**3:,} cells) <<<")
    print("=" * 115)

    # Save JSON report
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "analysis_parameters": {
                    "steps": args.steps,
                    "h100_gpu_glups": estimator.gpu_glups,
                    "crossover_grid_N": crossover_N,
                },
                "scaling_curve": results,
            },
            f,
            indent=2,
        )
    print(f"\n[Saved Scaling Report] -> {out_path}")


if __name__ == "__main__":
    main()
