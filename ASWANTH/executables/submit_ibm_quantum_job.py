"""Physical Quantum Hardware Transpilation & Submission Tool for QLBM Circuits.

Targets actual superconducting IBM Quantum processors (127-qubit IBM Eagle / Heron
architectures via FakeSherbrooke, FakeBrisbane, or live Qiskit Runtime Service).

Features:
1. Translates algorithmic QLBM circuits (Draper QFT Streaming, Rotation Collision, QPE Velocity Profiler)
   to physical hardware basis sets (ECR / CZ, RZ, SX, X).
2. Performs heavy-hex topological routing and depth optimization (Levels 1-3).
3. Evaluates hardware coherence budgets (T1/T2 vs circuit execution duration).
4. Analyzes error mitigation readiness (TREX twirling, ZNE scaling, Dynamical Decoupling).
5. Optionally executes on noisy physical backends and decodes quantum measurement counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C, CS_SQ, WEIGHTS
from research.quantum_fluid_evolution.collision.rotation_collision import RotationCollision3D
from research.quantum_fluid_evolution.evolution.quantum_phase_estimation_velocity import (
    QuantumPhaseEstimationVelocityProfiler,
)


def get_target_backend(backend_name: str, token: Optional[str] = None):
    """Obtain target Qiskit backend (FakeProvider or live Qiskit Runtime Service)."""
    backend_lower = backend_name.lower()
    if backend_lower == "fake_sherbrooke":
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke()
    elif backend_lower == "fake_brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        return FakeBrisbane()
    elif backend_lower in ("aer", "aer_simulator"):
        from qiskit_aer import AerSimulator
        return AerSimulator()
    else:
        # Live IBM Quantum Service
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            api_token = token or os.environ.get("QISKIT_IBM_TOKEN")
            if not api_token:
                raise ValueError(
                    f"Live IBM backend '{backend_name}' requested but no API token provided. "
                    f"Pass --token or set QISKIT_IBM_TOKEN environment variable."
                )
            service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
            return service.backend(backend_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize IBM Quantum backend '{backend_name}': {e}")


def build_hardware_circuit(
    circuit_type: str,
    prepare_downward_state: bool = True,
) -> Tuple[QuantumCircuit, List[int]]:
    """Build algorithmic circuit and return (circuit, measured_qubits)."""
    if circuit_type == "qpe_velocity":
        # 7-qubit circuit (5 velocity qubits + 2 precision ancillas)
        profiler = QuantumPhaseEstimationVelocityProfiler(
            num_spatial_qubits=0,
            lattice_type="D3Q19",
            num_precision_qubits=2,
            axis=2,  # Vertical Z-axis
        )
        qc = QuantumCircuit(profiler.total_qubits, 2, name="QPE_Velocity_Readout_D3Q19")

        # Optional initial state preparation on velocity qubits (qubits 0..4)
        if prepare_downward_state:
            # Prepare state with downward velocity (direction 6: c_z = -1)
            # Encode amplitude into basis state |6> = |00110>_2
            qc.x(1)
            qc.x(2)

        # Append QPE circuit
        qc.compose(profiler.circuit, inplace=True)

        # Measure precision register (qubits 5, 6) into classical bits 0, 1
        qc.measure([5, 6], [0, 1])
        return qc, [5, 6]

    elif circuit_type == "rotation_collision":
        # 5-qubit rotation collision circuit (9 Givens rotations)
        rot = RotationCollision3D(theta=0.1)
        base_qc = rot.build_circuit()
        qc = QuantumCircuit(5, 5, name="Rotation_Collision_D3Q19")
        if prepare_downward_state:
            qc.x(1)
            qc.x(2)
        qc.compose(base_qc, inplace=True)
        qc.measure(range(5), range(5))
        return qc, list(range(5))

    elif circuit_type == "coupled_step":
        # 1-step hydrodynamic streaming + rotation collision
        from qlbm.components.vof.streaming_3d import QuantumD3Q19StreamingOperator
        stream_op = QuantumD3Q19StreamingOperator(nx=2, ny=2, nz=2)
        stream_qc = stream_op.circuit
        rot = RotationCollision3D(theta=0.08)
        coll_qc = rot.build_circuit()

        total_q = stream_qc.num_qubits
        qc = QuantumCircuit(total_q, 5, name="Coupled_LBM_Step_1QPU")
        qc.compose(stream_qc, inplace=True)
        # Apply collision on velocity register (lower 5 qubits)
        qc.compose(coll_qc, qubits=list(range(5)), inplace=True)
        qc.measure(range(5), range(5))
        return qc, list(range(5))

    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


def analyze_hardware_readiness(
    transpiled_qc: QuantumCircuit,
    backend: Any,
) -> Dict[str, Any]:
    """Analyze physical circuit against target QPU calibration and coherence properties."""
    ops = transpiled_qc.count_ops()
    depth = transpiled_qc.depth()
    num_physical_qubits = transpiled_qc.num_qubits

    # Identify active physical qubits that contain gates
    active_qubit_indices = set()
    for instruction in transpiled_qc.data:
        for q in instruction.qubits:
            active_qubit_indices.add(transpiled_qc.find_bit(q).index)
    active_qubits = sorted(list(active_qubit_indices))

    # Determine native 2Q gates (ECR for Eagle, CZ for Heron, CX for Aer)
    num_2q = ops.get("ecr", 0) + ops.get("cz", 0) + ops.get("cx", 0)
    num_1q = ops.get("rz", 0) + ops.get("sx", 0) + ops.get("x", 0)
    num_measure = ops.get("measure", 0)

    # Estimate circuit execution duration
    # Typical superconducting timescales: 2Q ~ 300 ns, 1Q ~ 35 ns, Measure ~ 1000 ns
    t_2q_ns = 300.0
    t_1q_ns = 35.0
    t_meas_ns = 1000.0

    # Parallel execution duration: depth * average gate layer time
    estimated_duration_us = (depth * 0.7 * t_2q_ns + t_meas_ns) / 1000.0

    # Retrieve physical QPU properties if available
    t1_values = []
    t2_values = []
    e_2q_values = []

    has_props = hasattr(backend, "properties") and backend.properties() is not None
    if has_props:
        props = backend.properties()
        for q_idx in active_qubits:
            try:
                t1 = props.t1(q_idx)
                if t1:
                    t1_values.append(t1 * 1e6)  # microseconds
                t2 = props.t2(q_idx)
                if t2:
                    t2_values.append(t2 * 1e6)
            except Exception:
                pass

    median_t1_us = float(np.median(t1_values)) if t1_values else 280.0
    median_t2_us = float(np.median(t2_values)) if t2_values else 140.0

    # Coherence headroom: ratio of dephasing time T2 to total circuit duration
    coherence_ratio = median_t2_us / max(estimated_duration_us, 1e-3)

    # Estimated two-qubit error accumulation
    avg_2q_err = 0.0075  # Typical median Eagle ECR error rate ~ 0.75%
    estimated_success_fid = float((1.0 - avg_2q_err) ** num_2q) if num_2q > 0 else 1.0

    return {
        "backend_name": getattr(backend, "name", str(backend)),
        "backend_num_qubits": num_physical_qubits,
        "active_physical_qubits": len(active_qubits),
        "active_qubit_layout": active_qubits[:15],
        "transpiled_circuit_depth": depth,
        "native_2q_gates": num_2q,
        "native_1q_gates": num_1q,
        "measurement_gates": num_measure,
        "gate_breakdown": dict(ops),
        "estimated_execution_time_us": round(estimated_duration_us, 2),
        "median_t1_coherence_us": round(median_t1_us, 1),
        "median_t2_coherence_us": round(median_t2_us, 1),
        "coherence_budget_headroom": round(coherence_ratio, 2),
        "estimated_unmitigated_fidelity": round(estimated_success_fid, 4),
        "recommended_error_mitigation": [
            "Twirled Readout Error Mitigation (TREX)",
            "Zero Noise Extrapolation (ZNE) via Richardson / Exponential scaling",
            "Dynamical Decoupling (XY4 sequence) during qubit idling phases",
        ],
    }


def execute_transpiled_circuit(
    transpiled_qc: QuantumCircuit,
    backend: Any,
    shots: int = 4096,
) -> Dict[str, Any]:
    """Execute transpiled circuit on backend and extract bitstring counts."""
    t_start = time.perf_counter()
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    t_run = time.perf_counter() - t_start

    counts = result.get_counts(transpiled_qc)
    return {
        "shots": shots,
        "execution_time_s": round(t_run, 3),
        "counts": counts,
    }


def decode_qpe_velocity_counts(
    counts: Dict[str, int],
    shots: int,
) -> Dict[str, Any]:
    """Decode measured precision bitstrings to physical velocity distribution."""
    # Precision register bitstrings: '00' -> 0, '01' -> +1, '11' -> -1, '10' -> 0
    p_neg = 0.0
    p_zero = 0.0
    p_pos = 0.0

    for bitstr, cnt in counts.items():
        # Clean bitstring (take precision bits, little-endian or space-separated)
        clean_bits = bitstr.replace(" ", "")[-2:]
        frac = cnt / float(shots)
        if clean_bits == "11":
            p_neg += frac
        elif clean_bits in ("00", "10"):
            p_zero += frac
        elif clean_bits == "01":
            p_pos += frac

    mean_uz = float(p_pos * (+1.0) + p_neg * (-1.0))
    return {
        "p_downward_negative": round(p_neg, 4),
        "p_neutral_zero": round(p_zero, 4),
        "p_upward_positive": round(p_pos, 4),
        "measured_mean_velocity_uz": round(mean_uz, 4),
    }


def run_hardware_pipeline(
    backend_name: str = "fake_sherbrooke",
    circuit_type: str = "qpe_velocity",
    optimization_level: int = 3,
    shots: int = 4096,
    token: Optional[str] = None,
    dry_run: bool = False,
    output_dir: str = "data/hardware_transpilation_output",
) -> Dict[str, Any]:
    """Run full physical hardware transpilation and submission pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  PHYSICAL QUANTUM HARDWARE TRANSPILATION & READINESS PIPELINE")
    print(f"  Target QPU Architecture:  {backend_name.upper()}")
    print(f"  Circuit Type:             {circuit_type}")
    print(f"  Transpilation Opt Level:  {optimization_level}")
    print(f"  Measurement Shots:        {shots}")
    print("=" * 80)

    # 1. Initialize Backend
    print(f"\n[1/4] Connecting to Target Architecture [{backend_name}]...")
    backend = get_target_backend(backend_name, token=token)
    print(f"      Connected: {backend.name} ({backend.num_qubits} physical qubits)")

    # 2. Build Algorithmic Circuit
    print(f"\n[2/4] Constructing Algorithmic QLBM Circuit ({circuit_type})...")
    raw_qc, measured_qubits = build_hardware_circuit(circuit_type, prepare_downward_state=True)
    print(f"      Algorithmic Circuit: {raw_qc.num_qubits} qubits, {raw_qc.depth()} depth, {raw_qc.count_ops()}")

    # 3. Transpile Targeting Hardware Heavy-Hex Topology
    print(f"\n[3/4] Transpiling to Physical Native Gates (ECR/CZ, RZ, SX, X) [Level {optimization_level}]...")
    t_trans_start = time.perf_counter()
    transpiled_qc = transpile(raw_qc, backend=backend, optimization_level=optimization_level)
    t_trans = time.perf_counter() - t_trans_start
    print(f"      Transpilation Complete in {t_trans:.2f} s")

    # 4. Hardware Readiness Analysis
    analysis = analyze_hardware_readiness(transpiled_qc, backend)
    print("\n--- Physical Hardware Execution Analysis ---")
    print(f"  Target QPU:              {analysis['backend_name']} ({analysis['backend_num_qubits']} physical qubits)")
    print(f"  Active Physical Qubits:  {analysis['active_physical_qubits']}")
    print(f"  Transpiled Depth:        {analysis['transpiled_circuit_depth']}")
    print(f"  Native 2Q Gates (ECR):   {analysis['native_2q_gates']:,}")
    print(f"  Native 1Q Gates (SX/RZ): {analysis['native_1q_gates']:,}")
    print(f"  Est. Duration:           {analysis['estimated_execution_time_us']:.2f} µs")
    print(f"  Median QPU T1:           {analysis['median_t1_coherence_us']:.1f} µs")
    print(f"  Median QPU T2:           {analysis['median_t2_coherence_us']:.1f} µs")
    print(f"  Coherence Headroom (T2): {analysis['coherence_budget_headroom']:.2f}x")
    print(f"  Unmitigated Success Est: {analysis['estimated_unmitigated_fidelity']*100:.2f}%")
    print("  Mitigation Techniques:")
    for tech in analysis["recommended_error_mitigation"]:
        print(f"    • {tech}")

    # 5. Execution
    execution_res = None
    qpe_decoded = None
    if not dry_run:
        print(f"\n[4/4] Executing on Backend ({shots} shots)...")
        execution_res = execute_transpiled_circuit(transpiled_qc, backend, shots=shots)
        print(f"      Execution Complete in {execution_res['execution_time_s']:.2f} s")
        print(f"      Raw Measurement Counts: {dict(list(execution_res['counts'].items())[:6])}")

        if circuit_type == "qpe_velocity":
            qpe_decoded = decode_qpe_velocity_counts(execution_res["counts"], shots)
            print("\n--- Decoded Physical Velocity Readout (QPE) ---")
            print(f"  P(u_z = -1) [Downward Falling]: {qpe_decoded['p_downward_negative']:.4f}")
            print(f"  P(u_z =  0) [Rest / Horizontal]: {qpe_decoded['p_neutral_zero']:.4f}")
            print(f"  P(u_z = +1) [Upward Flow]:       {qpe_decoded['p_upward_positive']:.4f}")
            print(f"  Measured Quantum Mean <u_z>:     {qpe_decoded['measured_mean_velocity_uz']:+.4f}")
    else:
        print("\n[4/4] Dry Run Active: Skipped physical execution.")

    # Save JSON Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_analysis": analysis,
        "circuit_type": circuit_type,
        "optimization_level": optimization_level,
        "transpilation_time_s": round(t_trans, 3),
        "execution_results": execution_res,
        "decoded_qpe_velocity": qpe_decoded,
    }
    report_file = out_path / "ibm_quantum_hardware_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Hardware Report Saved] -> {report_file}")

    return report


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Physical IBM Quantum Hardware Transpilation & Submission Tool"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fake_sherbrooke",
        help="Target backend: 'fake_sherbrooke' (127Q Eagle), 'fake_brisbane', 'aer_simulator', or live backend name",
    )
    parser.add_argument(
        "--circuit-type",
        choices=["qpe_velocity", "rotation_collision", "coupled_step"],
        default="qpe_velocity",
        help="Type of QLBM circuit to transpile and run",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Transpiler optimization level (1-3)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of measurement shots",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="IBM Quantum Cloud API token (or set via QISKIT_IBM_TOKEN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform transpilation and hardware analysis without submitting execution",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hardware_transpilation_output",
        help="Directory to save hardware analysis report",
    )

    args = parser.parse_args()

    run_hardware_pipeline(
        backend_name=args.backend,
        circuit_type=args.circuit_type,
        optimization_level=args.opt_level,
        shots=args.shots,
        token=args.token,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
