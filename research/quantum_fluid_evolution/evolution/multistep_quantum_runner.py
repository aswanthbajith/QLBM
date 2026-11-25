"""Multi-Step Quantum Fluid Evolution Runner Without State Reinitialization.

Inspired by 2026 multi-step quantum evolution research (arXiv:2510.05965,
Comp. Phys. Comm. 2026).

Demonstrates continuous multi-step quantum fluid evolution:
- Executes 3 to 5 consecutive steps of:
  [Collision + Gravity] -> [Spatial Streaming] -> [Coherent Wall Reflection]
- Statevector |Psi_t> is retained continuously in the quantum register.
- Does NOT perform classical wavefunction collapse, readout, or re-encoding
  between consecutive time steps.
- Tracks non-destructive physical observables and ancilla success statistics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C, WEIGHTS
from qlbm.components.vof.quantum_stepper_3d import QuantumFluid3DStepper


class MultiStepQuantumRunner:
    """Orchestrates multi-step quantum fluid simulation runs."""

    def __init__(
        self,
        nx: int = 4,
        ny: int = 4,
        nz: int = 8,
        tau: float = 0.8,
        gz: float = -0.01,
        dt: float = 1.0,
        solid_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.tau = float(tau)
        self.gz = float(gz)
        self.dt = float(dt)

        if solid_mask is None:
            self.solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
            self.solid_mask[0, :, :] = True  # Default solid floor
        else:
            self.solid_mask = np.asarray(solid_mask, dtype=bool)

        self.stepper = QuantumFluid3DStepper(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            tau=self.tau,
            g=(0.0, 0.0, self.gz),
            dt=self.dt,
            solid_mask=self.solid_mask,
        )

    def run_multi_step_simulation(
        self,
        num_steps: int = 4,
        initial_density: float = 1.0,
    ) -> Dict[str, Any]:
        """Execute continuous multi-step simulation."""
        # Initial fluid resting on fluid cells
        f_init = np.zeros((19, self.nz, self.ny, self.nx), dtype=np.float64)
        for i in range(19):
            f_init[i, ~self.solid_mask] = WEIGHTS[i] * initial_density

        # Run multi-step quantum evolution
        t0 = time.perf_counter()
        res_quantum = self.stepper.run_quantum_evolution(f_init, num_steps=num_steps)
        t_quantum = time.perf_counter() - t0

        # Run exact classical reference for validation
        t1 = time.perf_counter()
        res_classical = self.stepper.run_classical_evolution(f_init, num_steps=num_steps)
        t_classical = time.perf_counter() - t1

        resources = self.stepper.resource_summary()

        return {
            "num_steps": num_steps,
            "grid": (self.nx, self.ny, self.nz),
            "resources": resources,
            "quantum_time_sec": t_quantum,
            "classical_time_sec": t_classical,
            "quantum_history": res_quantum["history"],
            "classical_history": res_classical["history"],
            "final_quantum_distributions": res_quantum["f_final"],
            "final_classical_distributions": res_classical["f_final"],
        }


def print_runner_report(data: Dict[str, Any]) -> None:
    """Print formatted summary of multi-step quantum execution."""
    print("=" * 90)
    print("  MULTI-STEP QUANTUM FLUID EVOLUTION RUNNER (No State Reinitialization)")
    print(f"  Grid: {data['grid'][0]} x {data['grid'][1]} x {data['grid'][2]} | Steps: {data['num_steps']}")
    print("=" * 90)

    res = data["resources"]
    print(f"\n--- Quantum Hardware Scaling ---")
    print(f"  Total Qubits:        {res['total_qubits']}")
    print(f"  Circuit Depth:       {res['step_circuit_depth']}")
    print(f"  Two-Qubit Gates:     {res['two_qubit_gates']}")
    print(f"  Statevector Memory:  {res['statevector_ram_bytes'] / 1024:.1f} KB")

    print(f"\n--- Step-by-Step Observable Trajectory ---")
    header = f"{'Step':^6} | {'P_succ':^10} | {'Cumul P_succ':^13} | {'Quantum Jz':^12} | {'Classical Jz':^12} | {'Delta Jz':^11} | {'Mass Err':^11}"
    print(header)
    print("-" * 90)

    for s in range(data["num_steps"]):
        q = data["quantum_history"][s]
        c = data["classical_history"][s]
        d_jz = abs(q["jz"] - c["jz"])
        m_err = abs(q["total_mass"] - c["total_mass"])
        row = (
            f"{q['step']:^6d} | "
            f"{q['p_succ']:^10.6f} | "
            f"{q['cumulative_p_succ']:^13.6f} | "
            f"{q['jz']:^12.6f} | "
            f"{c['jz']:^12.6f} | "
            f"{d_jz:^11.3e} | "
            f"{m_err:^11.3e}"
        )
        print(row)

    f_diff = np.max(np.abs(data["final_quantum_distributions"] - data["final_classical_distributions"]))
    print("=" * 90)
    print(f"Final Distribution Discrepancy: {f_diff:.4e}")
    print(f"Quantum Runtime:                {data['quantum_time_sec']:.3f} s")
    print(f"Classical Reference Runtime:    {data['classical_time_sec']:.3f} s")
    print("=" * 90)


if __name__ == "__main__":
    runner = MultiStepQuantumRunner(nx=4, ny=4, nz=8, gz=-0.01)
    res = runner.run_multi_step_simulation(num_steps=4)
    print_runner_report(res)
