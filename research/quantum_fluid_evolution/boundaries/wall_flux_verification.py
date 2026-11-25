"""Physical Wall Face Flux Verification for Quantum and Classical Boundary Conditions.

Quantifies fluid penetration flux across solid boundaries in 3D geometries:
1. Stationary Basin Floor: Planar horizontal wall at z = 0.
2. Elevated Shelf and Vertical Cliff: Multi-level shelf (z <= z_shelf, x <= x_shelf).
3. Closed Vertical Gate: Thin vertical barrier at x = x_gate.

For each solid face with outward normal n:
- Incoming mass flux toward wall: Phi_in = sum_{c_i . n > 0} f_i(x_fluid) * (c_i . n)
- Reflected mass flux returning:  Phi_out = sum_{c_i . n > 0} f_opp(i)(x_solid) * (c_i . n)
- Net penetrating flux:          Phi_net = Phi_in - Phi_out == 0.0

Verifies machine-precision no-penetration boundary confinement.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlbm.classical.d3q19 import C, OPPOSITE, WEIGHTS
from qlbm.components.vof.quantum_boundary_3d import QuantumD3Q19SpatialBoundaryOperator
from research.quantum_fluid_evolution.validation.independent_classical_d3q19 import IndependentClassicalD3Q19Solver


class WallFaceFluxVerifier:
    """Verifies face-normal mass flux across fluid-solid interfaces."""

    def __init__(self, nx: int = 8, ny: int = 4, nz: int = 8) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)

    def test_stationary_floor_flux(self) -> Dict[str, Any]:
        """Test zero-penetration flux across horizontal basin floor (z = 0)."""
        solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        solid_mask[0, :, :] = True

        # Fluid column with downward momentum toward the floor
        f = np.zeros((19, self.nz, self.ny, self.nx), dtype=np.float64)
        for i in range(19):
            f[i, ~solid_mask] = WEIGHTS[i] * 1.0
        # Inject downward velocity (-z, velocity index 6)
        f[6, 1, :, :] += 0.05
        f[5, 1, :, :] -= 0.05

        return self._evaluate_flux_on_mask(f, solid_mask, "Stationary Basin Floor (z=0)")

    def test_elevated_shelf_and_cliff_flux(self, z_shelf: int = 3, x_shelf: int = 3) -> Dict[str, Any]:
        """Test zero-penetration flux across elevated shelf and vertical cliff drop."""
        solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        solid_mask[0, :, :] = True  # Floor
        solid_mask[: z_shelf + 1, :, : x_shelf + 1] = True  # Shelf & cliff

        f = np.zeros((19, self.nz, self.ny, self.nx), dtype=np.float64)
        for i in range(19):
            f[i, ~solid_mask] = WEIGHTS[i] * 1.0
        # Fluid impacting the vertical cliff face (+x momentum toward x=x_shelf+1)
        f[1, 1:z_shelf+1, :, x_shelf+1] += 0.04
        f[2, 1:z_shelf+1, :, x_shelf+1] -= 0.04

        return self._evaluate_flux_on_mask(f, solid_mask, "Elevated Shelf & Cliff Drop")

    def test_closed_gate_flux(self, x_gate: int = 4) -> Dict[str, Any]:
        """Test zero-penetration flux across a closed vertical barrier gate."""
        solid_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        solid_mask[0, :, :] = True  # Floor
        solid_mask[:, :, x_gate] = True  # Vertical gate barrier

        f = np.zeros((19, self.nz, self.ny, self.nx), dtype=np.float64)
        for i in range(19):
            f[i, ~solid_mask] = WEIGHTS[i] * 1.0
        # Fluid pressing horizontally against the closed gate
        f[1, :, :, x_gate - 1] += 0.05
        f[2, :, :, x_gate - 1] -= 0.05

        return self._evaluate_flux_on_mask(f, solid_mask, "Closed Vertical Gate Barrier")

    def _evaluate_flux_on_mask(
        self, f_pre: np.ndarray, solid_mask: np.ndarray, name: str
    ) -> Dict[str, Any]:
        """Compute face-normal flux before and after reflection on solid nodes."""
        solver = IndependentClassicalD3Q19Solver(
            nx=self.nx, ny=self.ny, nz=self.nz, solid_mask=solid_mask
        )
        bound_op = QuantumD3Q19SpatialBoundaryOperator(
            nx=self.nx, ny=self.ny, nz=self.nz, solid_mask=solid_mask
        )

        # 1. Stream incoming populations into solid boundary nodes
        f_streamed = solver.stream(f_pre)

        # 2. Bounce-back reflection on solid nodes
        f_post = bound_op.apply_to_distribution_array(f_streamed)

        total_inward_flux = 0.0
        total_reflected_flux = 0.0
        max_face_discrepancy = 0.0

        # Scan all fluid cells adjacent to solid cells
        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    if solid_mask[z, y, x]:
                        continue
                    # Cell is fluid; check its 18 moving velocities
                    for i in range(1, 19):
                        cx, cy, cz = C[i]
                        nbr_z = (z + cz) % self.nz
                        nbr_y = (y + cy) % self.ny
                        nbr_x = (x + cx) % self.nx

                        if solid_mask[nbr_z, nbr_y, nbr_x]:
                            # Population f_i streamed from fluid (z,y,x) into solid (nbr_z, nbr_y, nbr_x)
                            f_incoming = f_pre[i, z, y, x]
                            # Reflected population at solid node in opposite direction
                            f_reflected = f_post[OPPOSITE[i], nbr_z, nbr_y, nbr_x]

                            total_inward_flux += f_incoming
                            total_reflected_flux += f_reflected
                            diff = abs(f_incoming - f_reflected)
                            if diff > max_face_discrepancy:
                                max_face_discrepancy = diff

        net_flux = abs(total_inward_flux - total_reflected_flux)
        mass_in_solid_after = float(np.sum(f_post[:, solid_mask]))
        mass_in_solid_before = float(np.sum(f_pre[:, solid_mask]))
        solid_mass_leak = abs(mass_in_solid_after - mass_in_solid_before)

        return {
            "name": name,
            "solid_cells": int(np.sum(solid_mask)),
            "fluid_cells": int(np.sum(~solid_mask)),
            "total_inward_flux": total_inward_flux,
            "total_reflected_flux": total_reflected_flux,
            "net_penetration_flux": net_flux,
            "max_face_discrepancy": max_face_discrepancy,
            "solid_mass_leakage": solid_mass_leak,
            "status": "PASSED" if net_flux < 1e-15 and max_face_discrepancy < 1e-15 else "FAILED",
        }


def run_wall_flux_suite() -> None:
    """Run all wall flux tests and print a clean scorecard."""
    print("=" * 85)
    print("  PHYSICAL WALL FACE FLUX & ZERO-PENETRATION VERIFICATION")
    print("=" * 85)
    print(f"{'Boundary Feature':<32} | {'Solid Cells':^11} | {'Net Penetration':^17} | {'Max Face Err':^14} | {'Status':^8}")
    print("-" * 85)

    verifier = WallFaceFluxVerifier(nx=8, ny=4, nz=8)
    tests = [
        verifier.test_stationary_floor_flux(),
        verifier.test_elevated_shelf_and_cliff_flux(),
        verifier.test_closed_gate_flux(),
    ]

    for t in tests:
        row = (
            f"{t['name']:<32} | "
            f"{t['solid_cells']:^11d} | "
            f"{t['net_penetration_flux']:^17.2e} | "
            f"{t['max_face_discrepancy']:^14.2e} | "
            f"{t['status']:^8}"
        )
        print(row)
    print("=" * 85)


if __name__ == "__main__":
    run_wall_flux_suite()
