"""Tests for the Körner (2005) / Thürey (2007) free-surface mass-exchange and cell-conversion scheme."""

import math
from pathlib import Path

import numpy as np
import pytest

from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver
from qlbm.classical.vof import CellType
from qlbm.components.vof.hybrid_stepper import HybridVOFStep


class TestKornerFreeSurface:
    """Validation and regression suite for the Körner free-surface scheme."""

    def test_no_mass_transfer_to_pure_gas(self):
        """In a single mass exchange step, no liquid mass transfers across pure GAS boundaries."""
        nx, ny = 16, 16
        solid = np.zeros((ny, nx), dtype=bool)
        solid[0, :] = solid[:, 0] = solid[:, -1] = True

        solver = VOFHydroSolver(
            nx=nx,
            ny=ny,
            tau=0.8,
            gravity=0.0,
            solid_mask=solid,
            free_surface_scheme="korner",
        )
        solver.initialize_dam_break(column_width=6, column_height=8, add_walls=False)

        # Before step, identify pure GAS cells
        pure_gas_before = solver.cell_types == CellType.GAS
        assert np.any(pure_gas_before)
        assert np.all(solver.mass[pure_gas_before] == 0.0)

        # Run 1 step
        solver.step()

        # Newly classified GAS cells must have zero mass
        pure_gas_after = solver.cell_types == CellType.GAS
        assert np.all(solver.mass[pure_gas_after] == 0.0)
        assert solver.mass_error() < 1e-12

    def test_long_term_dam_break_conservation_and_bounds(self):
        """Over 300 steps, mass error remains at machine precision, C in [0, 1], and no LIQUID touches GAS."""
        nx, ny = 32, 32
        solid = np.zeros((ny, nx), dtype=bool)
        solid[0, :] = solid[:, 0] = solid[:, -1] = True

        solver = VOFHydroSolver(
            nx=nx,
            ny=ny,
            tau=0.8,
            gravity=0.0005,
            solid_mask=solid,
            free_surface_scheme="korner",
        )
        solver.initialize_dam_break(column_width=8, column_height=16, add_walls=False)

        for step in range(1, 301):
            solver.step()
            assert solver.mass_error() < 1e-12, f"Mass error exceeded at step {step}: {solver.mass_error()}"
            assert np.all(solver.fill_level >= 0.0)
            assert np.all(solver.fill_level <= 1.0)

            # Check topological completeness: LIQUID must never directly contact GAS
            t = solver.cell_types
            is_liq = t == CellType.LIQUID
            is_gas = t == CellType.GAS
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    gas_nbr = np.roll(np.roll(is_gas, -dy, axis=0), -dx, axis=1)
                    assert not np.any(is_liq & gas_nbr), f"Direct liquid-gas contact at step {step}"

    def test_interface_sharpness_bounded_vs_legacy(self):
        """Körner scheme maintains a significantly sharper interface (fewer diffuse cells 0.01 < C < 0.5) than legacy."""
        nx, ny = 32, 32
        solid = np.zeros((ny, nx), dtype=bool)
        solid[0, :] = solid[:, 0] = solid[:, -1] = True

        # Run legacy
        s_leg = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.0005, solid_mask=solid, free_surface_scheme="legacy"
        )
        s_leg.initialize_dam_break(column_width=8, column_height=16, add_walls=False)
        for _ in range(150):
            s_leg.step()

        # Run korner
        s_kor = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.0005, solid_mask=solid, free_surface_scheme="korner"
        )
        s_kor.initialize_dam_break(column_width=8, column_height=16, add_walls=False)
        for _ in range(150):
            s_kor.step()

        diffuse_leg = int(np.sum((s_leg.fill_level > 0.01) & (s_leg.fill_level < 0.5)))
        diffuse_kor = int(np.sum((s_kor.fill_level > 0.01) & (s_kor.fill_level < 0.5)))

        # Körner should have fewer smeared cells than legacy
        assert diffuse_kor < diffuse_leg, f"Körner diffuse cells ({diffuse_kor}) not less than legacy ({diffuse_leg})"

    def test_legacy_scheme_bit_exact_reproducibility(self):
        """The 'legacy' free_surface_scheme reproduces saved regression fixture exactly."""
        ref_path = Path(__file__).parent / "data" / "legacy_dam_break_32x32_60steps.npz"
        if not ref_path.exists():
            pytest.skip("Reference fixture not found")

        ref = np.load(ref_path)
        nx, ny = 32, 32
        solid = np.zeros((ny, nx), dtype=bool)
        solid[0, :] = solid[:, 0] = solid[:, -1] = True

        solver = VOFHydroSolver(
            nx=nx,
            ny=ny,
            tau=0.8,
            gravity=0.0005,
            surface_tension=0.0,
            solid_mask=solid,
            free_surface_scheme="legacy",
        )
        solver.initialize_dam_break(column_width=8, column_height=12, add_walls=False)

        for _ in range(60):
            solver.step()

        assert np.allclose(solver.mass, ref["mass"], atol=1e-14)
        assert np.allclose(solver.fill_level, ref["fill"], atol=1e-14)
        assert np.allclose(solver.rho, ref["rho"], atol=1e-14)
        assert np.allclose(solver.u, ref["u"], atol=1e-14)
        assert np.allclose(solver.f, ref["f"], atol=1e-14)
        assert np.array_equal(solver.cell_types, ref["cell_types"])

    def test_classical_vs_hybrid_korner_equivalence(self):
        """HybridVOFStep (Option A) matches classical VOFHydroSolver to < 1e-11 under the Körner scheme."""
        nx, ny = 16, 16
        solid = np.zeros((ny, nx), dtype=bool)
        solid[0, :] = solid[:, 0] = solid[:, -1] = True

        s_clas = VOFHydroSolver(
            nx=nx,
            ny=ny,
            tau=0.8,
            gravity=0.0005,
            surface_tension=0.0,
            solid_mask=solid,
            free_surface_scheme="korner",
        )
        s_clas.initialize_dam_break(column_width=4, column_height=8, add_walls=False)

        s_hybr = VOFHydroSolver(
            nx=nx,
            ny=ny,
            tau=0.8,
            gravity=0.0005,
            surface_tension=0.0,
            solid_mask=solid,
            free_surface_scheme="korner",
        )
        s_hybr.initialize_dam_break(column_width=4, column_height=8, add_walls=False)
        stepper = HybridVOFStep(s_hybr, use_quantum_streaming=True)

        for _ in range(10):
            s_clas.step()
            stepper.step()

        diff_mass = float(np.max(np.abs(s_clas.mass - s_hybr.mass)))
        diff_fill = float(np.max(np.abs(s_clas.fill_level - s_hybr.fill_level)))
        diff_rho = float(np.max(np.abs(s_clas.rho - s_hybr.rho)))
        diff_u = float(np.max(np.abs(s_clas.u - s_hybr.u)))

        assert diff_mass < 1e-11, f"Mass discrepancy: {diff_mass}"
        assert diff_fill < 1e-11, f"Fill discrepancy: {diff_fill}"
        assert diff_rho < 1e-11, f"Density discrepancy: {diff_rho}"
        assert diff_u < 1e-11, f"Velocity discrepancy: {diff_u}"
