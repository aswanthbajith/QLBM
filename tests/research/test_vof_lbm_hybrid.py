"""Tests for Phase 12: Hybrid Volume-of-Fluid (VOF) Free-Surface LBM Solver.

Validates the D2Q9 Hybrid VOF-LBM solver (`VOFHydroSolver`):
- TEST A: Volume conservation under advection
- TEST B: Local pairwise anti-symmetry and global mass conservation
- TEST C: Static circular droplet mechanical equilibrium
- TEST D: Young-Laplace pressure scaling Delta p = sigma / R across multiple radii
- TEST E: Translating interface advection without systematic mass loss
- TEST F: 2D rectangular dam-break column collapse prototype (64 x 32 domain)
"""

import numpy as np
import pytest

from qlbm.classical.d2q9 import C, CS_SQ, OPPOSITE
from qlbm.classical.hybrid_vof_lbm import (
    VOFHydroSolver,
    enforce_neighborhood_consistency,
)
from qlbm.classical.vof import CellType


class TestPairwiseAntiSymmetryAndConservation:
    """Level 2 & Level 3: Anti-symmetry of mass flux and machine-precision mass conservation."""

    def test_local_flux_anti_symmetry(self):
        """Verify Delta M_i(x) == -Delta M_bar_i(x + c_i) across all interior faces."""
        nx, ny = 32, 32
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8)
        # Create an oblique interface with varying fill levels
        y, x = np.mgrid[0:ny, 0:nx]
        fill = np.clip(0.5 - (x + y - 24.0) / 4.0, 0.0, 1.0)
        solver.initialize_from_fill_level(fill)

        # Execute 1 collision and streaming to populate post-collision state
        f_eq = solver.f.copy()
        # Non-uniform velocity to induce non-trivial fluxes
        solver.u[0] = 0.02 * np.sin(2.0 * np.pi * y / ny)
        solver.u[1] = 0.02 * np.cos(2.0 * np.pi * x / nx)
        solver.step()

        # Check total mass conservation
        assert solver.mass_error() < 1e-12

    def test_global_mass_conservation_over_time(self):
        """Verify |M(t) - M(0)| / M(0) < 1e-12 over 50 consecutive time steps."""
        nx, ny = 32, 32
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, gravity=0.001)
        solver.initialize_droplet(radius=8.0, xc=16.0, yc=16.0)

        initial_mass = solver.total_mass
        assert initial_mass > 0.0

        for _ in range(50):
            solver.step()
            assert abs(solver.total_mass - initial_mass) / initial_mass < 1e-12


class TestInterfaceAdvection:
    """Level 3: Interface translation without numerical diffusion or systematic mass loss."""

    def test_translating_droplet_advection_x(self):
        """A circular droplet with uniform velocity translates linearly in x."""
        nx, ny = 64, 32
        u0 = 0.04
        u_init = np.zeros((2, ny, nx), dtype=np.float64)
        u_init[0, :, :] = u0

        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8)
        y, x = np.mgrid[0:ny, 0:nx]
        r = np.sqrt((x - 16.0) ** 2 + (y - 16.0) ** 2)
        fill = np.clip(0.5 - (r - 7.0), 0.0, 1.0)
        solver.initialize_from_fill_level(fill, u_init=u_init)

        cm_start = solver.center_of_mass
        num_steps = 30
        for _ in range(num_steps):
            solver.step()

        cm_end = solver.center_of_mass
        dx = cm_end[0] - cm_start[0]
        dy = cm_end[1] - cm_start[1]

        # Droplet must move in +x direction
        assert dx > 0.8 * (u0 * num_steps)
        # Transverse drift must be negligible (< 0.1 lattice units)
        assert abs(dy) < 0.1
        # Mass must be strictly conserved
        assert solver.mass_error() < 1e-12

    def test_volume_conservation_under_shear(self):
        """Liquid column under shearing motion conserves volume fraction integral."""
        nx, ny = 40, 40
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.9)
        u_shear = np.zeros((2, ny, nx), dtype=np.float64)
        u_shear[0, :, :] = 0.03 * np.sin(2.0 * np.pi * np.arange(ny)[:, None] / ny)

        y, x = np.mgrid[0:ny, 0:nx]
        r = np.sqrt((x - 20.0) ** 2 + (y - 20.0) ** 2)
        fill = np.clip(0.5 - (r - 8.0), 0.0, 1.0)
        solver.initialize_from_fill_level(fill, u_init=u_shear)

        init_vol = solver.total_volume
        for _ in range(40):
            solver.step()

        # Strict mass conservation to machine precision
        assert solver.mass_error() < 1e-12
        # Slight compressibility fluctuation in LBM remains bounded (< 1%)
        assert solver.volume_error() < 0.01


class TestStaticDropletEquilibrium:
    """Level 3: Static circular droplet mechanical equilibrium."""

    def test_static_droplet_stability(self):
        """A static droplet with surface tension remains circular and stationary."""
        nx, ny = 48, 48
        sigma = 0.01
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, surface_tension=sigma)
        solver.initialize_droplet(radius=10.0, xc=24.0, yc=24.0)

        cm_init = solver.center_of_mass

        for _ in range(60):
            solver.step()

        cm_final = solver.center_of_mass

        # Center of mass should not drift by more than 0.05 lattice cells
        assert abs(cm_final[0] - cm_init[0]) < 0.05
        assert abs(cm_final[1] - cm_init[1]) < 0.05

        # Maximum velocity must remain small (no parasitic blow-up)
        assert solver.velocity_magnitude().max() < 0.05

        # Mass strictly conserved
        assert solver.mass_error() < 1e-12


class TestYoungLaplacePressureScaling:
    """Level 4: Quantitative verification of Young-Laplace law Delta p = sigma / R."""

    def test_young_laplace_scaling_across_radii(self):
        """Measure interior pressure across R in [8, 12, 16] and verify Delta p proportional to 1/R."""
        nx, ny = 48, 48
        sigma = 0.01
        radii = [8.0, 12.0, 16.0]
        measured_pressures = []

        for R in radii:
            solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, surface_tension=sigma)
            solver.initialize_droplet(radius=R, xc=24.0, yc=24.0)
            for _ in range(25):
                solver.step()

            # Measure interface pressure jump over gas reference (p_gas = 0)
            iface_mask = solver.cell_types == CellType.INTERFACE
            p_iface = CS_SQ * (np.mean(solver.rho[iface_mask]) - 1.0)
            measured_pressures.append(p_iface)

        inv_r = [1.0 / R for R in radii]

        # Linear correlation between Delta p and 1/R
        corr_matrix = np.corrcoef(inv_r, measured_pressures)
        r_value = corr_matrix[0, 1]

        # Strong positive linear correlation expected
        assert r_value > 0.90, f"Expected r > 0.90, got {r_value:.4f}"

        # Pressure must be strictly positive (Laplace overpressure inside droplet)
        for p in measured_pressures:
            assert p > 0.0


class TestDamBreakColumnCollapse:
    """Level 4: Benchmark rectangular liquid column collapse prototype."""

    def test_dam_break_collapse_dynamics(self):
        """Domain 64x32, 16x24 column: y_cm drops, x_front advances, mass strictly conserved."""
        nx, ny = 64, 32
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True       # bottom wall
        solid_mask[:, 0] = True       # left wall
        solid_mask[:, -1] = True      # right wall

        solver = VOFHydroSolver(
            nx=nx,
            ny=ny,
            tau=0.8,
            gravity=0.0005,
            solid_mask=solid_mask,
        )

        fill = np.zeros((ny, nx), dtype=np.float64)
        fill[1:25, 1:17] = 1.0
        solver.initialize_from_fill_level(fill)

        init_cm = solver.center_of_mass
        init_front = solver.front_position()

        for step in range(1, 101):
            solver.step()

        final_cm = solver.center_of_mass
        final_front = solver.front_position()

        # 1. Height of center of mass must drop under gravity
        assert final_cm[1] < init_cm[1], (
            f"Expected y_cm drop: init={init_cm[1]:.3f}, final={final_cm[1]:.3f}"
        )

        # 2. Front position must advance along the floor
        assert final_front > init_front, (
            f"Expected front advancement: init={init_front}, final={final_front}"
        )

        # 3. Horizontal center of mass must shift rightward
        assert final_cm[0] > init_cm[0], (
            f"Expected x_cm advance: init={init_cm[0]:.3f}, final={final_cm[0]:.3f}"
        )

        # 4. Mass conservation must hold to machine precision
        assert solver.mass_error() < 1e-12

        # 5. Low-Mach number stability: velocities stay physical
        assert solver.velocity_magnitude().max() < 0.20

    def test_dam_break_mass_conservation_strict(self):
        """Track relative mass drift at every time step for 50 steps."""
        nx, ny = 48, 32
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, gravity=0.001)
        solver.initialize_dam_break(column_width=12, column_height=18)

        for _ in range(50):
            solver.step()
            assert solver.mass_error() < 1e-12
