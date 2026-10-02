"""Tests for the Shan-Chen Pseudopotential Multiphase Model (Phase 9).

Validates:
1. Pseudopotential evaluation (exponential and Carnahan-Starling).
2. Lattice force isotropy on uniform density fields (zero net force).
3. Attractive force orientation across density interfaces.
4. Subcritical stability: damping of perturbations when G > G_c = -4.0.
5. Supercritical spinodal decomposition: spontaneous liquid-vapor separation when G = -5.0.
6. Exact machine-precision mass conservation during phase separation.
7. Non-ideal equation of state van der Waals-like spinodal loop.
8. Gravity-driven phase stratification (liquid settles to bottom, gas rises).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qlbm.classical.d2q9 import CS_SQ
from qlbm.classical.multiphase import (
    ShanChenMultiphaseSolver,
    compute_pseudopotential,
    compute_shan_chen_force,
    shan_chen_pressure,
)


class TestPseudopotentialAndForceFormulation:
    """Validate pseudopotential mathematical properties and force calculation."""

    def test_exponential_pseudopotential_properties(self):
        """psi(0) = 0, psi strictly increasing, asymptotically approaching rho_0."""
        rho_vals = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 20.0])
        psi = compute_pseudopotential(rho_vals, form="exponential", rho_0=1.0)

        assert psi[0] == 0.0
        # Strictly increasing
        assert np.all(np.diff(psi) > 0.0)
        # Asymptote at rho_0 = 1.0
        assert np.isclose(psi[-1], 1.0, atol=1e-6)

    def test_uniform_density_force_isotropy(self):
        """On a uniform density field, Shan-Chen force must vanish identically."""
        ny, nx = 16, 16
        rho_uniform = np.full((ny, nx), 1.25)
        psi = compute_pseudopotential(rho_uniform)
        F_sc = compute_shan_chen_force(psi, G=-5.0)

        assert np.allclose(F_sc, 0.0, atol=1e-15)

    def test_attractive_force_direction(self):
        """For G < 0, interaction force points toward denser phase."""
        ny, nx = 8, 16
        rho = np.ones((ny, nx))
        # High density slab on left (x < 8), low density on right (x >= 8)
        rho[:, :8] = 1.5
        rho[:, 8:] = 0.5

        psi = compute_pseudopotential(rho)
        F_sc = compute_shan_chen_force(psi, G=-5.0)

        # At right interface: fluid at x=8 (dilute) and x=7 (dense) are pulled leftward into dense slab
        assert F_sc[0, ny // 2, 8] < -1e-4
        assert F_sc[0, ny // 2, 7] < -1e-4

        # At left interface (periodic wrap): fluid at x=15 (dilute) and x=0 (dense) are pulled rightward into dense slab
        assert F_sc[0, ny // 2, 15] > 1e-4
        assert F_sc[0, ny // 2, 0] > 1e-4

    def test_carnahan_starling_pseudopotential(self):
        """Carnahan-Starling pseudopotential produces valid non-negative real numbers."""
        rho = np.linspace(0.1, 0.5, 10)
        psi_cs = compute_pseudopotential(rho, form="carnahan_starling", G=-1.0)
        assert np.all(np.isfinite(psi_cs))
        assert np.all(psi_cs >= 0.0)


class TestPhaseSeparationAndThermodynamics:
    """Validate spinodal decomposition, coexistence densities, and mass conservation."""

    def test_subcritical_stability_g_greater_than_critical(self):
        """For G = -3.0 > G_c = -4.0, density perturbations damp out to uniform state."""
        nx, ny = 16, 16
        G = -3.0
        rng = np.random.RandomState(42)
        rho_init = 1.0 + 0.04 * (rng.rand(ny, nx) - 0.5)

        solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=G)
        solver.initialize(rho=rho_init)

        solver.run(200)

        # Difference between max and min density should decay below 0.005
        density_spread = np.max(solver.rho) - np.min(solver.rho)
        assert density_spread < 0.005, f"Subcritical perturbation failed to damp: spread={density_spread}"

    def test_supercritical_spinodal_decomposition(self):
        """For G = -5.0 < G_c = -4.0, spontaneous liquid-vapor phase separation occurs."""
        nx, ny = 32, 32
        G = -5.0
        rng = np.random.RandomState(42)
        rho_init = 1.0 + 0.05 * (rng.rand(ny, nx) - 0.5)

        solver = ShanChenMultiphaseSolver(nx=nx, ny=ny, tau=1.0, G=G)
        solver.initialize(rho=rho_init)

        m0 = solver.total_mass
        solver.run(500)
        m1 = solver.total_mass

        # 1. Exact mass conservation
        assert np.isclose(m1, m0, atol=1e-11)

        # 2. Distinct coexisting phases
        rho_liq = solver.liquid_density
        rho_vap = solver.vapor_density

        assert rho_liq > 1.7, f"Expected dense liquid phase > 1.7, got {rho_liq}"
        assert rho_vap < 0.2, f"Expected dilute vapor phase < 0.2, got {rho_vap}"
        assert solver.density_ratio > 10.0, f"Expected density ratio > 10, got {solver.density_ratio}"

    def test_shan_chen_equation_of_state_spinodal_loop(self):
        """Shan-Chen equation of state exhibits van der Waals-like non-monotonic loop."""
        rho_range = np.linspace(0.1, 2.5, 100)
        p = shan_chen_pressure(rho_range, G=-5.0)

        # Derivative dp/drho
        dp_drho = np.gradient(p, rho_range)

        # Spinodal region corresponds to negative compressibility: dp/drho < 0
        has_spinodal_instability = np.any(dp_drho < 0.0)
        assert has_spinodal_instability, "Expected spinodal instability region (dp/drho < 0)"

    def test_gravity_induced_phase_stratification(self):
        """Under gravity, dense liquid phase stratifies at bottom and vapor rises."""
        nx, ny = 8, 32
        G = -5.0
        g = 5e-5

        # Initialize with dense blob in middle
        rho_init = np.full((ny, nx), 0.2)
        rho_init[10:22, :] = 1.8

        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True       # bottom wall
        solid_mask[ny - 1, :] = True  # top wall

        solver = ShanChenMultiphaseSolver(
            nx=nx,
            ny=ny,
            tau=1.0,
            G=G,
            gravity=g,
            solid_mask=solid_mask,
        )
        solver.initialize(rho=rho_init)

        # Run 1000 steps to allow gravity sedimentation
        solver.run(1000)

        # Average density in settled liquid core (lower half) vs vapor phase (upper half)
        # Note: at neutral bounce-back walls without wetting potential, a thin gas cushion
        # separates the dense liquid slab from the bottom wall, placing the liquid core at y in [6, 13].
        rho_bottom = np.mean(solver.rho[6:13])
        rho_top = np.mean(solver.rho[20:29])

        assert rho_bottom > 1.7, f"Expected dense liquid in lower domain, got {rho_bottom}"
        assert rho_top < 0.2, f"Expected dilute vapor in upper domain, got {rho_top}"
