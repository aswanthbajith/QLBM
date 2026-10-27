"""Comprehensive numerical validation suite for the classical and hybrid quantum dam-break simulator."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qlbm.classical.d2q9 import (
    C,
    CS_SQ,
    OPPOSITE,
    WEIGHTS,
    bgk_collision,
    compute_equilibrium,
    compute_guo_force,
    compute_macroscopic,
    stream_and_bounce,
)
from qlbm.classical.hybrid_vof_lbm import VOFHydroSolver
from qlbm.classical.vof import CellType, classify_cells
from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.vof.boundary import QuantumD2Q9BounceBackOperator
from qlbm.components.vof.hybrid_stepper import HybridVOFStep
from qlbm.components.vof.streaming import QuantumD2Q9StreamingOperator


class TestInitialStateValidation:
    """Phase 3 / Section 10.1: Rigorous initial dam-break state verification."""

    def test_initial_geometry_and_mass(self):
        """Initial column geometry, zero velocity, and exact mass allocation."""
        nx, ny = 64, 32
        cw, ch = 16, 24
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[:, 0] = True
        solid_mask[:, -1] = True

        solver = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.0005, solid_mask=solid_mask
        )
        solver.initialize_dam_break(column_width=cw, column_height=ch, add_walls=False)

        # 1. Dimensions and grid shape
        assert solver.fill_level.shape == (ny, nx)
        assert solver.f.shape == (9, ny, nx)

        # 2. Liquid occupies exactly (cw x ch) unit cells
        # In dam break initialization: fill[1:ch+1, 1:cw+1] = 1.0 -> 16 * 24 = 384 cells
        liquid_cells = np.sum(solver.fill_level == 1.0)
        expected_cells = cw * ch
        assert liquid_cells == expected_cells, f"Expected {expected_cells} cells, got {liquid_cells}"

        # 3. No liquid inside solid walls
        assert np.all(solver.fill_level[solid_mask] == 0.0)
        assert np.all(solver.mass[solid_mask] == 0.0)

        # 4. Initial velocity is zero everywhere
        assert np.max(np.abs(solver.u)) == 0.0

        # 5. Total mass equals volume * rho_liquid
        expected_mass = float(expected_cells * 1.0)
        assert abs(solver.total_mass - expected_mass) < 1e-12

        # 6. Fill fractions bounded in [0, 1]
        assert np.all(solver.fill_level >= 0.0) and np.all(solver.fill_level <= 1.0)

        # 7. Distributions match equilibrium in active fluid cells
        f_eq_expected = compute_equilibrium(solver.rho, solver.u)
        active = solver.fluid_mask
        assert np.allclose(solver.f[:, active], f_eq_expected[:, active], atol=1e-14)
        assert np.all(solver.f[:, ~active] == 0.0)

        # 8. Deterministic initialization
        solver2 = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.0005, solid_mask=solid_mask
        )
        solver2.initialize_dam_break(column_width=cw, column_height=ch, add_walls=False)
        assert np.array_equal(solver.fill_level, solver2.fill_level)
        assert np.array_equal(solver.f, solver2.f)


class TestQuantumStreamingValidation:
    """Phase 6 / Section 10.2: Quantum-circuit streaming accuracy vs classical reference."""

    def test_single_occupied_basis_states(self):
        """Single occupied basis state |x, y, i> streams to |(x+cx)%Nx, (y+cy)%Ny, i>."""
        nx, ny = 4, 4
        streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)

        for v in range(9):
            cx, cy = C[v]
            x_in, y_in = 2, 2
            f_in = np.zeros((9, ny, nx), dtype=np.float64)
            f_in[v, y_in, x_in] = 1.0

            f_q = streamer.stream_distribution_array(f_in)
            expected_x = (x_in + cx) % nx
            expected_y = (y_in + cy) % ny

            # Must have probability 1.0 at destination
            assert abs(f_q[v, expected_y, expected_x] - 1.0) < 1e-12
            # Sum must be 1.0
            assert abs(np.sum(f_q) - 1.0) < 1e-12

    def test_boundary_adjacent_streaming(self):
        """Populations on boundary coordinates wrap periodically without leakage or distortion."""
        nx, ny = 4, 4
        streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)

        # Corner node (0, 0) with diagonal velocity 7 (-1, -1) -> wraps to (3, 3)
        f_in = np.zeros((9, ny, nx), dtype=np.float64)
        f_in[7, 0, 0] = 1.0
        f_q = streamer.stream_distribution_array(f_in)
        assert abs(f_q[7, 3, 3] - 1.0) < 1e-12

    def test_streaming_error_bound_random_fields(self):
        """Max difference E_max between quantum and classical periodic streaming < 1e-12."""
        nx, ny = 8, 4
        streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)

        rng = np.random.default_rng(999)
        for _ in range(5):
            f_in = rng.uniform(0.05, 2.0, size=(9, ny, nx))
            f_q = streamer.stream_distribution_array(f_in)
            f_c = stream_and_bounce(f_in, solid_mask=None)

            e_max = float(np.max(np.abs(f_q - f_c)))
            assert e_max < 1e-12, f"E_max = {e_max:.2e} exceeds tolerance 1e-12"

    def test_invalid_and_edge_inputs(self):
        """Safe handling of zero total distribution and rejection of negative inputs."""
        nx, ny = 4, 4
        streamer = QuantumD2Q9StreamingOperator(nx=nx, ny=ny)

        # Zero input
        f_zero = np.zeros((9, ny, nx), dtype=np.float64)
        f_out = streamer.stream_distribution_array(f_zero)
        assert np.max(np.abs(f_out)) == 0.0

        # Negative input must raise ValueError
        f_neg = np.zeros((9, ny, nx), dtype=np.float64)
        f_neg[0, 1, 1] = -0.5
        with pytest.raises(ValueError, match="Negative populations detected"):
            streamer.stream_distribution_array(f_neg)

        # Invalid shape must raise ValueError
        with pytest.raises(ValueError, match="Expected shape"):
            streamer.stream_distribution_array(np.ones((9, 8, 8)))


class TestCollisionConservationValidation:
    """Section 10.3: Mass and momentum conservation under BGK collision with Guo forcing."""

    def test_bgk_collision_mass_conservation(self):
        """BGK collision strictly conserves density sum(f*) = sum(f) = rho."""
        rng = np.random.default_rng(42)
        f = rng.uniform(0.1, 1.0, size=(9, 4, 4))
        rho, u = compute_macroscopic(f)
        f_eq = compute_equilibrium(rho, u)
        f_post = bgk_collision(f, f_eq, tau=0.8)

        rho_post = np.sum(f_post, axis=0)
        assert np.allclose(rho_post, rho, atol=1e-14)

    def test_guo_forcing_momentum_balance(self):
        """Momentum change under Guo forcing matches Delta t * F exactly."""
        tau = 0.85
        dt = 1.0
        force = np.array([0.001, -0.002])
        rng = np.random.default_rng(123)
        f = rng.uniform(0.1, 1.0, size=(9, 4, 4))
        rho, u = compute_macroscopic(f, force=force, dt=dt)
        f_eq = compute_equilibrium(rho, u)
        f_post = bgk_collision(f, f_eq, tau=tau, force=force, u=u, dt=dt)

        # Pre-collision raw momentum
        j_pre = np.empty((2, 4, 4), dtype=np.float64)
        j_pre[0] = np.sum(C[:, 0, None, None] * f, axis=0)
        j_pre[1] = np.sum(C[:, 1, None, None] * f, axis=0)

        # Post-collision raw momentum
        j_post = np.empty((2, 4, 4), dtype=np.float64)
        j_post[0] = np.sum(C[:, 0, None, None] * f_post, axis=0)
        j_post[1] = np.sum(C[:, 1, None, None] * f_post, axis=0)

        # Theoretical momentum relation in Guo scheme:
        # sum f_i* c_i = sum f_i c_i + dt * F
        j_expected = j_pre + dt * force[:, None, None]
        assert np.allclose(j_post, j_expected, atol=1e-13)


class TestBoundaryAndSolidInvariants:
    """Section 10.4: Solid obstacle non-penetration and bounce-back correctness."""

    def test_solid_obstacle_non_leakage(self):
        """Populations never penetrate or reside inside solid obstacle cells."""
        nx, ny = 8, 8
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[:, 0] = True
        solid_mask[:, -1] = True
        solid_mask[3:5, 3:5] = True  # Internal block obstacle

        solver = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.001, solid_mask=solid_mask
        )
        solver.initialize_dam_break(column_width=2, column_height=4, add_walls=False)

        for _ in range(15):
            solver.step()
            assert np.all(solver.mass[solid_mask] == 0.0)
            assert np.all(solver.fill_level[solid_mask] == 0.0)
            assert np.all(solver.f[:, solid_mask] == 0.0)
            assert np.all(solver.u[:, solid_mask] == 0.0)

    def test_quantum_bounce_back_parity(self):
        """Applying bounce-back reflection twice yields the identity transformation."""
        bb_op = QuantumD2Q9BounceBackOperator(encoding=ABEncodingType.AB)
        circ = bb_op.circuit
        # Double evolution
        for v in range(9):
            psi0 = Statevector.from_int(v, 16)
            psi1 = psi0.evolve(circ)
            psi2 = psi1.evolve(circ)
            probs = psi2.probabilities()
            assert int(np.argmax(probs)) == v
            assert abs(probs[v] - 1.0) < 1e-12


class TestFreeSurfaceInvariantsValidation:
    """Section 10.5: Fill fraction bounds, mass transfer anti-symmetry, and topological consistency."""

    def test_fill_level_strict_bounds(self):
        """Fill fraction remains strictly in [0, 1] during column collapse."""
        nx, ny = 32, 16
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, gravity=0.001)
        solver.initialize_dam_break(column_width=8, column_height=12)

        for _ in range(30):
            solver.step()
            assert np.all(solver.fill_level >= 0.0)
            assert np.all(solver.fill_level <= 1.0)

    def test_topological_completeness_invariant(self):
        """No LIQUID cell directly touches a GAS cell at any time step."""
        nx, ny = 32, 16
        solver = VOFHydroSolver(nx=nx, ny=ny, tau=0.8, gravity=0.001)
        solver.initialize_dam_break(column_width=8, column_height=12)

        for _ in range(30):
            solver.step()
            is_liquid = solver.cell_types == CellType.LIQUID
            is_gas = solver.cell_types == CellType.GAS
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    gas_nbr = np.roll(
                        np.roll(is_gas, shift=-dy, axis=0), shift=-dx, axis=1
                    )
                    assert not np.any(is_liquid & gas_nbr), "Liquid-Gas direct contact detected!"


class TestEndToEndDamBreakValidation:
    """Section 10.6: End-to-end dam-break collapse comparisons."""

    def test_classical_vs_hybrid_dam_break_match(self):
        """Classical solver and hybrid quantum stepper agree to numerical precision."""
        nx, ny = 16, 8
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[0, :] = True
        solid_mask[:, 0] = True
        solid_mask[:, -1] = True

        s_class = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.001, solid_mask=solid_mask
        )
        s_class.initialize_dam_break(column_width=4, column_height=6, add_walls=False)

        s_hybr = VOFHydroSolver(
            nx=nx, ny=ny, tau=0.8, gravity=0.001, solid_mask=solid_mask
        )
        s_hybr.initialize_dam_break(column_width=4, column_height=6, add_walls=False)

        stepper = HybridVOFStep(s_hybr, use_quantum_streaming=True)

        for _ in range(10):
            s_class.step()
            stepper.step()

        # Check identical fields
        diff_mass = float(np.max(np.abs(s_class.mass - s_hybr.mass)))
        diff_fill = float(np.max(np.abs(s_class.fill_level - s_hybr.fill_level)))
        diff_rho = float(np.max(np.abs(s_class.rho - s_hybr.rho)))
        diff_u = float(np.max(np.abs(s_class.u - s_hybr.u)))

        assert diff_mass < 1e-11
        assert diff_fill < 1e-11
        assert diff_rho < 1e-11
        assert diff_u < 1e-11
        assert s_hybr.mass_error() < 1e-12
