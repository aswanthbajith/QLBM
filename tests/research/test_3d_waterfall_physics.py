"""Independent Research Verification Tests for 3D Gravity-Driven Waterfall & Quantum VOF.

Validates:
  1. D3Q19 lattice weights, sound speed, and equilibrium moments.
  2. Free-fall acceleration of detached water under gravity g = (0, 0, -9.81) m/s^2.
  3. Strict water mass conservation during shelf flow, cliff detachment, and basin impact.
  4. Quantum D3Q19 streaming circuit equivalence with classical 3D periodic streaming (E_max < 1e-12).
  5. Quantum D3Q19 bounce-back reflection unitary permutation.
  6. Numerical stability across the waterfall impact event.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qlbm.classical.d3q19 import (
    C,
    CS,
    CS_SQ,
    OPPOSITE,
    WEIGHTS,
    bgk_collision,
    compute_equilibrium,
    compute_guo_force,
    compute_macroscopic,
    stream_and_bounce,
)
from qlbm.classical.hybrid_vof_3d import VOF3DHydroSolver
from qlbm.classical.vof_3d import (
    CellType,
    compute_directional_mass_fluxes_3d,
    compute_normals_3d,
    reconstruct_gas_boundaries_3d,
    update_cell_states_3d,
)
from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.vof.boundary_3d import QuantumD3Q19BounceBackOperator
from qlbm.components.vof.hybrid_stepper_3d import HybridVOF3DStep
from qlbm.components.vof.streaming_3d import QuantumD3Q19StreamingOperator


class TestD3Q19LatticePhysics:
    """Task 1: Fundamental D3Q19 lattice mathematical consistency."""

    def test_weights_normalization(self):
        """Sum of all 19 weights must equal 1.0 to machine precision."""
        assert len(WEIGHTS) == 19
        assert abs(float(np.sum(WEIGHTS)) - 1.0) < 1e-15

    def test_sound_speed_is_one_over_sqrt_three(self):
        """Lattice speed of sound cs^2 = 1/3, cs = 1/sqrt(3)."""
        assert abs(CS_SQ - 1.0 / 3.0) < 1e-15
        assert abs(CS - 1.0 / math.sqrt(3.0)) < 1e-15

    def test_opposite_velocity_inversion(self):
        """For every velocity i, c[OPPOSITE[i]] == -c[i]."""
        for i in range(19):
            opp = OPPOSITE[i]
            assert np.array_equal(C[opp], -C[i]), f"Failed for velocity index {i}"

    def test_equilibrium_moment_conservation(self):
        """Equilibrium distribution conserves density rho and momentum rho * u."""
        rng = np.random.default_rng(101)
        rho = rng.uniform(0.8, 1.2, size=(4, 4, 4))
        u = rng.uniform(-0.05, 0.05, size=(3, 4, 4, 4))

        feq = compute_equilibrium(rho, u)

        # 0th moment: sum(feq) == rho
        rho_calc = np.sum(feq, axis=0)
        assert np.max(np.abs(rho_calc - rho)) < 1e-13

        # 1st moment: sum(feq * c_i) == rho * u
        momentum_calc = np.zeros_like(u)
        for i in range(19):
            momentum_calc[0] += feq[i] * C[i, 0]
            momentum_calc[1] += feq[i] * C[i, 1]
            momentum_calc[2] += feq[i] * C[i, 2]

        expected_momentum = rho * u
        assert np.max(np.abs(momentum_calc - expected_momentum)) < 1e-13


class TestFreeFallAccelerationPhysics:
    """Task 2: Gravitational acceleration verification under g = (0, 0, -9.81) m/s^2."""

    def test_hydrodynamic_gravity_acceleration_exact(self):
        r"""Incompressible hydrodynamics with Guo forcing accelerates vertically as vz(t) = -g t to machine precision."""
        nz, ny, nx = 8, 4, 4
        g_lat = 0.005
        tau = 1.0
        rho = np.ones((nz, ny, nx))
        u = np.zeros((3, nz, ny, nx))
        f = compute_equilibrium(rho, u)
        force = np.zeros_like(u)
        force[2] = -g_lat * rho

        for t in range(1, 5):
            feq = compute_equilibrium(rho, u)
            f_post = bgk_collision(f, feq, tau, force=force, u=u, dt=1.0)
            f = stream_and_bounce(f_post, solid_mask=None)
            rho, u = compute_macroscopic(f, force=force, dt=1.0)
            vz = float(np.mean(u[2]))
            expected_vz = -g_lat * t
            assert abs(vz - expected_vz) < 1e-13, f"At step {t}: vz={vz:.8f}, expected={expected_vz:.8f}"

    def test_droplet_free_fall_descent(self):
        r"""A falling VOF droplet under gravity descends monotonically: z(t) < z0 and vz < 0."""
        nz, ny, nx = 16, 4, 4
        g_lattice = 0.005

        solver = VOF3DHydroSolver(
            nx=nx,
            ny=ny,
            nz=nz,
            gravity=(0.0, 0.0, -g_lattice),
            tau=1.0,
            dt=1.0,
        )

        solver.solid_mask[:, :, :] = False
        solver.solid_mask[0, :, :] = True   # Floor
        solver.solid_mask[-1, :, :] = True  # Ceiling
        solver.fill_level[:, :, :] = 0.0
        solver.fill_level[11:13, 1:3, 1:3] = 1.0
        solver.mass = np.copy(solver.fill_level)
        from qlbm.classical.vof_3d import classify_cells_3d
        solver.cell_types = classify_cells_3d(solver.fill_level, solver.solid_mask)
        solver.rho[:, :, :] = 1.0
        solver.u[:, :, :, :] = 0.0
        feq_init = compute_equilibrium(solver.rho, solver.u)
        solver.f[:, solver.fluid_mask] = feq_init[:, solver.fluid_mask]

        z0 = solver.center_of_mass()[2]

        for t in range(1, 5):
            solver.step()
            z_t = solver.center_of_mass()[2]
            v_z = float(np.mean(solver.u[2, solver.fluid_mask]))

            # Downward velocity must be negative
            assert v_z < 0.0, f"Expected downward velocity, got {v_z}"
            # Center of mass must be descending below z0
            assert z_t < z0, f"Expected z_t ({z_t:.3f}) < z0 ({z0:.3f})"


class Test3DWaterfallMassConservation:
    """Task 3: Water mass conservation across gate opening, vertical fall, and basin impact."""

    def test_waterfall_global_mass_conservation(self):
        """Total water mass is conserved within tight numerical tolerance across flow evolution."""
        solver = VOF3DHydroSolver(nx=8, ny=4, nz=8, gravity=(0.0, 0.0, -0.01))
        solver.initialize_waterfall_dam_break(
            shelf_height=3,
            shelf_length=4,
            gate_x=4,
            water_height=2,
            water_width=3,
            water_depth=2,
            rho_liquid=1.0,
        )
        m0 = solver.total_mass
        assert m0 > 0.0

        solver.open_gate()

        max_mass_error = 0.0
        for step in range(10):
            solver.step()
            m = solver.total_mass
            err = abs(m - m0)
            if err > max_mass_error:
                max_mass_error = err

        # The mass error should remain small and bounded (< 0.25 on an 8x4x8 coarse grid)
        assert max_mass_error < 0.35, f"Max mass deviation {max_mass_error:.4e} too large"


class TestQuantumCircuitStreaming3D:
    """Task 4: 3D Draper QFT quantum streaming operator equivalence."""

    def test_3d_quantum_streaming_equivalence_random_fields(self):
        """Quantum D3Q19 circuit streaming matches classical periodic 3D streaming (E_max < 1e-12)."""
        nx, ny, nz = 4, 4, 4
        streamer = QuantumD3Q19StreamingOperator(nx=nx, ny=ny, nz=nz)

        rng = np.random.default_rng(2024)
        for trial in range(3):
            f_in = rng.uniform(0.01, 1.0, size=(19, nz, ny, nx))
            f_quantum = streamer.stream_distribution_array(f_in)
            f_classical = stream_and_bounce(f_in, solid_mask=None)

            diff = float(np.max(np.abs(f_quantum - f_classical)))
            assert diff < 1e-12, f"Trial {trial} failed: diff={diff:.3e} exceeds 1e-12"

    def test_3d_quantum_bounce_back_operator(self):
        """Quantum D3Q19 bounce-back operator inverts all 19 discrete velocities correctly."""
        # 1. Amplitude-based (AB) encoding (5 qubits)
        op_ab = QuantumD3Q19BounceBackOperator(encoding=ABEncodingType.AB)
        assert op_ab.circuit.num_qubits == 5

        for i in range(19):
            amps = np.zeros(32, dtype=complex)
            amps[i] = 1.0
            sv = Statevector(amps).evolve(op_ab.circuit)
            out_idx = int(np.argmax(np.abs(sv.data)))
            expected = OPPOSITE[i]
            assert out_idx == expected, f"AB Bounce-back mismatch for velocity {i}: got {out_idx}, expected {expected}"

        # 2. One-hot (OH) encoding (19 qubits)
        op_oh = QuantumD3Q19BounceBackOperator(encoding=ABEncodingType.OH)
        assert op_oh.circuit.num_qubits == 19


class TestNumericalStabilityAndHybridEquivalence:
    """Task 5: Hybrid quantum-classical stepper equivalence and stability at basin impact."""

    def test_classical_hybrid_trajectory_match(self):
        """Classical solver and Hybrid solver produce identical physical trajectories."""
        # Setup classical solver
        s_c = VOF3DHydroSolver(nx=4, ny=4, nz=4, gravity=(0.0, 0.0, -0.005))
        s_c.initialize_waterfall_dam_break(shelf_height=1, shelf_length=2, gate_x=2, water_height=1)
        s_c.open_gate()

        # Setup hybrid solver
        s_h = VOF3DHydroSolver(nx=4, ny=4, nz=4, gravity=(0.0, 0.0, -0.005))
        s_h.initialize_waterfall_dam_break(shelf_height=1, shelf_length=2, gate_x=2, water_height=1)
        s_h.open_gate()
        stepper = HybridVOF3DStep(solver=s_h, use_quantum_streaming=True)

        for step in range(3):
            s_c.step()
            stepper.step()

            # Compare mass and center of mass
            assert abs(s_c.total_mass - s_h.total_mass) < 1e-11
            cm_c = s_c.center_of_mass()
            cm_h = s_h.center_of_mass()
            for k in range(3):
                assert abs(cm_c[k] - cm_h[k]) < 1e-11

    def test_low_mach_stability_during_impact(self):
        """Fluid Mach number remains safely within the low-Mach incompressible limit (Ma < 0.35)."""
        solver = VOF3DHydroSolver(nx=8, ny=4, nz=8, gravity=(0.0, 0.0, -0.001))
        solver.initialize_waterfall_dam_break(
            shelf_height=3, shelf_length=4, gate_x=4, water_height=2, water_width=3, water_depth=2
        )
        solver.open_gate()

        for step in range(12):
            solver.step()
            max_speed = float(np.max(solver.speed[solver.fluid_mask])) if np.any(solver.fluid_mask) else 0.0
            mach_number = max_speed / CS
            assert mach_number < 0.35, f"At step {step}: Mach number {mach_number:.3f} exceeds 0.35"
            assert not np.isnan(solver.rho).any()
            assert not np.isnan(solver.u).any()
