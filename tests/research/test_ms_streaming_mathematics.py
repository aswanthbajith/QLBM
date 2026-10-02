"""Deterministic validation tests for MSQLBM streaming mathematics.

Validates Phase 2 of the QLBM Research Program:
- Exact QFT-based position translation x' = (x + vx) mod Nx.
- All 4 directional quadrants (+x,+y), (+x,-y), (-x,+y), (-x,-y).
- Both speed tiers: slow (mag=0, delta=1) and fast (mag=1, delta=3).
- Mixed speed tiers across dimensions (vx != vy).
- Modular boundary wrapping around grid boundaries (0 and N-1).
- Multi-step streaming consistency.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlbm.components.ms import MSQLBM, MSStreamingOperator
from qlbm.lattice import MSLattice
from test.e2e.utils import (
    decode_state,
    get_nonzero_amplitudes,
    make_ms_qubit_layout,
    prepare_ms_particle,
    run_statevector,
)


@pytest.fixture
def lattice_8x8() -> MSLattice:
    """8x8 2D lattice, 4 velocities/dim, no obstacles (15 qubits)."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
            "geometry": [],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Single Substep Unit Increments
# ---------------------------------------------------------------------------


class TestMSSingleSubstepStreaming:
    def test_single_substep_positive(self, lattice_8x8):
        """Single substep streaming on velocity tier 0 moves +1 in both dimensions."""
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)

        qc = prepare_ms_particle(lat, (2, 3), (0, 0), (1, 1))
        stream_op = MSStreamingOperator(lat, [0])
        qc.compose(stream_op.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 3
        assert decoded["g_y"] == 4
        assert decoded["vd_x"] == 1
        assert decoded["vd_y"] == 1

    def test_single_substep_negative(self, lattice_8x8):
        """Single substep streaming on velocity tier 0 moves -1 in both dimensions."""
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)

        qc = prepare_ms_particle(lat, (2, 3), (0, 0), (0, 0))
        stream_op = MSStreamingOperator(lat, [0])
        qc.compose(stream_op.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 1
        assert decoded["g_y"] == 2
        assert decoded["vd_x"] == 0
        assert decoded["vd_y"] == 0


# ---------------------------------------------------------------------------
# Test 2: Full Macro-Step Across All Quadrants (Slow Tier: delta = 1)
# ---------------------------------------------------------------------------


class TestMSSlowMacroStepStreaming:
    @pytest.mark.parametrize(
        "v_dir, expected_pos",
        [
            ((1, 1), (3, 3)),  # (+x, +y) -> (2+1, 2+1) = (3, 3)
            ((1, 0), (3, 1)),  # (+x, -y) -> (2+1, 2-1) = (3, 1)
            ((0, 1), (1, 3)),  # (-x, +y) -> (2-1, 2+1) = (1, 3)
            ((0, 0), (1, 1)),  # (-x, -y) -> (2-1, 2-1) = (1, 1)
        ],
    )
    def test_slow_quadrants(self, lattice_8x8, v_dir, expected_pos):
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (2, 2), (0, 0), v_dir)
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        assert (decoded["g_x"], decoded["g_y"]) == expected_pos
        assert (decoded["vd_x"], decoded["vd_y"]) == v_dir
        # Ancillae must be reset
        assert decoded["a_v"] == 0


# ---------------------------------------------------------------------------
# Test 3: Full Macro-Step Across All Quadrants (Fast Tier: delta = 3)
# ---------------------------------------------------------------------------


class TestMSFastMacroStepStreaming:
    @pytest.mark.parametrize(
        "v_dir, expected_pos",
        [
            ((1, 1), (5, 5)),  # (+x, +y) -> (2+3, 2+3) = (5, 5)
            ((1, 0), (5, 7)),  # (+x, -y) -> (2+3, (2-3)%8) = (5, 7)
            ((0, 1), (7, 5)),  # (-x, +y) -> ((2-3)%8, 2+3) = (7, 5)
            ((0, 0), (7, 7)),  # (-x, -y) -> ((2-3)%8, (2-3)%8) = (7, 7)
        ],
    )
    def test_fast_quadrants(self, lattice_8x8, v_dir, expected_pos):
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (2, 2), (1, 1), v_dir)
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        assert (decoded["g_x"], decoded["g_y"]) == expected_pos
        assert (decoded["vd_x"], decoded["vd_y"]) == v_dir
        assert decoded["a_v"] == 0


# ---------------------------------------------------------------------------
# Test 4: Mixed Speed Tiers & Modular Boundary Wraparound
# ---------------------------------------------------------------------------


class TestMSMixedAndBoundaryStreaming:
    def test_mixed_speed_tiers(self, lattice_8x8):
        """x is fast (+3), y is slow (+1): (1, 1) -> (4, 2)."""
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (1, 1), (1, 0), (1, 1))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == (1 + 3) % 8  # 4
        assert decoded["g_y"] == (1 + 1) % 8  # 2

    def test_origin_wraparound_slow(self, lattice_8x8):
        """Slow particle moving (-1, -1) from (0, 0) wraps to (7, 7)."""
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (0, 0), (0, 0), (0, 0))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 7
        assert decoded["g_y"] == 7

    def test_upper_boundary_wraparound_slow(self, lattice_8x8):
        """Slow particle moving (+1, +1) from (7, 7) wraps to (0, 0)."""
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (7, 7), (0, 0), (1, 1))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 0
        assert decoded["g_y"] == 0

    def test_multi_step_trajectory(self, lattice_8x8):
        """Evolving 3 macro-timesteps: (0, 0) with slow (+1, +1) reaches (3, 3)."""
        lat = lattice_8x8
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (0, 0), (0, 0), (1, 1))
        for _ in range(3):
            qc.compose(alg.circuit.copy(), inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 3
        assert decoded["g_y"] == 3
