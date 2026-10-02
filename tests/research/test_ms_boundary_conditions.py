"""Deterministic validation tests for MSQLBM boundary conditions.

Validates Phase 3 of the QLBM Research Program:
- 2D Bounce-back reflection (complete velocity reversal on obstacle collision).
- 2D Specular reflection (normal component reversal, tangential preservation, clean ancilla reset).
- 2D Specular corner collision (bidirectional reversal and corner cleanup).
- 2D Particle streaming freely away from obstacles (no spurious reflections).
- 3D Bounce-back reflection on cuboid obstacles.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlbm.components.ms import MSQLBM
from qlbm.lattice import MSLattice
from test.e2e.utils import (
    decode_state,
    get_nonzero_amplitudes,
    make_ms_qubit_layout,
    prepare_ms_particle,
    run_statevector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lattice_2d_bounceback() -> MSLattice:
    """8x8 2D lattice with a bounceback cuboid at x=[3, 4], y=[2, 5]."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [3, 4],
                    "y": [2, 5],
                    "boundary": "bounceback",
                }
            ],
        }
    )


@pytest.fixture
def lattice_2d_specular() -> MSLattice:
    """8x8 2D lattice with a specular cuboid at x=[3, 4], y=[2, 5]."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [3, 4],
                    "y": [2, 5],
                    "boundary": "specular",
                }
            ],
        }
    )


@pytest.fixture
def lattice_2d_corner_specular() -> MSLattice:
    """8x8 2D lattice with a specular cuboid at x=[3, 4], y=[3, 4]."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [3, 4],
                    "y": [3, 4],
                    "boundary": "specular",
                }
            ],
        }
    )


@pytest.fixture
def lattice_3d_bounceback() -> MSLattice:
    """4x4x4 3D lattice with a bounceback cuboid at x=[2, 3], y=[1, 2], z=[1, 2]."""
    return MSLattice(
        {
            "lattice": {
                "dim": {"x": 4, "y": 4, "z": 4},
                "velocities": {"x": 4, "y": 4, "z": 4},
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 3],
                    "y": [1, 2],
                    "z": [1, 2],
                    "boundary": "bounceback",
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: 2D Bounce-Back Wall
# ---------------------------------------------------------------------------


class TestMS2DBounceBack:
    def test_bounceback_collision(self, lattice_2d_bounceback):
        """Particle at (2, 3) moving (+1, +1) hits wall at x=3 and reflects back to (2, 3) with (-1, -1)."""
        lat = lattice_2d_bounceback
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (2, 3), (0, 0), (1, 1))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        # Position must be preserved at (2, 3)
        assert decoded["g_x"] == 2
        assert decoded["g_y"] == 3
        # Direction must be completely reversed to (0, 0) == (-, -)
        assert decoded["vd_x"] == 0
        assert decoded["vd_y"] == 0

    def test_free_streaming_away_from_bounceback(self, lattice_2d_bounceback):
        """Particle at (1, 1) moving away (-1, +1) streams freely to (0, 2) with a_o=0."""
        lat = lattice_2d_bounceback
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (1, 1), (0, 0), (0, 1))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 0
        assert decoded["g_y"] == 2
        assert decoded["vd_x"] == 0
        assert decoded["vd_y"] == 1
        assert decoded["a_o"] == 0


# ---------------------------------------------------------------------------
# Test 2: 2D Specular Reflection Wall & Corner
# ---------------------------------------------------------------------------


class TestMS2DSpecular:
    def test_specular_wall_collision(self, lattice_2d_specular):
        """Particle at (2, 3) moving (+1, +1) hits x=3.

        Specular reflection inverts vdx (1 -> 0) while preserving vdy (1 -> 1).
        Final pos: (2, 4) with dir (0, 1) and a_o cleanly reset to 0.
        """
        lat = lattice_2d_specular
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (2, 3), (0, 0), (1, 1))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 2
        assert decoded["g_y"] == 4
        assert decoded["vd_x"] == 0  # inverted normal component
        assert decoded["vd_y"] == 1  # preserved tangential component
        assert decoded["a_o"] == 0  # cleanly reset ancilla

    def test_specular_corner_collision(self, lattice_2d_corner_specular):
        """Particle at (2, 2) moving (+1, +1) hits corner (3, 3).

        Both x and y walls trigger, inverting both vdx and vdy.
        Final pos: (2, 2) with dir (0, 0) and a_o cleanly reset to 0.
        """
        lat = lattice_2d_corner_specular
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (2, 2), (0, 0), (1, 1))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        assert decoded["g_x"] == 2
        assert decoded["g_y"] == 2
        assert decoded["vd_x"] == 0
        assert decoded["vd_y"] == 0
        assert decoded["a_o"] == 0


# ---------------------------------------------------------------------------
# Test 3: 3D Bounce-Back Reflection
# ---------------------------------------------------------------------------


class TestMS3DBounceBack:
    def test_3d_bounceback_collision(self, lattice_3d_bounceback):
        """Particle at (1, 1, 1) moving (+1, +1, +1) into cuboid at x=[2, 3].

        Reflects back to (1, 1, 1) with direction (0, 0, 0).
        """
        lat = lattice_3d_bounceback
        layout = make_ms_qubit_layout(lat)
        alg = MSQLBM(lat)

        qc = prepare_ms_particle(lat, (1, 1, 1), (0, 0, 0), (1, 1, 1))
        qc.compose(alg.circuit, inplace=True)

        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)

        pos = (decoded["g_x"], decoded["g_y"], decoded["g_z"])
        vd = (decoded["vd_x"], decoded["vd_y"], decoded["vd_z"])

        assert pos == (1, 1, 1)
        assert vd == (0, 0, 0)
