"""Diagnostic tests for MSLattice register allocation, qubit layout, and basis-state encoding.

Validates Phase 1 of the QLBM Research Program:
- Exact register counts and layout partitions for 2D and 3D configurations.
- Bit-level Little-Endian encoding within registers.
- Computational basis state preparation and statevector decoding.
- Error handling on out-of-range register queries.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path when test is invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.lattice import MSLattice
from qlbm.tools.exceptions import LatticeException
from test.e2e.utils import (
    decode_state,
    get_nonzero_amplitudes,
    make_ms_qubit_layout,
    prepare_ms_particle,
    run_statevector,
)


# ---------------------------------------------------------------------------
# Fixtures for 2D and 3D lattices
# ---------------------------------------------------------------------------


@pytest.fixture
def lattice_2d_8x8_no_obstacle() -> MSLattice:
    """8x8 2D lattice, 4 velocities/dim, no obstacles (15 qubits)."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
            "geometry": [],
        }
    )


@pytest.fixture
def lattice_2d_8x8_bounceback() -> MSLattice:
    """8x8 2D lattice, 4 velocities/dim, bounce-back obstacle (15 qubits)."""
    return MSLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [3, 4],
                    "y": [3, 4],
                    "boundary": "bounceback",
                }
            ],
        }
    )


@pytest.fixture
def lattice_2d_8x8_specular() -> MSLattice:
    """8x8 2D lattice, 4 velocities/dim, specular obstacle (16 qubits)."""
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
def lattice_3d_8x8x8_specular() -> MSLattice:
    """8x8x8 3D lattice, 4 velocities/dim, specular obstacle (25 qubits)."""
    return MSLattice(
        {
            "lattice": {
                "dim": {"x": 8, "y": 8, "z": 8},
                "velocities": {"x": 4, "y": 4, "z": 4},
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 4],
                    "y": [2, 4],
                    "z": [2, 4],
                    "boundary": "specular",
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Sizing and Register Count
# ---------------------------------------------------------------------------


class TestMSLatticeRegisterSizing:
    def test_2d_8x8_no_obstacle_counts(self, lattice_2d_8x8_no_obstacle):
        lat = lattice_2d_8x8_no_obstacle
        assert lat.num_dims == 2
        # Internal representation stores N - 1 for binary arithmetic compatibility
        assert lat.num_gridpoints == [7, 7]
        assert lat.num_velocities == [3, 3]

        # Register sizes:
        # a_v: 2 (d=2)
        # a_o: 1 (empty / bounceback default)
        # a_c: 2 (2*(d-1) = 2)
        # g_x, g_y: 3 each (ceil(log2(8)) = 3)
        # v_x, v_y: 1 each (ceil(log2(4)) - 1 = 1)
        # vd_x, vd_y: 1 each (1 per dim)
        assert lat.num_ancilla_qubits == 2 + 1 + 2  # 5
        assert lat.num_grid_qubits == 3 + 3  # 6
        assert lat.num_velocity_qubits == (1 + 1) + (1 + 1)  # 4
        assert lat.num_total_qubits == 15

        # Check circuit registers count
        reg_sizes = [(reg.name, reg.size) for reg in lat.registers]
        expected_regs = [
            ("a_v", 2),
            ("a_o", 1),
            ("a_c", 2),
            ("g_x", 3),
            ("g_y", 3),
            ("v_x", 1),
            ("v_y", 1),
            ("v_dir_x", 1),
            ("v_dir_y", 1),
        ]
        assert reg_sizes == expected_regs

    def test_2d_8x8_specular_obstacle_counts(self, lattice_2d_8x8_specular):
        lat = lattice_2d_8x8_specular
        # Specular obstacle requires d=2 obstacle ancillae
        assert lat.num_obstacle_qubits == 2
        assert lat.num_ancilla_qubits == 2 + 2 + 2  # 6
        assert lat.num_total_qubits == 16

        reg_sizes = [(reg.name, reg.size) for reg in lat.registers]
        assert reg_sizes[1] == ("a_o", 2)

    def test_3d_8x8x8_specular_obstacle_counts(self, lattice_3d_8x8x8_specular):
        lat = lattice_3d_8x8x8_specular
        assert lat.num_dims == 3
        # Ancillae: a_v=3, a_o=3, a_c=2*(3-1)=4 -> total 10 ancillae
        assert lat.num_ancilla_qubits == 3 + 3 + 4  # 10
        # Grid: 3+3+3 = 9
        assert lat.num_grid_qubits == 9
        # Velocities: 3 mag + 3 dir = 6
        assert lat.num_velocity_qubits == 6
        assert lat.num_total_qubits == 25


# ---------------------------------------------------------------------------
# Test 2: Qubit Index Partitioning (No overlaps, no gaps)
# ---------------------------------------------------------------------------


class TestMSLatticeIndexing:
    def test_2d_8x8_index_partition(self, lattice_2d_8x8_no_obstacle):
        lat = lattice_2d_8x8_no_obstacle

        a_v = lat.ancillae_velocity_index()
        a_o = lat.ancillae_obstacle_index()
        a_c = lat.ancillae_comparator_index()
        g_x = lat.grid_index(0)
        g_y = lat.grid_index(1)
        v_x = lat.velocity_index(0)
        v_y = lat.velocity_index(1)
        vd_x = lat.velocity_dir_index(0)
        vd_y = lat.velocity_dir_index(1)

        assert a_v == [0, 1]
        assert a_o == [2]
        assert a_c == [3, 4]
        assert g_x == [5, 6, 7]
        assert g_y == [8, 9, 10]
        assert v_x == [11]
        assert v_y == [12]
        assert vd_x == [13]
        assert vd_y == [14]

        # Verify contiguous partition
        all_indices = a_v + a_o + a_c + g_x + g_y + v_x + v_y + vd_x + vd_y
        assert all_indices == list(range(lat.num_total_qubits))

    def test_indexing_exceptions(self, lattice_2d_8x8_no_obstacle):
        lat = lattice_2d_8x8_no_obstacle
        with pytest.raises(LatticeException):
            lat.grid_index(2)  # dimension 2 does not exist in 2D
        with pytest.raises(LatticeException):
            lat.velocity_index(2)
        with pytest.raises(LatticeException):
            lat.velocity_dir_index(2)
        with pytest.raises(LatticeException):
            lat.ancillae_velocity_index(2)
        with pytest.raises(LatticeException):
            lat.ancillae_comparator_index(1)  # max index in 2D is 0


# ---------------------------------------------------------------------------
# Test 3: Computational Basis State Preparation & Statevector Decoding
# ---------------------------------------------------------------------------


class TestMSBasisStateEncoding:
    def test_encode_and_decode_single_particles(self, lattice_2d_8x8_no_obstacle):
        """Test multiple distinct particle states and verify statevector integer decoding."""
        lat = lattice_2d_8x8_no_obstacle
        layout = make_ms_qubit_layout(lat)

        test_cases = [
            # (pos_x, pos_y), (v_mag_x, v_mag_y), (v_dir_x, v_dir_y)
            ((0, 0), (0, 0), (1, 1)),  # origin, slow, (+, +)
            ((1, 2), (0, 1), (1, 0)),  # (1,2), x slow, y fast, (+, -)
            ((5, 3), (1, 0), (0, 1)),  # (5,3), x fast, y slow, (-, +)
            ((7, 7), (1, 1), (0, 0)),  # max corner, fast, (-, -)
            ((4, 6), (0, 0), (0, 1)),  # (4,6), slow, (-, +)
        ]

        for grid_pos, v_mag, v_dir in test_cases:
            qc = prepare_ms_particle(lat, grid_pos, v_mag, v_dir)
            sv = run_statevector(qc)
            amps = get_nonzero_amplitudes(sv)

            # Pure basis state: exactly 1 non-zero amplitude of unit magnitude
            assert len(amps) == 1
            basis_idx = list(amps.keys())[0]
            assert np.isclose(np.abs(amps[basis_idx]), 1.0)

            # Decode the integer basis index
            decoded = decode_state(basis_idx, layout)
            assert decoded["g_x"] == grid_pos[0]
            assert decoded["g_y"] == grid_pos[1]
            assert decoded["v_x"] == v_mag[0]
            assert decoded["v_y"] == v_mag[1]
            assert decoded["vd_x"] == v_dir[0]
            assert decoded["vd_y"] == v_dir[1]
            # Ancillae must be cleanly in |0>
            assert decoded["a_v"] == 0
            assert decoded["a_o"] == 0
            assert decoded["a_c"] == 0


# ---------------------------------------------------------------------------
# Test 4: Bit-Level Little-Endian Verification
# ---------------------------------------------------------------------------


class TestMSEndianConvention:
    def test_grid_register_bit_endianness(self, lattice_2d_8x8_no_obstacle):
        """Verify that within g_x: qubit 0 is LSB (2^0) and qubit 2 is MSB (2^2)."""
        lat = lattice_2d_8x8_no_obstacle
        g_x_qubits = lat.grid_index(0)  # [5, 6, 7]

        # For x = 5 = 101_2:
        # bit 0 = 1 -> qubit 5
        # bit 1 = 0 -> qubit 6
        # bit 2 = 1 -> qubit 7
        qc = prepare_ms_particle(lat, (5, 0), (0, 0), (0, 0))
        sv = run_statevector(qc)
        amps = get_nonzero_amplitudes(sv)
        basis_idx = list(amps.keys())[0]

        # Verify bits in basis_idx directly
        bit_5 = (basis_idx >> g_x_qubits[0]) & 1
        bit_6 = (basis_idx >> g_x_qubits[1]) & 1
        bit_7 = (basis_idx >> g_x_qubits[2]) & 1

        assert bit_5 == 1, "LSB (2^0) of g_x must be 1"
        assert bit_6 == 0, "Middle bit (2^1) of g_x must be 0"
        assert bit_7 == 1, "MSB (2^2) of g_x must be 1"

        # Check total basis index value formula
        # basis_idx = 1 * 2^5 + 1 * 2^7 = 32 + 128 = 160
        assert basis_idx == (1 << 5) | (1 << 7)
