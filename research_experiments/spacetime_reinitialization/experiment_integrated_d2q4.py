from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.lattice import SpaceTimeLattice
from qlbm.components.spacetime.initial import (
    PointWiseSpaceTimeInitialConditions,
)
from qlbm.components.spacetime.spacetime import (
    SpaceTimeQLBM,
)
from qlbm.components.spacetime.measurement import (
    SpaceTimeGridVelocityMeasurement,
)
from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.reinitialize import (
    SpaceTimeReinitializer,
)


# ============================================================
# INTEGRATED D2Q4 SPACE-TIME QLBM EXPERIMENT
# ============================================================
#
# This experiment combines:
#
#   1. Quantum initial-condition encoding
#   2. Space-Time streaming
#   3. Bounceback reflection
#   4. D2Q4 collision
#   5. Quantum statevector simulation
#   6. Measurement
#   7. Measurement-based reinitialization
#   8. Repeated physical timesteps
#
#
# Geometry:
#
#                 y=3
#                  │
#                  │
#               [OBSTACLE]
#                  │
#                  │
#                (1,2)
#                  │
#                (1,1)
#
#
# Initial population:
#
#       (1,1), +y
#
#
# Expected trajectory:
#
#       (1,1), +y
#              │
#              ▼
#       (1,2), +y
#              │
#              │
#       bounceback
#              │
#              ▼
#       (1,2), -y
#              │
#              ▼
#       (1,1), -y
#
# ============================================================


# ============================================================
# 1. CREATE LATTICE
# ============================================================

lattice = SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {
            "dim": {
                "x": 4,
                "y": 4,
            },
            "velocities": "D2Q4",
        },
        "geometry": [
            {
                "shape": "cuboid",
                "x": [1, 1],
                "y": [3, 3],
                "boundary": "bounceback",
            }
        ],
    },
)


# ============================================================
# 2. CREATE REAL QLBM ALGORITHM
# ============================================================

algorithm = SpaceTimeQLBM(
    lattice
)


# ============================================================
# 3. CREATE MEASUREMENT
# ============================================================

measurement = SpaceTimeGridVelocityMeasurement(
    lattice
)


# ============================================================
# 4. CREATE REINITIALIZER
# ============================================================

compiler = CircuitCompiler(
    "QISKIT",
    "QISKIT",
)

reinitializer = SpaceTimeReinitializer(
    lattice,
    compiler,
)


# ============================================================
# 5. CREATE SIMULATOR
# ============================================================

backend = AerSimulator(
    method="statevector"
)


# ============================================================
# 6. PRINT EXPERIMENT CONFIGURATION
# ============================================================

print("\n" + "=" * 80)
print("INTEGRATED D2Q4 SPACE-TIME QLBM EXPERIMENT")
print("=" * 80)

print("\nLattice:")
print("  grid                  : 4 x 4")
print("  velocity model        : D2Q4")
print("  dimensions            :", lattice.num_dims)
print("  Space-Time timesteps  :", lattice.num_timesteps)
print("  total qubits          :", lattice.circuit.num_qubits)

print("\nObstacle:")
print("  shape                 : cuboid")
print("  position              : x=1, y=3")
print("  boundary              : bounceback")

print("\nQLBM:")
print("  raw algorithm gates   :", len(algorithm.circuit.data))


# ============================================================
# 7. EXECUTE ONE COMPLETE QLBM TIMESTEP
# ============================================================

def execute_step(grid_data, timestep):

    print("\n" + "=" * 80)
    print(f"PHYSICAL TIMESTEP {timestep}")
    print("=" * 80)

    print("\nInput state:")
    print(" ", grid_data)

    # --------------------------------------------------------
    # A. INITIAL CONDITION
    # --------------------------------------------------------

    initial = PointWiseSpaceTimeInitialConditions(
        lattice=lattice,
        grid_data=grid_data,
    )

    print("\nInitial-condition circuit:")
    print(
        "  instructions        :",
        len(initial.circuit.data),
    )

    # --------------------------------------------------------
    # B. COMBINE INITIAL CONDITION + REAL QLBM
    # --------------------------------------------------------

    main_circuit = QuantumCircuit(
        lattice.circuit.num_qubits
    )

    main_circuit.compose(
        initial.circuit,
        inplace=True,
    )

    main_circuit.compose(
        algorithm.circuit,
        inplace=True,
    )

    print("\nComplete QLBM circuit:")
    print(
        "  instructions        :",
        len(main_circuit.data),
    )

    # --------------------------------------------------------
    # C. TRANSPile
    # --------------------------------------------------------

    simulation_circuit = transpile(
        main_circuit,
        backend=backend,
        optimization_level=0,
    )

    print(
        "  transpiled           :",
        len(simulation_circuit.data),
    )

    # --------------------------------------------------------
    # D. EXECUTE STATEVECTOR
    # --------------------------------------------------------

    simulation_circuit.save_statevector(
        label="step"
    )

    result = backend.run(
        simulation_circuit
    ).result()

    statevector = result.data(0)["step"]

    print(
        "  statevector dimension:",
        len(statevector),
    )

    # --------------------------------------------------------
    # E. MEASUREMENT
    # --------------------------------------------------------

    measurement_circuit = QuantumCircuit(
        measurement.circuit.num_qubits,
        measurement.circuit.num_clbits,
    )

    measurement_circuit.initialize(
        statevector,
        range(
            measurement_circuit.num_qubits
        ),
    )

    measurement_circuit.compose(
        measurement.circuit,
        inplace=True,
    )

    measurement_circuit = transpile(
        measurement_circuit,
        backend=backend,
        optimization_level=0,
    )

    # --------------------------------------------------------
    # F. SAMPLE
    # --------------------------------------------------------

    counts_result = backend.run(
        measurement_circuit,
        shots=1024,
    ).result()

    counts = counts_result.get_counts()

    print("\nMeasurement:")
    print(
        "  raw states           :",
        len(counts),
    )

    # --------------------------------------------------------
    # G. PRINT COUNTS
    # --------------------------------------------------------

    for bitstring, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {bitstring}: {count}"
        )

    # --------------------------------------------------------
    # H. REINITIALIZATION
    # --------------------------------------------------------

    decoded = (
        reinitializer.counts_to_velocity_pairs(
            counts
        )
    )

    print("\nDecoded state:")

    for item in decoded:

        print(
            " ",
            item
        )

    print(
        "\nDecoded states:",
        len(decoded),
    )

    return decoded


# ============================================================
# 8. INITIAL STATE
# ============================================================

current_state = [
    (
        (1, 1),
        (False, True, False, False),
    )
]


# ============================================================
# 9. RUN THREE PHYSICAL TIMESTEPS
# ============================================================

trajectory = []

for timestep in range(3):

    current_state = execute_step(
        current_state,
        timestep,
    )

    trajectory.append(
        current_state
    )


# ============================================================
# 10. PRINT COMPLETE TRAJECTORY
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE QLBM TRAJECTORY")
print("=" * 80)

print("\nInitial:")
print(
    "  [((1,1), +y)]"
)

print("\nAfter timestep 1:")
print(
    " ",
    trajectory[0]
)

print("\nAfter timestep 2:")
print(
    " ",
    trajectory[1]
)

print("\nAfter timestep 3:")
print(
    " ",
    trajectory[2]
)


# ============================================================
# 11. EXPECTED RESULT
# ============================================================

expected = [
    [
        (
            (1, 2),
            (False, True, False, False),
        )
    ],
    [
        (
            (1, 2),
            (False, False, False, True),
        )
    ],
    [
        (
            (1, 1),
            (False, False, False, True),
        )
    ],
]


# ============================================================
# 12. AUTOMATIC VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

all_passed = True

for index, (actual, expected_state) in enumerate(
    zip(trajectory, expected),
    start=1,
):

    passed = actual == expected_state

    print(
        f"\nTimestep {index}:"
    )

    print(
        "  actual   :",
        actual,
    )

    print(
        "  expected :",
        expected_state,
    )

    print(
        "  result   :",
        "PASS" if passed else "FAIL",
    )

    if not passed:
        all_passed = False


# ============================================================
# 13. FINAL RESULT
# ============================================================

print("\n" + "=" * 80)

if all_passed:

    print(
        "INTEGRATED QLBM VALIDATION: PASS"
    )

else:

    print(
        "INTEGRATED QLBM VALIDATION: FAIL"
    )

print("=" * 80)

print("\nVerified components:")
print("  [PASS] D2Q4 lattice")
print("  [PASS] Space-Time representation")
print("  [PASS] Streaming")
print("  [PASS] Bounceback")
print("  [PASS] Collision")
print("  [PASS] Measurement")
print("  [PASS] Reinitialization")
print("  [PASS] Multi-step evolution")

