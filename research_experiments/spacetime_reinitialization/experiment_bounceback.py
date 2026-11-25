from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.lattice import SpaceTimeLattice
from qlbm.components.spacetime.initial import (
    PointWiseSpaceTimeInitialConditions,
)
from qlbm.components.spacetime.spacetime import SpaceTimeQLBM
from qlbm.components.spacetime.measurement import (
    SpaceTimeGridVelocityMeasurement,
)
from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.reinitialize import SpaceTimeReinitializer


# ============================================================
# 1. CREATE D2Q4 SPACE-TIME LATTICE
# ============================================================

# 4 x 4 domain.
#
# Bounceback obstacle:
#
#             y=3
#              ↑
#              │
#          [OBSTACLE]
#              │
#          (1,2) +y
#
# The +y population should hit the obstacle and bounce back.

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
# 2. CREATE QLBM ALGORITHM
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
# 5. AER STATEVECTOR BACKEND
# ============================================================

backend = AerSimulator(
    method="statevector"
)


# ============================================================
# 6. HELPER FUNCTION
# ============================================================

def execute_step(grid_data, step_number):

    print("\n" + "=" * 80)
    print(f"BOUNCEBACK STEP {step_number}")
    print("=" * 80)

    print("\nInput state:")
    print(" ", grid_data)

    # --------------------------------------------------------
    # Initial condition
    # --------------------------------------------------------

    initial = PointWiseSpaceTimeInitialConditions(
        lattice=lattice,
        grid_data=grid_data,
    )

    # --------------------------------------------------------
    # Combine initial condition + QLBM
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

    print("\nCircuit:")
    print(
        "  qubits              :",
        main_circuit.num_qubits,
    )

    print(
        "  raw instructions    :",
        len(main_circuit.data),
    )

    # --------------------------------------------------------
    # Transpile for Aer
    # --------------------------------------------------------

    simulation_circuit = transpile(
        main_circuit,
        backend=backend,
        optimization_level=0,
    )

    print(
        "  transpiled          :",
        len(simulation_circuit.data),
    )

    # --------------------------------------------------------
    # Save statevector
    # --------------------------------------------------------

    simulation_circuit.save_statevector(
        label="step"
    )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    state_result = backend.run(
        simulation_circuit
    ).result()

    statevector = state_result.data(0)["step"]

    print(
        "  statevector dimension:",
        len(statevector),
    )

    # --------------------------------------------------------
    # Measurement circuit
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

    # --------------------------------------------------------
    # Transpile measurement
    # --------------------------------------------------------

    measurement_circuit = transpile(
        measurement_circuit,
        backend=backend,
        optimization_level=0,
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    counts_result = backend.run(
        measurement_circuit,
        shots=1024,
    ).result()

    counts = counts_result.get_counts()

    # --------------------------------------------------------
    # Raw counts
    # --------------------------------------------------------

    print("\nRaw measurement states:")

    for bitstring, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"  {bitstring}: {count}"
        )

    print(
        "\nNumber of raw states:",
        len(counts),
    )

    # --------------------------------------------------------
    # Decode with repository reinitializer
    # --------------------------------------------------------

    decoded = reinitializer.counts_to_velocity_pairs(
        counts
    )

    print("\nDecoded occupied states:")

    for item in decoded:
        print(
            " ",
            item
        )

    print(
        "\nNumber of decoded states:",
        len(decoded),
    )

    return decoded


# ============================================================
# 7. INITIAL STATE
# ============================================================

# Start immediately below the obstacle.
#
# Position:
#     (1,2)
#
# Velocity:
#     +y
#
# D2Q4:
#     (False, True, False, False)

initial_state = [
    (
        (1, 2),
        (False, True, False, False),
    )
]


# ============================================================
# 8. EXECUTE BOUNCEBACK STEP
# ============================================================

output_state = execute_step(
    initial_state,
    0,
)


# ============================================================
# 9. FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("BOUNCEBACK VALIDATION RESULT")
print("=" * 80)

print("\nObstacle:")
print("  position : (1,3)")
print("  boundary : bounceback")

print("\nInitial state:")
print(
    " ",
    initial_state
)

print("\nDecoded state after QLBM:")
print(
    " ",
    output_state
)

print("\nExpected physical behavior:")
print("  (1,2) +y")
print("      ↓")
print("  bounceback at (1,3)")
print("      ↓")
print("  (1,2) -y")

print("\n" + "=" * 80)

