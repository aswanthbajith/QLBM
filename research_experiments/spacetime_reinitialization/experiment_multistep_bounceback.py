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
# 1. LATTICE
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
# 2. QLBM COMPONENTS
# ============================================================

algorithm = SpaceTimeQLBM(lattice)

measurement = SpaceTimeGridVelocityMeasurement(
    lattice
)

compiler = CircuitCompiler(
    "QISKIT",
    "QISKIT",
)

reinitializer = SpaceTimeReinitializer(
    lattice,
    compiler,
)

backend = AerSimulator(
    method="statevector"
)


# ============================================================
# 3. EXECUTE ONE PHYSICAL TIMESTEP
# ============================================================

def execute_step(grid_data, step):

    print("\n" + "=" * 80)
    print(f"PHYSICAL TIMESTEP {step}")
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
    # QLBM circuit
    # --------------------------------------------------------

    circuit = QuantumCircuit(
        lattice.circuit.num_qubits
    )

    circuit.compose(
        initial.circuit,
        inplace=True,
    )

    circuit.compose(
        algorithm.circuit,
        inplace=True,
    )

    print("\nCircuit:")
    print(
        "  raw instructions       :",
        len(circuit.data),
    )

    # --------------------------------------------------------
    # Transpile
    # --------------------------------------------------------

    simulation_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=0,
    )

    print(
        "  transpiled instructions:",
        len(simulation_circuit.data),
    )

    # --------------------------------------------------------
    # Statevector
    # --------------------------------------------------------

    simulation_circuit.save_statevector(
        label="step"
    )

    result = backend.run(
        simulation_circuit
    ).result()

    statevector = result.data(0)["step"]

    # --------------------------------------------------------
    # Measurement
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

    counts_result = backend.run(
        measurement_circuit,
        shots=1024,
    ).result()

    counts = counts_result.get_counts()

    # --------------------------------------------------------
    # Raw measurement summary
    # --------------------------------------------------------

    print("\nRaw measurement states:")
    print(
        "  number of states:",
        len(counts),
    )

    for bitstring, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"  {bitstring}: {count}"
        )

    # --------------------------------------------------------
    # Repository decoder
    # --------------------------------------------------------

    decoded = reinitializer.counts_to_velocity_pairs(
        counts
    )

    print("\nDecoded occupied state:")

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
# 4. INITIAL STATE
# ============================================================

current_state = [
    (
        (1, 1),
        (False, True, False, False),
    )
]


# ============================================================
# 5. RUN THREE PHYSICAL TIMESTEPS
# ============================================================

results = []

for step in range(3):

    current_state = execute_step(
        current_state,
        step,
    )

    results.append(current_state)


# ============================================================
# 6. FINAL TRAJECTORY
# ============================================================

print("\n" + "=" * 80)
print("MULTI-STEP BOUNCEBACK TRAJECTORY")
print("=" * 80)

print("\nInitial state:")
print(
    " ",
    [
        (
            (1, 1),
            (False, True, False, False),
        )
    ]
)

print("\nAfter timestep 1:")
print(
    " ",
    results[0]
)

print("\nAfter timestep 2:")
print(
    " ",
    results[1]
)

print("\nAfter timestep 3:")
print(
    " ",
    results[2]
)

print("\nExpected trajectory:")
print("  (1,1), +y")
print("      ↓")
print("  (1,2), +y")
print("      ↓")
print("  bounceback")
print("      ↓")
print("  (1,2), -y")
print("      ↓")
print("  (1,1), -y")

print("\n" + "=" * 80)
