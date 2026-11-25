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
        "geometry": [],
    },
)


# ============================================================
# 2. CREATE QLBM COMPONENTS
# ============================================================

algorithm = SpaceTimeQLBM(
    lattice
)

measurement = SpaceTimeGridVelocityMeasurement(
    lattice
)


# ============================================================
# 3. CREATE REINITIALIZER
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
# 4. AER BACKEND
# ============================================================

backend = AerSimulator(
    method="statevector"
)


# ============================================================
# 5. HELPER FUNCTION
# ============================================================

def execute_one_step(grid_data, step_number):

    print("\n" + "=" * 80)
    print(f"EXECUTING STEP {step_number}")
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

    print("\nRaw circuit:")
    print(
        "  instructions :",
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
        "  transpiled   :",
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
        "  statevector dimension :",
        len(statevector),
    )

    # --------------------------------------------------------
    # Create measurement circuit
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
    # Transpile measurement circuit
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
    # Print raw counts
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
    # Reinitialize using repository logic
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
# 6. STEP 0
# ============================================================

step_0_input = [
    (
        (1, 1),
        (False, True, False, False),
    )
]

step_0_output = execute_one_step(
    step_0_input,
    0,
)


# ============================================================
# 7. USE STEP 0 OUTPUT AS STEP 1 INPUT
# ============================================================

step_1_input = step_0_output


# ============================================================
# 8. STEP 1
# ============================================================

step_1_output = execute_one_step(
    step_1_input,
    1,
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TWO-STEP REINITIALIZATION EXPERIMENT")
print("=" * 80)

print("\nStep 0 input:")
print(
    " ",
    step_0_input
)

print("\nStep 0 decoded output:")
print(
    " ",
    step_0_output
)

print("\nStep 1 input:")
print(
    " ",
    step_1_input
)

print("\nStep 1 decoded output:")
print(
    " ",
    step_1_output
)

print("\n" + "=" * 80)
