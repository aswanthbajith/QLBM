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


# ============================================================
# 1. CREATE SPACE-TIME D2Q4 LATTICE
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
# 2. DEFINE INITIAL CONDITION
# ============================================================

# One velocity population is initially activated at (1, 1).
#
# D2Q4 velocity ordering:
#
#   v0 = +x
#   v1 = +y
#   v2 = -x
#   v3 = -y
#
# Therefore:
#
# (False, True, False, False)
#
# means that v1 (+y) is initially occupied.

grid_data = [
    (
        (1, 1),
        (False, True, False, False),
    )
]


initial = PointWiseSpaceTimeInitialConditions(
    lattice=lattice,
    grid_data=grid_data,
)


# ============================================================
# 3. CREATE ONE-STEP SPACE-TIME QLBM ALGORITHM
# ============================================================

algorithm = SpaceTimeQLBM(
    lattice
)


# ============================================================
# 4. CREATE MEASUREMENT OPERATOR
# ============================================================

measurement = SpaceTimeGridVelocityMeasurement(
    lattice
)


# ============================================================
# 5. COMBINE INITIAL CONDITION + QLBM ALGORITHM
# ============================================================

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


# ============================================================
# 6. PRINT BASIC CIRCUIT INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("ONE-STEP SPACE-TIME D2Q4 QLBM EXPERIMENT")
print("=" * 80)

print("\nLattice:")
print("  user grid             : 4 x 4")
print("  internal max          :", lattice.num_gridpoints)
print("  dimensions            :", lattice.num_dims)
print("  timesteps             :", lattice.num_timesteps)
print("  total qubits          :", lattice.circuit.num_qubits)

print("\nInitial condition:")
print("  grid data             :", grid_data)

print("\nInitial-condition circuit:")
print("  qubits                :", initial.circuit.num_qubits)
print("  instructions          :", len(initial.circuit.data))

print("\nQLBM algorithm:")
print("  qubits                :", algorithm.circuit.num_qubits)
print("  instructions          :", len(algorithm.circuit.data))

print("\nCombined circuit:")
print("  qubits                :", main_circuit.num_qubits)
print("  instructions          :", len(main_circuit.data))


# ============================================================
# 7. CREATE AER STATEVECTOR SIMULATOR
# ============================================================

backend = AerSimulator(
    method="statevector"
)


# ============================================================
# 8. TRANSPile QLBM CIRCUIT FOR AER
# ============================================================

# SpaceTimeD2Q4CollisionOperator contains Qiskit's high-level
# MCMT instruction.
#
# Aer cannot execute the raw "mcmt" instruction directly in
# this configuration.
#
# Therefore, transpile the circuit into Aer-supported gates.

simulation_circuit = transpile(
    main_circuit,
    backend=backend,
    optimization_level=0,
)


print("\nSimulation circuit:")
print(
    "  original instructions   :",
    len(main_circuit.data),
)

print(
    "  transpiled instructions :",
    len(simulation_circuit.data),
)

print(
    "  qubits                  :",
    simulation_circuit.num_qubits,
)


# ============================================================
# 9. SAVE FINAL STATEVECTOR
# ============================================================

simulation_circuit.save_statevector(
    label="step"
)


# ============================================================
# 10. EXECUTE QLBM CIRCUIT
# ============================================================

state_result = backend.run(
    simulation_circuit
).result()


# Retrieve the statevector generated after:
#
# initial condition
#        +
# QLBM streaming
#        +
# reflection
#        +
# collision
#
statevector = state_result.data(0)["step"]


print("\nStatevector:")
print("  dimension             :", len(statevector))
print(
    "  expected dimension    :",
    2 ** lattice.circuit.num_qubits,
)


# ============================================================
# 11. CREATE MEASUREMENT CIRCUIT
# ============================================================

measurement_circuit = QuantumCircuit(
    measurement.circuit.num_qubits,
    measurement.circuit.num_clbits,
)


# Load the QLBM statevector into the measurement circuit.

measurement_circuit.initialize(
    statevector,
    range(measurement_circuit.num_qubits),
)


# Add the repository's measurement circuit.

measurement_circuit.compose(
    measurement.circuit,
    inplace=True,
)


print("\nMeasurement circuit:")
print(
    "  qubits                :",
    measurement_circuit.num_qubits,
)

print(
    "  classical bits        :",
    measurement_circuit.num_clbits,
)


# ============================================================
# 12. TRANSPile MEASUREMENT CIRCUIT
# ============================================================

measurement_circuit = transpile(
    measurement_circuit,
    backend=backend,
    optimization_level=0,
)


print(
    "  transpiled instructions :",
    len(measurement_circuit.data),
)


# ============================================================
# 13. SAMPLE THE QUANTUM STATE
# ============================================================

counts_result = backend.run(
    measurement_circuit,
    shots=1024,
).result()


counts = counts_result.get_counts()


# ============================================================
# 14. PRINT MEASUREMENT INFORMATION
# ============================================================

print("\nMeasurement:")
print(
    "  measured qubits       :",
    lattice.grid_index() + lattice.velocity_index(0),
)

print(
    "  classical bits        :",
    measurement.circuit.num_clbits,
)


# ============================================================
# 15. PRINT RAW COUNTS
# ============================================================

print("\nCounts:")

for bitstring, count in sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True,
):
    print(
        f"  {bitstring}: {count}"
    )


# ============================================================
# 16. VERIFY SHOT COUNT
# ============================================================

print(
    "\nTotal shots:",
    sum(counts.values()),
)


# ============================================================
# 17. BASIC PROBABILITY ANALYSIS
# ============================================================

print("\nProbabilities:")

total_shots = sum(counts.values())

for bitstring, count in sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True,
):
    probability = count / total_shots

    print(
        f"  {bitstring}: "
        f"{probability:.6f}"
    )


# ============================================================
# 18. EXPERIMENT COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("ONE-STEP EXPERIMENT COMPLETE")
print("=" * 80)
