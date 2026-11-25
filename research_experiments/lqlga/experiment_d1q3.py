from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.lattice import LQLGALattice
from qlbm.components.lqlga import (
    LQLGA,
    LQGLAInitialConditions,
    LQLGAGridVelocityMeasurement,
)

print("=== LQLGA D1Q3 EXPERIMENT ===")

lattice = LQLGALattice(
    {
        "lattice": {
            "dim": {"x": 8},
            "velocities": "D1Q3",
        },
        "geometry": [],
    }
)

print("\n--- LATTICE ---")
print("Dimensions:", lattice.num_dims)
print("Grid points:", [n + 1 for n in lattice.num_gridpoints])
print("Velocities per point:", lattice.num_velocities_per_point)
print("Base qubits:", lattice.num_base_qubits)
print("Total qubits:", lattice.num_total_qubits)

print("\n--- VELOCITY REGISTERS ---")
for gp in range(8):
    print(
        gp,
        [
            lattice.velocity_index_tuple((gp,), v)
            for v in range(lattice.num_velocities_per_point)
        ],
    )

initial = LQGLAInitialConditions(
    lattice,
    [
        ((3,), (False, True, False)),
    ],
)

print("\n--- INITIAL CONDITIONS ---")
print("Qubits:", initial.circuit.num_qubits)
print("Instructions:", len(initial.circuit.data))

algorithm = LQLGA(lattice)

print("\n--- LQLGA ALGORITHM ---")
print("Qubits:", algorithm.circuit.num_qubits)
print("Instructions:", len(algorithm.circuit.data))

measurement = LQLGAGridVelocityMeasurement(lattice)

print("\n--- MEASUREMENT ---")
print("Qubits:", measurement.circuit.num_qubits)
print("Classical bits:", measurement.circuit.num_clbits)

# Build complete circuit.
circuit = QuantumCircuit(lattice.circuit.num_qubits)
circuit.compose(initial.circuit, inplace=True)
circuit.compose(algorithm.circuit, inplace=True)

print("\n--- COMPLETE CIRCUIT ---")
print("Raw instructions:", len(circuit.data))

backend = AerSimulator()

compiled = transpile(
    circuit,
    backend=backend,
    optimization_level=0,
)

print("Transpiled instructions:", len(compiled.data))

measurement_circuit = QuantumCircuit(
    lattice.circuit.num_qubits,
    lattice.num_base_qubits,
)

measurement_circuit.compose(compiled, inplace=True)

measurement_circuit.measure(
    list(range(lattice.num_base_qubits)),
    list(range(lattice.num_base_qubits)),
)

result = backend.run(
    measurement_circuit,
    shots=1024,
).result()

counts = result.get_counts()

print("\n--- MEASUREMENT ---")
print("Number of states:", len(counts))

for state, count in sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True,
)[:20]:
    print(f"{state}: {count}")

print("\n=== END ===")
