from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.lattice import SpaceTimeLattice
from qlbm.components.spacetime.collision.d2q4_old import (
    SpaceTimeD2Q4CollisionOperator,
)


# ============================================================
# 1. CREATE D2Q4 SPACE-TIME LATTICE
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
# 2. CREATE LOCAL D2Q4 COLLISION OPERATOR
# ============================================================

collision = SpaceTimeD2Q4CollisionOperator(
    lattice=lattice,
    timestep=1,
)


# ============================================================
# 3. PRINT BASIC INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("D2Q4 SPACE-TIME COLLISION EXPERIMENT")
print("=" * 80)

print("\nLattice:")
print("  grid                 : 4 x 4")
print("  dimensions           :", lattice.num_dims)
print("  timesteps            :", lattice.num_timesteps)
print("  total qubits         :", lattice.circuit.num_qubits)

print("\nCollision operator:")
print("  qubits               :", collision.circuit.num_qubits)
print("  instructions         :", len(collision.circuit.data))


# ============================================================
# 4. LOCAL VELOCITY REGISTER
# ============================================================

# For D2Q4 the local velocity register contains four bits:
#
#   q0 = v0 = +x
#   q1 = v1 = +y
#   q2 = v2 = -x
#   q3 = v3 = -y
#
# We will test the local collision transformation directly.


local_circuit = QuantumCircuit(4)


# ============================================================
# 5. INITIAL LOCAL STATE
# ============================================================

# State:
#
#   |0101>
#
# This corresponds to:
#
#   v0 = 0
#   v1 = 1
#   v2 = 0
#   v3 = 1
#
# i.e. +y and -y are occupied.
#
# This state has zero net momentum.

local_circuit.x(1)
local_circuit.x(3)


# ============================================================
# 6. EMBED LOCAL STATE INTO A 24-QUBIT CIRCUIT
# ============================================================

main_circuit = QuantumCircuit(
    lattice.circuit.num_qubits
)

# For timestep 1:
#
# grid qubits = 4
#
# First local velocity block:
#
# q4, q5, q6, q7
#
# correspond to the origin point.

main_circuit.compose(
    local_circuit,
    qubits=[4, 5, 6, 7],
    inplace=True,
)


# ============================================================
# 7. APPLY COLLISION TO THE ORIGIN
# ============================================================

# The complete collision operator applies collision locally
# to every velocity block.
#
# For this experiment we apply only the local collision circuit
# to q4-q7 so that the transformation can be studied directly.

local_collision = collision.local_collision_circuit(
    reset_state=False
)

local_collision.compose(
    collision.gate_to_apply.control(
        3,
        label="collision"
    ),
    qubits=[1, 2, 3, 0],
    inplace=True,
)

local_collision.compose(
    collision.local_collision_circuit(
        reset_state=True
    ),
    inplace=True,
)


main_circuit.compose(
    local_collision,
    qubits=[4, 5, 6, 7],
    inplace=True,
)


# ============================================================
# 8. PRINT CIRCUIT SIZE
# ============================================================

print("\nLocal collision circuit:")
print(
    "  instructions       :",
    len(local_collision.data),
)

print("\nComplete test circuit:")
print(
    "  qubits              :",
    main_circuit.num_qubits,
)

print(
    "  instructions        :",
    len(main_circuit.data),
)


# ============================================================
# 9. TRANSPile FOR AER
# ============================================================

backend = AerSimulator(
    method="statevector"
)

simulation_circuit = transpile(
    main_circuit,
    backend=backend,
    optimization_level=0,
)


print("\nTranspiled circuit:")
print(
    "  instructions        :",
    len(simulation_circuit.data),
)


# ============================================================
# 10. SAVE STATEVECTOR
# ============================================================

simulation_circuit.save_statevector(
    label="collision"
)


# ============================================================
# 11. EXECUTE
# ============================================================

result = backend.run(
    simulation_circuit
).result()

statevector = result.data(0)["collision"]


# ============================================================
# 12. EXTRACT LOCAL STATE PROBABILITIES
# ============================================================

# Because the complete circuit contains 24 qubits, we will
# inspect the amplitudes of the computational basis states.
#
# We are particularly interested in the local velocity register
# q4-q7.


print("\nStatevector dimension:")
print(
    " ",
    len(statevector)
)


# ============================================================
# 13. PRINT NON-ZERO BASIS STATES
# ============================================================

print("\nNon-zero computational basis states:")

threshold = 1e-10

shown = 0

for index, amplitude in enumerate(statevector):

    probability = abs(amplitude) ** 2

    if probability > threshold:

        bitstring = format(
            index,
            f"0{main_circuit.num_qubits}b"
        )

        print(
            f"  {bitstring}: "
            f"probability={probability:.6f}, "
            f"amplitude={amplitude}"
        )

        shown += 1

        if shown >= 20:
            print(
                "\n  ... output limited to first 20 states ..."
            )
            break


# ============================================================
# 14. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("COLLISION EXPERIMENT COMPLETE")
print("=" * 80)

print("\nInitial local velocity state:")
print("  |0101>")

print("\nPhysical interpretation:")
print("  +y + -y")
print("  zero net momentum")

print("\nThe next step is to determine how the")
print("collision operator transforms this state.")
print("=" * 80)
