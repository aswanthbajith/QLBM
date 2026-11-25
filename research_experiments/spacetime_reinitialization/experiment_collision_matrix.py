from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.lattice import SpaceTimeLattice
from qlbm.components.spacetime.collision.d2q4_old import (
    SpaceTimeD2Q4CollisionOperator,
)


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
        "geometry": [],
    },
)


# ============================================================
# 2. COLLISION OPERATOR
# ============================================================

collision = SpaceTimeD2Q4CollisionOperator(
    lattice=lattice,
    timestep=1,
)

backend = AerSimulator(
    method="statevector"
)


# ============================================================
# 3. BUILD THE LOCAL 4-QUBIT COLLISION CIRCUIT
# ============================================================

def build_local_collision():

    circuit = collision.local_collision_circuit(
        reset_state=False
    )

    circuit.compose(
        collision.gate_to_apply.control(
            3,
            label="collision"
        ),
        qubits=[1, 2, 3, 0],
        inplace=True,
    )

    circuit.compose(
        collision.local_collision_circuit(
            reset_state=True
        ),
        inplace=True,
    )

    return circuit


# ============================================================
# 4. RUN ONE BASIS STATE
# ============================================================

def run_basis_state(bitstring):

    circuit = QuantumCircuit(4)

    # Qiskit uses little-endian qubit ordering.
    #
    # Reverse the displayed bitstring so that:
    #
    # |0101>
    #
    # means q0=1, q1=0, q2=1, q3=0
    #
    # according to the displayed computational basis.

    for q, bit in enumerate(reversed(bitstring)):

        if bit == "1":
            circuit.x(q)

    collision_circuit = build_local_collision()

    circuit.compose(
        collision_circuit,
        inplace=True,
    )

    simulation_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=0,
    )

    simulation_circuit.save_statevector(
        label="collision"
    )

    result = backend.run(
        simulation_circuit
    ).result()

    statevector = result.data(0)["collision"]

    return statevector


# ============================================================
# 5. EXTRACT AMPLITUDES
# ============================================================

def print_state(bitstring, statevector):

    print("\n" + "-" * 70)
    print(f"Input state: |{bitstring}>")
    print("-" * 70)

    for index, amplitude in enumerate(statevector):

        probability = abs(amplitude) ** 2

        if probability > 1e-10:

            output = format(
                index,
                "04b"
            )

            print(
                f"|{output}> : "
                f"amplitude = {amplitude.real:+.12f}"
                f"{amplitude.imag:+.12f}i"
                f"   probability = {probability:.12f}"
            )


# ============================================================
# 6. TEST |0101>
# ============================================================

state_0101 = run_basis_state(
    "0101"
)

print_state(
    "0101",
    state_0101,
)


# ============================================================
# 7. TEST |1010>
# ============================================================

state_1010 = run_basis_state(
    "1010"
)

print_state(
    "1010",
    state_1010,
)


# ============================================================
# 8. EXTRACT THE 2x2 COLLISION MATRIX
# ============================================================

# Basis ordering:
#
#     |0101>
#     |1010>
#
# We calculate the amplitudes of these two states
# after collision.

def amplitude_for(statevector, bitstring):

    index = int(
        bitstring,
        2
    )

    return statevector[index]


a11 = amplitude_for(
    state_0101,
    "0101",
)

a21 = amplitude_for(
    state_0101,
    "1010",
)

a12 = amplitude_for(
    state_1010,
    "0101",
)

a22 = amplitude_for(
    state_1010,
    "1010",
)


# ============================================================
# 9. PRINT MATRIX
# ============================================================

print("\n" + "=" * 80)
print("EFFECTIVE D2Q4 COLLISION MATRIX")
print("=" * 80)

print("\nBasis:")
print("  |0101>")
print("  |1010>")

print("\nMatrix:")

print(
    f"""
        | {a11.real:+.6f}   {a12.real:+.6f} |
    C = |                         |
        | {a21.real:+.6f}   {a22.real:+.6f} |
"""
)


# ============================================================
# 10. CHECK COLUMN NORMS
# ============================================================

column_1_norm = (
    abs(a11) ** 2
    +
    abs(a21) ** 2
)

column_2_norm = (
    abs(a12) ** 2
    +
    abs(a22) ** 2
)


print("Column norms:")

print(
    "  column 1 :",
    column_1_norm,
)

print(
    "  column 2 :",
    column_2_norm,
)


# ============================================================
# 11. CHECK ORTHOGONALITY
# ============================================================

orthogonality = (
    a11.conjugate() * a12
    +
    a21.conjugate() * a22
)


print("\nColumn inner product:")
print(
    " ",
    orthogonality
)


# ============================================================
# 12. FINAL
# ============================================================

print("\n" + "=" * 80)
print("COLLISION MATRIX EXPERIMENT COMPLETE")
print("=" * 80)
