from qlbm.lattice import SpaceTimeLattice
from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.reinitialize import SpaceTimeReinitializer


# ============================================================
# 1. CREATE THE SAME LATTICE
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
# 2. CREATE REINITIALIZER
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
# 3. REAL COUNTS FROM THE ONE-STEP EXPERIMENT
# ============================================================

counts = {
    "00000001": 81,
    "00001110": 75,
    "00001011": 75,
    "00000011": 70,
    "00000110": 65,
    "00001101": 65,
    "00000111": 64,
    "00000100": 64,
    "00000101": 64,
    "00000010": 64,
    "00001100": 64,
    "00101001": 63,
    "00001010": 59,
    "00000000": 53,
    "00001111": 50,
    "00001000": 48,
}


# ============================================================
# 4. DECODE USING REPOSITORY IMPLEMENTATION
# ============================================================

velocity_pairs = reinitializer.counts_to_velocity_pairs(
    counts
)


# ============================================================
# 5. PRINT RESULT
# ============================================================

print("\n" + "=" * 80)
print("DECODED SPACE-TIME QLBM MEASUREMENT")
print("=" * 80)

print("\nNumber of raw measured states:")
print(" ", len(counts))

print("\nNumber of decoded occupied states:")
print(" ", len(velocity_pairs))

print("\nDecoded states:")

for item in velocity_pairs:
    print(" ", item)

print("\n" + "=" * 80)

