# Source-level reuse audit

Input: uploaded `qlbm-main.zip`, not a live Git checkout. The original source is included unchanged.

| Path in upstream repository | Observed source behavior | Reuse decision |
|---|---|---|
| `qlbm/lattice/lattices/ab_lattice.py` | Binary grid and velocity registers, obstacle/comparator ancillas | Reuse convention; adapter constructs native lattice |
| `qlbm/components/ab/streaming.py` | D2Q9 QFT/controlled phase shifts/inverse QFT with direction indices 1,5,8 (+x), 3,6,7 (-x), 2,5,6 (+y), 4,7,8 (-y) | Direct import in optional adapter for periodic domains; parity test required |
| `qlbm/components/ab/reflection/standard_reflection.py` | Controlled geometry flags, reflection and streaming utilities | Candidate for geometry integration; not used in executed demonstration |
| `qlbm/components/ab/ab.py` | Documentation explicitly says no collision; composes streaming and reflection | Not sufficient as a physical fluid solver |
| `qlbm/components/ab/measurement.py` | `measure_velocity_qubits=True` maps x,y then velocity into classical bits | Direct import in optional adapter |
| `qlbm/infra/result/amplitude_result.py` | 2D converter writes counts into grid array | New writer keeps physical proxy and raw samples separate; existing converter not used |

The original VTI scalar counts are not proof of water depth. This extension writes new, clearly labeled fields. It does not overwrite the uploaded 18 VTI files or claim their provenance has been conclusively established.
