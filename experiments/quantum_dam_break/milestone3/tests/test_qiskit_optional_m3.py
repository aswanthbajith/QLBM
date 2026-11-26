import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
qiskit=pytest.importorskip('qiskit')
from qiskit.quantum_info import Statevector,DensityMatrix,partial_trace
from arithmetic import *
from physics import qiskit_force,force_kernel

def test_pressure_circuit():
    qc=qiskit_pressure(); v=np.zeros(512,complex); v[:8]=1/np.sqrt(8)
    assert np.allclose(Statevector(v).evolve(qc).data,statevector_apply(v,compile_pressure(),3))

def test_bed_source_circuit():
    qc=qiskit_bed_source(); v=np.zeros(256,complex); v[:8]=1/np.sqrt(8); expected=np.zeros_like(v)
    for h in range(8): expected[apply_controlled(h,compile_bed_source())]=v[h]
    assert np.allclose(Statevector(v).evolve(qc).data,expected)
# Force circuit uses 14 qubits with resets; avoid dense 14-qubit density matrix.
# Exact kernel tests run without Qiskit; native force execution still pending.
