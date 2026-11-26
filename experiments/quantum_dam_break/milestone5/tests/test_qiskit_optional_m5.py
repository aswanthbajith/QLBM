import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pytest
pytest.importorskip('qiskit')
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
from milestone5.solver2d import Format,table

def test_native_nonzero_oracle_gate():
    f=Format(4); lut=table(f,1);pattern=int(np.flatnonzero(lut)[0]);n=f.cell
    # Test one actual nonzero pattern-controlled oracle gate (one target).
    assert int(lut[pattern])!=0
    qc=QuantumCircuit(n+1)
    for i in range(n):
        if not pattern>>i&1:qc.x(i)
    qc.mcx(list(range(n)),n)
    for i in range(n):
        if not pattern>>i&1:qc.x(i)
    v=Statevector.from_int(pattern,2**(n+1)).evolve(qc)
    assert np.isclose(abs(v.data[pattern|(1<<n)]),1)
