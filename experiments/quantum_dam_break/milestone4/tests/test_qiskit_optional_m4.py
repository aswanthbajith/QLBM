import pytest
pytest.importorskip('qiskit')
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solver import add_gates

def test_native_signed_add():
    qc=QuantumCircuit(7); qc.x(0); qc.x(1); qc.x(2) # scratch = -1
    qc.x(4) # accumulator = 2
    for mask,pattern,target in add_gates(0,3,3,4):
        cs=[i for i in range(7) if mask>>i&1]; qc.mcx(cs,target)
    assert Statevector.from_instruction(qc).probabilities_dict()=={'0001111':1.0}
