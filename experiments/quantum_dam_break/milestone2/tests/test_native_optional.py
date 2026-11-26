"""Optional parity with upstream; a skipped test is not reuse verification."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
pytest.importorskip('qiskit')
from qiskit.quantum_info import Operator, DensityMatrix
from collision import U,K,upstream_native, qiskit_step,transition,M,P

def test_native_born_kernel():
    # Original upstream implementation may differ: fail loudly if so.
    actual=abs(Operator(upstream_native()).data)**2
    assert np.allclose(actual,K,atol=1e-9)

def test_single_cell_channel():
    circuit=qiskit_step(1,1,np.zeros((1,1),bool),.7,periodic=True)
    state=DensityMatrix.from_int(10,2**10).evolve(circuit)
    assert np.allclose(state.probabilities()[:512],transition(.7)[:,10],atol=1e-9)
