import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
pytest.importorskip('qiskit')
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from quantum_core import Grid, initial, stream_permutation, apply_permutation, relaxation_channel
from qiskit_bridge import preparation, closed_streaming, relaxation, native_components

def test_qiskit_closed_stream():
    g=Grid(); a,_=initial(g)
    actual=Statevector.from_instruction(preparation(g,a)).evolve(closed_streaming(g))
    assert actual.equiv(Statevector(apply_permutation(a,stream_permutation(g))))

def test_qiskit_relaxation():
    g=Grid(2,2); a,_=initial(g); rho=np.outer(a,a.conj())
    # Embed system in low bits, 5 ancillary |0> qubits in high bits.
    full=np.zeros(32*g.dim,complex); full[:g.dim]=a
    r=DensityMatrix(Statevector(full)).evolve(relaxation(g,.5))
    actual=partial_trace(r,list(range(g.nq,g.nq+5))).data
    assert np.allclose(actual,relaxation_channel(g,rho,.5),atol=1e-9)

def test_native_periodic_streaming():
    g=Grid(); a,_=initial(g); lattice,qc,_=native_components(g)
    full=np.zeros(2**qc.num_qubits,complex); full[:g.dim]=a
    expected=np.zeros_like(full); expected[:g.dim]=apply_permutation(a,stream_permutation(g,True))
    assert Statevector(full).evolve(qc).equiv(Statevector(expected))
