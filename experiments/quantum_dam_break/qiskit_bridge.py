"""Optional circuits: not executed in the delivery environment (Qiskit unavailable)."""
import numpy as np
from quantum_core import W, gate_pairs, stream_permutation


def native_components(g):
    """Direct reuse of upstream periodic D2Q9 streaming and joint measurement."""
    from qlbm.lattice import ABLattice
    from qlbm.components.ab import ABStreamingOperator, ABGridMeasurement
    if g.solid.any(): raise ValueError('This adapter is only for periodic empty domains')
    lattice=ABLattice({'lattice':{'dim':{'x':g.nx,'y':g.ny},'velocities':'d2q9'},'geometry':[]})
    return lattice, ABStreamingOperator(lattice).circuit, ABGridMeasurement(lattice,measure_velocity_qubits=True).circuit


def preparation(g,a):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    qc=QuantumCircuit(g.nq); qc.append(StatePreparation(a),range(g.nq)); return qc


def closed_streaming(g):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(g.nq)
    for x,y in gate_pairs(stream_permutation(g)):
        target=(x^y).bit_length()-1
        ctrl=[k for k in range(g.nq) if k!=target]
        zero=[k for k in ctrl if not (x>>k)&1]
        if zero: qc.x(zero)
        qc.mcx(ctrl,target)
        if zero: qc.x(zero)
    return qc


def relaxation(g,omega):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    if not 0<=omega<=1: raise ValueError('omega must be in [0,1]')
    qc=QuantumCircuit(g.nq+5); coin=g.nq; fresh=list(range(coin+1,coin+5))
    w=np.zeros(16); w[:9]=np.sqrt(W)
    qc.ry(2*np.arcsin(np.sqrt(omega)),coin)
    qc.append(StatePreparation(w),fresh)
    for a,b in zip(range(g.nq-4,g.nq),fresh): qc.cswap(coin,a,b)
    # Reset is a nonunitary operation (equivalent to tracing out and preparing |0>).
    qc.reset([coin]+fresh)
    return qc
