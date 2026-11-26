"""Explicit two-qubit dilation of the remote branch's signed-rail contraction.
Not a complete D3Q19 collision: projection, preparation and readout remain external.
"""
import numpy as np

def circuit(lam):
    if not np.isfinite(lam) or not -1<=lam<=1:raise ValueError('lambda outside [-1,1]')
    q=(1-lam)/2;theta=2*np.arcsin(np.sqrt(q));c,s=np.cos(theta/2),np.sin(theta/2)
    ry=np.array([[c,-s],[s,c]]);x=np.array([[0,1],[1,0]])
    cx=np.eye(4);cx[2:,2:]=x # environment high bit controls rail low bit
    return cx@np.kron(ry,np.eye(2))

def apply(rho,lam):
    u=circuit(lam); joint=np.kron(np.diag([1.,0.]),rho);out=u@joint@u.conj().T
    return np.trace(out.reshape(2,2,2,2),axis1=0,axis2=2)

def rail(delta,bound):
    if bound<=0 or abs(delta)>bound:raise ValueError('rail bounds exceeded')
    return np.diag([(1+delta/bound)/2,(1-delta/bound)/2])

def decode(rho,bound):return bound*(rho[0,0]-rho[1,1]).real

def shots(rho,bound,n,seed):
    k=np.random.default_rng(seed).binomial(n,float(np.clip(rho[0,0].real,0,1)))
    return bound*(2*k/n-1)

def qiskit_circuit(lam):
    from qiskit import QuantumCircuit
    circuit(lam);qc=QuantumCircuit(2);qc.ry(2*np.arcsin(np.sqrt((1-lam)/2)),1);qc.cx(1,0);return qc

def damping_circuit(lam):
    """Four-qubit unitary: rail plus/minus bits 0/1, environments 2/3.
    CRY then CX realizes |1,0> -> sqrt(a)|1,0>+sqrt(1-a)|0,1>.
    Negative lambda swaps the two rail bits. Explicit gate matrices, not a Kraus shortcut.
    """
    if not np.isfinite(lam) or not -1<=lam<=1:raise ValueError('invalid lambda')
    a=abs(lam);c,s=np.sqrt(a),np.sqrt(1-a);u=np.eye(16)
    for railbit,env in [(0,2),(1,3)]:
        for k in range(16):
            if (k>>railbit)&1 and not (k>>env)&1:
                j=k|(1<<env);u[[k,j]]=np.array([[c,-s],[s,c]])@u[[k,j]]
        for k in range(16):
            if (k>>env)&1 and not (k>>railbit)&1:
                j=k|(1<<railbit);u[[k,j]]=u[[j,k]]
    if lam<0:
        for k in range(16):
            if k&3==1:u[[k,k+1]]=u[[k+1,k]]
    return u

def dual_rail(delta,bound):
    if not np.isfinite(delta) or not np.isfinite(bound) or bound<=0 or abs(delta)>bound:raise ValueError('rail bounds exceeded')
    p=max(delta,0)/bound;n=max(-delta,0)/bound
    return np.kron(np.diag([1-n,n]),np.diag([1-p,p]))

def apply_damping(rho,lam):
    u=damping_circuit(lam);joint=np.kron(np.diag([1.,0,0,0]),rho);out=u@joint@u.conj().T
    return np.trace(out.reshape(4,4,4,4),axis1=0,axis2=2)

def dual_decode(rho,bound):return bound*(rho[1,1]-rho[2,2]).real

def dual_variance(rho,bound):return bound**2*(rho[1,1]+rho[2,2]).real-dual_decode(rho,bound)**2

def damping_qiskit(lam):
    from qiskit import QuantumCircuit
    damping_circuit(lam);qc=QuantumCircuit(4)
    for r,e in [(0,2),(1,3)]:qc.cry(2*np.arcsin(np.sqrt(1-abs(lam))),r,e);qc.cx(e,r)
    if lam<0:qc.swap(0,1)
    return qc
