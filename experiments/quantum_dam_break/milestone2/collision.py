"""D2Q9 occupancy collision channel, adapted from upstream equivalence-class design.
Global trajectories simulate an explicitly dephased quantum circuit, not BGK.
"""
from collections import defaultdict
import numpy as np
C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
MASS=np.array([2,1,1,1,1,1,1,1,1]) # upstream D2Q9 channel masses
OPP=np.array([0,3,4,1,2,7,8,5,6])
BITS=((np.arange(512)[:,None]>>np.arange(9))&1).astype(np.int8)
M=BITS@MASS; P=BITS@C
STRESS=np.einsum('si,ia,ib->sab',BITS,C,C)

def classes():
    groups=defaultdict(list)
    for s in range(512): groups[(int(M[s]),*map(int,P[s]))].append(s)
    return [np.array(v) for v in groups.values()]

GROUPS=classes()

def unitary():
    """Fourier block on each mass/momentum sector (including singleton identity)."""
    u=np.zeros((512,512),complex)
    for group in GROUPS:
        n=len(group); k=np.arange(n)
        u[np.ix_(group,group)]=np.exp(2j*np.pi*np.outer(k,k)/n)/np.sqrt(n)
    return u

U=unitary()
K=abs(U)**2

def transition(omega):
    if not 0<=omega<=1: raise ValueError('omega must be in [0,1]')
    return (1-omega)*np.eye(512)+omega*K

def channel(rho,omega):
    """Coin controlled Fourier, then full computational-basis dephasing.
    Valid on coherent inputs; the trajectory engine assumes diagonal inputs.
    """
    if not 0<=omega<=1: raise ValueError('omega must be in [0,1]')
    return np.diag(np.real(np.diag((1-omega)*rho+omega*(U@rho@U.conj().T))))

# Sampling tables derived from the executed local quantum unitary, not a fluid update.
MEMBERS=np.zeros((512,max(map(len,GROUPS))),dtype=np.int16)
LENGTH=np.zeros(512,dtype=np.int16)
for group in GROUPS:
    for s in group:
        MEMBERS[s,:len(group)]=group; LENGTH[s]=len(group)

def collide(states,omega,rng):
    if not 0<=omega<=1: raise ValueError('omega must be in [0,1]')
    # Since all inputs are diagonal, uniform sampling of |U_ba|^2 is exact.
    states=np.asarray(states)
    draw=(rng.random(states.shape)*LENGTH[states]).astype(int)
    return np.where(rng.random(states.shape)<omega,MEMBERS[states,draw],states)

def permutation(nx,ny,solid,periodic=False):
    """One wire per (cell,direction); outgoing wire -> destination wire."""
    p=np.arange(nx*ny*9)
    for y in range(ny):
        for x in range(nx):
            if solid[y,x]: continue
            for i,(cx,cy) in enumerate(C):
                a,b=x+cx,y+cy
                if periodic: a,b=a%nx,b%ny
                if not(0<=a<nx and 0<=b<ny) or solid[b,a]: a,b,j=x,y,OPP[i]
                else: j=i
                p[(y*nx+x)*9+i]=(b*nx+a)*9+j
    if not np.array_equal(np.sort(p),np.arange(len(p))): raise AssertionError('not bijective')
    return p

def swaps(p):
    seen=set()
    for start in range(len(p)):
        if start in seen: continue
        cur=start; cycle=[]
        while cur not in seen:
            seen.add(cur); cycle.append(cur); cur=int(p[cur])
        for j in cycle[1:]: yield start,j

def stream(states,p):
    bits=BITS[states].reshape(len(states),-1)
    bits=bits[:,np.argsort(p)].reshape(*states.shape,9)
    return (bits@(1<<np.arange(9))).astype(np.int16)

def upstream_native():
    """Direct import of existing QLBM operator. Requires upstream dependencies."""
    from qlbm.components.common.cbse_collision.cbse_collision import EQCCollisionOperator
    from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
    return EQCCollisionOperator(LatticeDiscretization.D2Q9).circuit

def qiskit_step(nx,ny,solid,omega,periodic=False,native=False):
    """Optional circuit; a reusable ancilla, local controlled U, reset, dephase, SWAPs.
    native=True selects upstream collision; parity must be checked independently.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    n=nx*ny*9; qc=QuantumCircuit(n+1); a=n
    if not 0<=omega<=1: raise ValueError('omega must be in [0,1]')
    if native: gate=upstream_native().to_gate().control(1)
    else: gate=UnitaryGate(U,label='EQC Fourier').control(1)
    for y in range(ny):
        for x in range(nx):
            if solid[y,x]: continue
            wires=list(range((y*nx+x)*9,(y*nx+x+1)*9))
            qc.ry(2*np.arcsin(np.sqrt(omega)),a); qc.append(gate,[a]+wires); qc.reset(a)
            for w in wires: qc.cx(w,a); qc.reset(a)
    for x,y in swaps(permutation(nx,ny,solid,periodic)): qc.swap(x,y)
    return qc
