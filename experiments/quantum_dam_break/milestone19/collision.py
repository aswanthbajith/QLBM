"""Batched four-qubit dual-rail circuit for momentwise dissipation.
Classical moment transforms and equilibrium; exact ideal expectation readout.
"""
import numpy as np
from milestone18.transport import C19,C7
from milestone18.channel import damping_circuit
from functools import lru_cache


def moment_basis(c):
    """Independent polynomial rows, first mass/momentum, then trace and deviatoric stress."""
    x,y,z=np.asarray(c).T
    candidates=[np.ones(len(c)),x,y,z,x*x+y*y+z*z,x*x-y*y,y*y-z*z,x*y,x*z,y*z]
    candidates += [x**i*y**j*z**k for i in range(3) for j in range(3) for k in range(3)]
    rows=[]
    for v in candidates:
        v=v.astype(float).copy()
        for r in rows:v-=np.dot(v,r)*r
        norm=np.linalg.norm(v)
        if norm>1e-9:rows.append(v/norm)
        if len(rows)==len(c):break
    return np.stack(rows)

M19=moment_basis(C19);M7=moment_basis(C7)

@lru_cache(None)
def unitary(lam):return damping_circuit(lam)


def relax_gate(delta,lam,bound=2.,shots=None,rng=None):
    """Prepare sign-split pure rail states and simulate actual 4-qubit gate unitary.
    A sign-split coherent preparation has the same diagonal expectation as the
    mixed encoding, but not the same off-diagonals. Only the rail-Z readout is used.
    The input signs, RY preparation angles and bound are classical parameters.
    """
    d=np.asarray(delta,float);shape=d.shape;d=d.ravel()
    if not np.isfinite(d).all() or np.max(np.abs(d),initial=0)>bound:raise ValueError('rail bound exceeded')
    p=np.abs(d)/bound
    psi=np.zeros((16,len(d)),complex)
    psi[0]=np.sqrt(1-p);idx=np.where(d>=0,1,2);psi[idx,np.arange(len(d))]=np.sqrt(p)
    out=unitary(float(lam))@psi
    probs=np.abs(out.reshape(4,4,len(d)))**2
    pr=probs.sum(axis=0);plus=pr[1]+pr[3];minus=pr[2]+pr[3]
    if shots is None:result=bound*(plus-minus)
    else:
        if rng is None:raise ValueError('seeded rng required')
        # Sign-split input and amplitude damping imply exclusive rail occupancy.
        if np.max(pr[3])>1e-12:raise RuntimeError('nonexclusive rails')
        p=np.clip(plus+minus,0,1);result=bound*np.sign(plus-minus)*rng.binomial(shots,p)/shots
    return result.reshape(shape)


def collide(f,eq,source,M,rates,mode='gate',shots=None,rng=None,bound=2.):
    """MRT with Guo/trapezoidal modewise source. All zero-rate conserved modes
    bypass dissipation; no readout noise is added to conserved moments.
    """
    q=f.shape[0];shape=f.shape
    m=M@f.reshape(q,-1);e=M@eq.reshape(q,-1);s=M@source.reshape(q,-1)
    d=m-e;out=m.copy();rates=np.asarray(rates)
    for w in np.unique(rates[rates!=0]):
        idx=rates==w
        if mode=='gate':r=relax_gate(d[idx],1-w,bound,shots,rng)
        elif mode=='reference':r=(1-w)*d[idx]
        else:raise ValueError(mode)
        out[idx]=e[idx]+r
    out+=(1-rates[:,None]/2)*s
    return (M.T@out).reshape(shape)


def qiskit_mode(delta,lam,bound=.25,measure=False):
    """Native circuit for one dissipative moment, optional dependency.
    System rail bits 0,1; environment bits 2,3; measurement of rails only.
    """
    from qiskit import QuantumCircuit
    from milestone18.channel import damping_qiskit
    if not np.isfinite(delta) or bound<=0 or abs(delta)>bound:raise ValueError('rail bound exceeded')
    qc=QuantumCircuit(4,2) if measure else QuantumCircuit(4)
    qc.ry(2*np.arcsin(np.sqrt(abs(delta)/bound)),0 if delta>=0 else 1)
    qc.compose(damping_qiskit(lam),inplace=True)
    if measure:qc.measure([0,1],[0,1])
    return qc
