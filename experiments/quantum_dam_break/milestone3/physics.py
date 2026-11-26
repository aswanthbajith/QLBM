"""Exact equilibrium diagnostics and biased quantum force channel for M2.
All occupancy quantities are lattice units; not calibrated water physics.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'milestone2'))
import numpy as np
from collision import BITS, M, P, STRESS, C, GROUPS, K, transition


def equilibrium(log_z, beta=(0.,0.)):
    """Grand canonical state exp(log_z*m + beta·p), fixed by M2 collision.
    Independent Bernoulli occupancies; stable normalization.
    """
    a=log_z*M+P@np.asarray(beta); w=np.exp(a-a.max()); return w/w.sum()


def eos(h):
    """Rest equilibrium at specified mean occupancy mass 0<h<10."""
    if not 0<h<10: raise ValueError('mass outside open interval (0,10)')
    lo,hi=-40.,40.
    for _ in range(80):
        mid=(lo+hi)/2
        if equilibrium(mid)@M<h: lo=mid
        else: hi=mid
    w=equilibrium((lo+hi)/2)
    pressure=w@STRESS[:,0,0]
    # derivative by exponential-family covariance, not finite differences
    cs2=(w@(STRESS[:,0,0]*M)-pressure*h)/(w@(M*M)-h*h)
    return w,pressure,cs2


def force_pairs(axis):
    if axis==0: return [(1,3),(5,6),(8,7)]
    if axis==1: return [(2,4),(5,8),(6,7)]
    raise ValueError('axis must be 0 or 1')


def force(states,axis,bias,rate,rng):
    """Conditional heat-bath channel on pairs with exactly one occupied wire.
    bias in [-1,1], positive drives positive axis. rate in [0,1].
    Equivalent to explicit pair measurement, conditional preparation, and
    dephasing. Does not compute a classical fluid field.
    """
    if not -1<=bias<=1 or not 0<=rate<=1: raise ValueError('invalid bias/rate')
    out=np.array(states,copy=True)
    for a,b in force_pairs(axis):
        single=((out>>a)^(out>>b))&1
        apply=(single==1)&(rng.random(out.shape)<rate)
        positive=rng.random(out.shape)<(1+bias)/2
        out=np.where(apply,(out&~((1<<a)|(1<<b)))|np.where(positive,1<<a,1<<b),out)
    return out.astype(np.int16)


def force_kernel(axis,bias,rate):
    """Exact 512-state channel on diagonal states, columns are input states."""
    if not -1<=bias<=1 or not 0<=rate<=1: raise ValueError('invalid bias/rate')
    k=np.eye(512); s=np.arange(512)
    for a,b in force_pairs(axis):
        pair=np.eye(512); active=s[(((s>>a)^(s>>b))&1)==1]; clear=active&~((1<<a)|(1<<b))
        pair[:,active]*=1-rate
        pair[clear|(1<<a),active]+=rate*(1+bias)/2
        pair[clear|(1<<b),active]+=rate*(1-bias)/2
        k=pair@k
    return k


def expected_impulse(states,axis,bias,rate):
    """Exact local expectation of momentum change for each input basis state."""
    out=np.zeros(np.shape(states),float)
    for a,b in force_pairs(axis):
        plus=((states>>a)&1).astype(int); minus=((states>>b)&1).astype(int)
        out+=rate*(bias*(plus^minus)-(plus-minus))
    return out


def qiskit_force(axis,bias,rate):
    """9 data + 5 reusable environment qubits; no classical feedback.
    Pair parity and coin control swapping with a one-excitation reservoir.
    Reset environment, then explicitly dephase all data wires.
    """
    if not -1<=bias<=1 or not 0<=rate<=1: raise ValueError('invalid bias/rate')
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(14); coin,parity,active,rp,rm=range(9,14)
    for a,b in force_pairs(axis):
        qc.ry(2*np.arcsin(np.sqrt(rate)),coin)
        qc.cx(a,parity); qc.cx(b,parity); qc.ccx(coin,parity,active)
        qc.ry(2*np.arcsin(np.sqrt((1+bias)/2)),rp); qc.x(rm); qc.cx(rp,rm)
        qc.cswap(active,a,rp); qc.cswap(active,b,rm)
        for w in (coin,parity,active,rp,rm): qc.reset(w)
    for w in range(9): qc.cx(w,coin); qc.reset(coin)
    return qc
