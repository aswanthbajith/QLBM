"""Small reversible fixed-point pressure oracle, NOT a fluid timestep.
Gate list consists solely of pattern-controlled X gates. No state-dependent
classical calculation is called during oracle execution. Truth table is
compiled offline; exponential scaling, suitable only as a proof of concept.
"""
from fractions import Fraction
import numpy as np


def pressure_word(h_word, h_frac=2, p_frac=4, gravity=Fraction(1,1)):
    h=Fraction(h_word,2**h_frac); p=gravity*h*h/2*2**p_frac
    return (2*p.numerator+p.denominator)//(2*p.denominator) # round half up


def compile_pressure(nh=3,np_=6,h_frac=2,p_frac=4,gravity=Fraction(1,1)):
    if nh<1 or np_<1 or h_frac<0 or p_frac<0 or gravity<0: raise ValueError('invalid encoding')
    gates=[]
    for h in range(1<<nh):
        p=pressure_word(h,h_frac,p_frac,gravity)
        if p>=1<<np_: raise OverflowError('pressure register too narrow')
        for j in range(np_):
            if p>>j&1: gates.append((h,nh+j))
    return gates


def basis_apply(word,gates,nh):
    for pattern,target in gates:
        if word&((1<<nh)-1)==pattern: word^=1<<target
    return word


def statevector_apply(state,gates,nh):
    state=np.array(state,dtype=complex,copy=True); words=np.arange(len(state))
    for pattern,target in gates:
        a=words[((words&((1<<nh)-1))==pattern)&(((words>>target)&1)==0)]; b=a^(1<<target)
        state[a],state[b]=state[b].copy(),state[a].copy()
    return state


def qiskit_pressure(nh=3,np_=6,**kwargs):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(nh+np_)
    for pattern,target in compile_pressure(nh,np_,**kwargs):
        zeros=[i for i in range(nh) if not pattern>>i&1]
        for i in zeros: qc.x(i)
        qc.mcx(list(range(nh)),target)
        for i in zeros: qc.x(i)
    return qc


def compile_bed_source(nh=3,nq=5,h_frac=2,q_frac=4,accel_dt=Fraction(-1,4)):
    """Reversible source update |h,q> -> |h,q+round(h*a*dt)>.
    q is signed two's complement; arithmetic is modulo 2**nq. Caller must
    ensure physical signed range (no overflow). Acceleration is prescribed
    and uniform, e.g. -g*bed_slope; no wet/dry or well-balanced flux here.
    Returns (control mask, control pattern, target bit) gates.
    """
    if min(nh,nq)<1 or min(h_frac,q_frac)<0: raise ValueError('invalid register')
    hm=(1<<nh)-1; gates=[]
    for h in range(1<<nh):
        v=Fraction(h,2**h_frac)*accel_dt*2**q_frac
        # round ties away from zero
        a=abs(v); inc=(2*a.numerator+a.denominator)//(2*a.denominator)
        if v<0: inc=-inc
        if not -(1<<(nq-1))<=inc<(1<<(nq-1)): raise OverflowError('source increment outside signed range')
        inc%=1<<nq
        for j in range(nq):
            if not inc>>j&1: continue
            for k in range(nq-1,j-1,-1):
                carry=((1<<(k-j))-1)<<(nh+j)
                gates.append((hm|carry,h|carry,nh+k))
    return gates


def apply_controlled(word,gates):
    for mask,pattern,target in gates:
        if word&mask==pattern: word^=1<<target
    return word


def signed(word,n):
    return word-(1<<n) if word&(1<<(n-1)) else word


def qiskit_bed_source(nh=3,nq=5,**kwargs):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(nh+nq)
    for mask,pattern,target in compile_bed_source(nh,nq,**kwargs):
        controls=[i for i in range(nh+nq) if mask>>i&1]; zeros=[i for i in controls if not pattern>>i&1]
        for i in zeros: qc.x(i)
        qc.mcx(controls,target)
        for i in zeros: qc.x(i)
    return qc
