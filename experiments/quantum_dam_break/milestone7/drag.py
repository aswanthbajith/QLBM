"""Reversible constant-coefficient linear bed-drag source primitive.
For clean workspace: |q>|y>|0> -> |q>|y XOR round(beta*q)>|0>.
No claim of molecular viscosity. beta = numerator / 2**shift.
"""
from functools import lru_cache
from milestone6.arithmetic import Builder,F,T
from milestone4.solver import gate,signed

class DragOracle:
    def __init__(self,width,numerator=7,shift=3):
        if width<2 or shift<1 or not 0<=numerator<=2**shift:raise ValueError('invalid drag parameters')
        self.width=width;self.numerator=numerator;self.shift=shift
        b=Builder(2*width);q=list(range(width));a=b.abs(q);W=width+shift+2
        product=b.mul(a,b.const(numerator,shift+1),W)
        rounded=b.add(product,b.const(1<<(shift-1),W),W)[0]
        mag=b.fit(rounded[shift:],width)
        result=b.select(q[-1],b.neg(mag,width),mag,width)
        for j,v in enumerate(result):
            if v==T:b.op(0,width+j)
            elif v!=F:b.op(1<<v,width+j)
        copy=[g for g in b.g if width<=g[2]<2*width];compute=[g for g in b.g if g[2]>=2*width]
        self.gates=tuple(compute+copy+compute[::-1]);self.nqubits=b.n;self.ancillas=b.n-2*width
    def apply(self,w,inverse=False):
        for m,p,t in (reversed(self.gates) if inverse else self.gates):w=gate(w,m,p,t)
        return w
    @lru_cache(maxsize=16384)
    def evaluate(self,q):
        if not -(1<<(self.width-1))<=q<(1<<(self.width-1)):raise ValueError('signed overflow')
        v=q&((1<<self.width)-1);z=self.apply(v)
        assert z&((1<<self.width)-1)==v and not z>>(2*self.width)
        return signed(z>>self.width,self.width)
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for m,p,t in self.gates:
            c=[i for i in range(m.bit_length()) if m>>i&1]
            if not c:qc.x(t)
            elif len(c)==1:qc.cx(c[0],t)
            else:qc.ccx(*c,t)
        return qc
