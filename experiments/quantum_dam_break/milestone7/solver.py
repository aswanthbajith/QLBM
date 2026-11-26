"""M6 transport with compact workspace and optional reversible linear drag.
The two stages retain input and intermediate registers. No fluid reference is
called by the gate simulation. Acceleration uses cached local gate evaluations.
"""
import numpy as np
from milestone6.solver import Arithmetic2D
from milestone4.solver import gate,signed
from milestone7.optimization import CompactFlux,controls
from milestone7.drag import DragOracle

class Compact2D(Arithmetic2D):
    def __init__(self,solid,scale=32):
        super().__init__(solid,scale)
        self.oracles={s:CompactFlux(scale,s) for s in (-1,1)}
        self.nqubits=self.working+max(a.ancillas for a in self.oracles.values())

class Damped2D:
    def __init__(self,solid,scale=32,numerator=7,shift=3):
        self.transport=c=Compact2D(solid,scale);self.f=c.f;self.solid=c.solid
        self.width=c.f.outwidths[1];self.drag=DragOracle(self.width,numerator,shift)
        # Extra outputs preserve pre-drag momentum; reuse the transport workspace.
        self.extra=c.nqubits;self.nqubits=self.extra+2*c.n*self.width
        assert self.drag.ancillas<=c.nqubits-c.working
        self.mask=(1<<self.width)-1
        self.pairs=[(c.n*c.f.cell+i*c.f.out+c.out_offsets[k],self.extra+(2*i+k-1)*self.width) for i in range(c.n) for k in (1,2)]
    def encode(self,s):return self.transport.encode(s)
    def _drag(self,w):
        for src,dst in self.pairs:
            q=signed((w>>src)&self.mask,self.width)
            w^=(self.drag.evaluate(q)&self.mask)<<dst
        return w
    def apply(self,w,inverse=False):
        c=self.transport;lo_mask=(1<<self.extra)-1
        if inverse:w=self._drag(w)
        w=(w&~lo_mask)|c.apply(w&lo_mask,inverse)
        if not inverse:w=self._drag(w)
        return w
    def decode(self,w):
        c=self.transport;z=c.decode(w)
        for i,(_,dst) in enumerate(self.pairs):z.reshape(-1,3)[i//2,1+i%2]=signed((w>>dst)&self.mask,self.width)
        return z
    def step(self,s,verify_inverse=False):
        c=self.transport;w=self.encode(s);z=self.apply(w)
        assert z&((1<<(c.n*c.f.cell))-1)==w
        assert ((z>>c.scratch)&((1<<(self.extra-c.scratch))-1))==0
        if verify_inverse:assert self.apply(z,True)==w
        out=self.decode(z);c.encode(out);return out
    def gate_count(self):return self.transport.gate_count()+len(self.pairs)*len(self.drag.gates)
    def iter_gates(self):
        yield from self.transport.iter_gates()
        for src,dst in self.pairs:
            m=tuple(range(src,src+self.width))+tuple(range(dst,dst+self.width))+tuple(range(self.transport.working,self.transport.working+self.drag.ancillas))
            for mask,p,t in self.drag.gates:
                mm=sum(1<<m[i] for i in controls(mask));yield mm,mm,m[t]
    def qiskit(self,max_gates=1000000):
        if self.gate_count()>max_gates:raise ValueError('logical gate export exceeds cap')
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for m,p,t in self.iter_gates():
            cs=list(controls(m));assert m==p
            if not cs:qc.x(t)
            elif len(cs)==1:qc.cx(cs[0],t)
            elif len(cs)==2:qc.ccx(*cs,t)
            else:qc.mcx(cs,t)
        return qc
