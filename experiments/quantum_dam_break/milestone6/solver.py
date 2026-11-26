"""Replace M5 LUTs with reversible arithmetic. Cache executed gate results only."""
from milestone5.solver2d import Circuit2D, Format, gate, gather
from milestone4.solver import copy_gates, add_gates
import numpy as np

class ExtendedFormat(Format):
    def __post_init__(self):
        if self.scale<2 or self.scale>128 or self.scale & (self.scale-1):raise ValueError("supported powers of two: 2..128")
from milestone6.arithmetic import FluxOracle

class Arithmetic2D(Circuit2D):
    def __init__(self,solid,scale=32):
        if scale<=32:
            super().__init__(solid,scale)
        else:
            # Same M5 layout and accumulation, extended without touching M5 source.
            self.solid=np.asarray(solid,bool).copy();self.ny,self.nx=self.solid.shape;self.solid.flags.writeable=False
            self.f=fmt=ExtendedFormat(scale);self.n=self.nx*self.ny
            self.scratch=self.n*(fmt.cell+fmt.out);self.nqubits=self.scratch+3*fmt.fb;self.blocks=[]
            self.in_offsets=np.cumsum((0,)+fmt.widths[:-1]).tolist();self.out_offsets=np.cumsum((0,)+fmt.outwidths[:-1]).tolist()
            for i in range(self.n):
                g=[]
                for k,(w,ow) in enumerate(zip(fmt.widths,fmt.outwidths)):
                    g.extend(copy_gates(i*fmt.cell+self.in_offsets[k],self.n*fmt.cell+i*fmt.out+self.out_offsets[k],w,ow-w,k!=0))
                self.blocks.append(('g',tuple(g)))
            for y,x in np.argwhere(~self.solid):
                i=int(y*self.nx+x)
                for axis,sign in ((0,1),(0,-1),(1,1),(1,-1)):
                    order=(0,axis+1,2-axis);bits=tuple(i*fmt.cell+self.in_offsets[k]+j for k in order for j in range(fmt.widths[k]))
                    oracle=('o',(sign,bits,sum(1<<b for b in bits)));self.blocks.append(oracle)
                    xx=int(x)+sign*(axis==0);yy=int(y)+sign*(axis==1);wall=not(0<=xx<self.nx and 0<=yy<self.ny) or self.solid[yy,xx];j=i if wall else yy*self.nx+xx;g=[]
                    for p,k in enumerate(order):
                        src=self.scratch+p*fmt.fb;w=fmt.outwidths[k]
                        dst=lambda cell:self.n*fmt.cell+cell*fmt.out+self.out_offsets[k]
                        g.extend(add_gates(src,fmt.fb,dst(i),w,True));g.extend(add_gates(src,fmt.fb,dst(j),w,wall and p==1))
                    self.blocks.append(('g',tuple(g)));self.blocks.append(oracle)
        self.oracles={s:FluxOracle(scale,s) for s in (-1,1)}
        self.working=self.nqubits
        self.nqubits+=max(a.ancillas for a in self.oracles.values())
    def apply(self,w,inverse=False):
        if w>>self.working:raise ValueError('arithmetic workspace must start clean')
        for typ,val in (reversed(self.blocks) if inverse else self.blocks):
            if typ=='g':
                for m,p,t in (reversed(val) if inverse else val):w=gate(w,m,p,t)
            else:
                sign,bits,mask=val;value=self.oracles[sign].evaluate(gather(w,bits))
                w^=value<<self.scratch
        return w
    def gate_count(self):return sum(len(v) if t=='g' else len(self.oracles[v[0]].gates) for t,v in self.blocks)
    def iter_gates(self):
        for typ,val in self.blocks:
            if typ=='g':yield from val
            else:
                sign,bits,_=val;a=self.oracles[sign]
                mapping=tuple(bits)+tuple(range(self.scratch,self.working))+tuple(range(self.working,self.working+a.ancillas))
                for m,p,t in a.gates:
                    mask=0
                    while m:
                        low=m&-m;mask|=1<<mapping[low.bit_length()-1];m^=low
                    yield mask,mask,mapping[t]
