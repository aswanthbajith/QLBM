"""Reversible 2D shallow-water split-flux circuit, not an LBM collision solver.
Offline integer LUT compiler; runtime is indexed basis controlled-X simulation.
The input is immutable and all temporary oracle registers are uncomputed.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone4.solver import gate, add_gates, copy_gates, signed, gather

@dataclass(frozen=True)
class Format:
    scale: int=32
    def __post_init__(self):
        if self.scale not in (2,4,8,16,32): raise ValueError('supported scale: 2,4,8,16,32')
    @property
    def hb(self): return self.scale.bit_length()
    @property
    def qb(self): return self.hb+1
    @property
    def cell(self): return self.hb+2*self.qb
    @property
    def fb(self): return self.qb+1
    @property
    def widths(self): return (self.hb,self.qb,self.qb)
    @property
    def outwidths(self): return tuple(w+4 for w in self.widths)
    @property
    def out(self): return sum(self.outwidths)
    def valid(self,h,x,y): return 0<=h<2*self.scale and all(-(1<<(self.qb-1))<=q<(1<<(self.qb-1)) and 2*abs(q)<=3*h for q in (x,y))

@lru_cache(None)
def table(fmt,sign):
    """Compile split transfer for all basis inputs, including invalid extension.
    In depth/discharge ticks, dt/dx=dt/dy=1/8, a=3, g=1.
    Floor nonnegative mass; nearest signed momentum; suppress zero-mass transfers.
    Normal output in global axis coordinates, transverse output unchanged.
    """
    H=np.arange(1<<fmt.hb,dtype=np.int64)[None,None,:]
    Q=np.arange(1<<fmt.qb,dtype=np.int64)
    Q=np.where(Q>=1<<(fmt.qb-1),Q-(1<<fmt.qb),Q)
    N=sign*Q[None,:,None]; T=Q[:,None,None]; D=np.maximum(H,1)
    M=(3*H+N)//16
    valid=(H>0)&(2*abs(N)<=3*H)&(2*abs(T)<=3*H)&(M>0)
    def rnd(num,den): return np.sign(num)*((2*abs(num)+den)//(2*den))
    P=sign*rnd(2*fmt.scale*(3*N*H+N*N)+H**3,32*fmt.scale*D)
    R=rnd(T*(3*H+N),16*D)
    mask=(1<<fmt.fb)-1
    v=np.where(valid,(M&mask)|((P&mask)<<fmt.fb)|((R&mask)<<(2*fmt.fb)),0).astype(np.uint32).ravel()
    v.flags.writeable=False
    return v

@lru_cache(None)
def oracle_count(fmt,sign):
    lookup=np.array([i.bit_count() for i in range(256)],np.uint8)
    return int(lookup[table(fmt,sign).view(np.uint8)].sum())

class Circuit2D:
    def __init__(self,solid,scale=32):
        self.solid=np.asarray(solid,bool).copy(); self.ny,self.nx=self.solid.shape
        self.solid.flags.writeable=False
        self.f=fmt=Format(scale); self.n=self.nx*self.ny
        self.scratch=self.n*(fmt.cell+fmt.out); self.nqubits=self.scratch+3*fmt.fb
        self.blocks=[]
        self.in_offsets=np.cumsum((0,)+fmt.widths[:-1]).tolist()
        self.out_offsets=np.cumsum((0,)+fmt.outwidths[:-1]).tolist()
        for i in range(self.n):
            g=[]
            for k,(w,ow) in enumerate(zip(fmt.widths,fmt.outwidths)):
                g.extend(copy_gates(i*fmt.cell+self.in_offsets[k],self.n*fmt.cell+i*fmt.out+self.out_offsets[k],w,ow-w,k!=0))
            self.blocks.append(('g',tuple(g)))
        for y,x in np.argwhere(~self.solid):
            i=int(y*self.nx+x)
            for axis,sign in ((0,1),(0,-1),(1,1),(1,-1)):
                order=(0,axis+1,2-axis)
                bits=tuple(i*fmt.cell+self.in_offsets[k]+j for k in order for j in range(fmt.widths[k]))
                oracle=('o',(sign,bits,sum(1<<b for b in bits)))
                self.blocks.append(oracle)
                xx=int(x)+sign*(axis==0); yy=int(y)+sign*(axis==1)
                wall=not(0<=xx<self.nx and 0<=yy<self.ny) or self.solid[yy,xx]
                j=i if wall else yy*self.nx+xx
                g=[]
                for p,k in enumerate(order):
                    src=self.scratch+p*fmt.fb; w=fmt.outwidths[k]
                    dst=lambda cell:self.n*fmt.cell+cell*fmt.out+self.out_offsets[k]
                    g.extend(add_gates(src,fmt.fb,dst(i),w,True))
                    g.extend(add_gates(src,fmt.fb,dst(j),w,wall and p==1))
                self.blocks.append(('g',tuple(g))); self.blocks.append(oracle)
    def encode(self,s):
        s=np.asarray(s)
        if s.shape!=(*self.solid.shape,3) or not np.issubdtype(s.dtype,np.integer): raise ValueError('expected integer ticks [y,x,h/qx/qy]')
        w=0
        for i,row in enumerate(s.reshape(-1,3)):
            v=tuple(map(int,row))
            if not self.f.valid(*v) or (self.solid.flat[i] and any(v)): raise ValueError(f'invalid cell {i}: {v}')
            for k,(a,b) in enumerate(zip(v,self.f.widths)): w|=(a&((1<<b)-1))<<(i*self.f.cell+self.in_offsets[k])
        return w
    def decode(self,w):
        rows=[]
        for i in range(self.n):
            v=[]
            for k,b in enumerate(self.f.outwidths):
                x=(w>>(self.n*self.f.cell+i*self.f.out+self.out_offsets[k]))&((1<<b)-1)
                v.append(signed(x,b))
            rows.append(v)
        return np.array(rows,dtype=np.int64).reshape(*self.solid.shape,3)
    def apply(self,w,inverse=False):
        for typ,val in (reversed(self.blocks) if inverse else self.blocks):
            if typ=='g':
                for m,p,t in (reversed(val) if inverse else val): w=gate(w,m,p,t)
            else:
                sign,bits,mask=val; key=gather(w,bits); value=int(table(self.f,sign)[key]); p=w&mask
                while value:
                    low=value&-value; w=gate(w,mask,p,self.scratch+low.bit_length()-1);value^=low
        return w
    def step(self,s,verify_inverse=False):
        w=self.encode(s); z=self.apply(w)
        if z&((1<<(self.n*self.f.cell))-1)!=w or z>>self.scratch: raise RuntimeError('register cleanup failed')
        if verify_inverse and self.apply(z,True)!=w: raise RuntimeError('inverse failed')
        out=self.decode(z); self.encode(out) # reject, never clip unsupported physical state
        return out
    def gate_count(self):
        return sum(len(v) if t=='g' else oracle_count(self.f,v[0]) for t,v in self.blocks)
    def iter_gates(self):
        for typ,val in self.blocks:
            if typ=='g': yield from val
            else:
                sign,bits,mask=val; lut=table(self.f,sign)
                for p in np.flatnonzero(lut):
                    p=int(p); v=int(lut[p]); pat=sum(1<<b for j,b in enumerate(bits) if p>>j&1)
                    while v:
                        low=v&-v;yield mask,pat,self.scratch+low.bit_length()-1;v^=low
    def qiskit(self,max_gates=1000000):
        if self.gate_count()>max_gates: raise ValueError('logical gate export exceeds safety cap; use a smaller test circuit')
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for mask,pattern,target in self.iter_gates():
            controls=[i for i in range(self.nqubits) if mask>>i&1]; zeros=[i for i in controls if not pattern>>i&1]
            for i in zeros:qc.x(i)
            if controls:qc.mcx(controls,target)
            else:qc.x(target)
            for i in zeros:qc.x(i)
        return qc
