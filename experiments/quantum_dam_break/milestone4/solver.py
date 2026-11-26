"""Reversible, table-compiled 1D shallow-water finite-volume step.
All runtime evolution in step_word uses controlled-X gates. Tables are compiled
OFFLINE, not evaluated as a classical flux during execution. This is a small,
expensive computational-basis quantum circuit prototype, not an efficient QLBM.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import sqrt

HB,QB=3,4
CELL=HB+QB
OH,OQ=5,6
OUT=OH+OQ
FB=6
SCALE=4
LAMBDA=F(1,4)
A=F(5,2)

def signed(v,n): return v-(1<<n) if v>>(n-1) else v

def pack(h,q): return h | ((q&15)<<HB)
def unpack(w): return w&7,signed((w>>HB)&15,QB)
def admissible(h,q): return 0<=h<=7 and -8<=q<=7 and abs(q)<=h

def rnd(x): return (1 if x>=0 else -1)*int(abs(x)+F(1,2))

def compile_flux_value(l,r):
    """Offline exact-rational flux compiler. Values in quarter-unit words.
    g=1, dx=1, dt=1/4. Rusanov a=2.5. Each face can export <= floor(h/2).
    Invalid encodings get zero flux (unitary extension outside physical domain).
    """
    h,q=l; H,Q=r
    if not admissible(h,q) or not admissible(H,Q): return 0,0
    def stress(h,q): return F(q*q,h)+F(h*h,8) if h else F(0)
    mass=rnd(LAMBDA*(F(q+Q,2)-A*F(H-h,2)))
    mass=max(-(H//2),min(h//2,mass))
    mom=rnd(LAMBDA*( (stress(h,q)+stress(H,Q))/2-A*F(Q-q,2)))
    return mass,mom

@lru_cache(None)
def oracle_patterns(kind):
    """Return input-pattern -> target bits of fixed, mutually exclusive MCX gates."""
    n=2*CELL if kind=='face' else CELL
    patterns=[]
    for w in range(1<<n):
        l=unpack(w&127)
        if kind=='face': r=unpack(w>>CELL)
        elif kind=='left': l,r=(l[0],-l[1]),l
        elif kind=='right': r=(l[0],-l[1])
        else: raise ValueError(kind)
        m,p=compile_flux_value(l,r)
        assert -(1<<(FB-1))<=m<(1<<(FB-1)) and -(1<<(FB-1))<=p<(1<<(FB-1))
        val=(m&63)|((p&63)<<FB)
        patterns.append(tuple(j for j in range(2*FB) if val>>j&1))
    return tuple(patterns)

def gate(word,mask,pattern,target):
    return word^(1<<target) if word&mask==pattern else word

def copy_gates(src,dst,width,extend=0,signed_copy=False):
    for j in range(width): yield 1<<(src+j),1<<(src+j),dst+j
    if signed_copy:
        for j in range(width,width+extend): yield 1<<(src+width-1),1<<(src+width-1),dst+j

def add_gates(src,sw,dst,dw,subtract=False):
    """Signed modular add of scratch into accumulator via controlled increments.
    Sign extension gives correct two's-complement arithmetic. Reversal subtracts.
    """
    g=[]
    for j in range(dw):
        s=src+min(j,sw-1)
        for k in range(dw-1,j-1,-1):
            carry=((1<<(k-j))-1)<<(dst+j)
            mask=(1<<s)|carry
            g.append((mask,mask,dst+k))
    return tuple(reversed(g)) if subtract else tuple(g)

def gather(word,bits): return sum(((word>>b)&1)<<i for i,b in enumerate(bits))

class StepCircuit:
    def __init__(self,n):
        if n<2: raise ValueError('need >=2 cells')
        self.n=n; self.scratch=n*(CELL+OUT); self.nqubits=self.scratch+2*FB
        self.blocks=[]
        for i in range(n):
            x=i*CELL; y=n*CELL+i*OUT
            self.blocks.append(('gates',tuple(copy_gates(x,y,HB))+tuple(copy_gates(x+HB,y+OH,QB,OQ-QB,True))))
        for face in range(n+1):
            if face==0: kind='left'; ix=[0]; targets=[(0,False)]
            elif face==n: kind='right'; ix=[n-1]; targets=[(n-1,True)]
            else: kind='face'; ix=[face-1,face]; targets=[(face-1,True),(face,False)]
            bits=tuple(i*CELL+j for i in ix for j in range(CELL))
            self.blocks.append(('oracle',(kind,bits)))
            for i,sub in targets:
                y=n*CELL+i*OUT
                self.blocks.append(('gates',add_gates(self.scratch,FB,y,OH,sub)+add_gates(self.scratch+FB,FB,y+OH,OQ,sub)))
            self.blocks.append(('oracle',(kind,bits)))
    def encode(self,cells):
        if len(cells)!=self.n or not all(admissible(h,q) for h,q in cells): raise ValueError('inadmissible input (range or |u|>1)')
        return sum(pack(h,q)<<(i*CELL) for i,(h,q) in enumerate(cells))
    def decode(self,word):
        return [(signed((word>>(self.n*CELL+i*OUT))&31,OH),signed((word>>(self.n*CELL+i*OUT+OH))&63,OQ)) for i in range(self.n)]
    def iter_gates(self):
        """Explicit entire gate stream (including all zero/one control patterns)."""
        for typ,val in self.blocks:
            if typ=='gates': yield from val
            else:
                kind,bits=val; mask=sum(1<<b for b in bits)
                for p,js in enumerate(oracle_patterns(kind)):
                    pat=sum(1<<b for i,b in enumerate(bits) if p>>i&1)
                    for j in js: yield mask,pat,self.scratch+j
    def apply(self,word,inverse=False):
        """Exact basis-gate simulator; index mutually exclusive immutable controls.
        Index selects which MCX gates can fire, NOT an evaluation of physical flux.
        """
        for typ,val in (reversed(self.blocks) if inverse else self.blocks):
            if typ=='gates':
                for m,p,t in (reversed(val) if inverse else val): word=gate(word,m,p,t)
            else:
                kind,bits=val; pat=gather(word,bits); mask=sum(1<<b for b in bits)
                p=word&mask
                for j in oracle_patterns(kind)[pat]: word=gate(word,mask,p,self.scratch+j)
        return word
    def step(self,cells):
        w=self.encode(cells); z=self.apply(w); assert z&((1<<(self.n*CELL))-1)==w
        assert z>>self.scratch==0
        out=self.decode(z)
        if not all(admissible(h,q) for h,q in out): raise OverflowError(f'next state outside supported encoding/domain: {out}')
        return out
    def gate_count(self):
        return sum(len(v) if t=='gates' else sum(map(len,oracle_patterns(v[0]))) for t,v in self.blocks)
    def qiskit(self):
        """Optional native circuit export; untranspiled, potentially large."""
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for mask,pattern,target in self.iter_gates():
            cs=[i for i in range(self.nqubits) if mask>>i&1]; zeros=[i for i in cs if not pattern>>i&1]
            for i in zeros: qc.x(i)
            if cs: qc.mcx(cs,target)
            else: qc.x(target)
            for i in zeros: qc.x(i)
        return qc
