"""D2Q9 probability encoding and exact small-system quantum channel simulator.
No nonlinear fluid solver is implemented here. Qiskit is optional.
"""
from dataclasses import dataclass
import numpy as np

C = np.array([(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)])
W = np.array([4/9]+[1/9]*4+[1/36]*4)
OPP = np.array([0,3,4,1,2,7,8,5,6])

@dataclass
class Grid:
    nx: int = 4
    ny: int = 4
    dx: float = 1.0
    dt: float = 1.0
    solid: np.ndarray | None = None
    def __post_init__(self):
        if any(n < 2 or n & (n-1) for n in (self.nx,self.ny)):
            raise ValueError('Grid dimensions must be powers of two >= 2')
        if self.dx <= 0 or self.dt <= 0: raise ValueError('dx and dt must be positive')
        self.solid = np.zeros((self.ny,self.nx),bool) if self.solid is None else np.asarray(self.solid,dtype=bool)
        if self.solid.shape != (self.ny,self.nx): raise ValueError('Invalid solid mask')
    @property
    def npos(self): return self.nx*self.ny
    @property
    def dim(self): return 16*self.npos
    @property
    def nq(self): return self.dim.bit_length()-1
    def index(self,x,y,i): return x+self.nx*y+self.npos*i


def encode(g, f):
    """f[y,x,i] nonnegative depth populations [m]; returns normalized ket and sum [m]."""
    f=np.asarray(f,float)
    if f.shape != (g.ny,g.nx,9) or not np.isfinite(f).all() or np.any(f<0) or np.any(f[g.solid]!=0):
        raise ValueError('Invalid populations')
    M=float(f.sum())
    if M<=0: raise ValueError('No wet population')
    a=np.zeros(g.dim,complex); a[:9*g.npos]=np.sqrt(f.transpose(2,0,1).ravel()/M)
    return a,M


def decode(g,p,M):
    p=np.asarray(p,float)
    if p.shape!=(g.dim,) or np.any(p < -1e-12) or not np.isclose(p.sum(),1) or np.sum(p[9*g.npos:])>1e-10:
        raise ValueError('Invalid quantum probability distribution')
    f=(p[:9*g.npos]*M).reshape(9,g.ny,g.nx).transpose(1,2,0)
    h=f.sum(axis=2)
    flux=f@C*(g.dx/g.dt)
    u=np.divide(flux,h[...,None],out=np.zeros_like(flux),where=h[...,None]>1e-12)
    return f,h,u


def probabilities(state):
    return abs(state)**2 if state.ndim==1 else np.real(np.diag(state))


def stream_permutation(g, periodic=False):
    """Closed link bounce-back or periodic transport; a bijective basis map.
    Fluid links hitting solid or outside closed domain reverse at source.
    Padded velocity codes and solid states are fixed.
    """
    p=np.arange(g.dim)
    for i,(cx,cy) in enumerate(C):
        for y in range(g.ny):
            for x in range(g.nx):
                if g.solid[y,x]: continue
                a,b=x+cx,y+cy
                outside=not(0<=a<g.nx and 0<=b<g.ny)
                if periodic: a,b=a%g.nx,b%g.ny; outside=False
                dest=(x,y,OPP[i]) if outside or g.solid[b,a] else (a,b,i)
                p[g.index(x,y,i)]=g.index(*dest)
    if not np.array_equal(np.sort(p),np.arange(g.dim)): raise AssertionError('Nonunitary map')
    return p


def transpositions(p):
    seen=set()
    for start in range(len(p)):
        if start in seen: continue
        cycle=[]; x=start
        while x not in seen:
            seen.add(x); cycle.append(x); x=int(p[x])
        for x in cycle[1:]: yield start,x


def gate_pairs(p):
    """Basis transpositions -> Gray path fully-controlled X gates.
    Each returned pair differs in exactly one bit: one MCX with open controls.
    """
    for a,b in transpositions(p):
        path=[a]; x=a; diff=a^b
        while diff:
            bit=diff & -diff; x^=bit; path.append(x); diff^=bit
        links=list(zip(path[:-1],path[1:]))
        yield from links; yield from links[-2::-1]


def apply_gates(a,p):
    out=a.copy()
    for x,y in gate_pairs(p): out[x],out[y]=out[y],out[x]
    return out


def apply_permutation(state,p):
    # Exact action of the same compiled unitary, not a fluid evolution formula.
    inv=np.argsort(p)
    return state[inv] if state.ndim==1 else state[np.ix_(inv,inv)]


def relaxation_channel(g,rho,omega):
    """Exact reduced density matrix of a quantum circuit:
    coin Ry(2 asin sqrt(omega)); fresh velocity |sqrt(W)>;
    coin-controlled velocity SWAP; discard coin and fresh register.
    Fixed zero-drift velocity reset, NOT momentum-conserving BGK.
    """
    if not 0<=omega<=1: raise ValueError('omega must be in [0,1]')
    if rho.shape!=(g.dim,g.dim): raise ValueError('Expected density matrix')
    r=rho.reshape(16,g.npos,16,g.npos)
    pos=np.einsum('iaib->ab',r)
    w=np.zeros(16); w[:9]=np.sqrt(W)
    return (1-omega)*rho + omega*np.kron(np.outer(w,w),pos)


def initial(g):
    h=np.zeros((g.ny,g.nx)); h[:,:g.nx//2]=1.0; h[g.solid]=0
    return encode(g,h[...,None]*W)


def counts(g,state,shots,seed):
    p=probabilities(state); p=np.maximum(p,0); p/=p.sum()
    return np.random.default_rng(seed).multinomial(shots,p)
