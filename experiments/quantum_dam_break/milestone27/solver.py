"""M27: finite-width D3Q19/D3Q7 dam release with all six walls closed."""
import numpy as np
from milestone23.solver import FaceDamBreak
from milestone21.solver import CS2,C19,W19,W7,weights,dot,hydrostatic
from milestone18.transport import LinkTransport,C7,opposites,gate_pairs
from milestone19.collision import M7,collide
from milestone20.limiter import limit

class ClosedTransport(LinkTransport):
    """Closed boundary permutation on all axes. No internal solids."""
    def __init__(self,shape,c):
        self.shape=tuple(shape);self.solid=np.zeros(shape,bool);self.c=np.asarray(c,int);self.q=len(c);self.opp=opposites(c)
        if any(n<2 or n&(n-1) for n in shape):raise ValueError('power-of-two grid required')
        self.n=int(np.prod(shape));self.dim=self.n*(1<<(self.q-1).bit_length());self.qubits=self.dim.bit_length()-1
        self.periodic=False;self.links=np.arange(self.dim);self.reverse=np.arange(self.dim);self.wall=[]
        loc=np.indices(shape).reshape(3,-1).T;pos=np.arange(self.n)
        for i,v in enumerate(c):
            dst=loc+v[::-1];valid=np.all((dst>=0)&(dst<shape),axis=1)
            src=i*self.n+pos;self.reverse[src]=self.opp[i]*self.n+pos
            self.links[src[valid]]=self.opp[i]*self.n+np.ravel_multi_index(dst[valid].T,shape)
            self.wall.extend(src[~valid])
        ids=np.arange(self.dim);assert np.array_equal(self.links[self.links],ids) and np.array_equal(self.reverse[self.reverse],ids)
        self.perm=self.reverse[self.links];assert np.array_equal(np.sort(self.perm),ids)
        self.gates=tuple(gate_pairs(self.links))+tuple(gate_pairs(self.reverse));self.direct_gates=tuple(gate_pairs(self.perm));self.wall=np.array(self.wall,int)
        self.invalid=ids>=self.q*self.n

class ThreeDDamBreak(FaceDamBreak):
    """All six walls closed. Both backends share physical equations and vector FCT."""
    def step(self,f,h,mode='gate',freeze=False):
        if mode not in ('gate','reference'):raise ValueError(mode)
        backend=mode
        ff,a,u,grad=self._collide_hydro(self,f,h,backend)
        if self.background is not None:ff=ff-self.reference_collision[backend]
        if not freeze:
            mag=np.linalg.norm(grad,axis=0);normal=np.divide(grad,mag,out=np.zeros_like(grad),where=mag>1e-10)
            heq=weights(W7)*a*(1+4*dot(C7,u));source=weights(W7)*dot(C7,normal)*4*a*(1-a)/self.xi
            hh=collide(h,heq,source,M7,np.array([0.]+[1/self.tau_phase]*6),backend,bound=.25)
            signed=2*hh-weights(W7);hh=((limit(signed)[0])+weights(W7))/2
        else:hh=h.copy()
        if mode=='gate':
            if self.t is None:self.t=[FastClosedTransport(self.shape,c) for c in (C19,C7)]
            def stream(tr,v):
                state,norm=tr.encode(v);return (tr.apply(state,direct=True)*norm)[:v.size].real.reshape(v.shape)
            ff=stream(self.t[0],ff);hh=h.copy() if freeze else stream(self.t[1],hh)
        else:
            from milestone19.solver import push_reference
            ff=push_reference(ff,C19);hh=h.copy() if freeze else push_reference(hh,C7)
        if self.background is not None:ff=ff+self.background[0]
        return ff,hh

# Compile the same ordered basis-pair swaps; no direct population remapping.
from numba import njit
@njit(cache=True)
def execute_pairs(state,pairs,inverse):
    out=state.copy()
    for k in range(len(pairs)):
        a,b=pairs[len(pairs)-1-k if inverse else k]
        tmp=out[a];out[a]=out[b];out[b]=tmp
    return out

class FastClosedTransport(ClosedTransport):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.pairs=np.asarray(self.gates,np.int64).reshape(-1,2);self.direct_pairs=np.asarray(self.direct_gates,np.int64).reshape(-1,2)
    def apply(self,state,inverse=False,direct=False):
        state=np.asarray(state)
        if state.shape!=(self.dim,):raise ValueError('state dimension')
        return execute_pairs(state,self.direct_pairs if direct else self.pairs,inverse)
