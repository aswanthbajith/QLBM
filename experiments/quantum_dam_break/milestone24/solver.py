"""M24: span-periodic D3Q19/D3Q7 dam release, closed x/z walls.
No physical equation is changed from M23; only boundary geometry and horizon.
"""
import numpy as np
from milestone23.solver import FaceDamBreak
from milestone21.solver import CS2,C19,W19,W7,weights,dot,hydrostatic
from milestone18.transport import LinkTransport,C7,opposites,gate_pairs
from milestone19.collision import M7,collide
from milestone20.limiter import limit,scalar_reference

class SpanTransport(LinkTransport):
    """Mixed boundary permutation; periodic y, closed x and z. No internal solids."""
    def __init__(self,shape,c):
        self.shape=tuple(shape);self.solid=np.zeros(shape,bool);self.c=np.asarray(c,int);self.q=len(c);self.opp=opposites(c)
        if any(n<2 or n&(n-1) for n in shape):raise ValueError('power-of-two grid required')
        self.n=int(np.prod(shape));self.dim=self.n*(1<<(self.q-1).bit_length());self.qubits=self.dim.bit_length()-1
        self.periodic='y';self.links=np.arange(self.dim);self.reverse=np.arange(self.dim);self.wall=[]
        loc=np.indices(shape).reshape(3,-1).T;pos=np.arange(self.n)
        for i,v in enumerate(c):
            dst=loc+v[::-1];valid=(dst[:,0]>=0)&(dst[:,0]<shape[0])&(dst[:,2]>=0)&(dst[:,2]<shape[2]);dst[:,1]%=shape[1]
            src=i*self.n+pos;self.reverse[src]=self.opp[i]*self.n+pos
            self.links[src[valid]]=self.opp[i]*self.n+np.ravel_multi_index(dst[valid].T,shape)
            self.wall.extend(src[~valid])
        ids=np.arange(self.dim);assert np.array_equal(self.links[self.links],ids) and np.array_equal(self.reverse[self.reverse],ids)
        self.perm=self.reverse[self.links];assert np.array_equal(np.sort(self.perm),ids)
        self.gates=tuple(gate_pairs(self.links))+tuple(gate_pairs(self.reverse));self.direct_gates=tuple(gate_pairs(self.perm));self.wall=np.array(self.wall,int)
        self.invalid=ids>=self.q*self.n

def span_push(f,c):
    """Independent slicing/rolling reference (not using compiled permutation)."""
    out=np.zeros_like(f);opp=opposites(c);nz,ny,nx=f.shape[1:]
    for i,(cx,cy,cz) in enumerate(c):
        src=(slice(max(0,-cz),min(nz,nz-cz)),slice(None),slice(max(0,-cx),min(nx,nx-cx)))
        dst=(slice(max(0,cz),min(nz,nz+cz)),slice(None),slice(max(0,cx),min(nx,nx+cx)))
        out[(i,)+dst]+=np.roll(f[(i,)+src],cy,axis=1);blocked=np.ones((nz,ny,nx),bool);blocked[src]=False;out[opp[i],blocked]+=f[i,blocked]
    return out

class LongDamBreak(FaceDamBreak):
    def __init__(self,*args,span_periodic=True,**kw):
        super().__init__(*args,**kw);self.span_periodic=span_periodic
    def set_hydrostatic_background(self,a):
        f,h=super().set_hydrostatic_background(a)
        if self.span_periodic:self.defect=f-span_push(self.reference_collision['reference'],C19)
        return f,h
    def step(self,f,h,mode='gate',freeze=False):
        if not self.span_periodic and mode!='fast':return super().step(f,h,mode,freeze)
        if mode not in ('gate','reference','fast'):raise ValueError(mode)
        # Only y-uniform fields supported in this quasi-2D experiment: M20
        # limiter and M23 derivatives need no y-periodic correction in that case.
        if self.span_periodic:
            for field,c in ((f,C19),(h,C7)):
                mirror=c.copy();mirror[:,1]*=-1
                indices=[int(np.flatnonzero(np.all(c==v,axis=1))[0]) for v in mirror]
                if np.max(np.ptp(field,axis=2))>1e-10 or np.max(abs(field-field[indices]))>1e-10:
                    raise ValueError('span-uniform, mirror-symmetric fields required')
        backend='reference' if mode=='fast' else mode
        ff,a,u,grad=self._collide_hydro(self,f,h,backend)
        if self.background is not None:ff=ff-self.reference_collision[backend]
        if not freeze:
            mag=np.linalg.norm(grad,axis=0);normal=np.divide(grad,mag,out=np.zeros_like(grad),where=mag>1e-10)
            heq=weights(W7)*a*(1+4*dot(C7,u));source=weights(W7)*dot(C7,normal)*4*a*(1-a)/self.xi
            hh=collide(h,heq,source,M7,np.array([0.]+[1/self.tau_phase]*6),backend,bound=.25)
            signed=2*hh-weights(W7);hh=((scalar_reference(signed)[0] if mode=='reference' else limit(signed)[0])+weights(W7))/2
        else:hh=h.copy()
        if mode=='gate':
            if self.t is None:self.t=[FastSpanTransport(self.shape,c) for c in (C19,C7)]
            def stream(tr,v):
                state,norm=tr.encode(v);return (tr.apply(state,direct=True)*norm)[:v.size].real.reshape(v.shape)
            ff=stream(self.t[0],ff);hh=h.copy() if freeze else stream(self.t[1],hh)
        else:
            if not self.span_periodic:
                from milestone19.solver import push_reference
                push=push_reference
            else:push=span_push
            ff=push(ff,C19);hh=h.copy() if freeze else push(hh,C7)
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

class FastSpanTransport(SpanTransport):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.pairs=np.asarray(self.gates,np.int64).reshape(-1,2);self.direct_pairs=np.asarray(self.direct_gates,np.int64).reshape(-1,2)
    def apply(self,state,inverse=False,direct=False):
        state=np.asarray(state)
        if state.shape!=(self.dim,):raise ValueError('state dimension')
        return execute_pairs(state,self.direct_pairs if direct else self.pairs,inverse)
