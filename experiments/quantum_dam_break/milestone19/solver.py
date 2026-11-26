"""Closed-tank small-density-contrast two-phase dam release, D3Q19/D3Q7.
Not water/air at physical density ratio. All units lattice/nondimensional.
"""
import numpy as np
from milestone18.transport import C19,C7,W19,W7,LinkTransport,opposites
from milestone19.collision import M19,M7,collide


def derivatives(p):
    # Neumann ghost extension at six outer walls, cell-centred.
    pp=np.pad(p,1,mode='edge');z,y,x=p.shape
    a=[pp[2:,1:-1,1:-1],pp[:-2,1:-1,1:-1],pp[1:-1,2:,1:-1],pp[1:-1,:-2,1:-1],pp[1:-1,1:-1,2:],pp[1:-1,1:-1,:-2]]
    return np.stack([(a[4]-a[5])/2,(a[2]-a[3])/2,(a[0]-a[1])/2]),sum(a)-6*p


def push_reference(f,c):
    """Independent slice-based push, closed stationary halfway bounce-back."""
    out=np.zeros_like(f);opp=opposites(c);shape=f.shape[1:]
    for i,v in enumerate(c):
        shift=v[::-1];src=tuple(slice(max(0,-d),min(n,n-d)) for n,d in zip(shape,shift));dst=tuple(slice(max(0,d),min(n,n+d)) for n,d in zip(shape,shift))
        out[(i,)+dst]+=f[(i,)+src];blocked=np.ones(shape,bool);blocked[src]=False
        out[opp[i],blocked]+=f[i,blocked]
    return out


def initial(shape):
    z,y,x=np.indices(shape,dtype=float);nx=shape[2];nz=shape[0]
    # Rounded rectangular heavy-fluid column at left and floor, no initial gate cells.
    p=-np.tanh(np.maximum(x+.5-nx*.33,z+.5-nz*.65)/1.5)
    f=W19[:,None,None,None]*np.ones(shape);g=W7[:,None,None,None]*p
    return f,g


def force_field(p,gprime,sigma,xi):
    grad,lap=derivatives(p);pc=np.clip(p,-1,1)
    mu=3*sigma/(2*xi)*(pc**3-pc)-3*sigma*xi/4*lap
    force=mu[None]*grad;force[2]-=gprime*(pc+1)/2
    return force,grad


def fields(f,g,gprime=.008,sigma=.003,xi=1.50):
    p=g.sum(0);rho=f.sum(0);force,_=force_field(p,gprime,sigma,xi)
    mom=np.einsum('id,izyx->dzyx',C19,f)
    if np.min(rho)<=0:raise RuntimeError('nonpositive density')
    return p,rho,(mom+force/2)/rho[None],force


class DamBreak:
    def __init__(self,shape=(16,4,32),gprime=.008,sigma=.003,xi=1.5,tau=.85,tau_phase=.95):
        self.shape=shape;self.gprime=gprime;self.sigma=sigma;self.xi=xi;self.tau=tau;self.tau_phase=tau_phase
        solid=np.zeros(shape,bool);self.t=[LinkTransport(solid,c,periodic=False) for c in (C19,C7)]
        # M19 rows 4 bulk, 5:10 deviatoric stresses. Remaining kinetic modes rate 1.2.
        self.rf=np.array([0.]*4+[1.0]+[1/tau]*5+[1.2]*9)
        self.rg=np.array([0.]+[1/tau_phase]*6)
    def equilibria(self,f,g):
        p,rho,u,force=fields(f,g,self.gprime,self.sigma,self.xi);_,grad=force_field(p,self.gprime,self.sigma,self.xi)
        cu=np.einsum('id,dzyx->izyx',C19,u);cf=np.einsum('id,dzyx->izyx',C19,force);uf=(u*force).sum(0)
        feq=W19[:,None,None,None]*rho*(1+3*cu+4.5*cu**2-1.5*(u*u).sum(0))
        sf=W19[:,None,None,None]*(3*(cf-uf)+9*cu*cf)
        norm=np.linalg.norm(grad,axis=0);normal=np.divide(grad,norm[None],out=np.zeros_like(grad),where=norm[None]>1e-8)
        cu7=np.einsum('id,dzyx->izyx',C7,u);cn=np.einsum('id,dzyx->izyx',C7,normal)
        geq=W7[:,None,None,None]*p*(1+4*cu7)
        sg=W7[:,None,None,None]*cn*(1-np.clip(p,-1,1)**2)/self.xi
        return feq,geq,sf,sg,force
    def step(self,f,g,mode='gate',shots=None,rng=None):
        feq,geq,sf,sg,force=self.equilibria(f,g)
        if mode=='gate':
            ff=collide(f,feq,sf,M19,self.rf,mode,shots,rng,bound=.25);gg=collide(g,geq,sg,M7,self.rg,mode,shots,rng,bound=.25)
        elif mode=='reference':
            # Separate population-space matrix algebra, not the circuit readout path.
            def classical(a,e,b,M,rates):
                A=M.T@np.diag(rates)@M;B=np.eye(len(rates))-.5*A
                return a+np.einsum('ij,jzyx->izyx',A,e-a)+np.einsum('ij,jzyx->izyx',B,b)
            ff=classical(f,feq,sf,M19,self.rf);gg=classical(g,geq,sg,M7,self.rg)
        else:raise ValueError(mode)
        # Conserved momentum update is the full force (not half-force).
        mass=max(np.max(np.abs(ff.sum(0)-f.sum(0))),np.max(np.abs(gg.sum(0)-g.sum(0))))
        mom=np.max(np.abs(np.einsum('id,izyx->dzyx',C19,ff-f)-force))
        if mode=='gate':out=[]
        else:out=[push_reference(ff,C19),push_reference(gg,C7)]
        if mode=='gate':
            for tr,arr in zip(self.t,(ff,gg)):
                state,norm=tr.encode(arr);out.append((tr.apply(state,direct=True)*norm)[:arr.size].real.reshape(arr.shape))
        # Global momentum balance includes impulses from closed-wall reflections.
        ids=self.t[0].wall;direction=ids//np.prod(self.shape)
        impulse=-2*np.sum(C19[direction]*ff.ravel()[ids][:,None],axis=0)
        before=np.einsum('id,izyx->d',C19,f);after=np.einsum('id,izyx->d',C19,out[0]);external=force.sum(axis=(1,2,3))
        balance=float(np.max(np.abs(after-before-external-impulse)))
        return *out,float(mass),float(mom),balance
