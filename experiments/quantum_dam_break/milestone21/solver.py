"""Velocity-based D3Q19 pressure LBM + bounded D3Q7 phase: M21 preflight.
Fakhari et al. PRE 96 053301 (2017), adapted to 3D, explicit FD viscous force,
M20 limiter, signed amplitude transport. Not a reproduction of their full MRT model.
"""
import numpy as np
from milestone18.transport import C19,C7,W19,W7,LinkTransport
from milestone19.collision import M19,M7,collide
from milestone19.solver import push_reference,initial
from milestone20.limiter import limit,scalar_reference
CS2=1/3

def weights(w):return w[:,None,None,None]
def dot(c,u):return np.einsum('id,dzyx->izyx',c,u)
def moment(c,f):return np.einsum('id,izyx->dzyx',c,f)
def derivatives(a):
    """Isotropic D3Q19 stencil, zero normal derivative ghost extension."""
    pad=np.pad(a,1,mode='edge');n=a.shape
    vals=np.stack([pad[tuple(slice(1+d,1+d+s) for d,s in zip(v[::-1],n))] for v in C19])
    return moment(C19,weights(W19)*vals)/CS2,2*(np.einsum('i,izyx->zyx',W19,vals)-a)/CS2

def materials(alpha,ratio=998/1.2,mu_ratio=.001/.000018,nu_liquid=.02):
    if not np.isfinite(alpha).all() or np.min(alpha)<-1e-10 or np.max(alpha)>1+1e-10:raise ValueError('inadmissible liquid fraction')
    if ratio<1 or mu_ratio<=0 or nu_liquid<=0:raise ValueError('invalid materials')
    rho=1/ratio+(1-1/ratio)*alpha
    mu=nu_liquid*(1/mu_ratio+(1-1/mu_ratio)*alpha)
    return rho,mu,mu/rho

def equilibrium(P,u):
    cu=dot(C19,u)
    return weights(W19)*(P+3*cu+4.5*cu**2-1.5*(u*u).sum(0))

def rail_variable(delta,lam,bound=.25):
    """Batched actual 4-qubit CRY, CX, rail SWAP gate actions with per-cell angles.
    Exact expectation; classical parameter preparation/readout. No scalar shortcut.
    """
    d,l=np.broadcast_arrays(delta,lam);shape=d.shape;d=d.ravel();l=l.ravel()
    if not np.isfinite(d).all() or np.max(abs(d),initial=0)>bound or np.max(abs(l),initial=0)>1:raise ValueError('rail range')
    p=abs(d)/bound;psi=np.zeros((16,d.size),complex);psi[0]=np.sqrt(1-p);psi[np.where(d>=0,1,2),np.arange(d.size)]=np.sqrt(p)
    c=np.sqrt(abs(l));s=np.sqrt(1-abs(l))
    for r,e in [(0,2),(1,3)]:
        for k in range(16):
            if (k>>r)&1 and not (k>>e)&1:
                j=k|(1<<e);a,b=psi[k].copy(),psi[j].copy();psi[k]=c*a-s*b;psi[j]=s*a+c*b
        for k in range(16):
            if (k>>e)&1 and not (k>>r)&1:
                j=k|(1<<r);psi[[k,j]]=psi[[j,k]]
    neg=l<0
    for k in range(16):
        if k&3==1:
            a=psi[k,neg].copy();psi[k,neg]=psi[k+1,neg];psi[k+1,neg]=a
    pr=(abs(psi)**2).reshape(4,4,-1).sum(0)
    return (bound*(pr[1]-pr[2])).reshape(shape)

def hydro_collision(f,eq,src,rates,mode):
    if mode=='reference':
        # Independent population-space application of the variable MRT operator.
        diff=np.einsum('ij,jzyx->izyx',M19,eq-f-src/2)
        return f+np.einsum('ji,jzyx->izyx',M19,rates*diff)+src
    m=np.einsum('ij,jzyx->izyx',M19,f);e=np.einsum('ij,jzyx->izyx',M19,eq);s=np.einsum('ij,jzyx->izyx',M19,src)
    out=m.copy();out[4:]=e[4:]+rail_variable((m-e)[4:],1-rates[4:]);out+=(1-rates/2)*s
    return np.einsum('ji,jzyx->izyx',M19,out)

class PressureDamBreak:
    def __init__(self,shape=(8,2,16),ratio=998/1.2,mu_ratio=.001/.000018,nu_liquid=.02,gravity=1e-5,sigma=0.,xi=3.,tau_phase=.95,pressure_force='product'):
        self.shape=shape;self.ratio=ratio;self.mu_ratio=mu_ratio;self.nu_liquid=nu_liquid;self.gravity=gravity;self.sigma=sigma;self.xi=xi;self.tau_phase=tau_phase;self.pressure_force=pressure_force
        self.t=None
    def mat(self,a):return materials(a,self.ratio,self.mu_ratio,self.nu_liquid)
    def decode(self,f,h):
        a=h.sum(0);rho,mu,nu=self.mat(a);P=f.sum(0);p=rho*CS2*P
        grad_rho,_=derivatives(rho);grad_a,lap=derivatives(a)
        if self.pressure_force=='product':
            gradP,_=derivatives(P);gradp,_=derivatives(p);acc=CS2*gradP-gradp/rho
        elif self.pressure_force=='paper':acc=-CS2*P*grad_rho/rho
        else:raise ValueError('pressure force')
        chem=48*self.sigma/self.xi*a*(a-1)*(a-.5)-1.5*self.sigma*self.xi*lap
        acc+=chem*grad_a/rho;acc[2]-=self.gravity
        # Explicit FD viscous density-gradient correction, using force-corrected predictor.
        pred=moment(C19,f)+acc/2
        grad_u=np.stack([derivatives(ui)[0] for ui in pred])
        acc+=nu*np.einsum('ijzyx,jzyx->izyx',grad_u+grad_u.swapaxes(0,1),grad_rho)/rho
        u=moment(C19,f)+acc/2
        return a,rho,mu,nu,p,u,acc,grad_a
    def initialize(self,a,p):
        rho,_,_=self.mat(a);h=weights(W7)*a;f=weights(W19)*(p/(rho*CS2))
        # Initialize transformed populations with half-force shift (predictor iterations).
        for _ in range(3):
            acc=self.decode(f,h)[6];f=weights(W19)*(p/(rho*CS2))-weights(W19)*dot(C19,acc)/(2*CS2)
        return f,h
    def step(self,f,h,mode='gate',freeze=False):
        a,rho,mu,nu,p,u,acc,grad=self.decode(f,h)
        rates=np.ones((19,)+self.shape);rates[:4]=0;rates[5:10]=1/(.5+nu/CS2);rates[10:]=1.2
        eq=equilibrium(f.sum(0),u);src=weights(W19)*dot(C19,acc)/CS2
        ff=hydro_collision(f,eq,src,rates,mode)
        if freeze:hh=h
        else:
            mag=np.linalg.norm(grad,axis=0);normal=np.divide(grad,mag,out=np.zeros_like(grad),where=mag>1e-10)
            heq=weights(W7)*a*(1+4*dot(C7,u));source=weights(W7)*dot(C7,normal)*4*a*(1-a)/self.xi
            hh=collide(h,heq,source,M7,np.array([0.]+[1/self.tau_phase]*6),mode,bound=.25)
            signed=2*hh-weights(W7)
            corrected=limit(signed)[0] if mode=='gate' else scalar_reference(signed)[0]
            hh=(corrected+weights(W7))/2
        if mode=='gate':
            if self.t is None:self.t=[LinkTransport(np.zeros(self.shape,bool),c,periodic=False) for c in (C19,C7)]
            def stream(tr,v):
                s,n=tr.encode(v);return (tr.apply(s,direct=True)*n)[:v.size].real.reshape(v.shape)
            return stream(self.t[0],ff),h.copy() if freeze else stream(self.t[1],hh)
        return push_reference(ff,C19),h.copy() if freeze else push_reference(hh,C7)

def hydrostatic(a,rho,g):
    """Cell-centred pressure from top boundary p=0, piecewise constant density."""
    return g*(np.cumsum(rho[::-1],axis=0)[::-1]-.5*rho)
