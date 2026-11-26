"""Independent small hybrid CAC/D3Q19 experiment, based on audited branch equations.
Classical collision/forcing and exact amplitude readout, quantum gate transport only.
Matched density, neutral wall extension for derivative; not a validated waterfall.
"""
import numpy as np
from milestone18.transport import C7,C19,W7,W19,LinkTransport,reference_stream,legacy_stream

def neighbors(phi,solid):
    vals=[]
    for a in range(3):
        for d in (-1,1):
            q=np.roll(phi,d,axis=a);mask=np.roll(solid,d,axis=a);vals.append(np.where(mask,phi,q))
    return vals

def init(solid):
    z,y,x=np.indices(solid.shape);p=.65*np.cos(2*np.pi*x/solid.shape[2])+.15*np.sin(2*np.pi*z/solid.shape[0]);p=np.where(solid,0,p)
    f=W19[:,None,None,None]*np.broadcast_to((~solid).astype(float),solid.shape)
    return f,W7[:,None,None,None]*p

def collision(f,g,solid,tau=.8,sigma=.01,xi=1.2):
    p=g.sum(0);near=neighbors(p,solid);lap=sum(near)-6*p
    grad=np.stack([(near[2*a]-near[2*a+1])/2 for a in (2,1,0)])
    pc=np.clip(p,-1,1);mu=3*sigma/(2*xi)*(pc**3-pc)-3*sigma*xi/4*lap
    force=mu[None]*grad;force[:,solid]=0
    rho=f.sum(0);mom=np.einsum('id,izyx->dzyx',C19,f)+force/2
    u=np.divide(mom,rho[None],out=np.zeros_like(mom),where=(rho[None]>1e-12)&(~solid[None]));cu=np.einsum('id,dzyx->izyx',C19,u);cf=np.einsum('id,dzyx->izyx',C19,force);uf=(u*force).sum(0);cs=1/3
    feq=W19[:,None,None,None]*rho*(1+cu/cs+cu**2/(2*cs**2)-(u*u).sum(0)/(2*cs))
    ff=f-(f-feq)/tau+(1-.5/tau)*W19[:,None,None,None]*((cf-uf)/cs+cu*cf/cs**2)
    norm=np.linalg.norm(grad,axis=0);normal=np.divide(grad,norm[None],out=np.zeros_like(grad),where=norm[None]>1e-4);cu7=np.einsum('id,dzyx->izyx',C7,u);cn=np.einsum('id,dzyx->izyx',C7,normal)
    geq=W7[:,None,None,None]*p*(1+cu7/.25);source=(1-.5/tau)*W7[:,None,None,None]*cn*(1-pc**2)/(2*xi*.25)
    gg=g-(g-geq)/tau+source
    ff[:,solid]=0;gg[:,solid]=0
    return ff,gg

def step(f,g,solid,mode,transports=None):
    ff,gg=collision(f,g,solid)
    if mode=='gate':return transports[0].stream(ff),transports[1].stream(gg)
    if mode=='reference':return reference_stream(ff,solid,C19),reference_stream(gg,solid,C7)
    if mode=='legacy':return legacy_stream(ff,solid,C19),legacy_stream(gg,solid,C7)
    raise ValueError(mode)
