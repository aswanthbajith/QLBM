"""Independent floating-point finite-volume baseline; never called by solver."""
import numpy as np

def face(l,r,quantized=False):
    h,q=l; H,Q=r
    f=lambda h,q: np.array([q,q*q/h+.5*h*h]) if h>0 else np.zeros(2)
    d=.25*(.5*(f(h,q)+f(H,Q))-1.25*(np.array(r)-l))
    if quantized:
        d=np.sign(d)*np.floor(abs(d)*4+.5+1e-12)/4
        d[0]=np.clip(d[0],-np.floor(H*2)/4,np.floor(h*2)/4)
    return d

def step(s,quantized=False):
    s=np.asarray(s,float); left=s[0]*[1,-1]; right=s[-1]*[1,-1]
    f=np.array([face(a,b,quantized) for a,b in zip(np.vstack((left,s)),np.vstack((s,right)))])
    return s+f[:-1]-f[1:]
