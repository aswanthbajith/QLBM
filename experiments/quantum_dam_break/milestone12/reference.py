"""Independent integer implementation for checking the compiled circuit."""
import numpy as np
from milestone11 import reference as r
V=r.V

def greedy(y,x):
    col=list(x);out=[]
    for a in y:
        for j,c in enumerate(col):
            z=min(a,c);out.append(z);a-=z;col[j]-=z
    return np.array(out,dtype=np.int64)

def project(p,axes):
    a=p.reshape(5,5).copy()
    for axis in axes:
        z=a if axis==0 else a.T
        for i,j in ((0,4),(1,3)):
            z[2]+=abs(z[i]-z[j]);z[i]=z[j]=np.minimum(z[i],z[j])
    return a.ravel()

def collision(f,S,w,omega_num=2,symmetry=True):
    f=np.asarray(f,dtype=np.int64);p,bad=r.collision(f,S,w)
    if bad:return f.copy(),True
    if omega_num!=4:
        p=((4-omega_num)*f+omega_num*p)//4
        R=int(f.sum()-p.sum());J=(f-p)@V
        p+=greedy(r.adjacent(R,int(J[1])),r.adjacent(R,int(J[0])))
    if symmetry:
        a=f.reshape(5,5);p=project(p,[d for d in (0,1) if np.array_equal(a,np.flip(a,d))])
    return p,False

def initialize(h,S,w,ux=None,uy=None):
    ux=np.zeros_like(h) if ux is None else ux;uy=np.zeros_like(h) if uy is None else uy
    f=np.zeros((*h.shape,25),np.int64)
    for idx in np.ndindex(h.shape):
        H=int(round(h[idx]*S));x=r.marginal(H,round(2*H*ux[idx]),S,w);y=r.marginal(H,round(2*H*uy[idx]),S,w);p=r.couple(y,x)
        f[idx]=project(p,[d for d,u in ((0,uy[idx]),(1,ux[idx])) if u==0])
    return f
