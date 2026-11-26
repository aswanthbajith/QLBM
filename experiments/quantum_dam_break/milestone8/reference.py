"""Independent floating and integer D2Q9 shallow-water BGK references."""
import numpy as np
from quantum_core import C,OPP
G=1/8
FREE=(3,4,5,6,7,8)
def equilibrium(h,u):
    h=np.asarray(h);u=np.asarray(u);d=u@C.T;u2=np.sum(u*u,axis=-1)
    w=np.array([0]+[1/9]*4+[1/36]*4)
    f=h[...,None]*w*(1.5*G*h[...,None]+3*d+4.5*d*d-1.5*u2[...,None])
    f[...,0]=h-5*G*h*h/6-2*h*u2/3
    return f

def moments(f):
    h=f.sum(axis=-1);q=f@C;u=np.divide(q,h[...,None],out=np.zeros_like(q,dtype=float),where=h[...,None]!=0)
    return h,u

def rnd(n,d):return (1 if n>=0 else -1)*((2*abs(n)+d)//(2*d))
def integer_collision(f,S,omega_num):
    f=list(map(int,f));H=sum(f);X=sum(int(c[0])*v for c,v in zip(C,f));Y=sum(int(c[1])*v for c,v in zip(C,f))
    if H==0:return f,False
    a=[0]*9
    for i in FREE:
        D=int(C[i,0])*X+int(C[i,1])*Y
        n=H**3+8*S*(2*D*H+3*D*D-X*X-Y*Y)
        d=48*S*H*(1 if i<5 else 4)
        e=rnd(n,d);a[i]=f[i]+rnd(omega_num*(e-f[i]),2)
    a[1]=X-sum(int(C[i,0])*a[i] for i in FREE);a[2]=Y-sum(int(C[i,1])*a[i] for i in FREE);a[0]=H-sum(a[1:])
    bad=min(a)<0 or max(a)>=2*S
    return (f if bad else a),bad

def initialize(h,u,S):
    # Initial preparation only; enforce exact integer depth and rounded momentum.
    h=np.asarray(h);u=np.asarray(u);f=equilibrium(h,u);a=np.rint(S*f).astype(np.int64)
    H=np.rint(S*h).astype(np.int64);Q=np.rint(S*h[...,None]*u).astype(np.int64)
    a[...,1]=Q[...,0]-sum(int(C[i,0])*a[...,i] for i in FREE)
    a[...,2]=Q[...,1]-sum(int(C[i,1])*a[...,i] for i in FREE)
    a[...,0]=H-a[...,1:].sum(axis=-1)
    if np.any(a<0) or np.any(a>=2*S):raise ValueError('initial equilibrium not representable')
    return a

def stream(f,solid,periodic=False):
    ny,nx=solid.shape;out=np.zeros_like(f)
    for y in range(ny):
        for x in range(nx):
            for i,(cx,cy) in enumerate(C):
                if solid[y,x]:out[y,x,i]=f[y,x,i];continue
                xx=x+cx;yy=y+cy
                if periodic:xx%=nx;yy%=ny
                if not(0<=xx<nx and 0<=yy<ny) or solid[yy,xx]:out[y,x,OPP[i]]=f[y,x,i]
                else:out[yy,xx,i]=f[y,x,i]
    return out

def float_step(f,solid,omega,periodic=False):
    h,u=moments(f);e=equilibrium(h,u);a=f+omega*(e-f);a[solid]=0
    return stream(a,solid,periodic)

def integer_step(f,solid,S,omega_num,periodic=False):
    a=np.zeros_like(f);bad=0
    for y,x in np.argwhere(~solid):
        a[y,x],flag=integer_collision(f[y,x],S,omega_num);bad+=int(flag)
    return stream(a,solid,periodic),bad
