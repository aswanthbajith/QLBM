"""Independent integer and floating D2Q25 references. Not the gate backend."""
import numpy as np
from milestone10.kinetic import equilibrium2d, SETS, G
K=np.arange(-2,3,dtype=np.int64)
V=np.array([(x,y) for y in K for x in K],dtype=np.int64)

def adjacent(H,J):
    a=[0]*5;sgn=1 if J>=0 else -1;j=abs(J)
    if j<=H:a[2]=H-j;a[2+sgn]=j
    else:a[2+sgn]=2*H-j;a[2+2*sgn]=j-H
    return a

def marginal(H,J,S,w):
    H,J=int(H),int(J)
    if not H:return [0]*5
    j=abs(J);lo=adjacent(H,j);n,r=divmod(j+2*H,4);hi=[0]*5;hi[4]=n;hi[0]=H-n-bool(r)
    if r:hi[r]+=1
    p0=sum(k*k*a for k,a in zip(range(-2,3),lo));p1=sum(k*k*a for k,a in zip(range(-2,3),hi));d=4*S*H*(p1-p0);n=4*S*j*j+H**3-4*S*H*p0
    Q=1<<w;alpha=0 if d==0 else min(Q,max(0,n*Q//d));a=[((Q-alpha)*x+alpha*y)//Q for x,y in zip(lo,hi)]
    res=adjacent(H-sum(a),j-sum(k*x for k,x in zip(range(-2,3),a)));a=[x+y for x,y in zip(a,res)]
    return a if J>=0 else a[::-1]

def couple(y,x):
    """Rounded cumulative proportional coupling; exact row and column totals."""
    col=list(x);rows=[]
    for r in y[:-1]:
        total=sum(col);cum=0;prev=0;row=[]
        for c in col:
            cum+=c;z=(2*r*cum+total)//(2*total) if total else 0;row.append(z-prev);prev=z
        col=[c-a for c,a in zip(col,row)];rows.append(row)
    return np.array(rows+[col],dtype=np.int64).ravel()

def collision(f,S,w):
    H=int(sum(f));Jx,Jy=np.asarray(f)@V
    if H>=1<<w:return np.asarray(f).copy(),True
    return couple(marginal(H,Jy,S,w),marginal(H,Jx,S,w)),False

def initialize(h,S,w):
    f=np.zeros((*h.shape,25),np.int64)
    for idx in np.ndindex(h.shape):
        H=int(round(h[idx]*S));p=marginal(H,0,S,w);f[idx]=couple(p,p)
    return f

def moments(f):
    h=f.sum(-1);q=f@V/2;u=np.divide(q,h[...,None],out=np.zeros_like(q,dtype=float),where=h[...,None]>0);return h,q,u

def float_collision(f):
    h,q,u=moments(f);z,bad=equilibrium2d(SETS['D2Q25'],h,u[...,0],u[...,1],project=True);return z.swapaxes(-1,-2).reshape(f.shape),bad
