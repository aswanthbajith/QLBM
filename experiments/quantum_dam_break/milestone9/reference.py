"""Independent integer oracle reference and diagnostic floating-point limiter."""
import numpy as np
from quantum_core import C
from milestone8 import reference as old
FREE=old.FREE

def target(f,S,k):
    f=list(map(int,f)); H=sum(f); X=sum(int(c[0])*v for c,v in zip(C,f));Y=sum(int(c[1])*v for c,v in zip(C,f));a=f.copy()
    if H:
        for i in FREE:
            D=int(C[i,0])*X+int(C[i,1])*Y
            e=old.rnd(H**3+8*S*(2*D*H+3*D*D-X*X-Y*Y),48*S*H*(1 if i<5 else 4))
            a[i]=f[i]+old.rnd(k*(e-f[i]),2)
    a[1]=X-sum(int(C[i,0])*a[i] for i in FREE);a[2]=Y-sum(int(C[i,1])*a[i] for i in FREE);a[0]=H-sum(a[1:]);return a

def limit(f,a,S):
    f=list(map(int,f));a=list(map(int,a));cap=2*S-1
    if min(a)>=0 and max(a)<=cap:return a,0
    # Sequential conservative corrections along a basis of the moment nullspace.
    # Only apply fallback if the full BGK candidate is infeasible.
    z=f.copy()
    for i in FREE:
        cx,cy=map(int,C[i]); coeff={i:1,1:-cx,2:-cy,0:cx+cy-1}
        d=a[i]-z[i];sign=1 if d>=0 else -1;mag=abs(d)
        for j,k in coeff.items():
            k*=sign
            if k:mag=min(mag,((cap-z[j]) if k>0 else z[j])//abs(k))
        for j,k in coeff.items():z[j]+=k*sign*mag
    return z,1 if z!=f else 2

def collision(f,S,k):return limit(f,target(f,S,k),S)
def step(f,solid,S,k,periodic=False):
    a=np.zeros_like(f);counts=np.zeros(3,int)
    for y,x in np.argwhere(~solid):
        a[y,x],status=collision(f[y,x],S,k);counts[status]+=1
    return old.stream(a,solid,periodic),counts
