"""Independent integer calculation of the M13 collision map."""
import numpy as np
from milestone11 import reference as r
from milestone12.reference import greedy
V=r.V

def marginal(H,J,S,w):
    H,J=int(H),int(J);j=abs(J);p=r.marginal(H,j,S,w)
    if H:
        den=4*S*H;t=4*S*j*j+H**3;cur=den*sum(k*k*a for k,a in zip(range(-2,3),p));up=t>=cur
        budget=min(4*H,(abs(t-cur)+den)//(2*den))
        for i in (2,1,3):
            if i!=2 and j==0:continue
            n=min(budget,p[i]//2 if up else min(p[i-1],p[i+1]));n0=n if up else -n
            p[i-1]+=n0;p[i+1]+=n0;p[i]-=2*n0;budget-=n
    return p if J>=0 else p[::-1]

def project(p,axes):
    a=np.asarray(p).reshape(5,5).copy()
    for axis in axes:
        z=a if axis==0 else a.T
        for i,j in ((0,4),(1,3)):
            s=z[i]+z[j];z[i]=z[j]=s//2;z[2]+=s%2
    return a.ravel()

def collision(f,S,w,omega_num=2,symmetry=True):
    f=np.asarray(f,dtype=np.int64);H=int(f.sum());J=f@V
    if H>=1<<w:return f.copy(),True
    p=r.couple(marginal(H,J[1],S,w),marginal(H,J[0],S,w))
    if omega_num!=4:
        p=((4-omega_num)*f+omega_num*p)//4;R=int(H-p.sum());JR=(f-p)@V
        p+=greedy(r.adjacent(R,int(JR[1])),r.adjacent(R,int(JR[0])))
    if symmetry:
        a=f.reshape(5,5);p=project(p,[d for d in (0,1) if np.array_equal(a,np.flip(a,d))])
    return p,False
from milestone12.reference import initialize
