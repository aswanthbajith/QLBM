"""M13 pressure-targeted integer marginal repair and pair-average symmetry.
All collision operations compile to X/CX/CCX; no runtime numerical oracle.
"""
import numpy as np
from milestone6.arithmetic import F,T
from milestone8.collision import IndexedBuilder,sum_signed
from milestone11.reference import V
from milestone12.collision import (CollisionOracle as Base,plus,sub,accum,adjacent,second,moment,coupling,mom,greedy,marginal as old_marginal)

def minimum(b,a,c,w):return b.select(b.ge(a,c,w),c,a,w)

def marginal(b,H,J,S,w):
    # Work in positive-momentum orientation, then reflect.
    j=b.abs(J);p=old_marginal(b,H,j,S,w);W=3*w+12;sh=(4*S).bit_length()-1
    t=plus(b,[F]*sh+b.mul(j,j,W),b.mul(b.mul(H,H,2*w),H,W),W)
    D=[F]*sh+H;cur=b.mul(D,second(b,p,w+3),W);delta,up=b.sub(t,cur,W)
    mag=b.select(up,delta,sub(b,cur,t,W),W);den=b.select(b.nz(H),D,b.const(1,len(D)),len(D))
    # A triple move changes the integer second moment by exactly 2.
    raw=b.div(plus(b,mag,den,W+1),[F]+den);bw=w+3
    bound=b.fit([F,F]+H,bw);budget=b.select(b.ge(raw,bound,len(raw)),bound,b.fit(raw,bw),bw)
    budget=b.select(b.nz(H),budget,[],bw);nzj=b.nz(j)
    for i in (2,1,3):
        a,c,d=p[i-1:i+2];capup=b.fit(c[1:],w);capdown=minimum(b,a,d,w)
        cap=b.select(up,capup,capdown,w);n=b.fit(minimum(b,budget,cap,bw),w)
        if i!=2:n=b.select(nzj,n,[],w) # preserve axial parity for zero mean where possible
        p[i-1]=b.select(up,plus(b,a,n,w),sub(b,a,n,w),w)
        p[i+1]=b.select(up,plus(b,d,n,w),sub(b,d,n,w),w)
        p[i]=b.select(up,sub(b,c,[F]+n,w),plus(b,c,[F]+n,w),w)
        budget=sub(b,budget,n,bw)
    return [b.select(J[-1],p[4-i],p[i],w) for i in range(5)]

def symmetrize(b,p,f,axis,w):
    ids=np.arange(25).reshape(5,5);ids=ids if axis==0 else ids.T
    same=T;out=p[:]
    for line in ids.T:
        center=p[line[2]]
        for i,j in ((line[0],line[4]),(line[1],line[3])):
            same=b.land(same,b.inv(b.nz([b.xor(a,c) for a,c in zip(f[i],f[j])])))
            s=plus(b,p[i],p[j],w+1);out[i]=out[j]=b.fit(s[1:],w)
            center=plus(b,center,[s[0]],w)
        out[line[2]]=center
    return [b.select(same,a,c,w) for a,c in zip(out,p)]

class CollisionOracle(Base):
    def __init__(self,S=128,w=None,omega_num=2,symmetry=True):
        if omega_num not in (1,2,3,4):raise ValueError("omega numerator must be 1..4 over 4")
        self.omega_num=omega_num;self.symmetry=symmetry
        if S<4 or S&(S-1):raise ValueError('S must be a power of two >=4')
        self.S=S;self.w=w=w or S.bit_length()+1
        if not 3<=w<=12 or S>=1<<w:raise ValueError('unsupported width')
        self.inputs=25*w;self.outputs=25*w+1;b=IndexedBuilder(self.inputs+self.outputs);f=[list(range(i*w,(i+1)*w)) for i in range(25)]
        H=accum(b,f,w+5);invalid=b.nz(H[w:]);H=H[:w];Js=[]
        for d in (0,1):
            terms=[]
            for i,k in enumerate(V[:,d]):
                if k:terms.append((b.fit(([F]+f[i]) if abs(k)==2 else f[i],w+7),int(np.sign(k))))
            J=sum_signed(b,terms,w+7);Js.append(b.fit(J,w+3))
        x=marginal(b,H,Js[0],S,w);y=marginal(b,H,Js[1],S,w);p=coupling(b,y,x,w)
        if omega_num!=4:
            p=[plus(b,b.mul(a,b.const(4-omega_num,3),w+2),b.mul(c,b.const(omega_num,3),w+2),w+2)[2:] for a,c in zip(f,p)]
            R=sub(b,H,accum(b,p,w),w);JR=[sub(b,Js[d],mom(b,p,d,w+3),w+3) for d in (0,1)]
            a=greedy(b,adjacent(b,R,JR[1],w),adjacent(b,R,JR[0],w),w)
            p=[plus(b,x,y,w) for x,y in zip(p,a)]
        if symmetry:
            for axis in (0,1):p=symmetrize(b,p,f,axis,w)
        out=[b.select(invalid,a,c,w) for a,c in zip(f,p)];compute=b.g;copy=[]
        for j,v in enumerate([a for p in out for a in p]+[invalid]):
            if v!=F:copy.append((self.inputs+j,-1 if v==T else v,-1))
        self.gates=np.asarray(compute+copy+compute[::-1],dtype=np.int32);self.nqubits=b.n;self.ancillas=b.n-self.inputs-self.outputs
