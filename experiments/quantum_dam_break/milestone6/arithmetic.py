"""Polynomial-width reversible Boolean arithmetic compiler (no input LUT).
Each fresh wire is computed by X/CX/CCX, then copied and uncomputed.
Registers are little endian. Integer arithmetic is exact modulo declared width.
"""
from functools import lru_cache
from milestone4.solver import gate, signed

F,T=-1,-2
class Builder:
    def __init__(self,n): self.n=n;self.g=[]
    def fresh(self): r=self.n;self.n+=1;return r
    def op(self,mask,target): self.g.append((mask,mask,target))
    def xor(self,a,b):
        if a==F:return b
        if b==F:return a
        if a==b:return F
        if a==T:return self.inv(b)
        if b==T:return self.inv(a)
        t=self.fresh();self.op(1<<a,t);self.op(1<<b,t);return t
    def inv(self,a):
        if a==F:return T
        if a==T:return F
        t=self.fresh();self.op(0,t);self.op(1<<a,t);return t
    def land(self,a,b):
        if F in (a,b):return F
        if a==T:return b
        if b==T or a==b:return a
        t=self.fresh();self.op((1<<a)|(1<<b),t);return t
    def lor(self,a,b):return self.xor(self.xor(a,b),self.land(a,b))
    def mux(self,c,a,b):return self.xor(b,self.land(c,self.xor(a,b)))
    def const(self,a,w):return [T if a>>i&1 else F for i in range(w)]
    def fit(self,a,w,sgn=False):return (list(a)+[a[-1] if sgn else F]*w)[:w]
    def add(self,a,b,w,carry=F):
        a=self.fit(a,w);b=self.fit(b,w);r=[]
        for x,y in zip(a,b):
            p=self.xor(x,y);r.append(self.xor(p,carry));carry=self.xor(self.land(x,y),self.land(p,carry))
        return r,carry
    def neg(self,a,w):return self.add([self.inv(x) for x in self.fit(a,w)],[],w,T)[0]
    def sub(self,a,b,w):return self.add(a,[self.inv(x) for x in self.fit(b,w)],w,T)
    def ge(self,a,b,w):return self.sub(a,b,w)[1]
    def nz(self,a):
        r=F
        for x in a:r=self.lor(r,x)
        return r
    def select(self,c,a,b,w):return [self.mux(c,x,y) for x,y in zip(self.fit(a,w),self.fit(b,w))]
    def abs(self,a):return self.select(a[-1],self.neg(a,len(a)),a,len(a))
    def mul(self,a,b,w):
        r=[F]*w
        for j,y in enumerate(b[:w]):r=self.add(r,[F]*j+[self.land(x,y) for x in a[:w-j]],w)[0]
        return r
    def div(self,a,b):
        # Unsigned restoring division; b=0 is defined (all quotient bits 1).
        w=len(b)+1;r=[F]*w;q=[]
        for bit in reversed(a):
            r=[bit]+r[:-1];d,c=self.sub(r,b,w);r=self.select(c,d,r,w);q.append(c)
        return q[::-1]
    def rounded_signed(self,num,den,w):
        # round ties away from zero, using floor((2*abs(n)+d)/(2*d))
        a=self.abs(num);n=self.add([F]+a,den,max(len(a)+2,len(den)+1))[0]
        q=self.fit(self.div(n,[F]+den),w)
        return self.select(num[-1],self.neg(q,w),q,w)

class FluxOracle:
    def __init__(self,scale,sign):
        if scale<2 or scale&(scale-1) or sign not in (-1,1):raise ValueError('power-of-two scale and sign required')
        self.scale=scale;self.sign=sign;hb=scale.bit_length();qb=hb+1;fb=qb+1
        self.widths=(hb,qb,qb);self.fb=fb;self.inputs=hb+2*qb;self.outputs=3*fb
        b=Builder(self.inputs+self.outputs);h=list(range(hb));x=list(range(hb,hb+qb));y=list(range(hb+qb,self.inputs));w=qb+4
        n=b.fit(x,w,True);t=b.fit(y,w,True)
        if sign<0:n=b.neg(n,w)
        threeh=b.add(h,[F]+h,w)[0];s=b.add(threeh,n,w)[0];m=b.fit(s[4:],fb)
        valid=b.land(b.nz(h),b.land(b.ge(threeh,[F]+b.abs(n),w+1),b.ge(threeh,[F]+b.abs(t),w+1)))
        valid=b.land(valid,b.nz(m))
        # Exact numerator P = 2*S*(3*N*H+N*N)+H^3 = 2*S*N*(3H+N)+H^3.
        # Wide two's complement intermediates avoid signed overflow even on invalid inputs.
        W=3*qb+8; prod=b.mul(b.fit(n,W,True),s,W);h2=b.mul(h,h,2*hb);h3=b.mul(h2,h,W)
        pnum=b.add([F]*hb+prod,h3,W)[0] # 2*S = 2**hb
        d=b.select(b.nz(h),h,b.const(1,hb),hb)
        pden=[F]*(hb+4)+d # 32*S = 2**(hb+4)
        p=b.rounded_signed(pnum,pden,fb)
        if sign<0:p=b.neg(p,fb)
        rnum=b.mul(b.fit(t,W,True),s,W);r=b.rounded_signed(rnum,[F]*4+d,fb)
        for j,a in enumerate(m+p+r):
            a=b.land(valid,a);dst=self.inputs+j
            if a==T:b.op(0,dst)
            elif a!=F:b.op(1<<a,dst)
        # Split compute from copy: output-target gates are last and never used as controls.
        copy=[g for g in b.g if self.inputs<=g[2]<self.inputs+self.outputs];compute=[g for g in b.g if g[2]>=self.inputs+self.outputs]
        self.gates=tuple(compute+copy+compute[::-1]);self.nqubits=b.n;self.ancillas=b.n-self.inputs-self.outputs
    def apply(self,w,inverse=False):
        for m,p,t in (reversed(self.gates) if inverse else self.gates):w=gate(w,m,p,t)
        return w
    @lru_cache(maxsize=65536)
    def evaluate(self,key):
        if not 0<=key<1<<self.inputs:raise ValueError('input width')
        z=self.apply(key)
        if z>> (self.inputs+self.outputs) or z&((1<<self.inputs)-1)!=key:raise RuntimeError('dirty arithmetic register')
        return z>>self.inputs
    def pack(self,h,x,y):
        hb,qb,_=self.widths;return h|((x&((1<<qb)-1))<<hb)|((y&((1<<qb)-1))<<(hb+qb))
    def unpack(self,v):return tuple(signed((v>>(j*self.fb))&((1<<self.fb)-1),self.fb) for j in range(3))
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for m,_,t in self.gates:
            cs=[i for i in range(m.bit_length()) if m>>i&1]
            if len(cs)==0:qc.x(t)
            elif len(cs)==1:qc.cx(cs[0],t)
            else:qc.ccx(*cs,t)
        return qc
