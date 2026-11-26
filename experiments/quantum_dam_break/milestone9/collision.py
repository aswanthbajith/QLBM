"""M9: conservative bounded coordinate fallback for the M8 BGK candidate.
X/CX/CCX only; flag means full candidate infeasible, not necessarily no update.
Derived from M8 collision.py; inherited arithmetic and bit-sliced simulator.
"""
import numpy as np
from numba import njit
from milestone6.arithmetic import Builder,F,T
from quantum_core import C

FREE=(3,4,5,6,7,8)
class IndexedBuilder(Builder):
    def op(self,mask,target):
        # Keep indexed X/CX/CCX instead of huge Python integer masks in gate storage.
        cs=[]
        while mask:
            low=mask&-mask;cs.append(low.bit_length()-1);mask^=low
        assert len(cs)<=2
        self.g.append((target,cs[0] if cs else -1,cs[1] if len(cs)==2 else -1))

def smul(b,a,d,w):
    mag=b.mul(b.abs(a),b.abs(d),w)
    return b.select(b.xor(a[-1],d[-1]),b.neg(mag,w),mag,w)

def sum_signed(b,terms,w):
    z=[F]*w
    for a,sgn in terms:
        z=(b.add(z,b.fit(a,w,True),w)[0] if sgn>0 else b.sub(z,b.fit(a,w,True),w)[0])
    return z

@njit(cache=True)
def _execute(gates, wires):
    for j in range(gates.shape[0]):
        t,a,b=gates[j]
        for k in range(wires.shape[1]):
            v=np.uint64(0xffffffffffffffff) if a<0 else wires[a,k]
            if b>=0:v &= wires[b,k]
            wires[t,k] ^= v

class LimitedCollisionOracle:
    def __init__(self,scale=1024,omega_num=2):
        if scale<8 or scale>16384 or scale&(scale-1) or omega_num not in (1,2,3):raise ValueError('scale power of two 8..16384, omega numerator 1,2,3 over 2')
        self.scale=scale;self.omega_num=omega_num;self.w=w=scale.bit_length();self.inputs=9*w;self.outputs=9*w+1
        b=IndexedBuilder(self.inputs+self.outputs)
        f=[list(range(i*w,(i+1)*w))+[F] for i in range(9)]
        hb=w+4; mw=hb+2; W=3*hb+9; rw=hb+6
        H=sum_signed(b,[(a,1) for a in f],hb)
        X=sum_signed(b,[(f[i],int(C[i,0])) for i in range(9) if C[i,0]],mw)
        Y=sum_signed(b,[(f[i],int(C[i,1])) for i in range(9) if C[i,1]],mw)
        # H unsigned; q signed; numerator H^3 + 8*S*(2*D*H + 3*D^2 - Q).
        # f_i^eq ticks = numerator/(48*S*H) for axial, /(192*S*H) diagonal.
        H2=b.mul(H,H,2*hb);H3=b.mul(H2,H,W)
        X2=b.mul(b.abs(X),b.abs(X),W);Y2=b.mul(b.abs(Y),b.abs(Y),W)
        Q=b.add(X2,Y2,W)[0]
        Hd=b.select(b.nz(H),H,b.const(1,hb),hb)
        den0=b.add(Hd,[F]+Hd,hb+2)[0] # 3*H; times 16*S =>48*S*H
        shift=scale.bit_length()+2 # 8*S
        denbase=[F]*(shift+1)+den0
        a=[None]*9
        for i in FREE:
            D=sum_signed(b,[(v,int(c)) for v,c in zip((X,Y),C[i]) if c],mw+1)
            DH=smul(b,D,H+[F],W)
            D2=b.mul(b.abs(D),b.abs(D),W)
            threeD2=b.add(D2,[F]+D2,W)[0]
            K=b.sub(b.add([F]+DH,threeD2,W)[0],Q,W)[0]
            num=b.add(H3,[F]*shift+K,W)[0]
            den=denbase if i<5 else [F,F]+denbase
            eq=b.rounded_signed(num,den,rw)
            if omega_num==2: a[i]=eq
            else:
                delta=b.sub(eq,b.fit(f[i],rw,True),rw)[0]
                v=delta if omega_num==1 else b.add(delta,[F]+delta,rw)[0]
                inc=b.rounded_signed(v,b.const(2,2),rw)
                a[i]=b.add(b.fit(f[i],rw,True),inc,rw)[0]
        # Exact integer conservation; remaining axial +x,+y and rest determined by moments.
        a[1]=b.sub(b.fit(X,rw,True),sum_signed(b,[(a[i],int(C[i,0])) for i in FREE if C[i,0]],rw),rw)[0]
        a[2]=b.sub(b.fit(Y,rw,True),sum_signed(b,[(a[i],int(C[i,1])) for i in FREE if C[i,1]],rw),rw)[0]
        a[0]=b.sub(b.fit(H,rw),sum_signed(b,[(a[i],1) for i in range(1,9)],rw),rw)[0]
        # Detect infeasible full candidate; construct a conservative fallback.
        invalid=F
        for z in a:invalid=b.lor(invalid,b.nz(z[w:]))
        # Each free population defines a nullspace direction with three compensators.
        # Limit its signed increment by the available capacity of every affected word.
        z=[v[:w] for v in f]
        for i in FREE:
            cx,cy=map(int,C[i]);coeff={i:1,1:-cx,2:-cy,0:cx+cy-1}
            d=b.sub(a[i],z[i],rw)[0];neg=d[-1];mag=b.abs(d)
            for j,k in coeff.items():
                if not k:continue
                decrease=neg if k>0 else b.inv(neg)
                capacity=b.select(decrease,z[j],[b.inv(v) for v in z[j]],w)
                if abs(k)==2:capacity=capacity[1:]
                elif abs(k)==3:capacity=b.div(capacity,b.const(3,2))
                bound=b.fit(capacity,rw)
                mag=b.select(b.ge(mag,bound,rw),bound,mag,rw)
            for j,k in coeff.items():
                if not k:continue
                v=b.fit(mag,w)
                if abs(k)==2:v=b.fit([F]+v,w)
                elif abs(k)==3:v=b.add(v,[F]+v,w)[0]
                decrease=neg if k>0 else b.inv(neg)
                z[j]=b.add(z[j],b.select(decrease,b.neg(v,w),v,w),w)[0]
        out=[b.select(invalid,z[i],a[i],w) for i in range(9)]
        compute=b.g[:];copy=[]
        for j,v in enumerate([v for z in out for v in z]+[invalid]):
            t=self.inputs+j
            if v==T:copy.append((t,-1,-1))
            elif v!=F:copy.append((t,v,-1))
        self.gates=np.asarray(compute+copy+compute[::-1],dtype=np.int32);self.nqubits=b.n;self.ancillas=b.n-self.inputs-self.outputs
    def _pack(self,f,out=None,flag=None):
        f=np.asarray(f,dtype=np.int64)
        if f.ndim!=2 or f.shape[1]!=9 or np.any(f<0) or np.any(f>=1<<self.w):raise ValueError('invalid populations')
        n=len(f);z=np.zeros((self.nqubits,(n+63)//64),np.uint64)
        for k in range(n):
            bit=np.uint64(1)<<np.uint64(k%64)
            for i in range(9):
                for j in range(self.w):
                    if (int(f[k,i])>>j)&1:z[i*self.w+j,k//64]|=bit
                    if out is not None and (int(out[k,i])>>j)&1:z[self.inputs+i*self.w+j,k//64]|=bit
            if flag is not None and flag[k]:z[self.inputs+9*self.w,k//64]|=bit
        return z
    def _unpack(self,z,n,offset):
        a=np.zeros((n,9),np.int64)
        for k in range(n):
            for i in range(9):
                for j in range(self.w):a[k,i]|=int((z[offset+i*self.w+j,k//64]>>np.uint64(k%64))&np.uint64(1))<<j
        return a
    def evaluate(self,f,verify=True):
        z=self._pack(f);n=len(f);_execute(self.gates,z)
        if np.any(z[self.inputs+self.outputs:]) or not np.array_equal(self._unpack(z,n,0),f):raise AssertionError('dirty scratch or changed input')
        out=self._unpack(z,n,self.inputs);flag=np.array([bool((z[self.inputs+9*self.w,k//64]>>np.uint64(k%64))&np.uint64(1)) for k in range(n)])
        if verify:
            _execute(self.gates[::-1].copy(),z)
            if not np.array_equal(z,self._pack(f)):raise AssertionError('inverse failed')
        return out,flag
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for t,a,b in self.gates:
            if a<0:qc.x(int(t))
            elif b<0:qc.cx(int(a),int(t))
            else:qc.ccx(int(a),int(b),int(t))
        return qc
