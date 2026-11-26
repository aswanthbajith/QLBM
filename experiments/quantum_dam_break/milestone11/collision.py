"""Structured reversible D2Q25 full-relaxation collision, no runtime lookup.
Uses M6 arithmetic and M8 bit-sliced gate executor. All output bits XORed.
"""
import numpy as np
from milestone6.arithmetic import F,T
from milestone8.collision import IndexedBuilder,_execute,sum_signed
from milestone11.reference import V

def plus(b,a,d,w):return b.add(a,d,w)[0]
def sub(b,a,d,w):return b.sub(a,d,w)[0]
def accum(b,vs,w):
    z=[F]*w
    for a in vs:z=plus(b,z,a,w)
    return z

def adjacent(b,H,J,w):
    # signed J; exact integer adjacent distribution on the five speeds.
    a=b.abs(J);large=b.ge(a,H,len(a));j=b.fit(a,w);two=[F]+H
    p0=b.select(large,[],sub(b,H,j,w),w)
    p1=b.select(large,sub(b,two,j,w),j,w)
    p2=b.select(large,sub(b,j,H,w),[],w)
    return [b.select(J[-1],p2,[],w),b.select(J[-1],p1,[],w),p0,b.select(J[-1],[],p1,w),b.select(J[-1],[],p2,w)]

def moment(b,p,w):
    return sum_signed(b,[(b.fit(p[0],w),-1),(b.fit(p[0],w),-1),(b.fit(p[1],w),-1),(b.fit(p[3],w),1),(b.fit(p[4],w),1),(b.fit(p[4],w),1)],w)

def second(b,p,w):return accum(b,[[F,F]+p[0],p[1],p[3],[F,F]+p[4]],w)

def marginal(b,H,J,S,w):
    j=b.abs(J);lo=adjacent(b,H,j+[F],w);total=plus(b,[F]+H,j,w+3);n=b.fit(total[2:],w);r=total[:2];nz=b.nz(r)
    hi=[sub(b,sub(b,H,n,w),[nz],w)]+[[F]*w for _ in range(3)]+[n]
    for k in (1,2,3):
        flag=b.land(r[0] if k&1 else b.inv(r[0]),r[1] if k&2 else b.inv(r[1]));hi[k]=[flag]+[F]*(w-1)
    # alpha = clamp((4S J² + H³ - 4S H Pmin)/(4S H (Pmax-Pmin)),0,1)
    pw=w+3;W=3*w+12;sh=(4*S).bit_length()-1
    p0,p1=second(b,lo,pw),second(b,hi,pw)
    h2=b.mul(H,H,2*w);h3=b.mul(h2,H,W);j2=b.mul(j,j,W)
    t=plus(b,[F]*sh+j2,h3,W);base=b.fit([F]*sh+b.mul(H,p0,W),W);num,ge=b.sub(t,base,W)
    den=b.fit([F]*sh+b.mul(H,sub(b,p1,p0,pw),W),W)
    nzden=b.nz(den);sat=b.ge(num,den,W);den1=b.select(nzden,den,b.const(1,W),W)
    # Trim known high zero bits for division only on valid inputs. Width W covers invalid inputs too.
    frac=b.fit(b.div([F]*w+num,den1),w+1)
    alpha=b.select(b.land(ge,nzden),b.select(sat,b.const(1<<w,w+1),frac,w+1),[],w+1)
    beta=sub(b,b.const(1<<w,w+1),alpha,w+1)
    p=[]
    for l,h in zip(lo,hi):
        val=plus(b,b.mul(l,beta,2*w+2),b.mul(h,alpha,2*w+2),2*w+2);p.append(b.fit(val[w:],w))
    R=sub(b,H,accum(b,p,w),w);Rj=sub(b,b.fit(j,w+3),moment(b,p,w+3),w+3);res=adjacent(b,R,Rj,w)
    p=[plus(b,a,c,w) for a,c in zip(p,res)]
    return [b.select(J[-1],p[4-i],p[i],w) for i in range(5)]

def coupling(b,y,x,w):
    col=x[:];rows=[]
    for r in y[:-1]:
        total=accum(b,col,w);den=b.select(b.nz(total),total,b.const(1,w),w);prev=[F]*w;cum=[F]*w;row=[]
        for c in col[:-1]:
            cum=plus(b,cum,c,w);prod=b.mul(r,cum,2*w);num=plus(b,[F]+prod,den,2*w+2)
            z=b.fit(b.div(num,[F]+den),w);row.append(sub(b,z,prev,w));prev=z
        row.append(sub(b,r,prev,w));col=[sub(b,c,a,w) for c,a in zip(col,row)];rows+=row
    return rows+col

class CollisionOracle:
    def __init__(self,S=128,w=None):
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
        out=[b.select(invalid,a,c,w) for a,c in zip(f,p)];compute=b.g;copy=[]
        for j,v in enumerate([a for p in out for a in p]+[invalid]):
            if v!=F:copy.append((self.inputs+j,-1 if v==T else v,-1))
        self.gates=np.asarray(compute+copy+compute[::-1],dtype=np.int32);self.nqubits=b.n;self.ancillas=b.n-self.inputs-self.outputs
    def pack(self,f,out=None,flag=None):
        f=np.asarray(f,dtype=np.int64)
        if f.ndim!=2 or f.shape[1]!=25 or np.any(f<0) or np.any(f>=1<<self.w):raise ValueError('invalid population words')
        n=len(f);z=np.zeros((self.nqubits,(n+63)//64),np.uint64)
        for offset,a in [(0,f),(self.inputs,out)]:
            if a is None:continue
            for i in range(25):
                for k in range(self.w):
                    bits=((a[:,i]>>k)&1).astype(np.uint8);raw=np.packbits(np.pad(bits,(0,(-n)%64)),bitorder='little');z[offset+i*self.w+k]=raw.view('<u8')
        if flag is not None:z[self.inputs+25*self.w]=np.packbits(np.pad(np.asarray(flag,dtype=np.uint8),(0,(-n)%64)),bitorder='little').view('<u8')
        return z
    def unpack(self,z,n,offset):
        a=np.zeros((n,25),np.int64)
        for i in range(25):
            for k in range(self.w):a[:,i]|=np.unpackbits(z[offset+i*self.w+k].view(np.uint8),bitorder='little')[:n].astype(np.int64)<<k
        return a
    def evaluate(self,f,verify=True):
        f=np.asarray(f,dtype=np.int64);z=self.pack(f);n=len(f);_execute(self.gates,z)
        # unused SIMD lanes are also zero-initialized and may have flag-free output; all scratch must be clean.
        assert not np.any(z[self.inputs+self.outputs:]) and np.array_equal(self.unpack(z,n,0),f)
        out=self.unpack(z,n,self.inputs);flag=np.unpackbits(z[self.inputs+25*self.w].view(np.uint8),bitorder='little')[:n].astype(bool)
        if verify:
            _execute(self.gates[::-1].copy(),z);assert np.array_equal(z,self.pack(f))
        return out,flag
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for t,a,b in self.gates:
            if a<0:qc.x(int(t))
            elif b<0:qc.cx(int(a),int(t))
            else:qc.ccx(int(a),int(b),int(t))
        return qc
