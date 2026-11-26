"""Small exact integer D1Q5 marginal equilibrium oracle.
Offline enumeration minimizes stress error under exact mass and momentum.
At execution only pattern-controlled X gates act; no optimization called.
"""
from fractions import Fraction
from itertools import product
import numpy as np
from milestone3.arithmetic import basis_apply,statevector_apply,signed
K=(-2,-1,0,1,2)

def compositions(h,n):
    if n==1:yield (h,);return
    for a in range(h+1):
        for rest in compositions(h-a,n-1):yield (a,)+rest

class MarginalOracle:
    def __init__(self,nh=3,scale=4):
        self.nh=nh;self.nj=nh+2;self.nin=nh+self.nj;self.nout=5*nh+1;self.nqubits=self.nin+self.nout;self.scale=scale;self.gates=[];self.table={}
        for H in range(1<<nh):
            bins={}
            for p in compositions(H,5):
                J=sum(k*a for k,a in zip(K,p));P=sum(k*k*a for k,a in zip(K,p));bins.setdefault(J,[]).append((P,p))
            for jw in range(1<<self.nj):
                J=signed(jw,self.nj);pat=H|(jw<<nh);valid=J in bins
                if valid:
                    target=Fraction(J*J,H)+Fraction(H*H,4*scale) if H else Fraction(0)
                    # tie break: smaller L2 norm then lexicographic; not isotropic in 2D
                    _,p=min(bins[J],key=lambda x:(abs(x[0]-target),sum(a*a for a in x[1]),x[1]))
                else:p=(0,)*5
                word=sum(a<<(i*nh) for i,a in enumerate(p))|((not valid)<<(5*nh));self.table[pat]=word
                for j in range(self.nout):
                    if word>>j&1:self.gates.append((pat,self.nin+j))
    def encode(self,H,J):
        if not 0<=H<1<<self.nh or not -(1<<(self.nj-1))<=J<(1<<(self.nj-1)):raise ValueError('outside register range')
        return H|((J%(1<<self.nj))<<self.nh)
    def decode(self,word):
        return np.array([(word>>(self.nin+i*self.nh))&((1<<self.nh)-1) for i in range(5)]),bool(word>>(self.nin+5*self.nh)&1)
    def execute(self,H,J):
        return self.decode(basis_apply(self.encode(H,J),self.gates,self.nin))
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for p,t in self.gates:
            zero=[i for i in range(self.nin) if not p>>i&1]
            for i in zero:qc.x(i)
            qc.mcx(list(range(self.nin)),t)
            for i in zero:qc.x(i)
        return qc
