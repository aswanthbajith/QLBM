"""Halfway link bounce-back for D3Q19/D3Q7. Coordinates z,y,x; velocities x,y,z.
The permutation acts on amplitude labels, not on probabilities. Solid and padded
basis states are fixed; physical input has zero amplitude there. No measurement.
"""
import numpy as np
from quantum_core import gate_pairs
C7=np.array([(0,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
C19=np.vstack([C7,[(1,1,0),(-1,-1,0),(1,-1,0),(-1,1,0),(1,0,1),(-1,0,-1),(1,0,-1),(-1,0,1),(0,1,1),(0,-1,-1),(0,1,-1),(0,-1,1)]])
W7=np.array([.25]+[.125]*6); W19=np.array([1/3]+[1/18]*6+[1/36]*12)

def opposites(c):return np.array([np.flatnonzero(np.all(c==-v,axis=1))[0] for v in c])

class LinkTransport:
    def __init__(self,solid,c,periodic=True):
        self.solid=np.asarray(solid,bool).copy();self.shape=self.solid.shape;self.c=np.asarray(c,int);self.q=len(c);self.opp=opposites(self.c)
        if self.solid.ndim!=3 or any(n<2 or n&(n-1) for n in self.shape):raise ValueError('power-of-two 3D dimensions >=2')
        self.n=int(self.solid.size);self.dim=self.n*(1<<(self.q-1).bit_length());self.qubits=self.dim.bit_length()-1
        self.periodic=periodic;self.links=np.arange(self.dim);self.reverse=np.arange(self.dim);self.wall=[]
        for i,v in enumerate(self.c):
            for loc in np.argwhere(~self.solid):
                src=tuple(loc);dst=loc+v[::-1];out=np.any((dst<0)|(dst>=self.shape));dst=tuple(dst%self.shape)
                k=i*self.n+np.ravel_multi_index(src,self.shape);op=self.opp[i]
                self.reverse[k]=op*self.n+np.ravel_multi_index(src,self.shape)
                if (out and not periodic) or self.solid[dst]:self.wall.append(k)
                else:self.links[k]=op*self.n+np.ravel_multi_index(dst,self.shape)
        ids=np.arange(self.dim)
        assert np.array_equal(self.links[self.links],ids) and np.array_equal(self.reverse[self.reverse],ids)
        self.perm=self.reverse[self.links];assert np.array_equal(np.sort(self.perm),ids)
        self.gates=tuple(gate_pairs(self.links))+tuple(gate_pairs(self.reverse))
        self.direct_gates=tuple(gate_pairs(self.perm))
        self.wall=np.array(self.wall,int)
        self.invalid=np.concatenate([np.tile(self.solid.ravel(),self.q),np.ones(self.dim-self.n*self.q,bool)])
    def apply(self,state,inverse=False,direct=False):
        """Execute compiled fully-controlled X gates on a complex statevector."""
        out=np.asarray(state).copy()
        if out.shape!=(self.dim,):raise ValueError('state dimension')
        gates=self.direct_gates if direct else self.gates
        for a,b in reversed(gates) if inverse else gates:out[a],out[b]=out[b],out[a]
        return out
    def encode(self,f):
        f=np.asarray(f); assert f.shape==(self.q,)+self.shape
        if np.any(f[:,self.solid]!=0):raise ValueError('solid populations not zero')
        v=np.zeros(self.dim,complex);v[:f.size]=f.ravel();norm=np.linalg.norm(v)
        return (v/norm if norm else v),norm
    def stream(self,f):
        """Hybrid exact amplitude encoding/readout; no claim of shot-efficient tomography."""
        state,norm=self.encode(f);out=self.apply(state)*norm
        return out[:self.q*self.n].real.reshape((self.q,)+self.shape)
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.qubits)
        for a,b in self.gates:
            target=(a^b).bit_length()-1;controls=[i for i in range(self.qubits) if i!=target]
            zeros=[i for i in controls if not ((a>>i)&1)]
            for i in zeros:qc.x(i)
            qc.mcx(controls,target)
            for i in zeros:qc.x(i)
        return qc

def reference_stream(f,solid,c,periodic=True):
    """Independent push-loop over populations; no compiled permutation."""
    out=np.zeros_like(f);opp=opposites(c);shape=solid.shape
    for i,(cx,cy,cz) in enumerate(c):
        for z,y,x in np.argwhere(~solid):
            zz,yy,xx=z+cz,y+cy,x+cx
            if not periodic and not(0<=zz<shape[0] and 0<=yy<shape[1] and 0<=xx<shape[2]):out[opp[i],z,y,x]+=f[i,z,y,x];continue
            zz%=shape[0];yy%=shape[1];xx%=shape[2]
            if solid[zz,yy,xx]:out[opp[i],z,y,x]+=f[i,z,y,x]
            else:out[i,zz,yy,xx]+=f[i,z,y,x]
    return out

def legacy_stream(f,solid,c):
    """Exact old branch transport logic, isolated from collision/initialization."""
    out=np.stack([np.roll(f[i],tuple(v[::-1]),axis=(0,1,2)) for i,v in enumerate(c)])
    for i in range(1,len(c)):out[opposites(c)[i],solid]=f[i,solid]
    return out
