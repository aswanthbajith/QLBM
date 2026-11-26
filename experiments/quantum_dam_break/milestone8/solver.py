"""Collision (reversible out-of-place) then wordwise streaming permutation.
Reuse M1 link permutation for population words rather than amplitude registers.
"""
import numpy as np
from quantum_core import Grid,stream_permutation,transpositions,C
from milestone8.collision import CollisionOracle
class LBM:
    def __init__(self,solid,scale=1024,omega_num=2,periodic=False,oracle=None):
        self.solid=np.asarray(solid,bool);ny,nx=self.solid.shape;self.grid=Grid(nx,ny,solid=self.solid);self.S=scale;self.omega_num=omega_num
        self.oracle=oracle or CollisionOracle(scale,omega_num)
        if (self.oracle.scale,self.oracle.omega_num)!=(scale,omega_num):raise ValueError('oracle format mismatch')
        self.active=np.argwhere(~self.solid);self.p=stream_permutation(self.grid,periodic)[:9*self.grid.npos];self.inv=np.argsort(self.p);self.swaps=tuple(transpositions(self.p));self.periodic=periodic
    def transport(self,f,inverse=False):
        flat=f.transpose(2,0,1).ravel();return flat[self.p if inverse else self.inv].reshape(9,self.grid.ny,self.grid.nx).transpose(1,2,0).copy()
    def step(self,f,verify=True):
        if np.any(f[self.solid]):raise ValueError('solid populations must be zero')
        a,flag=self.oracle.evaluate(f[~self.solid],verify=verify);post=np.zeros_like(f);post[~self.solid]=a
        assert np.array_equal(post.sum(-1),f.sum(-1)) and np.array_equal(post@C,f@C)
        out=self.transport(post)
        if verify:assert np.array_equal(self.transport(out,True),post)
        assert out.sum()==f.sum() and not np.any(out[self.solid])
        return out,int(flag.sum())
    def resources(self):
        n=self.grid.npos;w=self.oracle.w;active=len(self.active)
        return dict(local_qubits=self.oracle.nqubits,local_work_qubits=self.oracle.ancillas,local_gates=len(self.oracle.gates),global_qubits=2*n*9*w+active+self.oracle.ancillas,global_gates=active*len(self.oracle.gates)+3*w*len(self.swaps),streaming_word_swaps=len(self.swaps))
    def qiskit(self,max_gates=1000000):
        r=self.resources()
        if r['global_gates']>max_gates:raise ValueError('global circuit exceeds export cap')
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(r['global_qubits']);n=self.grid.npos;w=self.oracle.w;stride=9*w;work=2*n*stride+len(self.active)
        for j,(y,x) in enumerate(self.active):
            k=int(y)*self.grid.nx+int(x);mapping=list(range(k*stride,(k+1)*stride))+list(range(n*stride+k*stride,n*stride+(k+1)*stride))+[2*n*stride+j]+list(range(work,work+self.oracle.ancillas))
            for t,a,b in self.oracle.gates:
                if a<0:qc.x(mapping[t])
                elif b<0:qc.cx(mapping[a],mapping[t])
                else:qc.ccx(mapping[a],mapping[b],mapping[t])
        for a,b in self.swaps:
            ia,pa=divmod(a,n);ib,pb=divmod(b,n)
            for k in range(w):qc.swap(n*stride+pa*stride+ia*w+k,n*stride+pb*stride+ib*w+k)
        return qc
