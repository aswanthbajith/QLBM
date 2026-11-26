"""D2Q25 collision and reversible word transport. dx=.5, dt=1 by default."""
import numpy as np
from milestone10.transport import permutation
from quantum_core import transpositions
from milestone11.collision import CollisionOracle
from milestone11.reference import V
class Solver:
    def __init__(self,solid,S=128,stride=1,periodic=False,oracle=None):
        self.solid=np.asarray(solid,bool);self.ny,self.nx=self.solid.shape;self.n=self.nx*self.ny;self.S=S;self.stride=stride;self.periodic=periodic
        if stride<1 or int(stride)!=stride:raise ValueError('positive integer stride required')
        self.dx=.5/stride;self.oracle=oracle or CollisionOracle(S)
        if self.oracle.S!=S:raise ValueError('scale mismatch')
        if periodic:
            if self.solid.any():raise ValueError('periodic mode supports obstacle-free grids only')
            p=np.arange(25*self.n).reshape(25,self.ny,self.nx)
            p=np.array([np.roll(p[i],(-int(cy)*stride,-int(cx)*stride),axis=(0,1)) for i,(cx,cy) in enumerate(V)]).ravel()
        else:p=permutation(self.solid,range(-2*stride,2*stride+1,stride))
        self.p=p;self.inv=np.argsort(p);self.swaps=tuple(transpositions(p));assert np.array_equal(np.sort(p),np.arange(len(p)))
    def transport(self,f,inverse=False):
        flat=f.transpose(2,0,1).ravel();return flat[self.p if inverse else self.inv].reshape(25,self.ny,self.nx).transpose(1,2,0).copy()
    def step(self,f,verify=True):
        f=np.asarray(f,np.int64)
        if f.shape!=(*self.solid.shape,25) or np.any(f[self.solid]):raise ValueError('invalid solid or shape')
        z,flags=self.oracle.evaluate(f[~self.solid],verify);post=np.zeros_like(f);post[~self.solid]=z
        assert np.array_equal(post.sum(-1),f.sum(-1)) and np.array_equal(post@V,f@V)
        out=self.transport(post)
        assert out.sum()==f.sum() and not np.any(out[self.solid])
        if verify:assert np.array_equal(self.transport(out,True),post)
        return out,int(flags.sum())
    def resources(self):
        active=int((~self.solid).sum());w=self.oracle.w
        return dict(local_qubits=self.oracle.nqubits,local_scratch=self.oracle.ancillas,local_x_cx_ccx=int(len(self.oracle.gates)),global_qubits=2*self.n*25*w+active+self.oracle.ancillas,global_x_cx_ccx_equivalent=active*len(self.oracle.gates)+3*w*len(self.swaps),streaming_word_swaps=len(self.swaps))
