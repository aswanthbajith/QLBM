"""D4 covariant collision channel, using the M13 reversible collision as a black box.
Uniform three-qubit selector, controlled word permutations and compute/copy/uncompute.
Selector is retained (or traced out); it is NOT claimed to be clean after forward use.
"""
import numpy as np
from collections import defaultdict
from milestone13.collision import CollisionOracle
from milestone8.collision import _execute
from quantum_core import transpositions

# p maps a source population index to a destination index. Group generators
# act on the velocity square; the same permutation is used on both word registers.
I = np.arange(25).reshape(5,5)
GENERATORS = tuple(np.argsort(a.ravel()) for a in (np.rot90(I),np.rot90(I,2),np.fliplr(I)))
GROUP = tuple(np.array([j for j in range(25)]) for _ in range(8))
for g in range(8):
    for bit,p in enumerate(GENERATORS):
        if (g>>bit)&1: GROUP[g][:]=p[GROUP[g]]
assert len({tuple(p) for p in GROUP}) == 8

def action(f,p):
    return np.asarray(f)[...,np.argsort(p)]

def _controlled_permutation(p,offset,w,ctrl):
    gates=[]
    for i,j in transpositions(p):
        for k in range(w):
            a=offset+i*w+k; b=offset+j*w+k
            # Controlled SWAP = CX(a,b) CCX(ctrl,b,a) CX(a,b).
            gates.extend(((b,a,-1),(a,ctrl,b),(b,a,-1)))
    return gates

class TwirledCollision:
    def __init__(self,S=256,omega_num=4,w=None):
        self.base=CollisionOracle(S,w=w,omega_num=omega_num)
        b=self.base
        self.S,self.w,self.omega_num=b.S,b.w,b.omega_num
        self.inputs,self.outputs=b.inputs,b.outputs
        self.selector=tuple(range(b.nqubits,b.nqubits+3));self.nqubits=b.nqubits+3
        self.ancillas=b.ancillas # excludes three intentionally retained selector qubits
        prefix=[]
        for c,p in zip(self.selector,GENERATORS):
            prefix.extend(_controlled_permutation(p,0,self.w,c))
            prefix.extend(_controlled_permutation(p,self.inputs,self.w,c))
        self.gates=np.concatenate((np.asarray(prefix,np.int32),b.gates,np.asarray(prefix[::-1],np.int32)))
        self.wrapper_gates=2*len(prefix)
    def pack(self,f,labels,out=None,flag=None):
        f=np.asarray(f,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
        if labels.shape!=(len(f),) or np.any((labels<0)|(labels>7)):raise ValueError('selector labels 0..7 required')
        z=np.pad(self.base.pack(f,out,flag),((0,3),(0,0)))
        for i,q in enumerate(self.selector):
            bits=((labels>>i)&1).astype(np.uint8)
            z[q]=np.packbits(np.pad(bits,(0,(-len(f))%64)),bitorder='little').view('<u8')
        return z
    def evaluate(self,f,labels,verify=True,out=None,flag=None):
        f=np.asarray(f,dtype=np.int64); n=len(f)
        z=self.pack(f,labels,out,flag);initial=z.copy() if verify else None
        _execute(self.gates,z)
        assert not np.any(z[self.inputs+self.outputs:self.base.nqubits])
        assert np.array_equal(self.base.unpack(z,n,0),f)
        y=self.base.unpack(z,n,self.inputs)
        flags=np.unpackbits(z[self.inputs+25*self.w].view(np.uint8),bitorder='little')[:n].astype(bool)
        if verify:
            _execute(self.gates[::-1].copy(),z);assert np.array_equal(z,initial)
        return y,flags
    def branches(self,f,verify=True):
        f=np.asarray(f,dtype=np.int64)
        y,flag=self.evaluate(np.repeat(f,8,axis=0),np.tile(np.arange(8),len(f)),verify=verify)
        return y.reshape(-1,8,25),flag.reshape(-1,8)
    def qiskit(self,prepare=True):
        """Optional native construction. Unitary dilation, not measured/reset here."""
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        if prepare:qc.h(list(self.selector))
        for t,a,b in self.gates:
            if a<0:qc.x(int(t))
            elif b<0:qc.cx(int(a),int(t))
            else:qc.ccx(int(a),int(b),int(t))
        return qc

def reference_branches(f,S,w,omega_num):
    from milestone13.reference import collision
    a=[];flags=[]
    for p in GROUP:
        y,bad=collision(action(f,p),S,w,omega_num)
        a.append(action(y,np.argsort(p)));flags.append(bad)
    return np.asarray(a),np.asarray(flags)

# Exact sparse statevector simulator for H plus a reversible permutation circuit.
# Keys consist of the input basis populations, output basis populations, flag, selector.
# All workspace begins and ends zero; the actual indexed gates are executed on each support term.
def _hadamard(state,bit):
    out=defaultdict(complex)
    for (f,y,bad,g),a in state.items():
        low=g&~(1<<bit); high=low|(1<<bit)
        out[f,y,bad,low]+=a/np.sqrt(2)
        out[f,y,bad,high]+=(-1 if g&(1<<bit) else 1)*a/np.sqrt(2)
    return {k:v for k,v in out.items() if abs(v)>1e-14}

def coherent_dilation(oracle,inputs,amplitudes):
    """Forward U H^3 and inverse H^3 U† on a sparse complex superposition.
    Exact for this gate set; not a dense full-register statevector allocation.
    """
    zero=(0,)*25
    initial={(tuple(map(int,f)),zero,False,0):complex(a) for f,a in zip(inputs,amplitudes)}
    if len(initial)!=len(inputs) or not np.isclose(sum(abs(a)**2 for a in initial.values()),1):raise ValueError('distinct normalized inputs required')
    state=initial
    for i in range(3):state=_hadamard(state,i)
    keys=list(state);f=np.array([k[0] for k in keys]);y=np.array([k[1] for k in keys]);flags=np.array([k[2] for k in keys]);g=np.array([k[3] for k in keys]);z=oracle.pack(f,g,y,flags)
    _execute(oracle.gates,z)
    assert not np.any(z[oracle.inputs+oracle.outputs:oracle.base.nqubits])
    assert np.array_equal(oracle.base.unpack(z,len(keys),0),f)
    outs=oracle.base.unpack(z,len(keys),oracle.inputs)
    outflags=np.unpackbits(z[oracle.inputs+25*oracle.w].view(np.uint8),bitorder='little')[:len(keys)].astype(bool)
    forward={(k[0],tuple(map(int,v)),bool(b),k[3]):state[k] for k,v,b in zip(keys,outs,outflags)}
    _execute(oracle.gates[::-1].copy(),z)
    assert np.array_equal(z,oracle.pack(f,g,y,flags))
    back=state
    for i in reversed(range(3)):back=_hadamard(back,i)
    error=max(abs(back.get(k,0)-initial.get(k,0)) for k in back.keys()|initial.keys())
    return forward,float(error)
