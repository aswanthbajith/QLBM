"""Reversible conservative 1D Cahn--Hilliard finite-difference demonstrator.
Signed fixed point; periodic, unit spacing, A=kappa=M=1, dt=1/64.
All edge fluxes and updates execute compiled X/CX/CCX gates. No lookup table.
The hydrodynamic velocity is zero; not a D3Q7 or coupled water-air solver.
"""
import numpy as np
from milestone6.arithmetic import F,T
from milestone8.collision import IndexedBuilder, _execute, smul

DT=1/64

def pack_words(z,offset,values,width):
    values=np.asarray(values,dtype=np.int64)
    if values.ndim==1: values=values[:,None]
    n=len(values)
    for i in range(values.shape[1]):
        for j in range(width):
            v=((values[:,i]>>j)&1).astype(np.uint8)
            z[offset+i*width+j]=np.packbits(np.pad(v,(0,(-n)%64)),bitorder='little').view('<u8')

def read_words(z,offset,n,count,width):
    bits=np.unpackbits(z[offset:offset+count*width].view(np.uint8),axis=1,bitorder='little')[:,:n].T.reshape(n,count,width).astype(np.int64)
    weights=1<<np.arange(width,dtype=np.int64);weights[-1]*=-1
    return bits@weights

def signed_fit(b,a,w):return b.fit(a,w,True)

def signed_range(x,w):return np.all((x>=-(1<<(w-1))) & (x<(1<<(w-1))))

class Oracle:
    def finish(self,b,out):
        compute=list(b.g);copy=[]
        for j,a in enumerate(out):
            if a==T:copy.append((self.inputs+j,-1,-1))
            elif a!=F:copy.append((self.inputs+j,a,-1))
        self.gates=np.asarray(compute+copy+compute[::-1],np.int32)
        self.nqubits=b.n;self.ancillas=b.n-self.inputs-self.outputs
    def run(self,values,verify=True,initial_output=None):
        values=np.asarray(values,np.int64);n=len(values)
        if values.ndim!=2 or values.shape[1]!=len(self.input_widths):raise ValueError('shape')
        z=np.zeros((self.nqubits,(n+63)//64),np.uint64);off=0
        for j,w in enumerate(self.input_widths):
            if not signed_range(values[:,j],w):raise ValueError('input overflow')
            pack_words(z,off,values[:,j],w);off+=w
        if initial_output is not None:
            raw=np.asarray(initial_output,np.int64)
            if raw.shape!=(n,) or np.any(raw<0) or np.any(raw>=1<<self.outputs):raise ValueError('invalid initial output')
            pack_words(z,self.inputs,raw,self.outputs)
        before=z.copy() if verify else None
        _execute(self.gates,z)
        if np.any(z[self.inputs+self.outputs:]):raise AssertionError('dirty scratch')
        out=read_words(z,self.inputs,n,1,self.outputs)[:,0] & ((1<<self.outputs)-1)
        if verify:
            _execute(self.gates[::-1].copy(),z)
            if not np.array_equal(z,before):raise AssertionError('inverse')
        return out
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        append_gates(qc,self.gates)
        return qc

def append_gates(qc,gates):
    for t,a,b in gates:
        if a<0:qc.x(int(t))
        elif b<0:qc.cx(int(a),int(t))
        else:qc.ccx(int(a),int(b),int(t))

def decode_signed(x,w):
    return np.where(x&(1<<(w-1)),x-(1<<w),x)

class EdgeFlux(Oracle):
    """p = S phi. Compute signed flux tick J from cell i to i+1.
    mu_i S^3 = p_i^3 + S^2(p_i-p_{i-1}-p_{i+1}).
    J = round((mu_i-mu_{i+1}) S / 64), ties away from zero.
    The one edge result is shared with both adjacent cells.
    """
    def __init__(self,S):
        if S<8 or S>65536 or S&(S-1):raise ValueError('S power of two 8..65536')
        self.S=S;k=S.bit_length()-1;self.w=w=k+3;self.fw=fw=w+2
        self.input_widths=(w,)*4;self.inputs=4*w;self.outputs=fw
        b=IndexedBuilder(self.inputs+self.outputs)
        l,p,q,r=[list(range(i*w,(i+1)*w)) for i in range(4)]
        W=3*w+4
        def cube(x):
            a=b.abs(x);a2=b.mul(a,a,2*w);v=b.mul(a2,a,W)
            return b.select(x[-1],b.neg(v,W),v,W)
        # mu_i - mu_{i+1}: p^3-q^3 + S^2(2p-2q-l+r)
        a=b.sub(signed_fit(b,p,W),signed_fit(b,q,W),W)[0]
        lin=b.add([F]+a,signed_fit(b,r,W),W)[0]
        lin=b.sub(lin,signed_fit(b,l,W),W)[0]
        num=b.add(b.sub(cube(p),cube(q),W)[0],[F]*(2*k)+lin,W)[0]
        shift=2*k+6
        # Exact rounding by dyadic denominator. Wide magnitude avoids overflow.
        mag=b.abs(num); rounded=b.add(mag,b.const(1<<(shift-1),W),W)[0]
        quotient=b.fit(rounded[shift:],fw)
        result=b.select(num[-1],b.neg(quotient,fw),quotient,fw)
        self.finish(b,result)
    def evaluate(self,x,verify=True):return decode_signed(self.run(x,verify),self.fw)

class CellUpdate(Oracle):
    """p_next = p + J_left - J_right, flag signed word overflow (no clipping)."""
    def __init__(self,w,fw):
        self.w=w;self.fw=fw;self.input_widths=(w,fw,fw);self.inputs=w+2*fw;self.outputs=w+1
        b=IndexedBuilder(self.inputs+self.outputs);p=list(range(w));l=list(range(w,w+fw));r=list(range(w+fw,self.inputs));W=fw+2
        v=b.sub(b.add(signed_fit(b,p,W),signed_fit(b,l,W),W)[0],signed_fit(b,r,W),W)[0]
        overflow=b.nz([b.xor(bit,v[w-1]) for bit in v[w:]])
        self.finish(b,v[:w]+[overflow])
    def evaluate(self,x,verify=True):
        out=self.run(x,verify);return decode_signed(out&((1<<self.w)-1),self.w),out>>self.w

class PhaseStepper:
    def __init__(self,S):self.edge=EdgeFlux(S);self.cell=CellUpdate(self.edge.w,self.edge.fw);self.S=S
    def step(self,p,verify=True):
        p=np.asarray(p,np.int64)
        if p.ndim!=1 or len(p)<4:raise ValueError('periodic grid N>=4')
        j=self.edge.evaluate(np.stack([np.roll(p,1),p,np.roll(p,-1),np.roll(p,-2)],axis=1),verify)
        new,flags=self.cell.evaluate(np.stack([p,np.roll(j,1),j],axis=1),verify)
        if np.any(flags):raise OverflowError('signed phase register overflow')
        if int(new.sum())!=int(p.sum()):raise AssertionError('phase mass')
        return new
    def resources(self,n):
        # Global reversible timestep: edge words computed, cell outputs computed, edges uncomputed.
        e,c=self.edge,self.cell
        return dict(edge_qubits=e.nqubits,edge_gates=len(e.gates),update_qubits=c.nqubits,update_gates=len(c.gates),scratch_qubits=max(e.ancillas,c.ancillas),global_step_qubits=2*n*e.w+n*e.fw+n+max(e.ancillas,c.ancillas),global_step_gates=n*(2*len(e.gates)+len(c.gates)))

class CoherentGrid:
    """Persistent global word registers for T timesteps. No intermediate readout.
    Edge flux ancillas are uncomputed using the unchanged previous time slice.
    Supplied distinct basis inputs + complex amplitudes; X/CX/CCX act as exact
    permutations of sparse basis support. Input preparation not synthesized.
    """
    def __init__(self,S,n,steps=2):
        if n<4 or steps<1:raise ValueError('invalid dimensions')
        self.model=PhaseStepper(S);e,c=self.model.edge,self.model.cell
        self.n=n;self.steps=steps;self.w=w=e.w;fw=e.fw
        self.hist=(steps+1)*n*w;self.flag=self.hist;self.edges=self.flag+steps*n;self.scratch=self.edges+n*fw
        self.nqubits=self.scratch+max(e.ancillas,c.ancillas);self.schedule=[]
        def words(base,i,width):return list(range(base+i*width,base+(i+1)*width))
        def remap(o,m):
            m=np.asarray(m,np.int32);assert len(m)==o.nqubits and len(set(m))==len(m)
            return np.where(o.gates<0,-1,m[np.maximum(o.gates,0)]).astype(np.int32)
        for t in range(steps):
            x=t*n*w;y=(t+1)*n*w;edge_g=[];update_g=[]
            for i in range(n):
                m=sum([words(x,j%n,w) for j in (i-1,i,i+1,i+2)],[])+words(self.edges,i,fw)+list(range(self.scratch,self.scratch+e.ancillas))
                edge_g.append(remap(e,m))
                m=words(x,i,w)+words(self.edges,(i-1)%n,fw)+words(self.edges,i,fw)+words(y,i,w)+[self.flag+t*n+i]+list(range(self.scratch,self.scratch+c.ancillas))
                update_g.append(remap(c,m))
            self.schedule.extend(edge_g+update_g+[g[::-1].copy() for g in edge_g[::-1]])
    def execute(self,states,amplitudes,copy_final=False):
        x=np.asarray(states,np.int64);a=np.asarray(amplitudes,np.complex128)
        if x.ndim!=2 or x.shape[1]!=self.n or len(set(map(tuple,x)))!=len(x) or not signed_range(x,self.w) or a.shape!=(len(x),) or not np.isclose(np.vdot(a,a),1):raise ValueError('invalid state')
        extra=self.n*self.w if copy_final else 0
        z=np.zeros((self.nqubits+extra,(len(x)+63)//64),np.uint64);pack_words(z,0,x,self.w);init=z.copy()
        for g in self.schedule:_execute(g,z)
        if np.any(z[self.edges:self.nqubits]):raise AssertionError('dirty edges/workspace')
        if np.any(z[self.flag:self.edges]):raise OverflowError('phase register overflow')
        h=read_words(z,0,len(x),(self.steps+1)*self.n,self.w).reshape(len(x),self.steps+1,self.n)
        if copy_final:
            copies=np.asarray([(self.nqubits+i,self.steps*self.n*self.w+i,-1) for i in range(extra)],np.int32)
            _execute(copies,z)
        for g in self.schedule[::-1]:_execute(g[::-1].copy(),z)
        if not np.array_equal(z[:self.nqubits],init[:self.nqubits]):raise AssertionError('inverse')
        if copy_final and not np.array_equal(read_words(z,self.nqubits,len(x),self.n,self.w),h[:,-1]):raise AssertionError('final copy not retained')
        return h,a.copy()
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        for g in self.schedule:append_gates(qc,g)
        return qc
