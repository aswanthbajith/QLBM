"""Two-step coherent D2Q25 circuit on a closed two-cell grid.
Sparse complex amplitudes + bit-sliced *global* basis labels. No intermediate
measurement/repreparation; local gates are remapped onto persistent global wires.
"""
from collections import defaultdict
import numpy as np
from milestone14.twirl import TwirledCollision, GROUP, action
from milestone8.collision import _execute
from milestone11.solver import Solver
from milestone11.reference import V
from milestone13.reference import collision as reference_collision

class Circuit:
    def __init__(self,S=16,omega_num=4,steps=2):
        if not 1<=steps<=2: raise ValueError('small exact sparse benchmark: 1 or 2 steps')
        self.oracle=o=TwirledCollision(S,omega_num=omega_num)
        self.steps=steps; self.w=w=o.w; self.cells=2; self.words=50
        self.block=50*w; self.history=(steps+1)*self.block
        self.flags=self.history; self.selectors=self.flags+2*steps
        self.scratch=self.selectors+6*steps
        self.nqubits=self.scratch+o.ancillas
        self.transport=Solver(np.zeros((1,2),bool),S,oracle=o.base)
        self.schedule=[]
        for t in range(steps):
            for c in range(2):
                a=t*self.block+c*25*w; b=(t+1)*self.block+c*25*w
                m=np.concatenate((np.arange(a,a+25*w),np.arange(b,b+25*w),
                    [self.flags+2*t+c],np.arange(self.scratch,self.nqubits),
                    np.arange(self.selectors+6*t+3*c,self.selectors+6*t+3*c+3)))
                assert len(m)==o.nqubits and len(set(m))==len(m)
                self.schedule.append(np.where(o.gates<0,-1,m[np.maximum(o.gates,0)]).astype(np.int32))
            # transport index convention: velocity-major -> cell-major wire offsets.
            gates=[]
            for i,j in self.transport.swaps:
                a=(t+1)*self.block+(i%2*25+i//2)*w
                b=(t+1)*self.block+(j%2*25+j//2)*w
                for k in range(w):gates.extend(((b+k,a+k,-1),(a+k,b+k,-1),(b+k,a+k,-1)))
            self.schedule.append(np.asarray(gates,dtype=np.int32).reshape(-1,3))
    def pack(self,states):
        states=np.asarray(states,dtype=np.int64)
        if states.ndim!=3 or states.shape[1:]!=(2,25) or (states<0).any() or (states>=1<<self.w).any():raise ValueError('invalid two-cell inputs')
        n=len(states); k=6*self.steps; count=1<<k
        # H gates act on fresh independent selectors only; commute to circuit start.
        # Expand exact support after H^k, without ever measuring selector wires.
        labels=np.tile(np.arange(count,dtype=np.int64),n)
        f=np.repeat(states.reshape(n,50),count,axis=0)
        z=np.zeros((self.nqubits,(len(labels)+63)//64),np.uint64)
        def put(q,v):z[q]=np.packbits(np.pad(np.asarray(v,np.uint8),(0,(-len(v))%64)),bitorder='little').view('<u8')
        for i in range(50):
            for b in range(self.w):put(i*self.w+b,(f[:,i]>>b)&1)
        for b in range(k):put(self.selectors+b,(labels>>b)&1)
        return z,labels
    def bits(self,z,start,end,n):
        return np.unpackbits(z[start:end].view(np.uint8),axis=1,bitorder='little')[:,:n].T.astype(np.int64)
    def unpack_history(self,z,n):
        bits=self.bits(z,0,self.history,n).reshape(n,self.steps+1,2,25,self.w)
        return bits @ (1<<np.arange(self.w,dtype=np.int64))
    def execute(self,states,amplitudes):
        states=np.asarray(states,np.int64); a=np.asarray(amplitudes,np.complex128)
        if a.shape!=(len(states),) or not np.isclose(np.vdot(a,a),1) or len({tuple(f.ravel()) for f in states})!=len(states):raise ValueError('distinct normalized input states required')
        z,labels=self.pack(states);n=len(labels);count=1<<(6*self.steps)
        init=z[:self.scratch].copy();amps=np.repeat(a/np.sqrt(count),count)
        for g in self.schedule:
            _execute(g,z)
            if np.any(z[self.scratch:]):raise AssertionError('dirty workspace')
        hist=self.unpack_history(z,n);flags=self.bits(z,self.flags,self.selectors,n)
        if not np.array_equal(self.bits(z,self.selectors,self.scratch,n)@(1<<np.arange(6*self.steps)),labels):raise AssertionError('selector modified')
        # No basis label collision: retained input + selectors uniquely identify support.
        # Each X/CX/CCX permutes labels without changing their complex amplitude.
        for g in self.schedule[::-1]:_execute(g[::-1].copy(),z)
        if not np.array_equal(z[:self.scratch],init) or np.any(z[self.scratch:]):raise AssertionError('inverse failure')
        # Actual inverse H transform on restored selector amplitudes (Walsh-Hadamard).
        back=amps.reshape(len(a),count).copy()
        for b in range(6*self.steps):
            v=back.reshape(len(a),-1,2,1<<b);x=v[:,:,0,:].copy();y=v[:,:,1,:].copy()
            v[:,:,0,:]=(x+y)/np.sqrt(2);v[:,:,1,:]=(x-y)/np.sqrt(2)
        expected=np.zeros_like(back);expected[:,0]=a
        return dict(history=hist,flags=flags,labels=labels,amplitudes=amps,
            inverse_error=float(np.max(np.abs(back-expected))),norm=float(np.vdot(amps,amps).real))
    def resources(self):
        return dict(qubits=self.nqubits,history_qubits=self.history,flag_qubits=2*self.steps,
            orientation_qubits=6*self.steps,scratch_qubits=self.oracle.ancillas,
            hadamards=6*self.steps,x_cx_ccx=sum(len(g) for g in self.schedule),
            streaming_gates=sum(len(self.schedule[t*3+2]) for t in range(self.steps)))
    def qiskit(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(self.nqubits)
        qc.h(list(range(self.selectors,self.scratch)))
        for g in self.schedule:
            for t,a,b in g:
                if a<0:qc.x(int(t))
                elif b<0:qc.cx(int(a),int(t))
                else:qc.ccx(int(a),int(b),int(t))
        return qc

def independent_step(f,labels,S,w,omega_num):
    post=np.empty_like(f);flags=[]
    for c,g in enumerate(labels):
        p=GROUP[g];v,b=reference_collision(action(f[c],p),S,w,omega_num)
        post[c]=action(v,np.argsort(p));flags.append(b)
    assert np.array_equal(post.sum(-1),f.sum(-1)) and np.array_equal(post@V,f@V)
    # Independent closed-wall streaming by link endpoints, no Solver permutation.
    out=np.zeros_like(f)
    for c in range(2):
        for i,(x,y) in enumerate(V):
            if y==0 and 0<=c+x<2:out[c+x,i]+=post[c,i]
            else:out[c,24-i]+=post[c,i]
    return out,flags

def reference_histories(states,labels,steps,S,w,omega_num):
    """Cache identical reference substeps; never used by circuit update."""
    cache={};hist=[];flags=[];count=1<<(6*steps)
    for k,label in enumerate(labels):
        f=states[k//count];h=[f];b=[]
        for t in range(steps):
            g=((int(label)>>(6*t))&7,(int(label)>>(6*t+3))&7)
            key=(tuple(f.ravel()),g)
            if key not in cache:cache[key]=independent_step(f,g,S,w,omega_num)
            f,flag=cache[key];h.append(f);b.extend(flag)
        hist.append(h);flags.append(b)
    return np.asarray(hist),np.asarray(flags)

def retained_output_cleanup_probability(hist,amps,steps):
    """Copy final computational-basis output then invert evolution incl. H.
    Probability fresh selectors return all zero, computed by summing amplitudes
    within identical copied outputs for each original basis input. Demonstrates
    why random environment generally cannot be erased while retaining result.
    """
    count=1<<(6*steps);prob=0.0
    for start in range(0,len(hist),count):
        groups=defaultdict(complex)
        for h,a in zip(hist[start:start+count],amps[start:start+count]):groups[tuple(h[-1].ravel())]+=a/np.sqrt(count)
        prob+=sum(abs(a)**2 for a in groups.values())
    return float(prob)
