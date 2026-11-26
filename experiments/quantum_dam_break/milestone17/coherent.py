"""Global reversible composition of coupled phase/shear word oracles.
The circuit retains input/history, computes all edge/reduction work, updates
both fields, and uncomputes the work. Exact sparse basis support simulation
with complex amplitudes; no intermediate word decoding in evolution.
"""
import numpy as np
from milestone17.coupling import CoupledStepper
from milestone16.phase_field import pack_words,read_words,signed_range,append_gates
from milestone8.collision import _execute

class CoherentCoupledGrid:
    def __init__(self,S,shape=(4,4,4),steps=2):
        if steps<1:raise ValueError('steps')
        self.model=c=CoupledStepper(S,shape);self.steps=steps;self.n=n=c.N;self.w=w=c.w
        self.slice=(n+1)*w;self.hist=(steps+1)*self.slice;self.flags=self.hist
        self.work=self.flags+steps*(n+1);off=self.work
        def alloc(k,width):
            nonlocal off
            a=np.arange(off,off+k*width,dtype=np.int32).reshape(k,width);off+=k*width;return a
        raw=alloc(n,c.mu.outputs);mu=alloc(n,c.scale_mu.outputs);edge=alloc(3*n,c.flux.outputs).reshape(3,n,-1)
        force=alloc(n,c.force.outputs);tree=alloc(n-1,c.sum.outputs)
        zero=alloc(1,w)[0];positive=alloc(1,2)[0];negative=alloc(1,2)[0]
        self.const_one=int(negative[0]);self.scratch=off;self.nqubits=off+max(o.ancillas for o in (c.mu,c.scale_mu,c.flux,c.force,c.sum,c.vel,c.update))
        idx=np.arange(n).reshape(shape);right=[np.roll(idx,-1,axis=i).ravel() for i in range(3)];left=[np.roll(idx,1,axis=i).ravel() for i in range(3)];neg=c.flags.ravel()
        self.schedule=[];self.count=2 # two X gates for fixed orientation constant
        def add(block,o,parts,inv=False):
            a=np.concatenate([np.asarray(p,dtype=np.int32).ravel() for p in parts]);assert len(a)==o.inputs+o.outputs and len(np.unique(a))==len(a)
            block.append((o,a,inv))
        for t in range(steps):
            cur=np.arange(t*self.slice,(t+1)*self.slice,dtype=np.int32).reshape(n+1,w)
            nxt=np.arange((t+1)*self.slice,(t+2)*self.slice,dtype=np.int32).reshape(n+1,w)
            flag=np.arange(self.flags+t*(n+1),self.flags+(t+1)*(n+1),dtype=np.int32)
            prep=[];updates=[]
            for i in range(n):
                add(prep,c.mu,[cur[i]]+[cur[j[i]] for j in sum([[right[a],left[a]] for a in range(3)],[])]+[raw[i]])
                add(prep,c.scale_mu,[raw[i],mu[i]])
            for a in range(3):
                for i,j in enumerate(right[a]):
                    s=negative if a==2 and neg[i] else positive
                    add(prep,c.flux,[cur[i],cur[j],mu[i],mu[j],cur[-1] if a==2 else zero,s,edge[a,i]])
            for i,j in enumerate(right[2]):
                add(prep,c.force,[cur[i],cur[j],mu[i],mu[j],negative if neg[i] else positive,force[i]])
            # Pair adder has widened operands; mapping narrower leaf force words
            # would alias their sign bit. Compute sign extensions with CNOTs to
            # distinct dedicated wires instead (see explicit helper below).
            # Use a separate sign-extension oracle with output sum-width.
            # This implementation builds the tree with a mixed-width first layer.
            if not hasattr(self,'leaf'):
                from milestone17.coupling import SignedOracle,setup,signed_fit
                class LeafSum(SignedOracle):
                    def __init__(self,iw,ow):
                        b,(x,y)=setup(self,[iw,iw],ow);self.finish(b,b.add(signed_fit(b,x,ow),signed_fit(b,y,ow),ow)[0])
                self.leaf=LeafSum(c.force.outputs,c.sum.outputs)
                self.nqubits=max(self.nqubits,self.scratch+self.leaf.ancillas)
            v=list(force);k=0;first=True
            while len(v)>1:
                r=[]
                for j in range(0,len(v),2):
                    add(prep,self.leaf if first else c.sum,[v[j],v[j+1],tree[k]]);r.append(tree[k]);k+=1
                v=r;first=False
            assert k==n-1
            add(updates,c.vel,[cur[-1],v[0],nxt[-1],[flag[-1]]])
            for i in range(n):add(updates,c.update,[cur[i]]+sum([[edge[a,left[a][i]],edge[a,i]] for a in range(3)],[])+[nxt[i],[flag[i]]])
            self.schedule.extend(prep+updates+[(o,m,not inv) for o,m,inv in prep[::-1]])
        self.count+=sum(len(o.gates) for o,m,inv in self.schedule)
    def gates(self,inverse=False):
        x=np.array([[self.const_one,-1,-1]],np.int32);yield x
        for o,m,inv in (self.schedule[::-1] if inverse else self.schedule):
            mapping=np.concatenate([m,np.arange(self.scratch,self.scratch+o.ancillas,dtype=np.int32)])
            g=o.gates[::-1] if inv^inverse else o.gates
            yield np.where(g<0,-1,mapping[np.maximum(g,0)]).astype(np.int32)
        yield x
    def execute(self,phase,velocity,amplitudes,copy_final=True):
        p=np.asarray(phase,np.int64);a=np.asarray(velocity,np.int64);amp=np.asarray(amplitudes,np.complex128)
        if p.shape!=(len(a),)+self.model.shape or amp.shape!=a.shape or not signed_range(p,self.w) or not signed_range(a,self.w) or not np.isclose(np.vdot(amp,amp),1):raise ValueError('invalid input')
        init_words=np.column_stack([p.reshape(len(a),-1),a]);assert len(set(map(tuple,init_words)))==len(a)
        extra=self.slice if copy_final else 0
        z=np.zeros((self.nqubits+extra,(len(a)+63)//64),np.uint64);pack_words(z,0,init_words,self.w);initial=z.copy()
        for g in self.gates():_execute(g,z)
        if np.any(z[self.flags:self.nqubits]):raise AssertionError('dirty work or overflow')
        h=read_words(z,0,len(a),(self.steps+1)*(self.n+1),self.w).reshape(len(a),self.steps+1,self.n+1)
        if copy_final:
            g=np.array([(self.nqubits+i,self.steps*self.slice+i,-1) for i in range(extra)],np.int32);_execute(g,z)
        for g in self.gates(inverse=True):_execute(g,z)
        if not np.array_equal(z[:self.nqubits],initial[:self.nqubits]):raise AssertionError('inverse')
        if copy_final and not np.array_equal(read_words(z,self.nqubits,len(a),self.n+1,self.w),h[:,-1]):raise AssertionError('copy')
        return h,amp.copy()
