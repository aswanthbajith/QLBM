"""CDKM ripple-carry reversible arithmetic, signed fixed point.
Gate tuples (target, control1, control2); -1 means absent control.
No LUT, host arithmetic only for input quantization and output scaling.
"""
import numpy as np
from milestone8.collision import _execute
from milestone16.phase_field import pack_words,read_words,signed_range

class Circuit:
    def __init__(self,inputs,outputs):
        self.input_widths=tuple(inputs);self.output_widths=tuple(outputs)
        self.inputs=sum(inputs);self.outputs=sum(outputs);self.nqubits=self.inputs+self.outputs;self.g=[]
        def words(widths,offset):
            out=[]
            for w in widths:out.append(list(range(offset,offset+w)));offset+=w
            return out
        self.ins=words(inputs,0);self.outs=words(outputs,self.inputs)
    def alloc(self,w):
        a=list(range(self.nqubits,self.nqubits+w));self.nqubits+=w;return a
    def x(self,t):self.g.append((t,-1,-1))
    def cx(self,a,t):self.g.append((t,a,-1))
    def ccx(self,a,b,t):self.g.append((t,a,b))
    def add(self,a,b,c,subtract=False):
        """CDKM modular addition (a,b,0)->(a,b+a,0), or inverse."""
        if len(a)!=len(b) or len(set(a+b+[c]))!=2*len(a)+1:raise ValueError('overlapping adder wires')
        start=len(self.g)
        for j in range(len(a)):
            x,y,z=a[j],b[j],c if j==0 else a[j-1]
            self.cx(x,y);self.cx(x,z);self.ccx(z,y,x)
        for j in reversed(range(len(a))):
            x,y,z=a[j],b[j],c if j==0 else a[j-1]
            self.ccx(z,y,x);self.cx(x,z);self.cx(z,y)
        if subtract:self.g[start:]=self.g[start:][::-1]
    def add_word(self,src,dst,tmp,carry,subtract=False):
        pairs=list(zip(src+[src[-1]]*(len(dst)-len(src)),tmp))
        for a,b in pairs:self.cx(a,b)
        self.add(tmp,dst,carry,subtract)
        for a,b in reversed(pairs):self.cx(a,b)
    def const(self,value,word):
        for j,t in enumerate(word):
            if value>>j&1:self.x(t)
    def finish(self):self.gates=np.asarray(self.g,np.int32).reshape(-1,3);del self.g
    def evaluate(self,values,verify=True):
        v=np.asarray(values,np.int64)
        if v.ndim!=2 or v.shape[1]!=len(self.input_widths) or not len(v):raise ValueError('input shape')
        n=len(v);z=np.zeros((self.nqubits,(n+63)//64),np.uint64);off=0
        for j,w in enumerate(self.input_widths):
            if not signed_range(v[:,j],w):raise ValueError('input range')
            pack_words(z,off,v[:,j],w);off+=w
        initial=z.copy() if verify else None
        _execute(self.gates,z)
        if np.any(z[self.inputs+self.outputs:]):raise AssertionError('dirty workspace')
        out=[];off=self.inputs
        for w in self.output_widths:out.append(read_words(z,off,n,1,w)[:,0]);off+=w
        if verify:
            if not np.array_equal(z[:self.inputs],initial[:self.inputs]):raise AssertionError('changed input')
            _execute(self.gates[::-1].copy(),z)
            if not np.array_equal(z,initial):raise AssertionError('inverse failure')
        return np.stack(out,axis=1)
    def resources(self):
        g=self.gates;return dict(qubits=self.nqubits,input_qubits=self.inputs,output_qubits=self.outputs,workspace=self.nqubits-self.inputs-self.outputs,X=int((g[:,1]<0).sum()),CX=int(((g[:,1]>=0)&(g[:,2]<0)).sum()),CCX=int((g[:,2]>=0).sum()),gates=len(g))
    def qiskit(self):
        from qiskit import QuantumCircuit
        from milestone16.phase_field import append_gates
        qc=QuantumCircuit(self.nqubits);append_gates(qc,self.gates);return qc

class Moments(Circuit):
    """Input signed f_i,h_i; outputs alpha, P and raw velocity moments x,y,z."""
    def __init__(self,k):
        from milestone18.transport import C19
        self.k=k;w=k+3;ow=w+5
        super().__init__([w]*26,[ow]*5)
        tmp=self.alloc(ow);carry=self.alloc(1)[0]
        for h in self.ins[19:]:self.add_word(h,self.outs[0],tmp,carry)
        for f in self.ins[:19]:self.add_word(f,self.outs[1],tmp,carry)
        for d in range(3):
            for i,c in enumerate(C19[:,d]):
                if c:self.add_word(self.ins[i],self.outs[2+d],tmp,carry,c<0)
        self.finish()

class Materials(Circuit):
    """Affine fixed-point interpolation, rho and dynamic viscosity mu.
    Coefficients quantized once; signed alpha. Round nearest ties toward +infinity.
    Outputs represent the exact integer formula (r0*S + A*(r1-r0)+S/2)//S.
    """
    def __init__(self,k,ratio,mu_ratio,nu_liquid):
        self.k=k;self.S=1<<k;w=k+3;ow=k+3;W=2*k+8
        super().__init__([w],[ow,ow]);tmp=self.alloc(W);acc=self.alloc(W);carry=self.alloc(1)[0]
        self.coefficients=[(round(self.S/ratio),self.S),(round(self.S*nu_liquid/mu_ratio),round(self.S*nu_liquid))]
        for dst,(lo,hi) in zip(self.outs,self.coefficients):
            start=len(self.g);self.const(lo*self.S+self.S//2,acc)
            for j,a in enumerate(self.ins[0]):
                val=(hi-lo)<<j;bits=[t for i,t in enumerate(tmp) if val>>i&1]
                for t in bits:self.cx(a,t)
                self.add(tmp,acc,carry,subtract=j==w-1)
                for t in reversed(bits):self.cx(a,t)
            comp=self.g[start:].copy()
            for a,b in zip(acc[k:k+ow],dst):self.cx(a,b)
            self.g.extend(comp[::-1])
        self.finish()

class JointMacros(Circuit):
    """Compose moments and materials without intermediate readout.
    Precondition: phase sum fits signed material input word. Scratch shared.
    """
    def __init__(self,k,ratio,mu_ratio,nu_liquid):
        self.moments=Moments(k);self.materials=Materials(k,ratio,mu_ratio,nu_liquid);m,r=self.moments,self.materials
        super().__init__(m.input_widths,m.output_widths+r.output_widths)
        scratch=self.alloc(max(m.resources()['workspace'],r.resources()['workspace']))
        mm=np.array(list(range(m.inputs+m.outputs))+scratch[:m.resources()['workspace']],np.int32)
        rm=np.array(self.outs[0][:r.inputs]+sum(self.outs[5:],[])+scratch[:r.resources()['workspace']],np.int32)
        def remap(g,mp):return np.where(g>=0,mp[np.maximum(g,0)],-1)
        self.gates=np.concatenate((remap(m.gates,mm),remap(r.gates,rm)));del self.g
    def evaluate(self,values,verify=True):
        out=super().evaluate(values,verify)
        if not signed_range(out[:,0],self.materials.inputs):raise ValueError('phase sum exceeds material input register')
        return out
