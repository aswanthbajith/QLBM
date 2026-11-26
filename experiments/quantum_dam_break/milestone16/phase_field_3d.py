"""3D periodic Cahn--Hilliard interface-only step, no hydrodynamic advection.
A=kappa=M=dx=1; dt=1/256. Gate-evaluated chemical potential, edge flux, update.
The independent reference is not called during circuit evolution.
"""
import numpy as np
from milestone6.arithmetic import F
from milestone8.collision import IndexedBuilder
from milestone16.phase_field import Oracle,signed_fit,decode_signed

DT3=1/256
class ChemicalPotential(Oracle):
    def __init__(self,S):
        if S<8 or S>65536 or S&(S-1):raise ValueError('invalid scale')
        self.S=S;k=S.bit_length()-1;self.w=w=k+3;self.mw=W=3*w+3
        self.input_widths=(w,)*7;self.inputs=7*w;self.outputs=W
        b=IndexedBuilder(self.inputs+self.outputs);words=[list(range(i*w,(i+1)*w)) for i in range(7)];p=words[0]
        a=b.abs(p);square=b.mul(a,a,2*w);v=b.mul(square,a,W);cube=b.select(p[-1],b.neg(v,W),v,W)
        # mu S^3 = p^3 + S^2 (5 p - sum of six neighbours).
        ps=signed_fit(b,p,W);lin=b.add(ps,[F,F]+ps,W)[0]
        for q in words[1:]:lin=b.sub(lin,signed_fit(b,q,W),W)[0]
        out=b.add(cube,[F]*(2*k)+lin,W)[0];self.finish(b,out)
    def evaluate(self,x,verify=True):return decode_signed(self.run(x,verify),self.mw)

class PotentialFlux(Oracle):
    def __init__(self,S,mw,w):
        self.fw=fw=w+2;self.input_widths=(mw,mw);self.inputs=2*mw;self.outputs=fw
        b=IndexedBuilder(self.inputs+self.outputs);W=mw+2
        p,q=[list(range(i*mw,(i+1)*mw)) for i in range(2)]
        d=b.sub(signed_fit(b,p,W),signed_fit(b,q,W),W)[0];shift=2*(S.bit_length()-1)+8
        mag=b.abs(d);r=b.add(mag,b.const(1<<(shift-1),W),W)[0];u=b.fit(r[shift:],fw)
        self.finish(b,b.select(d[-1],b.neg(u,fw),u,fw))
    def evaluate(self,x,verify=True):return decode_signed(self.run(x,verify),self.fw)

class Update3D(Oracle):
    def __init__(self,w,fw):
        self.w=w;self.input_widths=(w,)+(fw,)*6;self.inputs=w+6*fw;self.outputs=w+1
        b=IndexedBuilder(self.inputs+self.outputs);W=fw+4;v=signed_fit(b,list(range(w)),W)
        for i in range(6):
            q=signed_fit(b,list(range(w+i*fw,w+(i+1)*fw)),W)
            v=(b.add(v,q,W) if i%2==0 else b.sub(v,q,W))[0]
        overflow=b.nz([b.xor(q,v[w-1]) for q in v[w:]]);self.finish(b,v[:w]+[overflow])
    def evaluate(self,x,verify=True):
        out=self.run(x,verify);return decode_signed(out&((1<<self.w)-1),self.w),out>>self.w

class PhaseStepper3D:
    def __init__(self,S):
        self.S=S;self.mu=ChemicalPotential(S);self.flux=PotentialFlux(S,self.mu.mw,self.mu.w);self.update=Update3D(self.mu.w,self.flux.fw)
    def step(self,p,verify=True):
        p=np.asarray(p,np.int64);shape=p.shape
        if p.ndim!=3 or min(shape)<3:raise ValueError('3D periodic grid axes >=3')
        values=[p]+[np.roll(p,d,axis=a) for a in range(3) for d in (-1,1)]
        mu=self.mu.evaluate(np.stack(values,-1).reshape(-1,7),verify).reshape(shape)
        pairs=np.concatenate([np.stack([mu,np.roll(mu,-1,axis=a)],-1).reshape(-1,2) for a in range(3)])
        J=self.flux.evaluate(pairs,verify).reshape((3,)+shape)
        inp=[p]+[v for a in range(3) for v in (np.roll(J[a],1,axis=a),J[a])]
        new,flags=self.update.evaluate(np.stack(inp,-1).reshape(-1,7),verify)
        if np.any(flags):raise OverflowError('phase register overflow')
        if int(new.sum())!=int(p.sum()):raise AssertionError('phase mass')
        return new.reshape(shape)
    def resources(self,n):
        m,f,u=self.mu,self.flux,self.update;scratch=max(m.ancillas,f.ancillas,u.ancillas)
        return dict(chemical_potential_gates=len(m.gates),flux_gates=len(f.gates),update_gates=len(u.gates),scratch_qubits=scratch,global_step_qubits=n*(2*m.w+m.mw+3*f.fw+1)+scratch,global_step_gates=n*(2*len(m.gates)+6*len(f.gates)+len(u.gates)))
