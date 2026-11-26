"""Reversible, matched-density CH + one projected incompressible shear mode.
Periodic grid (z,y,x), y=4, s=(1,1,-1,-1), u=(a*s,0,0).
All evolving arithmetic is X/CX/CCX; host only arranges fixed connectivity.
This is a finite-difference Galerkin demonstrator, NOT D3Q19 or full NSCH.
"""
import numpy as np
from milestone6.arithmetic import F
from milestone8.collision import IndexedBuilder, smul
from milestone16.phase_field import Oracle, decode_signed, signed_fit, signed_range
from milestone16.phase_field_3d import ChemicalPotential, Update3D

DT=1/256

def words(widths):
    off=0; out=[]
    for w in widths:out.append(list(range(off,off+w)));off+=w
    return out

def setup(o,widths,out):
    o.input_widths=tuple(widths);o.inputs=sum(widths);o.outputs=out
    return IndexedBuilder(o.inputs+out),words(widths)

def shift_round(b,v,shift,w):
    # Widen before abs/round so even most-negative input is defined.
    W=len(v)+2;v=signed_fit(b,v,W);a=b.abs(v)
    r=b.add(a,b.const(1<<(shift-1),W),W)[0]
    q=b.fit(r[shift:],w)
    return b.select(v[-1],b.neg(q,w),q,w)

class SignedOracle(Oracle):
    def evaluate(self,x,verify=True):return decode_signed(self.run(x,verify),self.outputs)

class ScaledChemical(SignedOracle):
    def __init__(self,S,raw_width,w):
        b,(p,)=setup(self,[raw_width],w+5)
        self.finish(b,shift_round(b,p,2*(S.bit_length()-1),self.outputs))

class FaceFlux(SignedOracle):
    """J ticks = round([2*S*(m_l-m_r)+(p_l+p_r)*a*s]/(512*S)).
    s=-1 if flag=1, otherwise +1. m=round(S*mu). a=S*velocity.
    One rounding of the total face flux, no clipping.
    """
    def __init__(self,S,w,mw):
        b,(p,q,m,n,a,flag)=setup(self,[w,w,mw,mw,w,2],w+2)
        W=2*w+mw+2;k=S.bit_length()-1
        pp=b.add(signed_fit(b,p,w+1),signed_fit(b,q,w+1),w+1)[0]
        adv=smul(b,pp,a,2*w+2);adv=b.select(flag[0],b.neg(adv,len(adv)),adv,len(adv))
        d=b.sub(signed_fit(b,m,mw+1),signed_fit(b,n,mw+1),mw+1)[0]
        v=b.add([F]*(k+1)+signed_fit(b,d,W),signed_fit(b,adv,W),W)[0]
        self.finish(b,shift_round(b,v,k+9,self.outputs))

class ProjectedForce(SignedOracle):
    """Unrounded force numerator s*(p_l+p_r)*(m_l-m_r)."""
    def __init__(self,w,mw):
        b,(p,q,m,n,flag)=setup(self,[w,w,mw,mw,2],w+mw+2)
        pp=b.add(signed_fit(b,p,w+1),signed_fit(b,q,w+1),w+1)[0]
        d=b.sub(signed_fit(b,m,mw+1),signed_fit(b,n,mw+1),mw+1)[0]
        v=smul(b,pp,d,self.outputs)
        self.finish(b,b.select(flag[0],b.neg(v,self.outputs),v,self.outputs))

class AddPair(SignedOracle):
    def __init__(self,w):
        b,(a,c)=setup(self,[w,w],w);self.finish(b,b.add(a,c,w)[0])

class VelocityUpdate(Oracle):
    """a_next = a + round((force_sum - N*S*a/2)/(512*N*S)).
    Matched density=1; nu=1/8, mode Laplacian=-2 s.
    """
    def __init__(self,S,N,w,sw):
        self.w=w; b,(a,f)=setup(self,[w,sw],w+1)
        k=S.bit_length()-1; l=N.bit_length()-1;W=max(sw,w+k+l)+2
        num=b.sub(signed_fit(b,f,W),[F]*(k+l-1)+signed_fit(b,a,W),W)[0]
        delta=shift_round(b,num,k+l+9,W)
        v=b.add(signed_fit(b,a,W),delta,W)[0]
        overflow=b.nz([b.xor(x,v[w-1]) for x in v[w:]])
        self.finish(b,v[:w]+[overflow])
    def evaluate(self,x,verify=True):
        v=self.run(x,verify);return decode_signed(v&((1<<self.w)-1),self.w),v>>self.w

class CoupledStepper:
    def __init__(self,S,shape):
        self.S=S;self.shape=tuple(shape);self.N=N=int(np.prod(shape))
        if len(shape)!=3 or shape[1]!=4 or min(shape)<3 or N&(N-1):raise ValueError('periodic power-of-two cell count, y=4, all axes >=3')
        self.mu=ChemicalPotential(S);self.w=w=self.mu.w
        self.scale_mu=ScaledChemical(S,self.mu.mw,w);mw=self.scale_mu.outputs
        self.flux=FaceFlux(S,w,mw);self.force=ProjectedForce(w,mw)
        self.sum=AddPair(self.force.outputs+N.bit_length()-1)
        self.vel=VelocityUpdate(S,N,w,self.sum.outputs);self.update=Update3D(w,self.flux.outputs)
        self.flags=np.broadcast_to(np.array([0,0,1,1])[None,:,None],shape)
    def step(self,p,a,verify=True,feedback=True):
        p=np.asarray(p,np.int64);a=int(a)
        if p.shape!=self.shape or not signed_range(p,self.w) or not signed_range(np.array([a]),self.w):raise ValueError('input')
        raw=self.mu.evaluate(np.stack([p]+[np.roll(p,d,axis=i) for i in range(3) for d in (-1,1)],-1).reshape(-1,7),verify)
        m=self.scale_mu.evaluate(raw[:,None],verify).reshape(self.shape)
        fs=[]
        for i in range(3):
            ar=np.full(self.shape,a if i==2 else 0,dtype=np.int64)
            flag=self.flags if i==2 else np.zeros(self.shape,np.int64)
            inp=np.stack([p,np.roll(p,-1,axis=i),m,np.roll(m,-1,axis=i),ar,flag],-1).reshape(-1,6)
            fs.append(self.flux.evaluate(inp,verify).reshape(self.shape))
        force=self.force.evaluate(np.stack([p,np.roll(p,-1,axis=2),m,np.roll(m,-1,axis=2),self.flags],-1).reshape(-1,5),verify)
        total=force
        while len(total)>1:total=self.sum.evaluate(total.reshape(-1,2),verify)
        # Control run removes capillary forcing but retains viscosity.
        a1,flag=self.vel.evaluate(np.array([[a,int(total[0]) if feedback else 0]],np.int64),verify)
        inp=[p]+[v for i in range(3) for v in (np.roll(fs[i],1,axis=i),fs[i])]
        p1,flags=self.update.evaluate(np.stack(inp,-1).reshape(-1,7),verify)
        if np.any(flags) or np.any(flag):raise OverflowError('state register overflow')
        if int(p1.sum())!=int(p.sum()):raise AssertionError('phase mass')
        return p1.reshape(self.shape),int(a1[0]),int(total[0])
    def resources(self):
        n=self.N; ops={'mu':(self.mu,n),'scale_mu':(self.scale_mu,n),'face':(self.flux,3*n),'force':(self.force,n),'sum':(self.sum,n-1),'velocity':(self.vel,1),'phase_update':(self.update,n)}
        return {'local_oracles':{k:{'qubits':o.nqubits,'gates':len(o.gates),'calls_per_step':c} for k,(o,c) in ops.items()},'forward_oracle_gate_total':sum(len(o.gates)*c for o,c in ops.values()),'max_local_oracle_qubits':max(o.nqubits for o,c in ops.values())}
