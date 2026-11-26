"""Bound-preserving extension of M19; only phase face fluxes are changed."""
import numpy as np
from milestone19.solver import DamBreak,initial,fields,push_reference
from milestone19.collision import M19,M7,collide
from milestone18.transport import C19,C7
from milestone20.limiter import limit,scalar_reference

class BoundedDamBreak(DamBreak):
    def step(self,f,g,mode='gate',shots=None,rng=None):
        feq,geq,sf,sg,force=self.equilibria(f,g)
        if mode=='gate':
            ff=collide(f,feq,sf,M19,self.rf,mode,shots,rng,bound=.25)
            gg=collide(g,geq,sg,M7,self.rg,mode,shots,rng,bound=.25)
            gg,pred,info=limit(gg)
        elif mode=='reference':
            def classical(a,e,b,M,rates):
                A=M.T@np.diag(rates)@M;B=np.eye(len(rates))-.5*A
                return a+np.einsum('ij,jzyx->izyx',A,e-a)+np.einsum('ij,jzyx->izyx',B,b)
            ff=classical(f,feq,sf,M19,self.rf);gg=classical(g,geq,sg,M7,self.rg)
            gg,_=scalar_reference(gg);pred=None;info={}
        else:raise ValueError(mode)
        mass=max(np.max(abs(ff.sum(0)-f.sum(0))),np.max(abs(gg.sum(0)-g.sum(0))))
        mom=np.max(abs(np.einsum('id,izyx->dzyx',C19,ff-f)-force))
        if mode=='gate':
            out=[]
            for tr,arr in zip(self.t,(ff,gg)):
                state,norm=tr.encode(arr);out.append((tr.apply(state,direct=True)*norm)[:arr.size].real.reshape(arr.shape))
            info['flux_update_error']=float(np.max(abs((1+out[1].sum(0))/2-pred)))
        else:out=[push_reference(ff,C19),push_reference(gg,C7)]
        ids=self.t[0].wall;direction=ids//np.prod(self.shape)
        impulse=-2*np.sum(C19[direction]*ff.ravel()[ids][:,None],axis=0)
        before=np.einsum('id,izyx->d',C19,f);after=np.einsum('id,izyx->d',C19,out[0]);external=force.sum(axis=(1,2,3))
        balance=float(np.max(abs(after-before-external-impulse)))
        if np.max(abs(out[1].sum(0)))>1+1e-10:raise RuntimeError('phase bound violated')
        return *out,float(mass),float(mom),balance,info
