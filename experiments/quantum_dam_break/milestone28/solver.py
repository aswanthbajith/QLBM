"""Integrate fixed-point quantum moments and affine materials in M27.
The hydrostatic background remains the original precomputed classical setup.
No changes to governing equations, limiter or dissipative channel.
"""
import numpy as np
from milestone27.solver import ThreeDDamBreak
from milestone23.solver import face_terms
from milestone21.solver import CS2,C19,W19,weights,dot,derivatives,equilibrium,hydro_collision
from milestone22.solver import BalancedDamBreak
from milestone28.arithmetic import JointMacros

class QuantumMacroDamBreak(ThreeDDamBreak):
    def __init__(self,*args,k=40,**kw):
        super().__init__(*args,**kw);self.k=k;self.S=1<<k
        self.circuit=JointMacros(k,self.ratio,self.mu_ratio,self.nu_liquid)
        self.macro_calls=0
    def macros(self,f,h):
        self.macro_calls+=1
        vals=np.concatenate((f,h)).reshape(26,-1).T
        if not np.isfinite(vals).all() or np.max(abs(vals))>=4:raise ValueError('fixed-point population range')
        m=self.circuit.evaluate(np.rint(vals*self.S).astype(np.int64),verify=False)
        a,P,*j=(m[:,:5].T/self.S).reshape(5,*self.shape);j=np.stack(j)
        rho,mu=(m[:,5:].T/self.S).reshape(2,*self.shape)
        if np.min(a)<-1e-10 or np.max(a)>1+1e-10 or np.min(rho)<=0 or np.min(mu)<=0:raise ValueError('inadmissible quantized materials')
        return a,P,j,rho,mu
    def decode_details(self,f,h):
        a,P,j,rho,mu=self.macros(f,h);nu=mu/rho;p=rho*CS2*P
        acc,rg=face_terms(P,rho)
        if not self.face_pressure:acc=CS2*derivatives(P)[0]-derivatives(p)[0]/rho
        if self.viscous=='original':rg=derivatives(rho)[0]/rho
        elif self.viscous=='none':rg=np.zeros_like(rg)
        elif self.viscous!='face':raise ValueError(self.viscous)
        grad,lap=derivatives(a);chem=48*self.sigma/self.xi*a*(a-1)*(a-.5)-1.5*self.sigma*self.xi*lap
        acc+=chem*grad/rho;acc[2]-=self.gravity
        pred=j+acc/2;gu=np.stack([derivatives(ui)[0] for ui in pred]);acc+=nu*np.einsum('ijzyx,jzyx->izyx',gu+gu.swapaxes(0,1),rg)
        return (a,rho,mu,nu,p,j+acc/2,acc,grad),P
    def decode(self,f,h):return self.decode_details(f,h)[0]
    @staticmethod
    def _collide_hydro(model,f,h,mode):
        if not isinstance(model,QuantumMacroDamBreak):return BalancedDamBreak._collide_hydro(model,f,h,mode)
        (a,rho,mu,nu,p,u,acc,grad),P=model.decode_details(f,h)
        rates=np.ones((19,)+model.shape);rates[:4]=0;rates[5:10]=1/(.5+nu/CS2);rates[10:]=1.2
        eq=equilibrium(P,u);src=weights(W19)*dot(C19,acc)/CS2
        return hydro_collision(f,eq,src,rates,mode),a,u,grad
