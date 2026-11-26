"""Face-density pressure correction and bounded discrete density gradient.
Experimental alternative, not an exact reproduction of a published LBM.
"""
import numpy as np
from milestone22.solver import BalancedDamBreak
from milestone21.solver import (CS2,C19,W19,W7,weights,dot,moment,derivatives,hydrostatic,push_reference)

def neighbors(a):
    pad=np.pad(a,1,mode='edge')
    return np.stack([pad[tuple(slice(1+d,1+d+s) for d,s in zip(v[::-1],a.shape))] for v in C19])

def face_terms(P,rho):
    rn=neighbors(rho);pn=neighbors(P)
    contrast=(rn-rho)/(rn+rho)
    # cs2 grad(P) - linkwise (p_j-p_i)/rho_face, rho_face arithmetic mean
    acc=-moment(C19,weights(W19)*contrast*(pn+P))
    # Centered, bounded approximation to grad(rho)/rho; smooth limit agrees.
    relative_grad=moment(C19,weights(W19)*2*contrast)/CS2
    return acc,relative_grad

class FaceDamBreak(BalancedDamBreak):
    def __init__(self,*args,face_pressure=True,viscous='face',**kw):
        super().__init__(*args,**kw);self.face_pressure=face_pressure;self.viscous=viscous
    def decode(self,f,h):
        a=h.sum(0);rho,mu,nu=self.mat(a);P=f.sum(0);p=rho*CS2*P
        acc,rg=face_terms(P,rho)
        if not self.face_pressure:acc=CS2*derivatives(P)[0]-derivatives(p)[0]/rho
        if self.viscous=='original':rg=derivatives(rho)[0]/rho
        elif self.viscous=='none':rg=np.zeros_like(rg)
        elif self.viscous!='face':raise ValueError(self.viscous)
        grad,lap=derivatives(a);chem=48*self.sigma/self.xi*a*(a-1)*(a-.5)-1.5*self.sigma*self.xi*lap
        acc+=chem*grad/rho;acc[2]-=self.gravity
        pred=moment(C19,f)+acc/2
        gu=np.stack([derivatives(ui)[0] for ui in pred]);acc+=nu*np.einsum('ijzyx,jzyx->izyx',gu+gu.swapaxes(0,1),rg)
        return a,rho,mu,nu,p,moment(C19,f)+acc/2,acc,grad
    def set_hydrostatic_background(self,a):
        a=np.asarray(a,float)
        if a.shape!=self.shape or not np.allclose(a,a[:,0:1,0:1],rtol=0,atol=1e-14):raise ValueError('background must be horizontally uniform')
        base=FaceDamBreak(shape=self.shape,ratio=self.ratio,mu_ratio=self.mu_ratio,nu_liquid=self.nu_liquid,gravity=self.gravity,sigma=0.,xi=self.xi,tau_phase=self.tau_phase,face_pressure=self.face_pressure,viscous=self.viscous)
        rho=base.mat(a)[0];p=hydrostatic(a,rho,self.gravity);P=p/(rho*CS2);h=weights(W7)*a
        acc=face_terms(P,rho)[0] if self.face_pressure else CS2*derivatives(P)[0]-derivatives(p)[0]/rho
        acc[2]-=self.gravity
        f=weights(W19)*(P-dot(C19,acc)/(2*CS2))
        assert np.max(abs(base.decode(f,h)[5]))<1e-12
        self.background=(f.copy(),h.copy());self.reference_collision={m:self._collide_hydro(base,f,h,m)[0] for m in ('reference','gate')}
        self.defect=f-push_reference(self.reference_collision['reference'],C19)
        return f.copy(),h.copy()
