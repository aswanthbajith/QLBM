"""M22: explicit pressure gauge and fixed hydrostatic residual subtraction.
Preserves M21 hybrid collision and streaming. Reference correction is a classical,
precomputed stationary defect, not a new quantum operator or general balance proof.
"""
import numpy as np
from milestone21.solver import (PressureDamBreak, CS2, C19, W19, W7, weights,
    dot, derivatives, hydrostatic, equilibrium, hydro_collision, moment, push_reference)
from milestone18.transport import LinkTransport, C7
from milestone19.collision import M7, collide
from milestone20.limiter import limit, scalar_reference

class BalancedDamBreak(PressureDamBreak):
    def __init__(self,*args,pressure_reference=0.,**kwargs):
        super().__init__(*args,**kwargs)
        self.pressure_reference=float(pressure_reference)
        if not np.isfinite(self.pressure_reference):raise ValueError('nonfinite pressure reference')
        self.background=None
        self.defect=None
        self.reference_collision={}

    def initialize(self,a,p):
        # Physical pressure offset is metadata; the internal gauge pressure is p-p_ref.
        return super().initialize(a,np.asarray(p)-self.pressure_reference)

    def initialize_gauge(self,a,p_gauge):
        # Avoid cancellation when an absolute pressure much larger than p_gauge is supplied.
        return super().initialize(a,p_gauge)

    def absolute_pressure(self,f,h):
        return self.decode(f,h)[4]+self.pressure_reference

    def set_hydrostatic_background(self,a):
        """Fixed, horizontally uniform reference, no capillary equilibrium claimed.
        Reference is hydrostatic for gravity only; capillary terms in the actual
        solution are deliberately not subtracted.
        """
        a=np.asarray(a,float)
        if a.shape!=self.shape or not np.allclose(a,a[:,0:1,0:1],rtol=0,atol=1e-14):
            raise ValueError('background must be horizontally uniform')
        # Exact rest state of the M21 half-force velocity definition, with gravity
        # and pressure correction only. For sigma != 0, the dynamic step retains
        # capillary forcing; the reference calculation uses sigma=0 explicitly.
        base=PressureDamBreak(shape=self.shape,ratio=self.ratio,mu_ratio=self.mu_ratio,
            nu_liquid=self.nu_liquid,gravity=self.gravity,sigma=0.,xi=self.xi,
            tau_phase=self.tau_phase,pressure_force=self.pressure_force)
        rho=base.mat(a)[0];p=hydrostatic(a,rho,self.gravity);P=p/(rho*CS2)
        if self.pressure_force=='product':acc=CS2*derivatives(P)[0]-derivatives(p)[0]/rho
        elif self.pressure_force=='paper':acc=-CS2*P*derivatives(rho)[0]/rho
        else:raise ValueError('pressure force')
        acc[2]-=self.gravity
        f=weights(W19)*(P-dot(C19,acc)/(2*CS2));h=weights(W7)*a
        assert np.max(abs(base.decode(f,h)[5]))<1e-12
        # Cache both backends once; subtract the baseline collision before streaming.
        # This is algebraically S(C(f)-C(f_H))+f_H.
        self.background=(f.copy(),h.copy())
        self.reference_collision={mode:self._collide_hydro(base,f,h,mode)[0] for mode in ('reference','gate')}
        self.defect=f-push_reference(self.reference_collision['reference'],C19)
        assert abs(self.defect.sum())<1e-10
        return f.copy(),h.copy()

    @staticmethod
    def _collide_hydro(model,f,h,mode):
        a,rho,mu,nu,p,u,acc,grad=model.decode(f,h)
        rates=np.ones((19,)+model.shape);rates[:4]=0;rates[5:10]=1/(.5+nu/CS2);rates[10:]=1.2
        eq=equilibrium(f.sum(0),u);src=weights(W19)*dot(C19,acc)/CS2
        return hydro_collision(f,eq,src,rates,mode),a,u,grad

    def step(self,f,h,mode='gate',freeze=False):
        if mode not in ('gate','reference'):raise ValueError(mode)
        ff,a,u,grad=self._collide_hydro(self,f,h,mode)
        if self.background is not None:
            # Stream the *perturbation* to avoid numerical cancellation in the
            # exact stationary state. Gate and reference cache their own baseline
            # collision, so roundoff is not amplified by a mixed-backend residual.
            ff=ff-self.reference_collision[mode]
        if freeze:hh=h.copy()
        else:
            mag=np.linalg.norm(grad,axis=0);normal=np.divide(grad,mag,out=np.zeros_like(grad),where=mag>1e-10)
            heq=weights(W7)*a*(1+4*dot(C7,u));source=weights(W7)*dot(C7,normal)*4*a*(1-a)/self.xi
            hh=collide(h,heq,source,M7,np.array([0.]+[1/self.tau_phase]*6),mode,bound=.25)
            signed=2*hh-weights(W7)
            corrected=limit(signed)[0] if mode=='gate' else scalar_reference(signed)[0]
            hh=(corrected+weights(W7))/2
        if mode=='gate':
            if self.t is None:self.t=[LinkTransport(np.zeros(self.shape,bool),c,periodic=False) for c in (C19,C7)]
            def stream(tr,v):
                state,norm=tr.encode(v)
                if norm==0:return np.zeros_like(v)  # zero perturbation has no normalized amplitude state
                return (tr.apply(state,direct=True)*norm)[:v.size].real.reshape(v.shape)
            ff=stream(self.t[0],ff);hh=h.copy() if freeze else stream(self.t[1],hh)
        else:
            ff=push_reference(ff,C19);hh=h.copy() if freeze else push_reference(hh,C7)
        if self.background is not None:ff=ff+self.background[0]
        return ff,hh
