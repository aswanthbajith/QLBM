import numpy as np
import pytest
from milestone23.solver import FaceDamBreak, face_terms
from milestone21.solver import CS2,C19,W19,weights,dot,hydrostatic,derivatives
from milestone19.solver import initial

def test_scalar_face_reference():
    rng=np.random.default_rng(23);rho=np.exp(rng.uniform(-np.log(832),0,(4,3,5)));P=rng.normal(size=rho.shape)
    acc,grad=face_terms(P,rho);ar=np.zeros_like(acc);gr=np.zeros_like(grad)
    for z,y,x in np.ndindex(rho.shape):
        r=rho[z,y,x];p=P[z,y,x]
        for c,w in zip(C19,W19):
            j=tuple(max(0,min(n-1,i+d)) for i,d,n in zip((z,y,x),c[::-1],rho.shape));s=rho[j];q=P[j]
            ar[:,z,y,x]+=w*c*((q-p)-2*(s*q-r*p)/(s+r))
            gr[:,z,y,x]+=w*c*2*(s-r)/(s+r)/CS2
    np.testing.assert_allclose(acc,ar,atol=2e-15);np.testing.assert_allclose(grad,gr,atol=2e-15)

def test_uniform_density():
    P=np.random.default_rng(1).normal(size=(3,4,5));a,b=face_terms(P,np.ones_like(P))
    assert np.max(abs(a))==np.max(abs(b))==0

@pytest.mark.parametrize('ratio',[1.,10.,998/1.2])
def test_bounded_relative_gradient(ratio):
    rho=np.ones((4,3,5));rho[2:]=1/ratio
    P=np.ones_like(rho)*.02;a,g=face_terms(P,rho)
    assert np.max(abs(g))<=1+1e-14
    assert np.max(abs(a))<=.02/3+1e-14

def test_smooth_limit():
    # dx is in physical units. Stencils use lattice units; divide by dx.
    errors=[]
    for n in [16,32,64]:
        dx=1/n;x=(np.arange(n)+.5)*dx;rho=np.broadcast_to((1+.2*np.sin(2*np.pi*x))[None,None,:],(2,2,n));P=np.broadcast_to((.5+.1*np.cos(2*np.pi*x))[None,None,:],rho.shape)
        a,g=face_terms(P,rho);exact=(.4*np.pi*np.cos(2*np.pi*x))/(1+.2*np.sin(2*np.pi*x))
        errors.append(np.max(abs(g[0,0,0,1:-1]/dx-exact[1:-1])))
        assert np.max(abs(a[0,0,0,1:-1]/dx+CS2*P[0,0,1:-1]*exact[1:-1]))<.02
    assert errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3.5

@pytest.mark.parametrize('width',[.1,.5,1.])
@pytest.mark.parametrize('perturb',[0.,1e-5])
def test_hydrostatic(width,perturb):
    s=FaceDamBreak(shape=(8,2,2),gravity=1e-5);z,y,x=np.indices(s.shape);a=.5*(1-np.tanh((z+.5-4)/width));f,h=s.set_hydrostatic_background(a);u=np.zeros((3,)+s.shape);u[2]=perturb*np.sin(np.pi*(z+.5)/8);f+=weights(W19)*dot(C19,u)/CS2
    for _ in range(64):f,h=s.step(f,h,'gate',freeze=True)
    assert np.max(abs(s.decode(f,h)[5]))<max(1e-12,perturb*.1)
    assert abs(s.defect.sum())<1e-12

def test_coupled_gate_reference():
    s=FaceDamBreak(shape=(4,2,8),gravity=.0005,sigma=1e-5);_,g=initial(s.shape);a=(1+g.sum(0))/2;p=hydrostatic(a,s.mat(a)[0],s.gravity);s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),a.shape));f,h=s.initialize_gauge(a,p);ff,hh=f.copy(),h.copy()
    for _ in range(12):f,h=s.step(f,h,'gate');ff,hh=s.step(ff,hh,'reference')
    np.testing.assert_allclose(f,ff,atol=1e-12);np.testing.assert_allclose(h,hh,atol=1e-12)
    assert abs(h.sum()-a.sum())<1e-10;assert h.sum(0).min()>=-1e-10 and h.sum(0).max()<=1+1e-10
