import numpy as np
import pytest
from milestone21.solver import *
from milestone18.channel import damping_circuit
@pytest.mark.parametrize('ratio',[1,10,100,998/1.2])
def test_materials(ratio):
    a=np.linspace(0,1,10);r,m,n=materials(a,ratio);assert np.isclose(r[-1]/r[0],ratio);assert np.isclose(m[-1]/m[0],.001/.000018);assert np.allclose(n,m/r)
    with pytest.raises(ValueError):materials(np.array([-.1]),ratio)

def test_variable_rail():
    rng=np.random.default_rng(71);d=rng.uniform(-.25,.25,128);l=rng.uniform(-1,1,128);v=rail_variable(d,l);assert np.max(abs(v-l*d))<1e-15
    for a,b,x in zip(d,l,v):
        p=np.zeros(16);p[0]=np.sqrt(1-abs(a)/.25);p[1 if a>=0 else 2]=np.sqrt(abs(a)/.25);q=(damping_circuit(b)@p).reshape(4,4);pr=(abs(q)**2).sum(0);assert abs(.25*(pr[1]-pr[2])-x)<1e-15

def test_pressure_moments():
    rng=np.random.default_rng(9);u=rng.uniform(-.01,.01,(3,2,2,2));p=rng.uniform(-.1,.1,(2,2,2));f=equilibrium(p,u);assert np.allclose(f.sum(0),p);assert np.allclose(moment(C19,f),u)
    e2=np.einsum('ia,ib,izyx->abzyx',C19,C19,f);assert np.allclose(e2,CS2*np.eye(3)[:,:,None,None,None]*p+u[:,None]*u[None,:])

def test_mrt():
    rng=np.random.default_rng(18);f=rng.normal(0,.001,(19,2,2,2));e=equilibrium(f.sum(0),moment(C19,f));src=weights(W19)*dot(C19,np.full((3,2,2,2),.0001))/CS2;rr=rng.uniform(.4,1.99,f.shape);rr[:4]=0
    a=hydro_collision(f,e,src,rr,'gate');b=hydro_collision(f,e,src,rr,'reference');assert np.max(abs(a-b))<1e-15;assert np.allclose(a.sum(0),f.sum(0));assert np.allclose(moment(C19,a-f),moment(C19,src))

@pytest.mark.parametrize('ratio',[1,998/1.2])
def test_hydrostatic(ratio):
    s=PressureDamBreak(shape=(8,2,2),ratio=ratio);z,y,x=np.indices(s.shape);a=.5*(1-np.tanh((z+.5-4)));p=hydrostatic(a,s.mat(a)[0],s.gravity);f,h=s.initialize(a,p)
    for _ in range(32):f,h=s.step(f,h,'gate',freeze=True)
    aa,r,mu,nu,pp,u,*_=s.decode(f,h);assert np.max(abs(u))<2e-6;assert np.array_equal(h,weights(W7)*a);assert np.sqrt(np.mean((pp-p)**2))<2e-6

def test_coupled_high_ratio():
    s=PressureDamBreak(shape=(4,2,8),gravity=.0005,sigma=1e-5);_,g=initial(s.shape);a=(1+g.sum(0))/2;f,h=s.initialize(a,hydrostatic(a,s.mat(a)[0],s.gravity));f1,h1=f.copy(),h.copy()
    for _ in range(8):f,h=s.step(f,h,'gate');f1,h1=s.step(f1,h1,'reference')
    assert np.max(abs(f-f1))<1e-12;assert np.max(abs(h-h1))<1e-12;assert abs(h.sum()-a.sum())<1e-10;assert h.sum(0).min()>=-1e-10;assert h.sum(0).max()<=1+1e-10
