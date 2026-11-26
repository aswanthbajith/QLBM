import numpy as np
import pytest
from milestone22.solver import *
from milestone19.solver import initial

@pytest.mark.parametrize('width',[.1,.5,1.])
@pytest.mark.parametrize('ratio',[1.,998/1.2])
def test_stationary(width,ratio):
    s=BalancedDamBreak(shape=(8,2,2),ratio=ratio,gravity=1e-5)
    z,y,x=np.indices(s.shape);a=.5*(1-np.tanh((z+.5-4)/width));f,h=s.set_hydrostatic_background(a);f0=f.copy();p0=s.decode(f,h)[4]
    for _ in range(32):f,h=s.step(f,h,'gate',freeze=True)
    assert np.max(abs(f-f0))<1e-12
    assert np.max(abs(s.decode(f,h)[5]))<1e-12
    assert np.max(abs(s.decode(f,h)[4]-p0))<1e-12
    assert abs(s.defect.sum())<1e-12

@pytest.mark.parametrize('offset',[-.001,1e-5,.001])
def test_gauge(offset):
    s=BalancedDamBreak(shape=(4,2,8),gravity=.0005,sigma=1e-5)
    t=BalancedDamBreak(shape=s.shape,gravity=s.gravity,sigma=s.sigma,pressure_reference=offset)
    _,g=initial(s.shape);a=(1+g.sum(0))/2;p=hydrostatic(a,s.mat(a)[0],s.gravity)
    f,h=s.initialize(a,p);ff,hh=t.initialize(a,p+offset)
    for _ in range(8):f,h=s.step(f,h,'reference');ff,hh=t.step(ff,hh,'gate')
    assert np.max(abs(f-ff))<1e-12;assert np.max(abs(h-hh))<1e-12
    assert np.max(abs(t.absolute_pressure(ff,hh)-s.absolute_pressure(f,h)-offset))<1e-12

def test_dynamic_background():
    s=BalancedDamBreak(shape=(4,2,8),gravity=.0005,sigma=1e-5)
    _,g=initial(s.shape);a=(1+g.sum(0))/2;p=hydrostatic(a,s.mat(a)[0],s.gravity)
    s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),a.shape))
    f,h=s.initialize_gauge(a,p);ff,hh=f.copy(),h.copy();f0=f.copy()
    for _ in range(8):f,h=s.step(f,h,'gate');ff,hh=s.step(ff,hh,'reference')
    assert np.max(abs(f-ff))<1e-12;assert np.max(abs(h-hh))<1e-12
    assert np.max(abs(f-f0))>1e-6;assert abs(h.sum()-a.sum())<1e-10
    assert h.sum(0).min()>=-1e-10 and h.sum(0).max()<=1+1e-10

def test_invalid_background():
    s=BalancedDamBreak(shape=(4,2,8));a=np.zeros(s.shape);a[:,:,0]=1
    with pytest.raises(ValueError):s.set_hydrostatic_background(a)
