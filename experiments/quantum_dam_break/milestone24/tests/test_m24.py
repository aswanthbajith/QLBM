import numpy as np
import pytest
from milestone24.solver import (SpanTransport,FastSpanTransport,span_push,LongDamBreak,hydrostatic,C19,C7)
from milestone18.transport import LinkTransport
from milestone19.solver import initial,push_reference
from milestone23.solver import FaceDamBreak

@pytest.mark.parametrize('c',[C19,C7])
def test_complex_gate_and_inverse(c):
    t=FastSpanTransport((4,2,8),c);rng=np.random.default_rng(24)
    f=rng.normal(size=(len(c),)+t.shape)+1j*rng.normal(size=(len(c),)+t.shape)
    state,n=t.encode(f);out=t.apply(state,direct=True)
    assert np.max(abs(out[:f.size].reshape(f.shape)*n-span_push(f,c)))<1e-14
    assert np.array_equal(t.apply(out,inverse=True,direct=True),state)
    assert np.array_equal(out,t.apply(state))
    assert np.array_equal(out,LinkTransport.apply(t,state,direct=True))
    assert np.all(out[f.size:]==0)

@pytest.mark.parametrize('c',[C19,C7])
def test_wrap_and_wall(c):
    shape=(4,2,4);t=SpanTransport(shape,c)
    f=np.zeros((len(c),)+shape);f[3,1,1,1]=1;f[6,0,0,2]=-2;f[1,2,0,3]=3
    a=t.stream(f);b=span_push(f,c);assert np.allclose(a,b)
    assert abs(a[3,1,0,1]-1)<1e-14
    assert abs(a[5,0,0,2]+2)<1e-14
    assert abs(a[2,2,0,3]-3)<1e-14
    assert abs(a.sum()-f.sum())<1e-14

def setup(s):
    _,g=initial(s.shape);a=(1+g.sum(0))/2
    f,h=s.initialize(a,hydrostatic(a,s.mat(a)[0],s.gravity))
    s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),s.shape));return f,h

def test_integrated_gate_scalar_reference():
    s=LongDamBreak(shape=(8,2,16),gravity=.00025,sigma=1e-5);f,h=setup(s);f2,h2=f.copy(),h.copy();mass=h.sum()
    for _ in range(24):
        f,h=s.step(f,h,'gate');f2,h2=s.step(f2,h2,'reference')
    assert np.max(abs(f-f2))<2e-13 and np.max(abs(h-h2))<2e-13
    assert abs(h.sum()-mass)<2e-11
    assert np.max(np.ptp(h,axis=2))<1e-12
    a=h.sum(0);assert a.min()>=-1e-12 and a.max()<=1+1e-12

def test_closed_fast_preserves_baseline():
    s=LongDamBreak(shape=(8,2,16),gravity=.00025,sigma=1e-5,span_periodic=False);f,h=setup(s)
    old=FaceDamBreak(shape=s.shape,gravity=s.gravity,sigma=s.sigma);f2,h2=setup(old)
    for _ in range(8):f,h=s.step(f,h,'fast');f2,h2=old.step(f2,h2,'reference')
    assert np.max(abs(f-f2))<1e-13 and np.max(abs(h-h2))<1e-13

def test_reject_nonplanar_fields():
    s=LongDamBreak(shape=(8,2,16));f,h=setup(s);h[0,0,0,0]+=.001
    with pytest.raises(ValueError,match='span-uniform'):s.step(f,h,'fast')

def test_background_stationary():
    s=LongDamBreak(shape=(8,2,4),gravity=.00025);z=np.indices(s.shape)[0];a=.5*(1-np.tanh((z+.5-4)))
    f,h=s.set_hydrostatic_background(a);f0=f.copy()
    for _ in range(5):f,h=s.step(f,h,'gate',freeze=True)
    assert np.max(abs(f-f0))<1e-14
    assert np.max(abs(s.decode(f,h)[5]))<1e-12
