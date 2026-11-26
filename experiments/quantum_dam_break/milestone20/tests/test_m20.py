import numpy as np
import pytest
from milestone20.limiter import limit,scalar_reference
from milestone20.solver import BoundedDamBreak
from milestone19.solver import DamBreak,initial,push_reference
from milestone18.transport import C7,W7

@pytest.mark.parametrize('seed',range(6))
def test_random_conservation(seed):
    rng=np.random.default_rng(seed);a=rng.uniform(0,1,(3,4,5));a.flat[::7]=0;a.flat[1::7]=1
    g=rng.normal(0,.6,(7,)+a.shape);g[0]+=2*a-1-g.sum(0)
    out,pred,info=limit(g);ref,_=scalar_reference(g)
    actual=(1+push_reference(out,C7).sum(0))/2
    assert np.allclose(out,ref,rtol=0,atol=2e-14)
    assert np.max(abs(g.sum(0)-out.sum(0)))<2e-14
    assert np.allclose(actual,pred,rtol=0,atol=2e-14)
    assert actual.min()>=-2e-14 and actual.max()<=1+2e-14
    assert abs(actual.sum()-a.sum())<2e-12
    assert info['limited_faces']>0

@pytest.mark.parametrize('p',[-1.,0.,1.])
def test_uniform_noop(p):
    g=W7[:,None,None,None]*np.full((4,4,4),p)
    out,pred,info=limit(g)
    assert np.array_equal(out,g);assert np.allclose(pred,(1+p)/2);assert not info['limited_faces']

def test_pure_phase_extreme_flux():
    g=np.zeros((7,2,2,2));g[1]=4;g[0]=-3
    out,pred,info=limit(g);assert np.allclose(pred,1);assert np.allclose(push_reference(out,C7).sum(0),1)

def test_invalid_input():
    g=np.ones((7,2,2,2));
    with pytest.raises(ValueError):limit(g)
    with pytest.raises(ValueError):limit(np.zeros_like(g),.2)

def test_axis_reflection_covariance():
    rng=np.random.default_rng(5);g=rng.normal(0,.7,(7,4,3,5));g[0]+=rng.uniform(-1,1,g.shape[1:])-g.sum(0)
    def reflect(a):return a[[0,2,1,3,4,5,6],:,:,::-1]
    out,_,_=limit(g);rot,_,_=limit(reflect(g));assert np.allclose(rot,reflect(out),atol=1e-14)

def test_coupled_gate_vs_reference():
    s=BoundedDamBreak((4,4,8));f,g=initial(s.shape);r,h=f.copy(),g.copy();m0=f.sum();p0=g.sum()
    for _ in range(12):
        f,g,m,j,b,info=s.step(f,g);r,h,*_=s.step(r,h,'reference')
        assert max(m,j,b,info['flux_update_error'])<1e-10
        assert np.max(abs(g-h))<1e-12;assert np.max(abs(f-r))<1e-12
        assert abs(f.sum()-m0)<1e-9 and abs(g.sum()-p0)<1e-9
        assert np.max(abs(g.sum(0)))<=1+1e-10

def test_sampled_conservation():
    s=BoundedDamBreak((4,4,8));f,g=initial(s.shape);m0=f.sum();p0=g.sum();rng=np.random.default_rng(6)
    for _ in range(12):
        f,g,m,j,b,info=s.step(f,g,shots=64,rng=rng)
        assert max(m,j,b,info['flux_update_error'])<1e-10
        assert abs(f.sum()-m0)<1e-9 and abs(g.sum()-p0)<1e-9
        assert np.max(abs(g.sum(0)))<=1+1e-10
