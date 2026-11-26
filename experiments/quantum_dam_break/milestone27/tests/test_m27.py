import numpy as np
import pytest
from milestone27.solver import FastClosedTransport,ThreeDDamBreak,C19,C7,hydrostatic
from milestone27.run import initial_fraction
from milestone18.transport import LinkTransport,reference_stream
from milestone20.limiter import limit,scalar_reference
from milestone21.solver import weights,W7

@pytest.mark.parametrize('c',[C19,C7])
def test_closed_gate(c):
    shape=(4,4,8);tr=FastClosedTransport(shape,c);old=LinkTransport(np.zeros(shape,bool),c,periodic=False)
    np.testing.assert_array_equal(tr.perm,old.perm)
    rng=np.random.default_rng(9);v=rng.normal(size=tr.dim)+1j*rng.normal(size=tr.dim);v/=np.linalg.norm(v)
    out=tr.apply(v,direct=True);np.testing.assert_array_equal(out[tr.perm],v);np.testing.assert_array_equal(tr.apply(out,inverse=True,direct=True),v)
    f=rng.normal(size=(len(c),)+shape);np.testing.assert_allclose(tr.stream(f),reference_stream(f,tr.solid,c,periodic=False),atol=1e-14)

def test_nonuniform_fraction():
    a=initial_fraction((16,16,32));assert a.min()>=0 and a.max()<=1;assert np.ptp(a,axis=1).max()>.9;np.testing.assert_array_equal(a,a[:,::-1,:])

def test_matched_gate_step():
    shape=(4,4,8);s=ThreeDDamBreak(shape=shape,gravity=1e-5,nu_liquid=.01);a=initial_fraction(shape,2.);rho=s.mat(a)[0];f,h=s.initialize(a,hydrostatic(a,rho,s.gravity));s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),shape));g,k=f.copy(),h.copy();m=a.sum()
    for _ in range(8):f,h=s.step(f,h,'gate');g,k=s.step(g,k,'reference')
    np.testing.assert_allclose(f,g,atol=1e-13);np.testing.assert_allclose(h,k,atol=1e-13);assert abs(h.sum()-m)<1e-10

def test_limiter_nonuniform():
    rng=np.random.default_rng(7);a=rng.uniform(0,1,(4,4,8));g=weights(W7)*(2*a-1);g[3]+=.4;g[0]-=.4
    v,_,_=limit(g);r,_=scalar_reference(g);np.testing.assert_allclose(v,r,atol=1e-14)
