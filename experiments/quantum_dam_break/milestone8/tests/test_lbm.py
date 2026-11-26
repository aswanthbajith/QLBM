import numpy as np
import pytest
from quantum_core import C
from milestone8 import reference as r
from milestone8.collision import CollisionOracle,_execute
from milestone8.solver import LBM

@pytest.fixture(scope='module',params=[1,2,3])
def oracle(request):return CollisionOracle(128,request.param)

def test_equilibrium_moments():
    h=np.array([.4,1.,1.5]);u=np.array([[.02,.01],[.03,-.04],[0.,.05]]);f=r.equilibrium(h,u)
    assert np.allclose(f.sum(-1),h) and np.allclose(f@C,h[:,None]*u)
    pi=np.einsum('ni,ij,ik->njk',f,C,C);expect=h[:,None,None]*u[:,:,None]*u[:,None,:]+r.G*h[:,None,None]**2/2*np.eye(2)
    assert np.allclose(pi,expect)

def test_random_collision_and_conservation(oracle):
    f=np.random.default_rng(98).integers(0,256,(193,9));a,b=oracle.evaluate(f);rr=[r.integer_collision(x,128,oracle.omega_num) for x in f]
    assert np.array_equal(a,[x[0] for x in rr]) and np.array_equal(b,[x[1] for x in rr])
    assert np.array_equal(a.sum(-1),f.sum(-1)) and np.array_equal(a@C,f@C)

def test_clean_arbitrary_outputs(oracle):
    rng=np.random.default_rng(13);f=rng.integers(0,256,(65,9));out=rng.integers(0,256,(65,9));flag=rng.integers(0,2,65);a,b=oracle.evaluate(f);z=oracle._pack(f,out,flag);before=z.copy();_execute(oracle.gates,z)
    assert np.array_equal(oracle._unpack(z,len(f),oracle.inputs),a^out) and not np.any(z[oracle.inputs+oracle.outputs:]);_execute(oracle.gates[::-1].copy(),z);assert np.array_equal(z,before)

def test_zero(oracle):
    f=np.zeros((1,9),int);a,b=oracle.evaluate(f);assert not a.any() and not b.any()

def test_stream_swaps_and_inverse():
    mask=np.zeros((4,4),bool);mask[1,1]=True;c=LBM(mask,128);n=c.grid.npos;v=np.arange(9*n);z=v.copy()
    for a,b in c.swaps:z[a],z[b]=z[b],z[a]
    assert np.array_equal(z,v[c.inv]);f=v.reshape(9,4,4).transpose(1,2,0);assert np.array_equal(c.transport(c.transport(f),True),f);assert np.array_equal(c.transport(f),r.stream(f,mask))

def test_integrated_closed_and_dry():
    mask=np.zeros((4,4),bool);mask[1,2]=True;c=LBM(mask,1024);h=np.zeros(mask.shape);h[:,:2]=1.5;f=r.initialize(h,np.zeros((*h.shape,2)),c.S);m=f.sum()
    for _ in range(6):
        z,b=c.step(f);rr,rb=r.integer_step(f,mask,c.S,2);assert np.array_equal(z,rr) and b==rb and z.sum()==m and not z[mask].any();f=z
    assert np.count_nonzero(f.sum(-1))>8

def test_periodic_uniform():
    mask=np.zeros((4,4),bool);c=LBM(mask,1024,2,True);h=np.ones(mask.shape);f=r.initialize(h,np.zeros((*h.shape,2)),c.S);z,b=c.step(f);assert np.array_equal(z,f) and b==0
