import numpy as np
import pytest
from milestone12.collision import CollisionOracle
from milestone12 import reference as r
from milestone11.solver import Solver
from milestone11.run import ref_transport

@pytest.fixture(scope='module',params=[1,2,3,4])
def oracle(request):return CollisionOracle(32,omega_num=request.param)

def test_random_conservation(oracle):
    rng=np.random.default_rng(12);f=np.array([rng.multinomial(int(H),rng.dirichlet(np.ones(25))) for H in rng.integers(0,1<<oracle.w,321)])
    z,bad=oracle.evaluate(f);assert not bad.any();assert (z>=0).all();assert np.array_equal(z,[r.collision(a,oracle.S,oracle.w,oracle.omega_num)[0] for a in f]);assert np.array_equal(z.sum(1),f.sum(1));assert np.array_equal(z@r.V,f@r.V)

def test_reflection(oracle):
    rng=np.random.default_rng(9);a=rng.integers(0,2,(64,5,5));f=(a+np.flip(a,1)).reshape(-1,25);z,bad=oracle.evaluate(f);assert not bad.any();z=z.reshape(-1,5,5);assert np.array_equal(z,z[:,::-1,:])

def test_extreme_and_overflow(oracle):
    f=np.vstack([np.eye(25,dtype=int)*((1<<oracle.w)-1),np.zeros((1,25),int),np.full((1,25),1<< (oracle.w-1))]);z,bad=oracle.evaluate(f);assert np.array_equal(z,[r.collision(a,oracle.S,oracle.w,oracle.omega_num)[0] for a in f]);assert np.array_equal(z.sum(1),f.sum(1));assert np.array_equal(z@r.V,f@r.V);assert np.array_equal(z[-1],f[-1]);assert bad[-1] and not bad[:-1].any()

def test_strip():
    o=CollisionOracle(64,omega_num=2);solid=np.zeros((1,64),bool);c=Solver(solid,64,periodic=True,oracle=o);h=np.zeros(solid.shape);h[:,:32]=1.5;f=r.initialize(h,64,o.w);m=f.sum()
    for _ in range(5):
        z,bad=c.step(f);ref=np.array([r.collision(a,64,o.w,2)[0] for a in f[0]])[None];assert np.array_equal(z,ref_transport(ref,solid,1,True));assert not bad and z.sum()==m and not (z@r.V)[...,1].any();f=z
