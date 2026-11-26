import numpy as np
import pytest
from milestone13 import reference as r
from milestone13.collision import CollisionOracle
from milestone11 import reference as old
from milestone11.solver import Solver
from milestone11.run import ref_transport

@pytest.mark.parametrize('S',[4,8,32])
def test_marginal_exhaustive(S):
    w=S.bit_length()+1
    for H in range(1<<w):
        for J in range(-2*H,2*H+1):
            p=np.array(r.marginal(H,J,S,w));q=np.array(old.marginal(H,J,S,w));k=np.arange(-2,3)
            assert p.min()>=0 and p.sum()==H and p@k==J
            target=4*S*J*J+H**3;den=4*S*H
            assert abs(den*(p@(k*k))-target)<=abs(den*(q@(k*k))-target)
            assert np.array_equal(p,r.marginal(H,-J,S,w)[::-1]) if J else True

@pytest.fixture(scope='module',params=[1,2,3,4])
def oracle(request):return CollisionOracle(32,omega_num=request.param)

def test_gate_reference(oracle):
    rng=np.random.default_rng(13);f=np.array([rng.multinomial(int(h),rng.dirichlet(np.ones(25)*.3)) for h in rng.integers(0,1<<oracle.w,512)])
    z,bad=oracle.evaluate(f);assert not bad.any();assert np.array_equal(z,[r.collision(a,oracle.S,oracle.w,oracle.omega_num)[0] for a in f]);assert np.array_equal(z.sum(1),f.sum(1));assert np.array_equal(z@r.V,f@r.V);assert z.min()>=0

def test_symmetry_and_overflow(oracle):
    rng=np.random.default_rng(31);a=rng.integers(0,2,(64,5,5));f=(a+a[:,::-1,:]).reshape(-1,25);f=np.vstack([f,np.full((1,25),1<<(oracle.w-1))]);z,bad=oracle.evaluate(f);assert bad[-1] and not bad[:-1].any();assert np.array_equal(z[-1],f[-1]);assert np.array_equal(z[:-1].reshape(-1,5,5),z[:-1].reshape(-1,5,5)[:,::-1,:]);assert np.array_equal(z@r.V,f@r.V);assert np.array_equal(z,[r.collision(a,oracle.S,oracle.w,oracle.omega_num)[0] for a in f])

def test_pair_pressure():
    from milestone12.reference import project as prior
    rng=np.random.default_rng(17);k=r.V[:,1]**2
    for p in rng.integers(0,100,(128,25)):
        z=r.project(p,[0]);q=prior(p,[0]);assert z.sum()==p.sum();assert z@r.V[:,0]==p@r.V[:,0];assert z@k<=p@k;assert p@k-z@k<=25;assert p@k-z@k<=p@k-q@k

def test_step():
    from milestone12.reference import initialize
    mask=np.zeros((4,8),bool);mask[1:3,5]=True;h=np.zeros(mask.shape);h[:,:3]=1.5;o=CollisionOracle(64,omega_num=2);c=Solver(mask,64,oracle=o);f=initialize(h,64,o.w);mass=f.sum()
    for _ in range(4):
        z,bad=c.step(f);p=np.zeros_like(f);p[~mask]=[r.collision(a,64,o.w,2)[0] for a in f[~mask]];assert np.array_equal(z,ref_transport(p,mask,1,False));assert not bad and z.sum()==mass and not z[mask].any();f=z

def test_output_xor():
    from milestone8.collision import _execute
    o=CollisionOracle(4,omega_num=4);rng=np.random.default_rng(44);f=np.array([rng.multinomial(int(h),np.ones(25)/25) for h in range(16)]);out=rng.integers(0,1<<o.w,f.shape);flags=rng.integers(0,2,len(f));z=o.pack(f,out,flags);initial=z.copy();_execute(o.gates,z);expected=np.array([r.collision(a,4,o.w,4)[0] for a in f]);assert np.array_equal(o.unpack(z,len(f),o.inputs),out^expected);assert not np.any(z[o.inputs+o.outputs:]);_execute(o.gates[::-1].copy(),z);assert np.array_equal(z,initial)
