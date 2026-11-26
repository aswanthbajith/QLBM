import numpy as np
import pytest
from milestone11.collision import CollisionOracle,_execute
from milestone11 import reference as r
from milestone11.solver import Solver
from milestone11.run import reference_step,ref_transport

@pytest.fixture(scope='module')
def oracle():return CollisionOracle(32)

def test_marginal_exhaustive():
    for H in range(64):
        for J in range(-2*H,2*H+1):
            p=np.array(r.marginal(H,J,32,6));assert p.min()>=0 and p.sum()==H and p@r.K==J
            assert np.array_equal(p,r.marginal(H,-J,32,6)[::-1])

def test_coupling_exact():
    rng=np.random.default_rng(21)
    for H in rng.integers(0,512,500):
        a=rng.multinomial(H,[.2]*5);b=rng.multinomial(H,[.2]*5);z=r.couple(a,b).reshape(5,5);assert z.min()>=0 and np.array_equal(z.sum(1),a) and np.array_equal(z.sum(0),b)

def test_random_circuit(oracle):
    rng=np.random.default_rng(12);f=np.array([rng.multinomial(H,rng.dirichlet(np.ones(25))) for H in rng.integers(0,1<<oracle.w,257)]);z,flags=oracle.evaluate(f);assert not flags.any();assert np.array_equal(z,[r.collision(a,oracle.S,oracle.w)[0] for a in f]);assert np.array_equal(z.sum(1),f.sum(1)) and np.array_equal(z@r.V,f@r.V)

def test_extreme_velocity_and_dry(oracle):
    f=np.eye(25,dtype=int)*((1<<oracle.w)-1);f=np.vstack([f,np.zeros((1,25),int)]);z,flag=oracle.evaluate(f);assert np.array_equal(z,[r.collision(a,oracle.S,oracle.w)[0] for a in f]);assert not flag.any();assert not z[-1].any()

def test_overflow(oracle):
    f=np.full((3,25),1<< (oracle.w-1),int);z,flag=oracle.evaluate(f);assert np.array_equal(z,f) and flag.all()

def test_arbitrary_output_xor(oracle):
    rng=np.random.default_rng(13);f=rng.integers(0,5,(19,25));o=rng.integers(0,1<<oracle.w,(19,25));fl=rng.integers(0,2,19);z=oracle.pack(f,o,fl);before=z.copy();target,flag=oracle.evaluate(f);_execute(oracle.gates,z);assert np.array_equal(oracle.unpack(z,len(f),oracle.inputs),o^target);assert not np.any(z[oracle.inputs+oracle.outputs:]);assert np.array_equal(np.unpackbits(z[oracle.inputs+25*oracle.w].view(np.uint8),bitorder='little')[:len(f)],fl^flag);_execute(oracle.gates[::-1].copy(),z);assert np.array_equal(z,before)

def test_integrated_obstacle(oracle):
    mask=np.zeros((4,8),bool);mask[1,3]=True;c=Solver(mask,32,oracle=oracle);h=np.zeros(mask.shape);h[:,:2]=1.5;f=r.initialize(h,32,oracle.w);m=f.sum()
    for _ in range(5):
        z,flag=c.step(f);rr,rf=reference_step(f,c);assert np.array_equal(z,rr) and flag==rf==0 and z.sum()==m and not z[mask].any();f=z
    assert (f.sum(-1)>0).sum()>8

def test_periodic_transport(oracle):
    mask=np.zeros((1,32),bool);c=Solver(mask,32,4,True,oracle);f=np.arange(800).reshape(1,32,25);z=c.transport(f);assert np.array_equal(z,ref_transport(f,mask,4,True));assert np.array_equal(c.transport(z,True),f)

def test_moment_precision():
    errors=[]
    for S in (32,128,512):
        p=np.array(r.marginal(S,S//2,S,S.bit_length()+1))/S;errors.append(abs(p@(r.K/2)**2-(.25**2+.125/2)))
    assert errors[-1]<=errors[0]

def test_float_equilibrium_cross_moment():
    f=np.zeros((1,25));f[0,12]=.5;f[0,18]=.5;z,_=r.float_collision(f);assert np.allclose(z.sum(-1),f.sum(-1)) and np.allclose(z@r.V,f@r.V)
