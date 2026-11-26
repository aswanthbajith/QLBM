import numpy as np
import pytest
from quantum_core import C
from milestone8 import reference as old
from milestone8.solver import LBM
from milestone9 import reference as r
from milestone9.collision import LimitedCollisionOracle,_execute
from milestone9.run import bound
@pytest.fixture(scope='module',params=[1,2,3])
def oracle(request):return LimitedCollisionOracle(128,request.param)
def test_gate_reference_and_invariants(oracle):
    rng=np.random.default_rng(53);f=rng.integers(0,256,(257,9));f[:32]=rng.integers(0,10,(32,9));a,b=oracle.evaluate(f);rr=[r.collision(x,128,oracle.omega_num) for x in f]
    assert np.array_equal(a,[x[0] for x in rr]);assert np.array_equal(b,[x[1]!=0 for x in rr]);assert np.array_equal(a.sum(-1),f.sum(-1));assert np.array_equal(a@C,f@C);assert np.all((a>=0)&(a<256))
def test_arbitrary_output_xor(oracle):
    rng=np.random.default_rng(14);f=rng.integers(0,256,(17,9));out=rng.integers(0,256,(17,9));flag=rng.integers(0,2,17);a,b=oracle.evaluate(f);z=oracle._pack(f,out,flag);before=z.copy();_execute(oracle.gates,z)
    assert np.array_equal(oracle._unpack(z,len(f),oracle.inputs),a^out);assert not np.any(z[oracle.inputs+oracle.outputs:]);_execute(oracle.gates[::-1].copy(),z);assert np.array_equal(z,before)
def test_zero_and_rest(oracle):
    f=np.zeros((2,9),int);f[1]=old.initialize(np.array([1.]),np.zeros((1,2)),128)[0];a,b=oracle.evaluate(f);assert not a[0].any() and not b.any();assert np.array_equal(a,f)
def test_full_candidate_and_limiter():
    rng=np.random.default_rng(10);statuses=set()
    for f in rng.integers(0,70,(1000,9)):
        a=r.target(f,128,2);z,k=r.limit(f,a,128);statuses.add(k)
        assert sum(z)==sum(f) and np.array_equal(np.array(z)@C,f@C)
        if min(a)>=0 and max(a)<256:assert z==a and k==0
    assert 1 in statuses and 0 in statuses
def test_pressure_necessary_condition():
    f=np.zeros((1,9),int);f[0,0]=64;f[0,1]=64;bad,gap=bound(f,128);assert bad[0] and gap>0
    f=old.initialize(np.array([1.]),np.zeros((1,2)),128);bad,gap=bound(f,128);assert not bad[0] and gap==0
def test_integrated_dry():
    mask=np.zeros((4,8),bool);mask[1,3]=True;S=128;o=LimitedCollisionOracle(S);c=LBM(mask,S,2,False,o);h=np.zeros(mask.shape);h[:,:2]=1.5;f=old.initialize(h,np.zeros((*h.shape,2)),S);m=f.sum()
    for _ in range(6):
        z,_=c.step(f);r0,_=r.step(f,mask,S,2);assert np.array_equal(z,r0) and z.sum()==m and not z[mask].any();f=z
    assert (f.sum(-1)>0).sum()>8
