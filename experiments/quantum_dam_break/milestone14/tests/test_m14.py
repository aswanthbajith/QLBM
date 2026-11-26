import numpy as np
import pytest
from milestone14.twirl import TwirledCollision,GROUP,action,reference_branches,coherent_dilation
from milestone11.reference import V

@pytest.fixture(scope='module')
def oracle():return TwirledCollision(32,omega_num=2)

def samples(n=24):
    rng=np.random.default_rng(14)
    return np.array([rng.multinomial(int(h),rng.dirichlet(np.ones(25)*.4)) for h in rng.integers(1,64,n)])

def test_group():
    assert len({tuple(p) for p in GROUP})==8
    assert all(tuple(p[q]) in {tuple(r) for r in GROUP} for p in GROUP for q in GROUP)

def test_branch_reference(oracle):
    f=samples();y,flag=oracle.branches(f)
    assert not flag.any()
    assert np.array_equal(y,np.array([reference_branches(a,oracle.S,oracle.w,oracle.omega_num)[0] for a in f]))
    assert np.all(y>=0)
    assert np.all(y.sum(-1)==f.sum(-1)[:,None])
    assert np.all(y@V==(f@V)[:,None,:])

def test_covariant_distribution(oracle):
    f=samples(12);y,_=oracle.branches(f)
    for p in GROUP:
        z,_=oracle.branches(action(f,p));expect=action(y,p)
        assert all(sorted(map(tuple,a))==sorted(map(tuple,b)) for a,b in zip(z,expect))

def test_output_xor_and_overflow(oracle):
    f=samples(12);f[0]=2**oracle.w-1
    rng=np.random.default_rng(6);g=rng.integers(0,8,len(f));out=rng.integers(0,2**oracle.w,f.shape);flag=rng.integers(0,2,len(f)).astype(bool)
    y,b=oracle.evaluate(f,g);z,c=oracle.evaluate(f,g,out=out,flag=flag)
    assert b[0] and np.array_equal(y[0],f[0])
    assert np.array_equal(z,y^out) and np.array_equal(c,b^flag)

def test_coherence(oracle):
    f=samples(2);a=np.array([1,1j])/np.sqrt(2);state,error=coherent_dilation(oracle,f,a)
    assert error<1e-12 and len(state)==16
    assert np.isclose(sum(abs(x)**2 for x in state.values()),1)
    for (v,y,b,g),amp in state.items():
        i=0 if tuple(f[0])==v else 1
        assert np.isclose(amp,a[i]/np.sqrt(8))
        assert np.array_equal(y,reference_branches(v,oracle.S,oracle.w,oracle.omega_num)[0][g])

def test_qiskit_optional():
    pytest.importorskip('qiskit')
    # Avoid a dense simulation of the huge arithmetic workspace.
    o=TwirledCollision(4,w=3);qc=o.qiskit()
    assert qc.num_qubits==o.nqubits and qc.count_ops()['h']==3
