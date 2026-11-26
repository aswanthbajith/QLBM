import numpy as np
import pytest
from milestone18.transport import *
from milestone18.channel import circuit,apply,rail,decode,shots,qiskit_circuit
from milestone18.hybrid import init,step

@pytest.mark.parametrize('c',[C7,C19])
@pytest.mark.parametrize('periodic',[False,True])
def test_all_links(c,periodic):
    solid=np.zeros((4,4,4),bool);solid[0]=True;solid[1,1,1]=True
    t=LinkTransport(solid,c,periodic);rng=np.random.default_rng(42);f=rng.integers(-100,101,size=(len(c),)+solid.shape);f[:,solid]=0
    state=np.zeros(t.dim,complex);state[:f.size]=f.ravel();a=t.apply(state)
    np.testing.assert_array_equal(a[:f.size].real.reshape(f.shape),reference_stream(f,solid,c,periodic))
    assert not np.any(a[t.invalid]); assert a.sum()==state.sum();np.testing.assert_array_equal(t.apply(a,inverse=True),state)
    np.testing.assert_array_equal(t.apply(state,direct=True),a)
    assert all((int(x)^int(y)).bit_count()==1 for x,y in t.gates)

@pytest.mark.parametrize('c',[C7,C19])
def test_single_wall_link(c):
    mask=np.zeros((4,4,4),bool);mask[0]=True;t=LinkTransport(mask,c)
    f=np.zeros((len(c),)+mask.shape);f[6,1,1,1]=1
    q=t.stream(f);assert q[5,1,1,1]==1 and q.sum()==1 and not q[:,mask].any()
    old=legacy_stream(f,mask,c);assert old.sum()==0

def test_complex_superposition_and_padding():
    mask=np.zeros((4,4,4),bool);mask[0]=True;t=LinkTransport(mask,C19)
    rng=np.random.default_rng(2);v=rng.normal(size=t.dim)+1j*rng.normal(size=t.dim);v/=np.linalg.norm(v)
    q=t.apply(v);np.testing.assert_array_equal(q[t.perm],v);np.testing.assert_array_equal(q[t.invalid],v[t.invalid]);np.testing.assert_array_equal(t.apply(q,True),v)

@pytest.mark.parametrize('lam',[-1,-.25,0,.4,1])
def test_quantum_channel(lam):
    u=circuit(lam);np.testing.assert_allclose(u.T@u,np.eye(4),atol=1e-14)
    rho=np.array([[.7,.1+.2j],[.1-.2j,.3]]);x=np.array([[0,1],[1,0]])
    out=apply(rho,lam);np.testing.assert_allclose(out,(1+lam)/2*rho+(1-lam)/2*x@rho@x,atol=1e-14)
    assert np.min(np.linalg.eigvalsh(out))>=-1e-14 and abs(np.trace(out)-1)<1e-14
    assert abs(decode(apply(rail(-.17,.5),lam),.5)+.17*lam)<1e-14

def test_channel_reject_invalid():
    with pytest.raises(ValueError):rail(1,.5)
    with pytest.raises(ValueError):circuit(1.01)

def test_hybrid_matches_reference():
    mask=np.zeros((4,4,4),bool);mask[0]=True;mask[1,1,1]=True;t=(LinkTransport(mask,C19),LinkTransport(mask,C7));f,g=init(mask);fr,gr=f.copy(),g.copy();mass=f.sum();phase=g.sum()
    for _ in range(8):
        f,g=step(f,g,mask,'gate',t);fr,gr=step(fr,gr,mask,'reference');np.testing.assert_allclose(f,fr,atol=1e-13);np.testing.assert_allclose(g,gr,atol=1e-13)
        assert abs(f.sum()-mass)<1e-12 and abs(g.sum()-phase)<1e-12;assert not f[:,mask].any() and not g[:,mask].any()

def test_native_qiskit_optional():
    pytest.importorskip('qiskit');from qiskit.quantum_info import Operator
    np.testing.assert_allclose(Operator(qiskit_circuit(-.25)).data,circuit(-.25),atol=1e-14)
    t=LinkTransport(np.zeros((2,2,2),bool),C7);v=np.arange(t.dim,dtype=complex);np.testing.assert_allclose(Operator(t.qiskit()).data@v,t.apply(v),atol=1e-10)

@pytest.mark.parametrize('lam',[-1,-.25,0,.4,1])
def test_published_dual_rail_damping(lam):
    from milestone18.channel import damping_circuit,dual_rail,apply_damping,dual_decode
    u=damping_circuit(lam);np.testing.assert_allclose(u.T@u,np.eye(16),atol=1e-14)
    for d in [-.5,-.17,0,.21,.5]:
        r=dual_rail(d,.5)
        for t in range(1,5):
            r=apply_damping(r,lam);assert abs(np.trace(r)-1)<1e-14 and np.min(np.linalg.eigvalsh(r))>=-1e-14
            assert abs(dual_decode(r,.5)-d*lam**t)<1e-14
