import numpy as np
import pytest
from milestone19.collision import M19,M7,moment_basis,relax_gate,collide
from milestone19.solver import DamBreak,initial,push_reference,derivatives,fields
from milestone18.transport import C19,C7,reference_stream

@pytest.mark.parametrize('M,c',[(M19,C19),(M7,C7)])
def test_basis(M,c):
    assert np.max(abs(M@M.T-np.eye(len(c))))<1e-12
    assert np.allclose(M[0],np.ones(len(c))/np.sqrt(len(c)))
    assert np.allclose(M[1:4],c.T/np.linalg.norm(c,axis=0)[:,None])

@pytest.mark.parametrize('lam',[-1.,-.9,-.25,0.,.5,1.])
def test_rail(lam):
    d=np.linspace(-.25,.25,101);assert np.max(abs(relax_gate(d,lam,.25)-d*lam))<1e-15

def test_bound_and_shots():
    with pytest.raises(ValueError):relax_gate(np.array([.3]),.4,.25)
    d=np.array([-.15,0,.1]);a=relax_gate(d,.5,.25,256,np.random.default_rng(12));b=relax_gate(d,.5,.25,256,np.random.default_rng(12));assert np.array_equal(a,b);assert a[1]==0

@pytest.mark.parametrize('c',[C7,C19])
def test_push(c):
    shape=(4,4,8);f=np.random.default_rng(4).normal(size=(len(c),)+shape)
    assert np.array_equal(push_reference(f,c),reference_stream(f,np.zeros(shape,bool),c,False))

def test_uniform_derivatives():
    g,l=derivatives(np.ones((4,4,8)));assert not np.any(g);assert not np.any(l)

def test_mrt_bgk():
    rng=np.random.default_rng(31);f=rng.random((19,4,3,2));d=rng.normal(size=f.shape)*.003;d=(M19.T@np.r_[np.zeros((4,24)),(M19@d.reshape(19,-1))[4:]]).reshape(f.shape);e=f+d
    rates=np.r_[np.zeros(4),np.full(15,1.2)]
    assert np.allclose(collide(f,e,np.zeros_like(f),M19,rates,bound=.25),f+1.2*d,atol=2e-14)

def test_coupled_conservation_and_reference():
    s=DamBreak((4,4,8));f,g=initial(s.shape);r,h=f.copy(),g.copy();mass=f.sum();phase=g.sum()
    for _ in range(8):
        f,g,m,j,b=s.step(f,g);r,h,*_=s.step(r,h,'reference');assert max(m,j,b)<1e-10
        assert np.max(abs(f-r))<1e-12;assert np.max(abs(g-h))<1e-12
        assert abs(f.sum()-mass)<1e-9;assert abs(g.sum()-phase)<1e-9

def test_sampled_conserved_moments():
    s=DamBreak((4,4,8));f,g=initial(s.shape);f1,g1,m,j,b=s.step(f,g,shots=64,rng=np.random.default_rng(9));assert max(m,j,b)<1e-10
    assert abs(f1.sum()-f.sum())<1e-9;assert abs(g1.sum()-g.sum())<1e-9


def test_optional_native_circuit():
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    from milestone19.collision import qiskit_mode
    for d in [-.13,.0,.13]:
        for lam in [-.25,.5]:
            p=Statevector.from_instruction(qiskit_mode(d,lam)).probabilities().reshape(4,4).sum(0)
            assert abs(.25*(p[1]-p[2])-lam*d)<1e-13
