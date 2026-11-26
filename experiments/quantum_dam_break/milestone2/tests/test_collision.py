import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from collision import *
from run_collision import run

def test_unitary():
    assert np.allclose(U.conj().T@U,np.eye(512),atol=1e-12)

def test_all_basis_conservation():
    a,b=np.nonzero(K>1e-14)
    assert np.array_equal(M[a],M[b]); assert np.array_equal(P[a],P[b])

def test_transition():
    for w in [0,.3,1]:
        t=transition(w); assert np.all(t>=0); assert np.allclose(t.sum(0),1); assert np.allclose(t.sum(1),1)
        assert np.allclose(M@t,M); assert np.allclose(P.T@t,P.T)

def test_stress_relaxation():
    # E-W pair has anisotropic stress and belongs to a zero-momentum mass-2 class.
    s=(1<<1)|(1<<3); q=STRESS[:,0,0]-STRESS[:,1,1]
    assert q[s]==2
    for w in [.2,.7,1]: assert np.isclose(q@transition(w)[:,s],2*(1-w))

def test_coherent_input_channel():
    v=np.zeros(512,complex); v[1]=1/np.sqrt(2); v[2]=1j/np.sqrt(2)
    out=channel(np.outer(v,v.conj()),.7)
    assert np.allclose(out,np.diag(np.diag(out))); assert np.isclose(np.trace(out),1); assert np.diag(out).min()>=-1e-12

def test_swap_compilation():
    solid=np.zeros((4,8),bool); solid[1,3]=1
    for periodic in [False,True]:
        p=permutation(8,4,solid,periodic); a=np.arange(len(p))
        for x,y in swaps(p): a[x],a[y]=a[y],a[x]
        assert np.array_equal(a,np.argsort(p))

def test_stream_periodic_momentum():
    rng=np.random.default_rng(10); states=rng.integers(0,512,(20,4,8)); p=permutation(8,4,np.zeros((4,8),bool),True)
    out=stream(states,p); assert np.array_equal(M[out].sum((1,2)),M[states].sum((1,2))); assert np.array_equal(P[out].sum((1,2)),P[states].sum((1,2)))

def test_reflection():
    solid=np.zeros((4,4),bool); solid[1,2]=1; p=permutation(4,4,solid)
    s=np.zeros((1,4,4),int); s[0,1,1]=1<<1; out=stream(s,p); assert out[0,1,1]==1<<3; assert not out[:,solid].any()

def test_trajectory_kernel():
    rng=np.random.default_rng(42); s=np.full(50000,1); o=collide(s,.7,rng); empirical=np.bincount(o,minlength=512)/len(o)
    assert np.max(abs(empirical-transition(.7)[:,1]))<.01

def test_end_to_end(tmp_path):
    rows=run(tmp_path,shots=128,steps=4)
    assert all(r['max_mass_error_per_trajectory']==r['collision_mass_error']==r['collision_momentum_error']==r['solid_mass_lu']==0 for r in rows)
    assert rows[-1]['right_half_mass_lu']>0
    assert len(list(tmp_path.glob('*.vti')))==5
