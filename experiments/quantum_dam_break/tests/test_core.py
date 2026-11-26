import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from quantum_core import *
from run_demo import run

def test_encoding():
    g=Grid(); a,M=initial(g); f,h,u=decode(g,probabilities(a),M)
    assert np.allclose(h[:,:2],1) and np.allclose(h[:,2:],0)
    assert np.allclose(u,0) and np.isclose(np.linalg.norm(a),1)

def test_invalid_encoding():
    g=Grid(); f=np.zeros((4,4,9)); f[0,0,0]=-1
    with pytest.raises(ValueError): encode(g,f)

@pytest.mark.parametrize('periodic',[True,False])
def test_gate_compilation(periodic):
    g=Grid(); p=stream_permutation(g,periodic); rng=np.random.default_rng(2)
    a=rng.normal(size=g.dim)+1j*rng.normal(size=g.dim); a/=np.linalg.norm(a)
    assert np.allclose(apply_gates(a,p),apply_permutation(a,p))
    assert np.array_equal(np.sort(p),np.arange(g.dim))

def test_stream_and_reflect():
    solid=np.zeros((4,4),bool); solid[1,2]=True; g=Grid(solid=solid); p=stream_permutation(g)
    assert p[g.index(0,0,1)]==g.index(1,0,1)
    assert p[g.index(1,1,1)]==g.index(1,1,3)
    assert p[g.index(0,0,3)]==g.index(0,0,1)
    assert p[g.index(0,0,0)]==g.index(0,0,0)

def test_reset_channel():
    g=Grid(); a=np.zeros(g.dim,complex); a[g.index(1,2,1)]=1; rho=np.outer(a,a.conj())
    assert np.allclose(relaxation_channel(g,rho,0),rho)
    r=relaxation_channel(g,rho,1); f,h,u=decode(g,probabilities(r),1)
    assert np.allclose(f[2,1],W) and np.allclose(u,0)
    assert np.isclose(np.trace(r),1) and np.linalg.eigvalsh(r).min()>-1e-10

def test_shots():
    g=Grid(); a,M=initial(g); c=counts(g,a,4096,123)
    assert c.sum()==4096; assert np.count_nonzero(c[9*g.npos:])==0

def test_demo(tmp_path):
    rows=run(tmp_path,steps=4)
    assert all(np.isclose(r['encoded_volume_m3'],8) and np.isclose(r['trace'],1) and r['solid_population']<1e-10 for r in rows)
    assert len(list(tmp_path.glob('*.vti')))==5
    assert rows[-1]['max_depth_proxy_m']<rows[0]['max_depth_proxy_m']
