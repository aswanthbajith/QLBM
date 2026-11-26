import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from solver import *
import reference

@pytest.mark.parametrize('h',[0,1,4,7])
def test_uniform_rest(h):
    c=StepCircuit(3); assert c.step([(h,0)]*3)==[(h,0)]*3

def test_signed_adder_exhaustive():
    for sub in [False,True]:
        g=add_gates(0,3,3,4,sub)
        for w in range(128):
            z=w
            for m,p,t in g: z=gate(z,m,p,t)
            assert z&7==w&7
            assert z>>3==((w>>3)+(-1 if sub else 1)*signed(w&7,3))%16

def test_face_oracle_all_patterns():
    for w,js in enumerate(oracle_patterns('face')):
        l,r=unpack(w&127),unpack(w>>7)
        z=sum(1<<j for j in js)
        m,p=signed(z&63,6),signed(z>>6,6)
        if not (admissible(*l) and admissible(*r)):
            assert (m,p)==(0,0); continue
        d=reference.face(np.array(l)/4,np.array(r)/4,True)
        assert np.allclose(np.array([m,p])/4,d)
        assert -(r[0]//2)<=m<=l[0]//2

def test_full_gate_stream_and_inverse():
    c=StepCircuit(2); rng=np.random.default_rng(22)
    for _ in range(6):
        w=int.from_bytes(rng.bytes(8),'little')&((1<<c.nqubits)-1)
        z=w
        for m,p,t in c.iter_gates(): z=gate(z,m,p,t)
        assert z==c.apply(w); assert c.apply(z,True)==w

def test_coherent_sparse_permutation():
    c=StepCircuit(2); words=[c.encode([(6,0),(0,0)]),c.encode([(0,0),(6,0)])]
    state=dict(zip(words,[1/np.sqrt(2),1j/np.sqrt(2)])); out={c.apply(w):a for w,a in state.items()}
    assert len(out)==2 and np.isclose(sum(abs(a)**2 for a in out.values()),1)
    assert {c.apply(w,True):a for w,a in out.items()}==state

def test_dam_break():
    c=StepCircuit(6); s=[(6,0)]*3+[(0,0)]*3
    for i in range(8):
        w=c.encode(s); z=c.apply(w); assert c.apply(z,True)==w; assert z>>c.scratch==0
        ref=reference.step(np.array(s)/4,True); s=c.step(s)
        assert np.allclose(np.array(s)/4,ref); assert sum(h for h,q in s)==18
        assert all(admissible(h,q) for h,q in s)
    assert all(h>0 for h,q in s)

def test_invalid_rejected():
    c=StepCircuit(2)
    with pytest.raises(ValueError): c.step([(0,1),(4,0)])
    with pytest.raises(ValueError): c.step([(8,0),(4,0)])

def test_random_conservation_before_domain_check():
    c=StepCircuit(4); rng=np.random.default_rng(7)
    for _ in range(100):
        h=rng.integers(0,8,4); q=[rng.integers(-v,v+1) for v in h]; s=list(zip(map(int,h),map(int,q)))
        w=c.encode(s); z=c.apply(w); out=c.decode(z)
        assert sum(v[0] for v in out)==sum(h); assert min(v[0] for v in out)>=0
        assert c.apply(z,True)==w; assert z>>c.scratch==0
