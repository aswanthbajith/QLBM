import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from milestone6.arithmetic import FluxOracle
from milestone6.solver import Arithmetic2D
from milestone5.solver2d import Format,table,gate
from milestone5.reference2d import quantized_step

@pytest.mark.parametrize('sign',[-1,1])
def test_exhaustive_small(sign):
    a=FluxOracle(2,sign);t=table(Format(2),sign)
    for k,v in enumerate(t):assert a.evaluate(k)==v

@pytest.mark.parametrize('scale',[8,32])
@pytest.mark.parametrize('sign',[-1,1])
def test_random_large(scale,sign):
    a=FluxOracle(scale,sign);t=table(Format(scale),sign);rng=np.random.default_rng(7)
    for k in rng.integers(0,len(t),100):assert a.evaluate(int(k))==t[k]
    assert all(m.bit_count()<=2 for m,p,t in a.gates)

def test_dirty_register_inverse():
    a=FluxOracle(4,1);rng=np.random.default_rng(31)
    for _ in range(5):
        w=int.from_bytes(rng.bytes((a.nqubits+7)//8),'little')&((1<<a.nqubits)-1)
        assert a.apply(a.apply(w),True)==w

def test_superposition_permutation():
    a=FluxOracle(4,-1);keys=[a.pack(6,0,0),a.pack(5,2,-2)];state=dict(zip(keys,[2**-.5,1j*2**-.5]));out={a.apply(k):v for k,v in state.items()}
    assert len(out)==2 and np.isclose(sum(abs(v)**2 for v in out.values()),1)
    assert {a.apply(k,True):v for k,v in out.items()}==state

def test_integrated_gate_stream():
    c=Arithmetic2D(np.zeros((1,2),bool),2);w=c.encode(np.array([[[3,0,0],[0,0,0]]]))
    z=w;n=0
    for m,p,t in c.iter_gates():z=gate(z,m,p,t);n+=1
    assert n==c.gate_count() and z==c.apply(w) and c.apply(z,True)==w

def test_obstacle_and_reference():
    mask=np.zeros((3,5),bool);mask[1,3]=True;c=Arithmetic2D(mask,16);s=np.zeros((3,5,3),int);s[:,:2,0]=24;m=s[:,:,0].sum()
    for _ in range(8):
        z=c.step(s,True);assert np.array_equal(z,quantized_step(s,mask,16));assert z[:,:,0].sum()==m and not np.any(z[mask]);s=z

def test_scaling():
    a=FluxOracle(32,1);b=FluxOracle(64,1);assert len(b.gates)<2*len(a.gates)
