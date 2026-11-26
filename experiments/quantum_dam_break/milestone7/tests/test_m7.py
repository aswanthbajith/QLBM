import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from milestone7.optimization import CompactFlux
from milestone7.drag import DragOracle
from milestone7.solver import Damped2D
from milestone7.reference import quantized_step,drag_ticks,planar_reference
from milestone6.arithmetic import FluxOracle
from milestone4.solver import gate

@pytest.mark.parametrize('sign',[-1,1])
def test_compaction_exhaustive(sign):
    a=CompactFlux(2,sign);b=FluxOracle(2,sign)
    for k in range(1<<a.inputs):assert a.evaluate(k)==b.evaluate(k)
    assert a.ancillas<b.ancillas

@pytest.mark.parametrize('sign',[-1,1])
def test_compaction_sampled(sign):
    a=CompactFlux(32,sign);b=FluxOracle(32,sign);rng=np.random.default_rng(19)
    keys=list(rng.integers(0,1<<a.inputs,60))+[a.pack(48,x,y) for x,y in [(0,0),(2,-3),(-16,9),(24,24)]]
    for k in keys:assert a.evaluate(int(k))==b.evaluate(int(k))
    assert a.ancillas<.65*b.ancillas and len(a.gates)<1.1*len(b.gates)

@pytest.mark.parametrize('numerator',[0,1,7,8])
def test_drag_exhaustive(numerator):
    a=DragOracle(6,numerator)
    for q in range(-32,32):
        v=a.evaluate(q);assert v==drag_ticks(q,numerator);assert abs(v)<=abs(q)
        w=q&63;assert a.apply(a.apply(w),True)==w

def test_superposition_and_arbitrary_targets():
    a=DragOracle(4);state={1:2**-.5,15:1j*2**-.5};z={a.apply(k):v for k,v in state.items()};assert len(z)==2 and {a.apply(k,True):v for k,v in z.items()}==state
    rng=np.random.default_rng(33)
    for _ in range(4):
        w=int.from_bytes(rng.bytes((a.nqubits+7)//8),'little')&((1<<a.nqubits)-1);assert a.apply(a.apply(w),True)==w

def test_expanded_composite():
    c=Damped2D(np.zeros((1,2),bool),4);s=np.array([[[6,2,0],[0,0,0]]]);w=c.encode(s);z=w;n=0
    for m,p,t in c.iter_gates():z=gate(z,m,p,t);n+=1
    assert n==c.gate_count() and z==c.apply(w) and c.apply(z,True)==w
    assert np.array_equal(c.decode(z),quantized_step(s,c.solid,4))

def test_damped_obstacle():
    mask=np.zeros((3,6),bool);mask[1,3]=True;c=Damped2D(mask,32);s=np.zeros((3,6,3),int);s[:,:2,0]=48;m=s[:,:,0].sum()
    for _ in range(6):
        z=c.step(s,True);assert np.array_equal(z,quantized_step(s,mask,32));assert z[:,:,0].sum()==m and not np.any(z[mask]);s=z

def test_drag_rounding_pins_small_values():
    a=DragOracle(8);assert a.evaluate(4)==4 and a.evaluate(-4)==-4;assert a.evaluate(5)==4

def test_planar_reference():
    s=planar_reference(64,32);assert abs(s[:,0].sum()*.25-12)<1e-12 and s[:,0].min()>=0
