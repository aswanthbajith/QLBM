import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from milestone5.solver2d import Circuit2D,Format,table,gate,signed
from milestone5 import reference2d as ref

@pytest.mark.parametrize('scale',[8,16,32])
def test_dam_and_reference(scale):
    mask=np.zeros((4,8),bool);mask[1:3,5]=True
    c=Circuit2D(mask,scale);s=np.zeros((4,8,3),int);s[:,:3,0]=3*scale//2;m=s[:,:,0].sum()
    for i in range(24):
        z=c.step(s,verify_inverse=i==0);assert np.array_equal(z,ref.quantized_step(s,mask,scale));s=z
        assert s[:,:,0].sum()==m and np.min(s[:,:,0])>=0 and np.all(s[mask]==0)
    assert np.count_nonzero(s[:,:,0])>12
    if scale==32: assert np.any(s[:,:,2])

@pytest.mark.parametrize('scale',[2,8,32])
def test_uniform_rest(scale):
    mask=np.zeros((3,4),bool);mask[1,1]=True;c=Circuit2D(mask,scale)
    s=np.zeros((3,4,3),int);s[~mask,0]=scale
    assert np.array_equal(s,c.step(s,True))

def test_arbitrary_basis_roundtrip_and_gate_count():
    c=Circuit2D(np.zeros((1,2),bool),4);rng=np.random.default_rng(19);g=list(c.iter_gates());assert len(g)==c.gate_count()
    for i in range(5):
        w=int.from_bytes(rng.bytes((c.nqubits+7)//8),'little')&((1<<c.nqubits)-1);z=w
        for m,p,t in g:z=gate(z,m,p,t)
        assert z==c.apply(w) and c.apply(z,True)==w

def test_sparse_coherent_permutation():
    c=Circuit2D(np.zeros((1,2),bool),8)
    a=c.encode(np.array([[[12,0,0],[0,0,0]]]))
    b=c.encode(np.array([[[0,0,0],[12,0,0]]]))
    v={a:1/np.sqrt(2),b:1j/np.sqrt(2)};out={c.apply(k):z for k,z in v.items()}
    assert len(out)==2 and np.isclose(sum(abs(z)**2 for z in out.values()),1)
    assert {c.apply(k,True):z for k,z in out.items()}==v

def test_mass_outflow_bound_all_local_codes():
    f=Format(8);p=table(f,1);m=table(f,-1);msk=(1<<f.fb)-1
    for h in range(2*f.scale):
        for x in range(-16,16):
            for y in range(-16,16):
                if not f.valid(h,x,y):continue
                k=lambda x,y:h|((x&31)<<f.hb)|((y&31)<<(f.hb+f.qb))
                assert sum(int(v)&msk for v in (p[k(x,y)],m[k(x,y)],p[k(y,x)],m[k(y,x)]))<=h

def test_reject_invalid_no_wraparound():
    c=Circuit2D(np.zeros((1,1),bool),8)
    for v in [(15,20,0),(0,1,0),(16,0,0),(-1,0,0)]:
        with pytest.raises(ValueError):c.encode(np.array([[v]]))

def test_reflection_transverse_momentum():
    c=Circuit2D(np.array([[False,True],[True,True]]),32);s=np.zeros((2,2,3),int);s[0,0]=[48,12,8]
    z=c.step(s,True);assert z[0,0,0]==48 and np.all(z[c.solid]==0)
    assert np.array_equal(z,ref.quantized_step(s,c.solid,32))

def test_ritter_initial_and_front():
    x=np.array([-3.,0,3.]);h,u=ref.ritter(x,1)
    assert np.isclose(h[0],1.5) and np.isclose(h[1],2/3) and h[2]==0
    assert np.isclose(u[1],2*np.sqrt(1.5)/3)
