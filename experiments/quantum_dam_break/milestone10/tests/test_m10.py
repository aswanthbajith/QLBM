import numpy as np
import pytest
from milestone10.kinetic import *
from milestone10.oracle import MarginalOracle,K,compositions
from milestone10.transport import permutation,apply_word_swaps
from milestone3.arithmetic import basis_apply,statevector_apply,signed
from quantum_core import apply_gates

def test_d2q9_impossible():
    assert not feasible(SETS['D2Q9'],1.,.5)
    assert feasible(SETS['D2Q25'],1.,.5)

def test_no_finite_grid_universally_realizable():
    for v in SETS.values():
        assert not feasible(v,1e-7,(v[1]+v[2])/2)

def test_bounds():
    v=SETS['D2Q25'];lo,hi=bounds(v,.25);assert lo==.125 and hi==1

def test_marginal_moments():
    v=SETS['D2Q25'];f,b=marginal(v,1.,.5);assert not b and f.min()>=0;assert np.allclose([f.sum(),f@v,f@(v*v)],[1,.5,.3125])

def test_projection():
    v=SETS['D2Q9'];f,b=marginal(v,1.,.5,project=True);assert b and f.sum()==1 and f@v==.5 and f@(v*v)==.5
    with pytest.raises(ValueError):marginal(v,1.,.5)

def test_product_moments():
    v=SETS['D2Q25'];f,b=equilibrium2d(v,np.array([1.,0.]),np.array([.5,0.]),np.array([-.5,0.]));assert not b.any();assert np.allclose(f[0].sum(),1);assert np.allclose(v@f[0]@v,-.25);assert np.allclose((v*v)@f[0].sum(1),.3125);assert f[1].sum()==0

def test_strip():
    for v in SETS.values():
        x,h,r=kinetic_strip(v,steps=4);assert len(r)==4 and all(a['valid'] and abs(a['mass_error'])<1e-10 for a in r)

def test_oracle_all_inputs():
    o=MarginalOracle();ks=np.array(K)
    for pat in range(1<<o.nin):
        H=pat&7;J=signed(pat>>o.nh,o.nj);w=basis_apply(pat,o.gates,o.nin);p,flag=o.decode(w);assert basis_apply(w,o.gates[::-1],o.nin)==pat
        assert flag==(abs(J)>2*H)
        if not flag:
            assert p.sum()==H and p@ks==J
            target=J*J/H+H*H/(4*o.scale) if H else 0
            best=min(abs(np.array(q)@(ks*ks)-target) for q in compositions(H,5) if np.array(q)@ks==J)
            assert abs(p@(ks*ks)-target)<=best+1e-12

def test_oracle_arbitrary_output():
    o=MarginalOracle(2);w=o.encode(3,2)|((1<<o.nout)-1)<<o.nin;assert basis_apply(basis_apply(w,o.gates,o.nin),o.gates[::-1],o.nin)==w

def test_oracle_coherence():
    o=MarginalOracle(2);s=np.zeros(1<<o.nqubits,complex);a=o.encode(3,1);b=o.encode(2,-1);s[a]=.6;s[b]=.8j;z=statevector_apply(s,o.gates,o.nin);assert z[basis_apply(a,o.gates,o.nin)]==.6 and z[basis_apply(b,o.gates,o.nin)]==.8j;assert np.array_equal(statevector_apply(z,o.gates[::-1],o.nin),s)

def test_multispeed_transport():
    mask=np.zeros((4,4),bool);mask[1,2]=1;p=permutation(mask,range(-2,3));n=16;v=[(x,y) for y in range(-2,3) for x in range(-2,3)];idx=v.index((2,0))*n+4;assert p[idx]==v.index((-2,0))*n+4 # link crosses solid
    words=np.arange(len(p));words.reshape(25,-1)[:,mask.ravel()]=0;z=apply_word_swaps(words,p);assert z.sum()==words.sum() and not z.reshape(25,-1)[:,mask.ravel()].any();assert np.array_equal(apply_word_swaps(z,np.argsort(p)),words)

def test_transport_statevector():
    p=permutation(np.zeros((2,2),bool),range(-2,3));pad=np.arange(128);pad[:len(p)]=p;s=np.arange(128,dtype=complex);s/=np.linalg.norm(s);assert np.array_equal(apply_gates(s,pad),s[np.argsort(pad)])
