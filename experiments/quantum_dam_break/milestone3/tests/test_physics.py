import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from fractions import Fraction
from physics import *
from arithmetic import *
from run_physics import shear

@pytest.mark.parametrize('h',[.25,1,3,6,9.75])
def test_equilibrium(h):
    w,p,c=eos(h)
    assert np.isclose(w@M,h); assert np.allclose(w@P,0)
    assert np.allclose(transition(.7)@w,w)
    assert np.allclose(np.einsum('s,sab->ab',w,STRESS),p*np.eye(2))
    eps=1e-4; assert np.isclose(c,(eos(h+eps)[1]-eos(h-eps)[1])/(2*eps),rtol=1e-5)

def test_eos_mismatch():
    h=np.linspace(.25,9.75,39); p=np.array([eos(x)[1] for x in h]); v=h*h/2; g=v@p/(v@v)
    assert np.linalg.norm(p-g*v)/np.linalg.norm(p)>.2

@pytest.mark.parametrize('axis',[0,1])
def test_force_kernel(axis):
    k=force_kernel(axis,.4,.25); assert k.min()>=0; assert np.allclose(k.sum(0),1)
    assert np.allclose(M@k,M); assert np.allclose(P[:,1-axis]@k,P[:,1-axis])
    assert np.allclose(P[:,axis]@k-P[:,axis],expected_impulse(np.arange(512),axis,.4,.25))
    assert np.allclose(force_kernel(axis,0,0),np.eye(512))

def test_force_sampling():
    rng=np.random.default_rng(91); s=np.full(80000,(1<<1)|(1<<6),np.int16); z=force(s,0,.4,.25,rng)
    assert np.array_equal(M[s],M[z]); assert np.array_equal(P[s,1],P[z,1]); assert np.isclose((P[z,0]-P[s,0]).mean(),expected_impulse(s,0,.4,.25)[0],atol=.015)

def test_shear_conservation():
    rows,meta=shear(shots=128,steps=20)
    assert all(r['mass_error']==r['momentum_error']==0 for r in rows)
    assert rows[0]['shear_amplitude_lu']>rows[20]['shear_amplitude_lu']

def test_pressure_all_basis():
    g=compile_pressure()
    for w in range(512):
        h=w&7; q=w>>3; out=basis_apply(w,g,3)
        assert out&7==h; assert out>>3==q^pressure_word(h); assert basis_apply(out,g,3)==w
    assert max(abs(pressure_word(h)/16-h*h/32) for h in range(8))<=1/32

def test_pressure_coherent_uncompute():
    rng=np.random.default_rng(4); v=rng.normal(size=512)+1j*rng.normal(size=512); v/=np.linalg.norm(v); g=compile_pressure(); out=statevector_apply(v,g,3)
    assert np.isclose(np.linalg.norm(out),1); assert np.allclose(statevector_apply(out,g,3),v)

def test_bed_source_all_basis():
    for a in [Fraction(-1,4),Fraction(1,4),Fraction(0)]:
        g=compile_bed_source(accel_dt=a)
        for w in range(256):
            h=w&7; q=w>>3; out=apply_controlled(w,g)
            assert out&7==h; assert out>>3==(q+int(4*a*h))%32
            assert apply_controlled(out,g[::-1])==w

def test_overflow_guards():
    with pytest.raises(OverflowError): compile_pressure(np_=1)
    with pytest.raises(OverflowError): compile_bed_source(accel_dt=Fraction(10))
