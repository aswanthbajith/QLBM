import numpy as np
import pytest
from milestone16.phase_field import EdgeFlux,CellUpdate,PhaseStepper,CoherentGrid,decode_signed
from milestone16.reference import round_ratio,integer_step,energy,float_step

@pytest.mark.parametrize('S',[8,256,65536])
def test_edge_exact_signed(S):
    o=EdgeFlux(S);rng=np.random.default_rng(42);x=rng.integers(-4*S,4*S,size=(256,4));x[:2]=[[-4*S,-4*S,4*S-1,4*S-1],[0,0,0,0]]
    expected=[]
    for l,p,q,r in x.tolist():expected.append(round_ratio(p**3-q**3+S*S*(2*p-2*q-l+r),64*S*S))
    assert np.array_equal(o.evaluate(x),expected)
    assert np.array_equal(o.evaluate(x[:,::-1]),-np.asarray(expected))

def test_xor_and_inverse():
    o=EdgeFlux(16);x=np.array([[3,-7,19,5],[-20,16,-30,22]]);initial=np.array([123,85]);actual=o.run(x,initial_output=initial);assert np.array_equal(actual,o.run(x)^initial)

def test_overflow_flag_and_signed_update():
    o=CellUpdate(6,8);x=np.array([[31,1,0],[-32,-1,0],[-4,7,-9],[0,-128,127]])
    y,flag=o.evaluate(x);assert np.array_equal(flag,[1,1,0,1]);assert y[2]==12

@pytest.mark.parametrize('S',[256,4096])
def test_step_reference_mass_translation_reflection(S):
    s=PhaseStepper(S);p=np.random.default_rng(4).integers(-S,S+1,16);q=s.step(p)
    assert np.array_equal(q,integer_step(p,S));assert q.sum()==p.sum()
    assert np.array_equal(s.step(np.roll(p,3)),np.roll(q,3));assert np.array_equal(s.step(p[::-1]),q[::-1])
    for v in [-S,0,S,S//2]:assert np.array_equal(s.step(np.full(16,v)),np.full(16,v))

def test_coherent_two_steps():
    S=64;c=CoherentGrid(S,4,2);x=np.array([[40,0,-40,0],[20,-10,-20,10]]);a=np.array([1,1j])/np.sqrt(2);h,b=c.execute(x,a,copy_final=True)
    for i,p in enumerate(x):
        for t in range(2):p=integer_step(p,S);assert np.array_equal(h[i,t+1],p)
    assert np.array_equal(a,b);assert np.all(h.sum(axis=2)==x.sum(axis=1)[:,None])

def test_phase_energy_and_growth():
    S=65536;s=PhaseStepper(S);x=np.arange(32);p=np.rint(.05*np.cos(2*np.pi*3*x/32)*S).astype(int);e=energy(p/S);r=p/S
    for _ in range(32):p=s.step(p);r=float_step(r)
    assert energy(p/S)<e;assert np.max(np.abs(p/S-r))<.001;assert np.max(p)>np.rint(.05*S)

def test_optional_qiskit():
    pytest.importorskip('qiskit');from qiskit import QuantumCircuit
    o=EdgeFlux(8);q=o.qiskit();assert isinstance(q,QuantumCircuit);assert len(q.data)==len(o.gates)

@pytest.mark.parametrize('S',[256,65536])
def test_3d_chemical_and_step(S):
    from milestone16.phase_field_3d import ChemicalPotential,PhaseStepper3D
    from milestone16.reference import integer_step3
    rng=np.random.default_rng(5);o=ChemicalPotential(S);v=rng.integers(-4*S,4*S,(128,7));expected=v[:,0]**3+S*S*(5*v[:,0]-v[:,1:].sum(axis=1));assert np.array_equal(o.evaluate(v),expected)
    p=rng.integers(-S,S+1,(4,4,4));s=PhaseStepper3D(S);q=s.step(p);assert np.array_equal(q,integer_step3(p,S));assert q.sum()==p.sum()
    assert np.array_equal(s.step(p.transpose(2,0,1)),q.transpose(2,0,1));assert np.array_equal(s.step(p[::-1]),q[::-1]);assert np.array_equal(s.step(np.roll(p,1,axis=2)),np.roll(q,1,axis=2))

def test_3d_constant_state():
    from milestone16.phase_field_3d import PhaseStepper3D
    s=PhaseStepper3D(64);p=np.full((3,3,3),-32);assert np.array_equal(s.step(p),p)
