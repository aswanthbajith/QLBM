import numpy as np
import pytest
from milestone17.coupling import CoupledStepper,FaceFlux,ProjectedForce,VelocityUpdate,AddPair,ScaledChemical
from milestone17.reference import integer_step,float_rhs,balance,mode,rr
from milestone16.phase_field import pack_words,read_words
from milestone8.collision import _execute

@pytest.fixture(scope='module')
def solver():return CoupledStepper(4096,(4,4,4))

@pytest.mark.parametrize('a',[0,800,-1200])
def test_gate_trajectory(solver,a):
    S=solver.S;rng=np.random.default_rng(83);p=rng.integers(-S,S+1,solver.shape);mass=p.sum()
    for _ in range(5):
        expected=integer_step(p,a,S);q,v,F=solver.step(p,a)
        assert np.array_equal(q,expected[0]) and (v,F)==expected[1:]
        assert q.sum()==mass;p,a=q,v

def test_uniform_phase_and_zero_mean(solver):
    p=np.full(solver.shape,solver.S//2);q,a,F=solver.step(p,1000)
    assert np.array_equal(q,p) and F==0 and a==999
    assert (a*mode(p.shape)).sum()==0

def test_rest_and_no_feedback(solver):
    rng=np.random.default_rng(28);p=rng.integers(-solver.S,solver.S+1,solver.shape)
    q,a,F=solver.step(p,0,feedback=False)
    expected=integer_step(p,0,solver.S,False)
    assert np.array_equal(q,expected[0]) and a==0 and F==expected[2]

def test_unrounded_energy_exchange():
    rng=np.random.default_rng(10)
    for _ in range(10):
        p=rng.uniform(-1,1,(4,4,8));a=rng.uniform(-.3,.3);b=balance(p,a)
        assert abs(b['exchange_residual'])<1e-12
        assert abs(b['energy_rate_residual'])<1e-10
        assert b['dissipation']>=0 and b['viscous_dissipation']>=0

def test_force_reflection(solver):
    rng=np.random.default_rng(7);p=rng.integers(-solver.S,solver.S+1,solver.shape);a=300
    q,b,F=solver.step(p,a);r,c,G=solver.step(p[:,:,::-1],-a)
    assert np.array_equal(q[:,:,::-1],r) and b==-c and F==-G

def test_local_face_rounding_and_force(solver):
    S=solver.S;rng=np.random.default_rng(9);x=rng.integers(-S,S+1,(128,6));x[:,5]=rng.integers(0,2,128);x[:,2:4]*=3
    p,q,m,n,a,flag=x.T;s=1-2*flag
    assert np.array_equal(solver.flux.evaluate(x),rr(2*S*(m-n)+(p+q)*a*s,512*S))
    assert np.array_equal(solver.force.evaluate(x[:,[0,1,2,3,5]]),(p+q)*(m-n)*s)

def test_ties_and_dirty_output(solver):
    o=solver.flux;S=solver.S;x=np.array([[0,0,64,-64,0,0],[0,0,-64,64,0,0],[0,0,63,-64,0,0]])
    y=o.evaluate(x);assert list(y)==[1,-1,0]
    dirty=np.array([3,7,2]);raw=o.run(x,initial_output=dirty)
    assert np.array_equal(raw,dirty^(y&((1<<o.outputs)-1)))

def test_complex_sparse_composition(solver):
    # Same input words control both gate oracles. Distinct basis inputs remain
    # orthogonal; complex amplitude weights are unchanged by X/CX/CCX permutations.
    a=solver.flux;b=solver.force
    x=np.array([[300,-200,1000,-500,400,0],[-500,600,-1000,700,-300,1]],np.int64)
    amps=np.array([1,1j])/np.sqrt(2);offset=a.inputs+a.outputs;off2=offset+b.outputs;N=off2+max(a.ancillas,b.ancillas)
    z=np.zeros((N,1),np.uint64);off=0
    for j,w in enumerate(a.input_widths):pack_words(z,off,x[:,j],w);off+=w
    initial=z.copy();wa=np.cumsum((0,)+a.input_widths)
    ma=list(range(a.inputs+a.outputs))+list(range(off2,off2+a.ancillas))
    mb=sum([list(range(wa[j],wa[j+1])) for j in (0,1,2,3,5)],[])+list(range(offset,off2))+list(range(off2,off2+b.ancillas))
    def remap(g,m):return np.where(g<0,-1,np.asarray(m,np.int32)[np.maximum(g,0)]).astype(np.int32)
    ga,gb=remap(a.gates,ma),remap(b.gates,mb)
    _execute(ga,z);_execute(gb,z);assert not np.any(z[off2:])
    assert np.array_equal(read_words(z,a.inputs,2,1,a.outputs)[:,0],a.evaluate(x))
    assert np.array_equal(read_words(z,offset,2,1,b.outputs)[:,0],b.evaluate(x[:,[0,1,2,3,5]]))
    # Readout is validation only; reverse the full circuit.
    _execute(gb[::-1].copy(),z);_execute(ga[::-1].copy(),z);assert np.array_equal(z,initial)
    assert np.isclose(np.vdot(amps,amps),1) and amps[1]/amps[0]==1j

def test_qiskit_export():
    pytest.importorskip('qiskit');q=AddPair(3).qiskit();assert q.num_qubits==AddPair(3).nqubits

def test_global_two_step_circuit():
    from milestone17.coherent import CoherentCoupledGrid
    S=65536;s=CoherentCoupledGrid(S);z,y,x=np.indices((4,4,4))
    p=np.rint(.8*S*(.65*np.sin(2*np.pi*x/4)+.35*np.cos(2*np.pi*(x/4+y/4))+.15*np.cos(2*np.pi*z/4))).astype(np.int64)
    pp=np.stack([p,p[:,:,::-1]]);a=np.array([0,-10000]);amps=np.array([1,1j])/np.sqrt(2)
    h,out=s.execute(pp,a,amps,copy_final=True);assert np.array_equal(out,amps)
    assert h[0,1,-1]!=0 and h[0,2,-1]!=h[0,1,-1]
    for i in range(2):
        q=pp[i];v=a[i]
        for k in range(2):
            q,v,_=integer_step(q,v,S);assert np.array_equal(h[i,k+1,:-1],q.ravel()) and h[i,k+1,-1]==v
        assert np.all(h[i,:,:-1].sum(axis=1)==pp[i].sum())
