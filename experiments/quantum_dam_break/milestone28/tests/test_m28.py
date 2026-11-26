import numpy as np
import pytest
from milestone28.arithmetic import Circuit,Moments,Materials,JointMacros
from milestone18.transport import C19
from milestone28.solver import QuantumMacroDamBreak
from milestone27.solver import ThreeDDamBreak
from milestone21.solver import hydrostatic

@pytest.mark.parametrize('subtract',[False,True])
def test_cdkm_adder(subtract):
    c=Circuit([4],[4]);carry=c.alloc(1)[0];c.add(c.ins[0],c.outs[0],carry,subtract);c.finish()
    v=np.arange(-8,8)[:,None];o=c.evaluate(v)[:,0]
    assert np.array_equal(o,(((-v[:,0] if subtract else v[:,0])+8)%16)-8)
    # Exhaustively exercise nonzero accumulator and verify input/carry recovery.
    for a in range(16):
        for b in range(16):
            z=a|(b<<4)
            for t,x,y in c.gates:
                if (x<0 or (z>>int(x))&1) and (y<0 or (z>>int(y))&1):z^=1<<int(t)
            assert z==a|(((b+(-a if subtract else a))%16)<<4)

@pytest.mark.parametrize('k',[16,24,32,40])
def test_joint_exact(k):
    S=1<<k;m=JointMacros(k,998/1.2,.001/.000018,.00125)
    rng=np.random.default_rng(17);v=rng.integers(-S//4,S//4,(131,26));v[:,19:]=rng.integers(0,S//8,(131,7));v[0,19:]=0;v[1,19:]=0;v[1,19]=S
    out=m.evaluate(v);exact=np.column_stack([v[:,19:].sum(1),v[:,:19].sum(1),v[:,:19]@C19]);mat=np.array([[(lo*S+int(x)*(hi-lo)+S//2)//S for lo,hi in m.materials.coefficients] for x in exact[:,0]])
    assert np.array_equal(out,np.column_stack([exact,mat]));assert np.all(out[:,5:]>0)

def test_complex_sparse_superposition():
    # Actual reversible permutation gate action on two complex amplitudes; no dense statevector claim.
    k=16;S=1<<k;c=JointMacros(k,998/1.2,.001/.000018,.00125);w=c.input_widths[0]
    v=np.zeros((2,26),np.int64);v[0,1]=S//8;v[0,2]=-S//16;v[0,19]=S//2;v[1,5]=-S//4;v[1,19]=S
    def pack(row):return sum((int(x)&((1<<w)-1))<<(i*w) for i,x in enumerate(row))
    def permute(z,gates):
        for t,a,b in gates:
            if (a<0 or (z>>int(a))&1) and (b<0 or (z>>int(b))&1):z^=1<<int(t)
        return z
    state={pack(v[0]):1/np.sqrt(2),pack(v[1]):1j/np.sqrt(2)}
    out={permute(z,c.gates):a for z,a in state.items()};assert len(out)==2
    expected=c.evaluate(v)
    for i,z in enumerate(state):
        key=z;offset=c.inputs
        for x,width in zip(expected[i],c.output_widths):key|=(int(x)&((1<<width)-1))<<offset;offset+=width
        assert out[key]==state[z];assert key>>(c.inputs+c.outputs)==0
    assert {permute(z,c.gates[::-1]):a for z,a in out.items()}==state
    assert abs(sum(abs(a)**2 for a in out.values())-1)<1e-14

def test_integration_small():
    kw=dict(shape=(4,4,8),gravity=6.25e-5,nu_liquid=.00125,sigma=2.5e-6,xi=3.,tau_phase=.725)
    ref=ThreeDDamBreak(**kw);q=QuantumMacroDamBreak(**kw);z,y,x=np.indices(kw['shape']);a=.5*(1-np.tanh(np.maximum(x-2,z-2)/1.5));f,h=ref.initialize(a,hydrostatic(a,ref.mat(a)[0],kw['gravity']));fr,hr=f.copy(),h.copy()
    bg=np.broadcast_to(a.mean((1,2),keepdims=True),a.shape);q.set_hydrostatic_background(bg);ref.set_hydrostatic_background(bg)
    for _ in range(4):f,h=q.step(f,h);fr,hr=ref.step(fr,hr)
    assert np.max(abs(h-hr))<1e-8 and np.max(abs(f-fr))<1e-8
    assert abs(h.sum()-a.sum())<1e-9 and q.macro_calls==4

def test_optional_qiskit_export():
    pytest.importorskip('qiskit');c=Moments(16);qc=c.qiskit();assert qc.num_qubits==c.nqubits and sum(qc.count_ops().values())==len(c.gates)
