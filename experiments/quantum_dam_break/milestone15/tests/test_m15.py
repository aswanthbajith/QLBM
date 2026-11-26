import numpy as np
import pytest
from milestone15.coherent import Circuit, independent_step, reference_histories, retained_output_cleanup_probability
from milestone15.run import inputs
from milestone11.reference import V

@pytest.fixture(scope='module')
def data():
    c=Circuit(S=16,omega_num=4,steps=2);f=inputs(c.oracle.S,c.w);a=np.array([1,1j])/np.sqrt(2)
    return c,f,a,c.execute(f,a)

def test_full_coherent_reference(data):
    c,f,a,r=data;ref,flags=reference_histories(f,r['labels'],c.steps,c.oracle.S,c.w,c.oracle.omega_num)
    assert np.array_equal(r['history'],ref) and np.array_equal(r['flags'],flags)
    assert not flags.any()

def test_norm_phase_inverse(data):
    c,f,a,r=data;n=1<<(6*c.steps)
    assert len(r['amplitudes'])==2*n
    assert np.allclose(r['amplitudes'],np.repeat(a/np.sqrt(n),n),atol=1e-15)
    assert abs(r['norm']-1)<1e-12 and r['inverse_error']<1e-12

def test_conservation_and_nontrivial_transport(data):
    c,f,a,r=data;h=r['history'];m=h.sum((2,3))
    assert np.all(m==m[:,:1]) and h.min()>=0
    assert np.any(h[:,0]!=h[:,1]) and np.any(h[:,1]!=h[:,2])
    assert np.any(h[:,0].sum(-1)!=h[:,-1].sum(-1))

def test_branch_history_and_environment(data):
    c,f,a,r=data;n=1<<(6*c.steps);h=r['history']
    assert np.array_equal(r['labels'],np.tile(np.arange(n),2))
    assert len(np.unique(h[:n,-1].reshape(n,-1),axis=0))>=1
    p=retained_output_cleanup_probability(h,r['amplitudes'],c.steps)
    assert 0<p<=1+1e-12
    # A branch-dependent copied output cannot generally be uncomputed to clean selectors.
    assert p<1-1e-8

def test_wall_reference_transport(data):
    c,_,_,_=data;f=np.zeros((2,25),np.int64);i=np.where((V==[2,0]).all(axis=1))[0][0];f[0,i]=1
    # Directly execute streaming SWAP gates, no collision.
    z,_=c.pack(f[None]);z[c.block:c.block+c.block]=z[:c.block]
    from milestone8.collision import _execute
    _execute(c.schedule[2],z);h=c.unpack_history(z,1)
    assert h[0,1,0,24-i]==1 and h[0,1].sum()==1

def test_qiskit_optional():
    pytest.importorskip('qiskit');c=Circuit(S=4,steps=1);q=c.qiskit()
    assert q.num_qubits==c.nqubits and q.count_ops()['h']==6

def test_retained_output_probability_by_explicit_hadamard(data):
    c,_,a,r=data;n=1<<(6*c.steps);h=r['history'];prob=0.
    # Independent numerical simulation of the selector Hadamards after
    # a final register copy and reverse basis permutation. Group by copied output.
    for i in range(2):
        _,ids=np.unique(h[i*n:(i+1)*n,-1].reshape(n,-1),axis=0,return_inverse=True)
        v=np.zeros((ids.max()+1,n),complex);v[ids,np.arange(n)]=a[i]/np.sqrt(n)
        for b in range(6*c.steps):
            z=v.reshape(len(v),-1,2,1<<b);x=z[:,:,0,:].copy();y=z[:,:,1,:].copy()
            z[:,:,0,:]=(x+y)/np.sqrt(2);z[:,:,1,:]=(x-y)/np.sqrt(2)
        assert np.isclose(np.sum(abs(v)**2),abs(a[i])**2)
        prob+=float(np.sum(abs(v[:,0])**2))
    assert np.isclose(prob,retained_output_cleanup_probability(h,r['amplitudes'],c.steps),atol=1e-12)
