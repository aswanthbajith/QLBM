"""Execute a full global two-step circuit and copy/reverse history cleanup."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone17.coherent import CoherentCoupledGrid
from milestone17.reference import integer_step
OUT=Path(__file__).parent/'results'

def main():
    S=65536;s=CoherentCoupledGrid(S);z,y,x=np.indices((4,4,4))
    p=np.rint(.8*S*(.65*np.sin(2*np.pi*x/4)+.35*np.cos(2*np.pi*(x/4+y/4))+.15*np.cos(2*np.pi*z/4))).astype(np.int64)
    pp=np.stack([p,p[:,:,::-1]]);a=np.array([0,-10000]);amps=np.array([1,1j])/np.sqrt(2)
    h,out=s.execute(pp,a,amps,copy_final=True);assert np.array_equal(out,amps)
    max_error=0
    for i in range(2):
        q=pp[i];v=a[i]
        for k in range(2):
            q,v,_=integer_step(q,v,S);max_error=max(max_error,int(np.max(abs(h[i,k+1,:-1]-q.ravel()))),abs(int(h[i,k+1,-1])-v))
    assert max_error==0
    report=dict(grid_zyx=list(s.model.shape),steps=s.steps,scale=S,initial_components=len(amps),intermediate_measurements=0,intermediate_reinitializations=0,logical_qubits=s.nqubits,forward_X_CX_CCX_gates=s.count,copy_qubits=s.slice,compute_copy_reverse_gates=2*s.count+s.slice,max_integer_reference_error_ticks=max_error,max_mass_error_ticks=int(np.max(abs(h[:,:,:-1].sum(axis=2)-h[:,0,:-1].sum(axis=1)[:,None]))),velocities_ticks=h[:,:,-1].tolist(),velocity_amplitude_nd=(h[:,:,-1]/S).tolist(),arithmetic_workspace_cleanup=True,history_cleanup_with_original_input_retained=True,relative_phase_ratio_imaginary=float((out[1]/out[0]).imag),inverse_bit_error=0,simulator='exact sparse complex support, bit-sliced permutation gates; supplied initial amplitudes')
    OUT.mkdir(exist_ok=True);(OUT/'coherent_report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'coherent_history.npz',histories=h,amplitudes=out,scale=S,shape=s.model.shape);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
