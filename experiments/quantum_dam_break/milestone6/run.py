"""Run exact clean-workspace basis simulation and independent comparison."""
import sys,json,csv,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone6.solver import Arithmetic2D
from milestone6.arithmetic import FluxOracle
from milestone5.run_2d import write_vti
from milestone5 import reference2d as ref

def run():
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=True)
    previous=Path(__file__).parents[1]/'milestone5/results'
    old=np.load(previous/'history.npz');mask=old['solid'];s=(old['quantum'][0]*32).astype(int);c=Arithmetic2D(mask,32);r=old['reference'][0].copy();hist=[];refs=[];rows=[];start=time.monotonic()
    for j in range(33):
        h=s[:,:,0]/32;hist.append(s/32);refs.append(r.copy())
        rows.append(dict(step=j,time_nd=j/8,integrated_depth_nd=float(h.sum()),wet_cells=int(np.count_nonzero(h)),obstacle_depth=float(h[mask].sum()),depth_rmse_vs_float=float(np.sqrt(np.mean((h[~mask]-r[:,:,0][~mask])**2))),max_error_vs_m5=float(np.max(abs(s/32-old['quantum'][j])))))
        write_vti(out/f'step_{j:03}.vti',s/32,mask,r)
        if j<32:
            z=c.step(s,verify_inverse=True);assert np.array_equal(z,ref.quantized_step(s,mask,32));s=z;r=ref.step(r,mask)
        if j%8==0:print('step',j,'elapsed',round(time.monotonic()-start,1),flush=True)
    assert all(x['max_error_vs_m5']==0 for x in rows)
    with (out/'diagnostics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'history.npz',quantum=np.array(hist),reference=np.array(refs),solid=mask)
    scaling=[]
    for S in (2,4,8,16,32,64,128):
        a=FluxOracle(S,1);scaling.append(dict(scale=S,input_bits=a.inputs,arithmetic_gates=len(a.gates),arithmetic_ancillas=a.ancillas,lut_input_patterns=2**a.inputs))
    with (out/'arithmetic_scaling.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=scaling[0]);w.writeheader();w.writerows(scaling)
    old_report=json.loads((previous/'report.json').read_text());old_g=old_report['logical_pattern_controlled_x_per_step'];new_g=c.gate_count()
    report=dict(backend='clean-workspace reversible basis gate simulation; memoized local gate executions; no statevector/QPU',grid=[16,16],steps=32,initial=rows[0],final=rows[-1],max_mass_error=max(abs(x['integrated_depth_nd']-rows[0]['integrated_depth_nd']) for x in rows),max_error_vs_m5=max(x['max_error_vs_m5'] for x in rows),quantized_reference_agreement='all steps',inverse_and_cleanup='all steps',old_logical_gates=old_g,new_logical_gates=new_g,gate_reduction_factor=old_g/new_g,old_logical_qubits=old_report['logical_qubits_per_step'],new_logical_qubits=c.nqubits,arithmetic_work_qubits=c.nqubits-c.working,local_oracles={str(k):dict(gates=len(a.gates),qubits=a.nqubits,ancillas=a.ancillas) for k,a in c.oracles.items()},scaling=scaling,elapsed_seconds=time.monotonic()-start,native_qiskit='not executed',physical_viscosity='not implemented')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
