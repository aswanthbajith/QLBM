"""Higher-precision 1D dam-break benchmark using the 2D arithmetic circuit."""
import sys,csv,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone6.solver import Arithmetic2D
from milestone5 import reference2d as ref

def run():
    out=Path(__file__).parent/'results';mask=np.zeros((1,32),bool);x=(np.arange(32)-15.5)*.5;initial=np.zeros((1,32,3));initial[0,:16,0]=1.5
    r=initial.copy()
    for _ in range(16):r=ref.step(r,mask)
    exact,_=ref.ritter(x,1.);rows=[];data=dict(x=x,float_depth=r[0,:,0],exact_depth=exact)
    for S in (32,64,128):
        c=Arithmetic2D(mask,S);s=(initial*S).astype(int);mass=s[:,:,0].sum()
        for j in range(16):
            z=c.step(s,True);assert np.array_equal(z,ref.quantized_step(s,mask,S));assert z[:,:,0].sum()==mass;s=z
        h=s[0,:,0]/S;data[f'depth_{S}']=h;rows.append(dict(scale=S,rmse_vs_float=float(np.sqrt(np.mean((h-r[0,:,0])**2))),rmse_vs_ritter=float(np.sqrt(np.mean((h-exact)**2))),wet_cells=int(np.count_nonzero(h)),integrated_depth=float(h.sum()*.5),logical_gates=c.gate_count(),logical_qubits=c.nqubits));print(rows[-1],flush=True)
    with (out/'precision.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'precision.npz',**data)
if __name__=='__main__':run()
