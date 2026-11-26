"""Execute 3D signed-register phase field in bit-sliced local gate batches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,csv,time
import numpy as np
from milestone16.phase_field_3d import PhaseStepper3D,DT3
from milestone16.reference import integer_step3,float_step3,energy3,chemical3
OUT=Path(__file__).parent/'results'/'interface_3d'

def vtk(path,p,ref):
    nz,ny,nx=p.shape;extent=f'0 {nx} 0 {ny} 0 {nz}'
    fields={'phase_order_parameter':p,'chemical_potential_nd':chemical3(p),'reference_phase':ref,'phase_error':p-ref}
    with path.open('w') as f:
        f.write(f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian"><ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1"><Piece Extent="{extent}"><CellData Scalars="phase_order_parameter">\n')
        for name,v in fields.items():f.write(f'<DataArray type="Float64" Name="{name}" format="ascii">'+' '.join(format(x,'.12g') for x in v.ravel())+'</DataArray>\n')
        f.write('</CellData></Piece></ImageData></VTKFile>\n')

def main():
    S=65536;n=8;steps=256;OUT.mkdir(parents=True,exist_ok=True);s=PhaseStepper3D(S)
    z,y,x=np.indices((n,n,n));r=np.sqrt((x-(n-1)/2)**2+(y-(n-1)/2)**2+(z-(n-1)/2)**2)
    p=np.rint(S*np.tanh((2.5-r)/.7)).astype(np.int64);ref=p/S;mass0=int(p.sum());h=[p.copy()];rr=[ref.copy()];rows=[];start=time.perf_counter()
    for t in range(steps+1):
        f=p/S;rows.append(dict(step=t,time=t*DT3,phase_mass_ticks=int(p.sum()),energy=energy3(f),reference_energy=energy3(ref),rmse=float(np.sqrt(np.mean((f-ref)**2))),min_phi=float(f.min()),max_phi=float(f.max()),positive_cells=int(np.sum(f>0))))
        if t%32==0:vtk(OUT/f'step_{t:03}.vti',f,ref)
        if t==steps:break
        q=s.step(p);expected=integer_step3(p,S);assert np.array_equal(q,expected);p=q;ref=float_step3(ref);h.append(p.copy());rr.append(ref.copy())
    with (OUT/'diagnostics.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    np.savez_compressed(OUT/'history.npz',ticks=h,reference=rr,scale=S,time=np.arange(steps+1)*DT3)
    e=np.array([r['energy'] for r in rows]);delta=np.diff(e);err=np.array([r['rmse'] for r in rows]);h=np.asarray(h)
    report=dict(grid=[n]*3,steps=steps,dt=DT3,physical_time_nd=steps*DT3,scale=S,initial_mass_ticks=mass0,max_phase_mass_error_ticks=int(np.max(np.abs(h.sum(axis=(1,2,3))-mass0))),initial_energy=float(e[0]),final_energy=float(e[-1]),energy_increase_steps=int(np.sum(delta>1e-12)),max_energy_increase=float(max(0,delta.max())),final_rmse=float(err[-1]),max_rmse=float(err.max()),min_phi=float(h.min()/S),max_phi=float(h.max()/S),initial_positive_cells=rows[0]['positive_cells'],final_positive_cells=rows[-1]['positive_cells'],matched_reference_steps=steps,vtk_files=len(list(OUT.glob('step_*.vti'))),resources=s.resources(n**3),seconds=time.perf_counter()-start)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
