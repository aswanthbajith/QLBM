"""Reproduce matched gate trajectories and independent references."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,csv,time,argparse
import numpy as np
from milestone17.coupling import CoupledStepper,DT
from milestone17.reference import integer_step,float_step,balance,total_energy,mode
from milestone16.reference import energy3,chemical3
ROOT=Path(__file__).parent/'results'
SHAPE=(4,4,8)

def initial(S):
    z,y,x=np.indices(SHAPE)
    p=.65*np.sin(2*np.pi*x/8)+.35*np.cos(2*np.pi*(x/8+y/4))+.15*np.cos(2*np.pi*z/4)
    return np.rint(S*.8*p).astype(np.int64)

def vtk(path,p,a,ref):
    nz,ny,nx=p.shape;extent=f'0 {nx} 0 {ny} 0 {nz}'
    fields={'phase_order_parameter':p,'chemical_potential_nd':chemical3(p),'reference_phase':ref,'phase_error':p-ref,'velocity_x_nd':a*mode(p.shape)}
    with path.open('w') as f:
        f.write(f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian"><ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1"><Piece Extent="{extent}"><CellData Scalars="phase_order_parameter">\n')
        for name,v in fields.items():f.write(f'<DataArray type="Float64" Name="{name}" format="ascii">'+' '.join(format(x,'.12g') for x in v.ravel())+'</DataArray>\n')
        f.write('</CellData></Piece></ImageData></VTKFile>\n')

def run(S,feedback,steps=256,a0=0.,write_vtk=False):
    out=ROOT/f'{S}_{"coupled" if feedback else "control"}_{a0:g}';out.mkdir(parents=True,exist_ok=True)
    solver=CoupledStepper(S,SHAPE);p=initial(S);a=round(S*a0);ref=p/S;v=a/S;mass=int(p.sum());hist=[];rh=[];rows=[];start=time.perf_counter();forces=[]
    for t in range(steps+1):
        f=p/S;b=balance(f,a/S);hist.append(p.copy());rh.append(ref.copy())
        rows.append(dict(step=t,time=t*DT,phase_sum_ticks=int(p.sum()),mean_momentum=float((mode(SHAPE)*a/S).mean()),velocity_amplitude=a/S,reference_velocity=v,phase_energy=energy3(f),kinetic_energy=p.size*(a/S)**2/2,total_energy=total_energy(f,a/S),reference_energy=total_energy(ref,v),phase_rmse=float(np.sqrt(np.mean((f-ref)**2))),min_phi=float(f.min()),max_phi=float(f.max()),**b))
        if write_vtk and t%32==0:vtk(out/f'step_{t:03}.vti',f,a/S,ref)
        if t==steps:break
        p1,a1,F=solver.step(p,a,feedback=feedback);r=integer_step(p,a,S,feedback);assert np.array_equal(p1,r[0]) and (a1,F)==r[1:];forces.append(F/(2*S*S*p.size));p,a=p1,a1;ref,v=float_step(ref,v,feedback)
    with (out/'diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'history.npz',ticks=hist,reference=rh,scale=S,velocity=[r['velocity_amplitude'] for r in rows],reference_velocity=[r['reference_velocity'] for r in rows],force=forces,time=np.arange(steps+1)*DT)
    E=np.array([r['total_energy'] for r in rows]);A=np.array([r['velocity_amplitude'] for r in rows]);V=np.array([r['reference_velocity'] for r in rows]);W=np.array([r['exchange_residual'] for r in rows]);R=np.array([r['energy_rate_residual'] for r in rows]);d=np.diff(E)
    report=dict(scale=S,shape_zyx=list(SHAPE),steps=steps,dt=DT,feedback=feedback,initial_velocity=a0,final_velocity=float(A[-1]),reference_final_velocity=float(V[-1]),max_velocity_error=float(np.max(abs(A-V))),max_velocity=float(np.max(abs(A))),max_capillary_acceleration=float(max(abs(np.array(forces)))),initial_total_energy=float(E[0]),final_total_energy=float(E[-1]),energy_increase_steps=int((d>1e-12).sum()),max_energy_increase=float(max(0,d.max())),final_phase_rmse=rows[-1]['phase_rmse'],max_phase_mass_error_ticks=max(abs(r['phase_sum_ticks']-mass) for r in rows),max_mean_momentum=max(abs(r['mean_momentum']) for r in rows),max_unrounded_exchange_residual=float(max(abs(W))),max_unrounded_energy_rate_residual=float(max(abs(R))),matched_integer_steps=steps,min_phi=min(r['min_phi'] for r in rows),max_phi=max(r['max_phi'] for r in rows),resources=solver.resources(),runtime_seconds=time.perf_counter()-start)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(out.name,json.dumps({k:v for k,v in report.items() if k!='resources'}),flush=True);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=256);args=p.parse_args();runs=[]
    for S,fb,a in [(256,True,0.),(4096,True,0.),(65536,True,0.),(65536,False,0.),(65536,True,.15)]:runs.append(run(S,fb,args.steps,a,write_vtk=S==65536 and fb and a==0))
    (ROOT/'report.json').write_text(json.dumps(runs,indent=2)+'\n')
if __name__=='__main__':main()
