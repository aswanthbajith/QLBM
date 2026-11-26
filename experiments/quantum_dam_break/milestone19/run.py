"""Reproduce matched dam-release, gravity control and ideal finite-shot trajectories."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,platform
import numpy as np
from milestone19.solver import DamBreak,initial,fields
OUT=Path(__file__).parent/'results'

def write_vti(path,p,r,u,force):
    nz,ny,nx=p.shape;extent=f'0 {nx} 0 {ny} 0 {nz}'
    fields=[('phase_order_parameter',p,1),('formal_phase_fraction_unclipped',(p+1)/2,1),('lb_density_lu',r,1),('pressure_proxy_lu',r/3,1),('velocity_force_corrected_lu',u.transpose(1,2,3,0),3),('force_density_lu',force.transpose(1,2,3,0),3)]
    data='\n'.join(f'<DataArray type="Float64" Name="{n}" NumberOfComponents="{nc}" format="ascii">'+ ' '.join(f'{v:.17g}' for v in a.ravel())+'</DataArray>' for n,a,nc in fields)
    path.write_text(f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian"><ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1"><Piece Extent="{extent}"><CellData>{data}</CellData></Piece></ImageData></VTKFile>\n')

def run(steps=96):
    OUT.mkdir(exist_ok=True);s=DamBreak();f0,g0=initial(s.shape);z,y,x=np.indices(s.shape);n=int(x.size)
    reports={};rows=[];hist={}
    for name,model,mode,shots in [('reference',s,'reference',None),('quantum_exact',s,'gate',None),('no_gravity',DamBreak(gprime=0),'gate',None),('shots_256',s,'gate',256),('shots_4096',s,'gate',4096)]:
        f,g=f0.copy(),g0.copy();rng=np.random.default_rng(20261125);ph=[];rh=[];vel=[];poph=[];local=[]
        for t in range(steps+1):
            if t:f,g,m,j,b=model.step(f,g,mode,shots,rng);local.append([m,j,b])
            p,r,u,force=fields(f,g,model.gprime,model.sigma,model.xi);a=(p+1)/2;ph.append(p.copy());rh.append(r.copy());vel.append(u.copy());poph.append(np.r_[f.ravel(),g.ravel()])
            cells=np.argwhere(p[:,s.shape[1]//2,:]>0);front=int(cells[:,1].max()) if cells.size else -1
            rows.append(dict(case=name,step=t,lb_mass=float(r.sum()),phase_sum=float(p.sum()),formal_fraction_sum=float(a.sum()),formal_com_x=float((a*x).sum()/a.sum()),formal_com_z=float((a*z).sum()/a.sum()),midplane_front_x=front,max_speed=float(np.linalg.norm(u,axis=0).max()),phase_min=float(p.min()),phase_max=float(p.max()),out_of_range_cells=int(np.count_nonzero(np.abs(p)>1+1e-10))))
            if name=='quantum_exact' and t%16==0:write_vti(OUT/f'step_{t:03}.vti',p,r,u,force)
        ph=np.asarray(ph);rh=np.asarray(rh);vel=np.asarray(vel);poph=np.asarray(poph);hist[name]=(ph,rh,vel,poph)
        a0,a1=(ph[0]+1)/2,(ph[-1]+1)/2;ma=np.asarray(local)
        ref=hist['reference'];p_err=ph-ref[0];pop_err=poph-ref[3]
        reports[name]={'steps':steps,'shots_per_mode':shots,'seed':20261125 if shots else None,'max_lb_mass_drift':float(np.max(np.abs(rh.sum((1,2,3))-rh[0].sum()))),'max_phase_sum_drift':float(np.max(np.abs(ph.sum((1,2,3))-ph[0].sum()))),'max_local_conservation_error':float(ma[:,0].max()),'max_local_force_momentum_error':float(ma[:,1].max()),'max_wall_impulse_balance_error':float(ma[:,2].max()),'phase_rmse_final_vs_reference':float(np.sqrt(np.mean(p_err[-1]**2))) if name!='no_gravity' else None,'max_population_error_vs_reference':float(np.max(np.abs(pop_err))) if name!='no_gravity' else None,'velocity_rmse_final_vs_reference':float(np.sqrt(np.mean((vel[-1]-ref[2][-1])**2))) if name!='no_gravity' else None,'min_phase_all_steps':float(ph.min()),'max_phase_all_steps':float(ph.max()),'final_out_of_range_cells':int(np.count_nonzero(np.abs(ph[-1])>1+1e-10)),'max_speed':float(np.linalg.norm(vel,axis=1).max()),'initial_formal_com_x':float((a0*x).sum()/a0.sum()),'final_formal_com_x':float((a1*x).sum()/a1.sum()),'initial_formal_com_z':float((a0*z).sum()/a0.sum()),'final_formal_com_z':float((a1*z).sum()/a1.sum())}
        print(name,reports[name],flush=True)
    with (OUT/'diagnostics.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'history.npz',**{f'{name}_{key}':arr for name,val in hist.items() for key,arr in zip(('phase','rho','velocity'),val[:3])})
    r={'shape_zyx':s.shape,'grid_cells':n,'steps':steps,'reduced_gravity':s.gprime,'sigma_model':s.sigma,'xi':s.xi,'tau_shear':s.tau,'tau_phase':s.tau_phase,'model':'Boussinesq small-contrast two-phase dam release; weakly compressible, not real water-air','quantum_classification':'hybrid exact 4-qubit statevector moment relaxation + amplitude transport gates; classical equilibrium, forcing, transforms, preparation/readout','rates_f':s.rf.tolist(),'rates_g':s.rg.tolist(),'dissipative_modes_per_cell':21,'mode_circuit_evaluations_per_trajectory':n*steps*21,'naive_shots_4096':n*steps*21*4096,'transport_qubits':[a.qubits for a in s.t],'transport_mcx_per_step':[len(a.direct_gates) for a in s.t],'results':reports,'python':platform.python_version(),'numpy':np.__version__}
    (OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':run()
