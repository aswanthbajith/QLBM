"""Matched M19/M20 experiments. No hardware claims; see README."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import json,csv,platform
from milestone20.solver import BoundedDamBreak
from milestone19.solver import DamBreak,initial,fields
from milestone19.run import write_vti
OUT=Path(__file__).parent/'results'

def run(steps=96):
    OUT.mkdir(exist_ok=True);shape=(16,4,32);f0,g0=initial(shape);z,y,x=np.indices(shape);rows=[];hist={};reports={}
    for name,model,mode,shots in [('m19',DamBreak(shape),'reference',None),('reference',BoundedDamBreak(shape),'reference',None),('bounded',BoundedDamBreak(shape),'gate',None),('no_gravity',BoundedDamBreak(shape,gprime=0),'gate',None),('shots_256',BoundedDamBreak(shape),'gate',256),('shots_4096',BoundedDamBreak(shape),'gate',4096)]:
        f,g=f0.copy(),g0.copy();rng=np.random.default_rng(20261125);ph=[];rh=[];vel=[];pop=[];diags=[];stats=[]
        for t in range(steps+1):
            info={}
            if t:
                result=model.step(f,g,mode,shots,rng);f,g,m,j,b=result[:5];diags.append([m,j,b]);info=result[5] if len(result)>5 else {};stats.append(info)
            p,r,u,force=fields(f,g,model.gprime,model.sigma,model.xi);a=(p+1)/2;ph.append(p.copy());rh.append(r.copy());vel.append(u.copy());pop.append(np.r_[f.ravel(),g.ravel()]);
            cells=np.argwhere(p[:,shape[1]//2,:]>0);front=int(cells[:,1].max()) if cells.size else -1
            rows.append(dict(case=name,step=t,lb_mass=float(r.sum()),phase_sum=float(p.sum()),fraction_sum=float(a.sum()),com_x=float((a*x).sum()/a.sum()),com_z=float((a*z).sum()/a.sum()),front=front,max_speed=float(np.linalg.norm(u,axis=0).max()),phase_min=float(p.min()),phase_max=float(p.max()),out_of_range_cells=int(np.count_nonzero(abs(p)>1+1e-10)),**{key:info.get(key,'') for key in ('limited_faces','flux_l1','max_flux_change','unlimited_min','unlimited_max','flux_update_error')}))
            if name=='bounded' and t%16==0:write_vti(OUT/f'step_{t:03}.vti',p,r,u,force)
        ph=np.array(ph);rh=np.array(rh);vel=np.array(vel);pop=np.array(pop);hist[name]=(ph,rh,vel);a0,a1=(ph[0]+1)/2,(ph[-1]+1)/2;ma=np.asarray(diags)
        if name=='reference':ref_pop=pop.copy()
        ref=hist.get('reference');re={'steps':steps,'shots_per_mode':shots,'max_lb_mass_drift':float(np.max(abs(rh.sum((1,2,3))-rh[0].sum()))),'max_phase_sum_drift':float(np.max(abs(ph.sum((1,2,3))-ph[0].sum()))),'max_local_conservation_error':float(ma[:,0].max()),'max_local_force_momentum_error':float(ma[:,1].max()),'max_wall_impulse_balance_error':float(ma[:,2].max()),'min_phase_all_steps':float(ph.min()),'max_phase_all_steps':float(ph.max()),'final_out_of_range_cells':int(np.count_nonzero(abs(ph[-1])>1+1e-10)),'all_out_of_range_cell_steps':int(np.count_nonzero(abs(ph)>1+1e-10)),'max_speed':float(np.linalg.norm(vel,axis=1).max()),'initial_com_x':float((a0*x).sum()/a0.sum()),'final_com_x':float((a1*x).sum()/a1.sum()),'initial_com_z':float((a0*z).sum()/a0.sum()),'final_com_z':float((a1*z).sum()/a1.sum()),'limiter_active_steps':sum(d.get('limited_faces',0)>0 for d in stats),'limited_face_updates':sum(d.get('limited_faces',0) for d in stats),'total_abs_flux_change':sum(d.get('flux_l1',0) for d in stats),'max_flux_update_error':max((d.get('flux_update_error',0) for d in stats),default=0.)}
        if mode=='reference' and name=='reference':
            for key in ('limiter_active_steps','limited_face_updates','total_abs_flux_change','max_flux_update_error'):re[key]=None
        if ref is not None and name!='no_gravity':re.update(phase_rmse_final_vs_bounded_reference=float(np.sqrt(np.mean((ph[-1]-ref[0][-1])**2))),max_pop_error_vs_bounded_reference=float(np.max(abs(pop-ref_pop))))
        reports[name]=re;print(name,json.dumps(re),flush=True)
    with (OUT/'diagnostics.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'history.npz',**{f'{name}_{key}':arr for name,val in hist.items() for key,arr in zip(('phase','rho','velocity'),val)})
    report={'shape_zyx':shape,'steps':steps,'model':'M19 Boussinesq D3Q19/D3Q7 with classical conservative phase flux limiter','precision':'float64','results':reports,'python':platform.python_version(),'numpy':np.__version__,'seed':20261125}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run()
