"""Re-run M25 Re212 gate experiment; capture full 3-D fields and wall probes."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone25.run import configuration,initial_fraction
from milestone24.solver import LongDamBreak,hydrostatic
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; BASE=ROOT.parent/'milestone25/results'
STRIDE=64

def derive(a,rho,mu,p,u):
    speed=np.linalg.norm(u,axis=0)
    return dict(alpha=a,phi=2*a-1,rho=rho,mu=mu,pressure=p,velocity=u,speed=speed,jz=rho*u[2],dynamic_pressure=.5*rho*speed**2)

def wall_metrics(a,p,initial_p):
    wall=p[:,:,-1];excess=wall-initial_p
    return dict(wall_pressure_mean=float(wall.mean()),wall_pressure_max=float(wall.max()),wall_pressure_sum=float(wall.sum()),wall_excess_max=float(excess.max()),wall_alpha_max=float(a[:,:,-1].max()),wall_wet_fraction=float(np.mean(a[:,:,-1]>=.5)))

def run():
    OUT.mkdir(exist_ok=True)
    c=configuration(.0025,1,2);s=LongDamBreak(**{k:c[k] for k in ('shape','gravity','nu_liquid','sigma','xi','tau_phase')})
    a=initial_fraction(c);rho=s.mat(a)[0];f,h=s.initialize(a,hydrostatic(a,rho,c['gravity']));s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),c['shape']))
    fields={};rows=[];steps=[];wall_history=[];wall_alpha_history=[];p0=None;maxdrift=0.;mass=a.sum()
    for t in range(c['steps']+1):
        if t:f,h=s.step(f,h,'gate')
        a,rho,mu,nu,p,u,*_=s.decode(f,h)
        if p0 is None:p0=p[:,:,-1].copy()
        wall_history.append(p[:,:,-1].copy());wall_alpha_history.append(a[:,:,-1].copy())
        row={'step':t,'t_star':float(t*np.sqrt(c['gravity']/c['H'])),**wall_metrics(a,p,p0),'total_vertical_momentum':float(np.sum(rho*u[2])),'peak_speed':float(np.linalg.norm(u,axis=0).max())};rows.append(row)
        maxdrift=max(maxdrift,abs(a.sum()-mass)/mass)
        assert np.isfinite(f).all() and np.isfinite(h).all() and a.min()>=-1e-10 and a.max()<=1+1e-10 and row['peak_speed']<=.1
        if t%STRIDE==0 or t==c['steps']:
            steps.append(t)
            for k,v in derive(a,rho,mu,p,u).items():fields.setdefault(k,[]).append(v.copy())
        if t%320==0:print('step',t,flush=True)
    data={k:np.stack(v) for k,v in fields.items()}; wall=np.stack(wall_history);wa=np.stack(wall_alpha_history)
    np.savez_compressed(OUT/'fields.npz',steps=steps,wall_pressure=wall,wall_alpha=wa,**data)
    with (OUT/'wall_probes.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    old=np.load(BASE/'slow212_gate.npz');err={k:float(np.max(abs(v-old[k]))) for k,v in [('f',f),('h',h)]}
    old_indices=[int(np.flatnonzero(old['steps']==t)[0]) for t in steps];err['alpha_saved_slices']=float(np.max(abs(data['alpha'][:,:,c['shape'][1]//2,:]-old['alpha'][old_indices])))
    assert max(err.values())<1e-11,err
    contact=np.flatnonzero(np.max(wa,axis=(1,2))>=.5);peak=np.unravel_index(np.argmax(wall),wall.shape);ex=wall-wall[0];ep=np.unravel_index(np.argmax(ex),ex.shape)
    report=dict(configuration=c,mode='gate',snapshots=len(steps),stride=STRIDE,comparison_max_abs=err,max_phase_relative_error=maxdrift,first_wall_wetting_step=int(contact[0]) if len(contact) else None,peak_wall_pressure=float(wall[peak]),peak_wall_pressure_index=list(map(int,peak)),peak_wall_excess=float(ex[ep]),peak_wall_excess_index=list(map(int,ep)),max_speed_all_steps=max(r['peak_speed'] for r in rows),max_span_variation=float(np.max(np.ptp(data['alpha'],axis=2))),pressure_definition='Cell-centred gauge pressure, x=31 adjacent to wall at x=32; not validated wall traction or impact sensor.',vertical_momentum_definition='rho*u_z, positive upward; normalized material density and force-corrected velocity',units='lattice units; no calibrated SI impact loads')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(report,flush=True)
    from export import write_series
    write_series(OUT,data,steps,c)
if __name__=='__main__':run()
