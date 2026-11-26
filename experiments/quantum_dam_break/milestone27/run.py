"""Finite-width release using M23 physics and compiled M18 gates."""
from pathlib import Path
import sys,json,csv,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone27.solver import ThreeDDamBreak,hydrostatic
from milestone25.run import configuration
from milestone26.capture import derive,wall_metrics
from milestone26.export import write_series
OUT=Path(__file__).parent/'results'
def initial_fraction(shape,width=8.):
    z,y,x=np.indices(shape,dtype=float);nz,ny,nx=shape
    return .5*(1-np.tanh(np.maximum.reduce([x+.5-nx*.33,z+.5-nz*.65,abs(y+.5-ny/2)-width/2])/1.5))
def run(mode,gravity=True,N=1280):
    c=configuration(.0025,1,2);c.update(shape=(16,16,32),steps=N,reservoir_width=8.,gravity=c['gravity'] if gravity else 0.)
    s=ThreeDDamBreak(**{k:c[k] for k in ('shape','gravity','nu_liquid','sigma','xi','tau_phase')})
    a=initial_fraction(c['shape'],c['reservoir_width']);rho=s.mat(a)[0];f,h=s.initialize(a,hydrostatic(a,rho,c['gravity']));s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),c['shape']))
    name=mode if gravity else 'zero_gravity';out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    mass=a.sum();mass0=rho.sum();z,y,x=np.indices(c['shape'],dtype=float);y+=.5-c['shape'][1]/2;x+=.5;z+=.5
    fields={};rows=[];steps=[];wall=[];wa=[];start=time.monotonic();status='completed'
    for t in range(N+1):
        try:
            if t:f,h=s.step(f,h,mode)
            a,rho,mu,nu,p,u,*_=s.decode(f,h)
        except (ValueError,RuntimeError,FloatingPointError) as e:status=str(e);break
        speed=float(np.linalg.norm(u,axis=0).max());wet=a>=.5;inds=np.argwhere(wet);floor=np.argwhere(wet[0]);left=np.argwhere(wet[:,:,0]);
        if t==0:p0=p[:,:,-1].copy()
        wall.append(p[:,:,-1].copy());wa.append(a[:,:,-1].copy())
        row=dict(step=t,t_star=t*np.sqrt(c['gravity']/c['H']),front=float(inds[:,2].max()+1) if inds.size else 0.,floor_width=float(np.ptp(floor[:,0])+1) if floor.size else 0.,left_height=float(left[:,0].max()+1) if left.size else 0.,com_x=float((a*x).sum()/a.sum()),com_z=float((a*z).sum()/a.sum()),span_rms=float(np.sqrt((a*y*y).sum()/a.sum())),max_abs_uy=float(abs(u[1]).max()),liquid_weighted_abs_uy=float((a*abs(u[1])).sum()/a.sum()),span_momentum=float((rho*u[1]).sum()),span_mirror_error=float(np.max(abs(a-a[:,::-1,:]))),speed=speed,phase_relative_error=float(abs(a.sum()-mass)/mass),material_relative_error=float(abs(rho.sum()-mass0)/mass0),min_alpha=float(a.min()),max_alpha=float(a.max()),**wall_metrics(a,p,p0))
        rows.append(row)
        if t%64==0 or t==N:
            steps.append(t)
            for k,v in derive(a,rho,mu,p,u).items():fields.setdefault(k,[]).append(v.copy())
        if t%160==0:print(name,t,'front',row['front'],'width',row['floor_width'],'uy',row['max_abs_uy'],'speed',speed,flush=True)
        if not np.isfinite(speed) or speed>.1:status='speed guard exceeded';break
    data={k:np.stack(v) for k,v in fields.items()};wall=np.stack(wall);wa=np.stack(wa)
    np.savez_compressed(out/'fields.npz',steps=steps,wall_pressure=wall,wall_alpha=wa,**data);np.savez_compressed(out/'final_populations.npz',f=f,h=h)
    with (out/'history.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    contact=np.flatnonzero(wa.max((1,2))>=.5);peak=np.unravel_index(np.argmax(wall),wall.shape)
    r=dict(configuration=c,mode=mode,status=status,steps=rows[-1]['step'],initial=rows[0],final=rows[-1],max_speed_all_steps=max(q['speed'] for q in rows),max_abs_uy=max(q['max_abs_uy'] for q in rows),max_phase_relative_error=max(q['phase_relative_error'] for q in rows),max_material_relative_error=max(q['material_relative_error'] for q in rows),max_span_mirror_error=max(q['span_mirror_error'] for q in rows),min_alpha=min(q['min_alpha'] for q in rows),max_alpha=max(q['max_alpha'] for q in rows),first_wall_wetting_step=int(contact[0]) if len(contact) else None,peak_wall_pressure=float(wall[peak]),peak_wall_pressure_index=list(map(int,peak)),elapsed_seconds=time.monotonic()-start,pressure_definition='Cell-centred gauge pressure, not validated wall traction')
    if s.t:r['transport_qubits']=[q.qubits for q in s.t];r['logical_mcx']=[len(q.direct_gates) for q in s.t]
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(name,status,r['final'],flush=True)
    if mode=='gate':write_series(out,data,steps,c)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['gate','reference']);p.add_argument('--zero-gravity',action='store_true');p.add_argument('--steps',type=int,default=1280);a=p.parse_args();run(a.mode,not a.zero_gravity,a.steps)
