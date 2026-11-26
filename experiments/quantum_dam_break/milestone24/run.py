"""Actual long-horizon trajectories, matched sidewall and no-gravity controls."""
from pathlib import Path
import sys,json,csv,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone24.solver import LongDamBreak,hydrostatic
from milestone19.solver import initial
from milestone21.run import vti
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
SHAPE=(16,4,32);G=.00025;STEPS=640;H=.65*SHAPE[0]

def run_case(name,gravity,span,mode,steps=STEPS):
 s=LongDamBreak(shape=SHAPE,gravity=gravity,sigma=1e-5,span_periodic=span);_,g=initial(SHAPE);a=(1+g.sum(0))/2;rho=s.mat(a)[0]
 f,h=s.initialize(a,hydrostatic(a,rho,gravity));s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),SHAPE))
 z,y,x=np.indices(SHAPE);mass=a.sum();m0=rho.sum();rows=[];snap=[];ts=[];status='completed';start=time.monotonic()
 for t in range(steps+1):
  try:
   if t:f,h=s.step(f,h,mode)
   a,rho,mu,nu,p,u,*_=s.decode(f,h)
  except (ValueError,RuntimeError,FloatingPointError) as e:status=str(e);break
  speed=float(np.linalg.norm(u,axis=0).max());wet=a>.5;front=int(np.where(wet)[2].max()) if wet.any() else -1;floor=wet[0];floorfront=int(np.where(floor)[1].max()) if floor.any() else -1
  top=int(np.where(wet[:,:,0])[0].max())+1 if wet[:,:,0].any() else 0
  rows.append(dict(case=name,step=t,t_star=t*np.sqrt(gravity/H) if gravity else 0.,front=front,floor_front=floorfront,left_height=top,com_x=float((a*(x+.5)).sum()/a.sum()),com_z=float((a*(z+.5)).sum()/a.sum()),max_speed=speed,phase_sum=float(a.sum()),material_mass=float(rho.sum()),min_alpha=float(a.min()),max_alpha=float(a.max())))
  if t%32==0 or t==steps:
   snap.append(np.concatenate([a[None],rho[None],mu[None],p[None],u]));ts.append(t)
  if name=='span_gate' and t%128==0:vti(OUT/f'step_{t:04}.vti',a,rho,mu,p,u)
  if t%128==0:print(name,rows[-1],flush=True)
  if not np.isfinite(speed) or speed>.1:status='speed guard exceeded';break
 res=dict(status=status,steps=rows[-1]['step'],gravity=gravity,span_periodic=span,mode=mode,initial=rows[0],final=rows[-1],phase_drift=max(abs(r['phase_sum']-mass) for r in rows),material_drift=max(abs(r['material_mass']-m0) for r in rows),peak_speed=max(r['max_speed'] for r in rows),min_alpha=min(r['min_alpha'] for r in rows),max_alpha=max(r['max_alpha'] for r in rows),elapsed_s=time.monotonic()-start)
 if s.t:res.update(transport_qubits=[tr.qubits for tr in s.t],logical_mcx=[len(tr.direct_gates) for tr in s.t])
 print('RESULT',name,res,flush=True);return res,rows,np.asarray(snap),np.asarray(ts),f,h

def main():
 results={};rows=[];hist={};fin={}
 for name,g,span,mode in [('span_gate',G,True,'gate'),('span_reference',G,True,'fast'),('closed_reference',G,False,'fast'),('zero_gravity',0.,True,'fast')]:
  r,rr,ss,tt,f,h=run_case(name,g,span,mode);results[name]=r;rows+=rr;hist[name]=ss;hist[name+'_steps']=tt;fin[name]=(f,h)
  np.savez_compressed(OUT/'history.npz',**hist)
  (OUT/'report.json').write_text(json.dumps(results,indent=2)+'\n')
 a,b=hist['span_gate'],hist['span_reference'];results['span_gate']['snapshot_phase_rmse']=float(np.sqrt(np.mean((a[:,0]-b[:,0])**2)));results['span_gate']['final_population_max_error']=float(max(np.max(abs(x-y)) for x,y in zip(fin['span_gate'],fin['span_reference'])))
 with (OUT/'trajectory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'report.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
