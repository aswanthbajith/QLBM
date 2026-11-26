"""Matched-time viscosity and grid-scaling study, reusing M24 solver."""
from pathlib import Path
import sys,json,csv,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone24.solver import LongDamBreak,hydrostatic
from milestone21.run import vti
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
def configuration(nu=.02,r=1,slow=1):
    if r not in (1,2) or slow not in (1,2,4) or nu<=0:raise ValueError('invalid scale')
    # Acoustic refinement: dx and dt shrink together, U_lu constant.
    return dict(shape=(16*r,4,32*r),gravity=.00025/r/slow**2,nu_liquid=nu*r/slow,sigma=1e-5*r/slow**2,xi=3.*r,tau_phase=.5+.45*r/slow,steps=640*r*slow,H=10.4*r,refine=r,base_nu=nu,slow=slow)
def initial_fraction(c):
    z,y,x=np.indices(c['shape'],dtype=float);r=c['refine'];return .5*(1-np.tanh(np.maximum((x+.5)/r-32*.33,(z+.5)/r-16*.65)/1.5))
def run_case(name,nu=.02,r=1,mode='fast',write_vtk=False,steps=None,slow=1):
    c=configuration(nu,r,slow);H=c['H'];g=c['gravity'];N=c['steps'] if steps is None else steps
    s=LongDamBreak(**{k:c[k] for k in ('shape','gravity','nu_liquid','sigma','xi','tau_phase')});a=initial_fraction(c);rho=s.mat(a)[0];f,h=s.initialize(a,hydrostatic(a,rho,g));s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),c['shape']))
    z,y,x=np.indices(c['shape']);mass=a.sum();mass0=rho.sum();rows=[];snaps=[];ts=[];status='completed';start=time.monotonic()
    for t in range(N+1):
        try:
            if t:f,h=s.step(f,h,mode)
            a,rho,mu,v,p,u,*_=s.decode(f,h);speed=float(np.linalg.norm(u,axis=0).max())
        except (ValueError,RuntimeError,FloatingPointError) as e:status=str(e);break
        wet=a>.5;floor=np.flatnonzero(wet[0,c['shape'][1]//2]);left=np.flatnonzero(wet[:,c['shape'][1]//2,0]);front=float((floor[-1]+1)/H) if floor.size else 0.;height=float((left[-1]+1)/H) if left.size else 0.
        rows.append(dict(case=name,step=t,t_star=t*np.sqrt(g/H),front_H=front,left_height_H=height,com_x_H=float((a*(x+.5)).sum()/a.sum()/H),com_z_H=float((a*(z+.5)).sum()/a.sum()/H),speed=speed,mach=speed*np.sqrt(3),phase_relative_error=float(abs(a.sum()-mass)/mass),material_relative_error=float(abs(rho.sum()-mass0)/mass0),min_alpha=float(a.min()),max_alpha=float(a.max())))
        if t%(32*r*slow)==0 or t==N:snaps.append(a[:,c['shape'][1]//2].copy());ts.append(t)
        if write_vtk and (t%(160*r*slow)==0 or t==N):vti(OUT/f'{name}_{t:04}.vti',a,rho,mu,p,u)
        if not np.isfinite(speed) or speed>.1:status='speed guard exceeded';break
    if not rows:raise RuntimeError(status)
    result=dict(configuration=c,mode=mode,status=status,steps=rows[-1]['step'],Re=float(np.sqrt(g*H)*H/c['nu_liquid']),Bo=float((1-1/s.ratio)*g*H**2/c['sigma']),initial=rows[0],final=rows[-1],peak_speed=max(q['speed'] for q in rows),max_phase_relative_error=max(q['phase_relative_error'] for q in rows),max_material_relative_error=max(q['material_relative_error'] for q in rows),min_alpha=min(q['min_alpha'] for q in rows),max_alpha=max(q['max_alpha'] for q in rows),seconds=time.monotonic()-start)
    if s.t:result['transport_qubits']=[q.qubits for q in s.t];result['logical_mcx']=[len(q.direct_gates) for q in s.t]
    np.savez_compressed(OUT/f'{name}.npz',alpha=np.array(snaps),steps=np.array(ts),f=f,h=h)
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/f'{name}.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(name,status,result['steps'],result['Re'],result['final'],flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--nu',type=float,default=.02);p.add_argument('--refine',type=int,default=1);p.add_argument('--mode',choices=['fast','gate','reference'],default='fast');p.add_argument('--vtk',action='store_true');p.add_argument('--steps',type=int);p.add_argument('--slow',type=int,default=1);a=p.parse_args();run_case(a.name,a.nu,a.refine,a.mode,a.vtk,a.steps,a.slow)
