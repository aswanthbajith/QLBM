"""Precision sweep and matched continuation of M27 dam break checkpoint."""
from pathlib import Path
import sys,json,time,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone28.arithmetic import JointMacros
from milestone28.solver import QuantumMacroDamBreak
from milestone27.solver import ThreeDDamBreak
from milestone27.run import initial_fraction
from milestone25.run import configuration
from milestone21.solver import hydrostatic
from milestone18.transport import C19
from milestone26.capture import derive
from milestone26.export import write_series
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
BASE=Path(__file__).resolve().parents[1]/'milestone27/results/gate/final_populations.npz'

def main(N=16):
    c=configuration(.0025,1,2);c.update(shape=(16,16,32),reservoir_width=8.)
    kw={k:c[k] for k in ('shape','gravity','nu_liquid','sigma','xi','tau_phase')}
    ref=ThreeDDamBreak(**kw);a0=initial_fraction(c['shape']);f0,h0=ref.initialize(a0,hydrostatic(a0,ref.mat(a0)[0],c['gravity']))
    checkpoint=np.load(BASE);f1,h1=checkpoint['f'],checkpoint['h'];precision=[]
    for k in (16,24,32,40):
        m=JointMacros(k,ref.ratio,ref.mu_ratio,ref.nu_liquid);S=1<<k;maxerr=np.zeros(7);maxint=0;amin=1.;amax=0.;mumin=1.;rhomin=1.
        for f,h in ((f0,h0),(f1,h1)):
            v=np.concatenate((f,h)).reshape(26,-1).T;ticks=np.rint(v*S).astype(np.int64)
            out=m.evaluate(ticks)
            exact=np.column_stack((ticks[:,19:].sum(1),ticks[:,:19].sum(1),ticks[:,:19]@C19))
            mat=np.array([[(lo*S+int(x)*(hi-lo)+S//2)//S for lo,hi in m.materials.coefficients] for x in exact[:,0]],np.int64)
            maxint=max(maxint,int(np.max(abs(out-np.column_stack((exact,mat))))))
            a=h.sum(0);rho,mu,_=ref.mat(a);r=np.column_stack((a.ravel(),f.sum(0).ravel(),f.reshape(19,-1).T@C19,rho.ravel(),mu.ravel()))
            maxerr=np.maximum(maxerr,np.max(abs(out/S-r),axis=0));amin=min(amin,float(out[:,0].min()/S));amax=max(amax,float(out[:,0].max()/S));rhomin=min(rhomin,float(out[:,5].min()/S));mumin=min(mumin,float(out[:,6].min()/S))
        precision.append(dict(bits=k,errors=dict(zip(('alpha','P','jx','jy','jz','rho','mu'),map(float,maxerr))),integer_error=maxint,min_alpha=amin,max_alpha=amax,min_rho=rhomin,min_mu=mumin,resources=m.resources()))
        print('precision',k,precision[-1],flush=True)
    q=QuantumMacroDamBreak(**kw,k=40);bg=np.broadcast_to(a0.mean((1,2),keepdims=True),c['shape']);q.set_hydrostatic_background(bg);ref.set_hydrostatic_background(bg)
    f,h=f1.copy(),h1.copy();fr,hr=f1.copy(),h1.copy();mass=h.sum();rows=[];fields={};steps=[];wall=[];wa=[];t0=time.monotonic();maxpop=0
    for t in range(N+1):
        if t:f,h=q.step(f,h,'gate');fr,hr=ref.step(fr,hr,'gate')
        a,rho,mu,nu,p,u,*_=q.decode(f,h);ar,rr,mr,nr,pr,ur,*_=ref.decode(fr,hr)
        maxpop=max(maxpop,float(np.max(abs(f-fr))),float(np.max(abs(h-hr))))
        row=dict(step=1280+t,phase_rmse=float(np.sqrt(np.mean((a-ar)**2))),max_density_error=float(np.max(abs(rho-rr))),max_velocity_component_error=float(np.max(abs(u-ur))),max_population_error=maxpop,phase_relative_error=float(abs(h.sum()-mass)/mass),min_alpha=float(a.min()),max_alpha=float(a.max()),max_speed=float(np.linalg.norm(u,axis=0).max()),raw_min_alpha=float(h.sum(0).min()),raw_max_alpha=float(h.sum(0).max()))
        if not np.isfinite(row['max_speed']) or row['max_speed']>.1:raise RuntimeError('speed guard')
        rows.append(row);wall.append(p[:,:,-1].copy());wa.append(a[:,:,-1].copy());print('checkpoint',row,flush=True)
        if t%4==0 or t==N:
            steps.append(1280+t)
            for k,v in derive(a,rho,mu,p,u).items():fields.setdefault(k,[]).append(v.copy())
    data={k:np.stack(v) for k,v in fields.items()};np.savez_compressed(OUT/'fields.npz',steps=steps,wall_pressure=np.stack(wall),wall_alpha=np.stack(wa),**data);np.savez_compressed(OUT/'final_populations.npz',f=f,h=h,reference_f=fr,reference_h=hr)
    write_series(OUT,data,steps,c)
    with (OUT/'history.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(configuration=c,precision=precision,checkpoint_start=1280,continuation_steps=N,final=rows[-1],maximum_phase_rmse=max(r['phase_rmse'] for r in rows),max_population_error=maxpop,max_phase_relative_error=max(r['phase_relative_error'] for r in rows),macro_calls=q.macro_calls,elapsed_seconds=time.monotonic()-t0,resources=q.circuit.resources(),scope='Basis-gate simulation of joint macroscopic moment and affine material circuit; hybrid per-cell preparation and readout; matched M27 gate backend.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',report['final'],flush=True)
if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 16)
