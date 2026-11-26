"""Matched hydrostatic, gauge, perturbation and dam-release experiments."""
from pathlib import Path
import sys,csv,json,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone22.solver import BalancedDamBreak,PressureDamBreak,hydrostatic,CS2,W19,C19,weights,dot
from milestone19.solver import initial
from milestone21.run import vti
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
def csv_save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def hydro_case(width,ratio,balanced,perturb=0.,mode='gate',offset=0.,steps=64):
    cls=BalancedDamBreak if balanced else PressureDamBreak
    s=cls(shape=(8,2,2),ratio=ratio,gravity=1e-5,**({'pressure_reference':offset} if balanced else {}));z,y,x=np.indices(s.shape)
    a=.5*(1-np.tanh((z+.5-4)/width));p0=hydrostatic(a,s.mat(a)[0],s.gravity)
    if balanced:f,h=s.set_hydrostatic_background(a)
    else:f,h=s.initialize(a,p0+offset)
    if perturb:
        u=np.zeros((3,)+s.shape);u[2]=perturb*np.sin(np.pi*(z+.5)/8);f+=weights(W19)*dot(C19,u)/CS2
    history=[];status='completed';p=p0.copy();speed=0.
    for t in range(steps+1):
        if t:
            try:f,h=s.step(f,h,mode,freeze=True)
            except (ValueError,RuntimeError,FloatingPointError) as e:status=str(e);break
        _,_,_,_,p,u,*_=s.decode(f,h);speed=float(np.linalg.norm(u,axis=0).max())
        err=float(np.sqrt(np.mean((p-p0-(0 if balanced else offset))**2)))
        history.append(dict(width=width,ratio=ratio,balanced=balanced,perturbation=perturb,mode=mode,offset=offset,step=t,max_speed=speed,pressure_rmse=err))
        if not np.isfinite(speed) or speed>.1:status='low-Mach speed guard exceeded';break
    row=dict(width=width,ratio=ratio,balanced=balanced,perturbation=perturb,mode=mode,offset=offset,steps=history[-1]['step'],status=status,peak_speed=max(x['max_speed'] for x in history),final_speed=speed,pressure_rmse=history[-1]['pressure_rmse'],defect_max=float(np.max(abs(s.defect))) if balanced else 0.,defect_sum=float(s.defect.sum()) if balanced else 0.)
    print('hydro',row,flush=True);return row,history

def dam_case(name,mode,background=False,offset=0.,gravity=.0005,steps=96,legacy=False):
    shape=(16,4,32);cls=PressureDamBreak if legacy else BalancedDamBreak
    s=cls(shape=shape,gravity=gravity,sigma=1e-5,**({} if legacy else {'pressure_reference':offset}));_,g=initial(shape);a=(1+g.sum(0))/2;p0=hydrostatic(a,s.mat(a)[0],gravity)
    if background:s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),shape))
    f,h=s.initialize(a,p0+offset);rows=[];hist=[];z,y,x=np.indices(shape);status='completed'
    for t in range(steps+1):
        if t:
            try:f,h=s.step(f,h,mode)
            except (ValueError,RuntimeError,FloatingPointError) as e:status=str(e);break
        a,rho,mu,nu,p,u,*_=s.decode(f,h);speed=float(np.linalg.norm(u,axis=0).max());hist.append(np.concatenate([a[None],rho[None],mu[None],p[None],u]));rows.append(dict(case=name,step=t,phase_sum=float(a.sum()),material_mass=float(rho.sum()),pressure_sum=float(f.sum()),min_alpha=float(a.min()),max_alpha=float(a.max()),max_speed=speed,com_x=float((a*x).sum()/a.sum()),com_z=float((a*z).sum()/a.sum()),front=int(np.argwhere(a>.5)[:,2].max()) if np.any(a>.5) else -1))
        if name=='balanced_gate' and t%24==0:vti(OUT/f'step_{t:03}.vti',a,rho,mu,p,u)
        if not np.isfinite(speed) or speed>.1:status='low-Mach speed guard exceeded';break
    r0,r1=rows[0],rows[-1];res=dict(status=status,steps=r1['step'],background=background,offset=offset,gravity=gravity,peak_speed=max(q['max_speed'] for q in rows),max_phase_drift=max(abs(q['phase_sum']-r0['phase_sum']) for q in rows),max_material_mass_drift=max(abs(q['material_mass']-r0['material_mass']) for q in rows),max_pressure_sum_drift=max(abs(q['pressure_sum']-r0['pressure_sum']) for q in rows),min_alpha=min(q['min_alpha'] for q in rows),max_alpha=max(q['max_alpha'] for q in rows),initial_com_x=r0['com_x'],final_com_x=r1['com_x'],initial_com_z=r0['com_z'],final_com_z=r1['com_z'],final_front=r1['front'],background_defect_max=float(np.max(abs(s.defect))) if background else 0.)
    print('dam',name,res,flush=True);return res,rows,np.array(hist),f,h

def run():
    hydro=[];trace=[];ratio=998/1.2
    for width in [.1,.5,1.]:
        for b in [False,True]:
            for pert in [0.,1e-7]:
                r,h=hydro_case(width,ratio,b,pert);hydro.append(r);trace+=h
    csv_save('hydrostatic.csv',hydro);csv_save('hydrostatic_history.csv',trace)
    cases=[('baseline','reference',False,0.,.0005,96,True),('legacy_offset','reference',False,1e-5,.0005,96,True),('gauge_offset','reference',False,1e-5,.0005,96,False),('balanced_reference','reference',True,0.,.0005,96,False),('balanced_gate','gate',True,0.,.0005,96,False),('balanced_offset','reference',True,.001,.0005,96,False),('strong_balanced','reference',True,0.,.003,96,False)]
    results={};hist={};rows=[];final={}
    for name,*args in cases:
        r,rr,hh,f,g=dam_case(name,*args);results[name]=r;rows+=rr;hist[name]=hh;final[name]=(f,g)
    for x,y in [('gauge_offset','baseline'),('balanced_gate','balanced_reference'),('balanced_offset','balanced_reference'),('balanced_reference','baseline')]:
        n=min(len(hist[x]),len(hist[y]));results[x]['comparison_to']=y;results[x]['comparison_steps']=n-1;results[x]['max_field_error']=float(np.max(abs(hist[x][:n]-hist[y][:n])));results[x]['phase_rmse']=float(np.sqrt(np.mean((hist[x][n-1,0]-hist[y][n-1,0])**2)))
        if len(hist[x])==len(hist[y]):results[x]['final_population_max_error']=float(max(np.max(abs(a-b)) for a,b in zip(final[x],final[y])))
    csv_save('dam_release.csv',rows);np.savez_compressed(OUT/'history.npz',**hist)
    report=dict(model='M22 explicit pressure gauge and fixed hydrostatic background residual subtraction',hydrostatic=hydro,dam_release=results,python=platform.python_version(),numpy=np.__version__)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run()
