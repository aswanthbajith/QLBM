"""Reproducible hydrostatic, pressure-gauge and dam-release preflight."""
from pathlib import Path
import sys,csv,json,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone21.solver import PressureDamBreak,hydrostatic,initial,CS2
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)

def save_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def vti(path,a,rho,mu,p,u):
    nz,ny,nx=a.shape;ex=f'0 {nx} 0 {ny} 0 {nz}'
    fields=[('liquid_fraction',a,1),('material_density_lu',rho,1),('dynamic_viscosity_lu',mu,1),('pressure_lu',p,1),('velocity_lu',np.moveaxis(u,0,-1),3)]
    data='\n'.join(f'<DataArray type="Float64" Name="{k}" NumberOfComponents="{nc}" format="ascii">'+ ' '.join(f'{v:.16g}' for v in arr.ravel())+'</DataArray>' for k,arr,nc in fields)
    path.write_text(f'<?xml version="1.0"?><VTKFile type="ImageData" version="1.0" byte_order="LittleEndian"><ImageData WholeExtent="{ex}" Origin="0 0 0" Spacing="1 1 1"><Piece Extent="{ex}"><CellData>{data}</CellData></Piece></ImageData></VTKFile>\n')

def hydro_tests():
    rows=[];ratio=998/1.2
    cases=[(r,method,1.,0.,'reference') for r in [1,10,100,ratio] for method in ['paper','product']]
    cases += [(ratio,'product',1.,0.,'gate'),(ratio,'product',.1,0.,'reference'),(ratio,'paper',.1,0.,'reference'),(ratio,'product',1.,1e-5,'reference'),(ratio,'paper',1.,1e-5,'reference')]
    for r,method,width,offset,mode in cases:
        s=PressureDamBreak(ratio=r,pressure_force=method);z,y,x=np.indices(s.shape);a=.5*(1-np.tanh((z+.5-4)/width));rho=s.mat(a)[0];p0=hydrostatic(a,rho,s.gravity)+offset;f,h=s.initialize(a,p0);peak=0.;status='completed';t=0
        try:
            for t in range(1,33):
                f,h=s.step(f,h,mode,freeze=True);aa,rho,mu,nu,p,u,*_=s.decode(f,h);peak=max(peak,float(np.linalg.norm(u,axis=0).max()))
                if not np.isfinite(peak) or peak>.1:raise ValueError('speed guard')
        except (ValueError,RuntimeError,FloatingPointError) as e:status=str(e)
        rows.append(dict(ratio=r,force=method,width=width,pressure_offset=offset,mode=mode,steps=t,status=status,peak_speed=peak,final_speed=float(np.linalg.norm(u,axis=0).max()),pressure_rmse=float(np.sqrt(np.mean((p-p0)**2))),pressure_relative_rmse=float(np.linalg.norm(p-p0)/np.linalg.norm(p0)),phase_drift=float(abs(h.sum()-a.sum()))))
        print('hydro',rows[-1],flush=True)
    save_csv('hydrostatic.csv',rows);return rows

def dam_case(name,gravity,mode,steps=96,shape=(16,4,32)):
    s=PressureDamBreak(shape=shape,gravity=gravity,sigma=1e-5);_,g=initial(shape);a=(1+g.sum(0))/2;rho=s.mat(a)[0];f,h=s.initialize(a,hydrostatic(a,rho,gravity));z,y,x=np.indices(shape);rows=[];hist=[];pop=[];status='completed'
    for t in range(steps+1):
        if t:
            try:f,h=s.step(f,h,mode)
            except (ValueError,RuntimeError,FloatingPointError) as e:status=str(e);break
        a,rho,mu,nu,p,u,*_=s.decode(f,h);speed=float(np.linalg.norm(u,axis=0).max());hist.append(np.concatenate([a[None],rho[None],mu[None],p[None],u]));pop.append(np.r_[f.ravel(),h.ravel()]);cells=np.argwhere(a>.5)
        rows.append(dict(case=name,step=t,phase_sum=float(a.sum()),material_mass=float(rho.sum()),pressure_population_sum=float(f.sum()),com_x=float((a*x).sum()/a.sum()),com_z=float((a*z).sum()/a.sum()),front=int(cells[:,2].max()) if cells.size else -1,min_alpha=float(a.min()),max_alpha=float(a.max()),max_speed=speed,physical_momentum_x=float((rho*u[0]).sum()),physical_momentum_z=float((rho*u[2]).sum()),min_pressure=float(p.min()),max_pressure=float(p.max())))
        if name=='gate' and t%16==0:vti(OUT/f'step_{t:03}.vti',a,rho,mu,p,u)
        if not np.isfinite(speed) or speed>.1:status='low-Mach speed guard exceeded';break
    hist=np.array(hist);pop=np.array(pop);a0=rows[0];a1=rows[-1];res=dict(status=status,steps=a1['step'],gravity=gravity,shape_zyx=shape,initial_com_x=a0['com_x'],final_com_x=a1['com_x'],initial_com_z=a0['com_z'],final_com_z=a1['com_z'],initial_front=a0['front'],final_front=a1['front'],max_phase_drift=max(abs(q['phase_sum']-a0['phase_sum']) for q in rows),max_material_mass_drift=max(abs(q['material_mass']-a0['material_mass']) for q in rows),min_alpha=min(q['min_alpha'] for q in rows),max_alpha=max(q['max_alpha'] for q in rows),max_speed=max(q['max_speed'] for q in rows),final_max_speed=a1['max_speed'])
    if s.t:res['transport_qubits']=[t.qubits for t in s.t];res['direct_transport_mcx']=[len(t.direct_gates) for t in s.t]
    print(name,res,flush=True);return res,rows,hist,pop

def run():
    hydro=hydro_tests();r={};rows=[];hists={}
    for name,g,mode in [('reference',.0005,'reference'),('gate',.0005,'gate'),('zero_gravity',0.,'gate'),('strong_gravity',.003,'reference')]:
        res,rr,hh,pp=dam_case(name,g,mode);r[name]=res;rows.extend(rr);hists[name]=hh
        if name=='reference':ref=pp
        if name=='gate':
            assert res['status']=='completed';res['max_population_error_vs_reference']=float(np.max(abs(pp-ref)));res['final_alpha_rmse_vs_reference']=float(np.sqrt(np.mean((hh[-1,0]-hists['reference'][-1,0])**2)))
    save_csv('dam_release.csv',rows);np.savez_compressed(OUT/'history.npz',**hists)
    report=dict(model='M21 velocity-based pressure D3Q19 + bounded D3Q7, adapted from Fakhari et al.',density_ratio=998/1.2,dynamic_viscosity_ratio=.001/.000018,nu_liquid_lu=.02,sigma_lu=1e-5,xi_lu=3.,pressure_force='discrete product-rule',hydrostatic=hydro,dam_release=r,python=platform.python_version(),numpy=np.__version__)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run()
