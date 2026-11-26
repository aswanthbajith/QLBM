"""Execute reversible D2Q9 experiments; floating solvers are independent comparisons."""
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone8.solver import LBM
from milestone8 import reference as ref
from milestone5.run_2d import csvout
from xml.etree.ElementTree import Element,SubElement,ElementTree
from quantum_core import C
OUT=Path(__file__).parent/'results'

def vtk(path,f,solid,r):
    h,u=ref.moments(f);q=f@C;ny,nx=solid.shape;ext=f'0 {nx} 0 {ny} 0 0';root=Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian');im=SubElement(root,'ImageData',WholeExtent=ext,Origin='0 0 0',Spacing='1 1 1');p=SubElement(im,'Piece',Extent=ext);SubElement(p,'PointData');d=SubElement(p,'CellData',Scalars='depth_lu')
    vec=lambda a:np.concatenate((a,np.zeros((*h.shape,1))),axis=2)
    fields=[('depth_lu',h,1),('discharge_lu',vec(q),3),('velocity_lu',vec(u),3),('hydrostatic_flux_lu',ref.G*h*h/2,1),('wet_mask',h>0,1),('solid_mask',solid,1),('reference_depth_lu',r,1)]+[(f'f{i}_lu',f[...,i],1) for i in range(9)]
    for name,arr,nc in fields:
        node=SubElement(d,'DataArray',type='Float64',Name=name,NumberOfComponents=str(nc),format='ascii');node.text=' '.join(f'{a:.12g}' for a in np.asarray(arr,float).ravel())
    ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def experiment(c,low,steps,name):
    S=c.S;mask=c.solid;h=np.full(mask.shape,low);h[:,:5]=1.5;h[mask]=0;u=np.zeros((*h.shape,2));f=ref.initialize(h,u,S);ff=ref.equilibrium(h,u);m=f.sum();hist=[];rh=[];rows=[];bad=0;first_negative=None;float_failed=None;max_negative=0.;folder=OUT/name;folder.mkdir(exist_ok=True)
    for j in range(steps+1):
        a=f/S;h,u=ref.moments(a);r=ff.sum(-1) if ff is not None else np.full(mask.shape,np.nan)
        if ff is not None:
            minf=float(ff.min());max_negative=min(max_negative,minf)
            if minf < -1e-12 and first_negative is None:first_negative=j
        rows.append(dict(step=j,total_depth=float(h.sum()),wet_cells=int(np.count_nonzero(h)),max_depth=float(h.max()),max_speed=float(np.linalg.norm(u,axis=-1).max()),max_transverse_velocity=float(abs(u[:,:,1]).max()),rejected_collisions_last_step=bad,depth_rmse_vs_float=None if ff is None else float(np.sqrt(np.mean((h[~mask]-r[~mask])**2))),float_reference_valid=ff is not None,obstacle_depth=float(h[mask].sum())))
        hist.append(a);rh.append(r.copy());vtk(folder/f'step_{j:03}.vti',a,mask,r)
        if j<steps:
            z,bad=c.step(f);rint,rb=ref.integer_step(f,mask,S,c.omega_num);assert np.array_equal(z,rint) and bad==rb and z.sum()==m;f=z
            if ff is not None:
                ff=ref.float_step(ff,mask,c.omega_num/2)
                depth=ff.sum(-1)
                if not np.isfinite(ff).all() or np.max(abs(ff))>4 or depth.min() < -1e-9:
                    float_failed=j+1;ff=None
    csvout(folder/'diagnostics.csv',rows);np.savez_compressed(folder/'history.npz',populations=np.array(hist),reference_depth=np.array(rh),solid=mask)
    report=dict(initial=rows[0],final=rows[-1],steps=steps,quantized_reference_agreement=steps,reversibility_verified=steps,conservation_max_error=max(abs(v['total_depth']-rows[0]['total_depth']) for v in rows),rejected_collisions_total=sum(v['rejected_collisions_last_step'] for v in rows),collision_attempts=steps*len(c.active),float_first_negative_population=first_negative,float_reference_stopped_step=float_failed,min_float_population_before_stop=max_negative)
    print(name,json.dumps(report),flush=True);return report

def shear():
    S=16384;mask=np.zeros((32,4),bool);wave=np.sin(2*np.pi*np.arange(32)/32);h=np.ones(mask.shape);u=np.zeros((*h.shape,2));u[:,:,0]=.01*wave[:,None];f0=ref.initialize(h,u,S);rows=[];summary=[];k2=(2*np.pi/32)**2
    for k in (1,2,3):
        c=LBM(mask,S,k,True);f=f0.copy();ff=f/S;A=[];B=[];bad=0
        for j in range(65):
            h,u=ref.moments(f/S);A.append(float(np.sum(u[:,:,0]*wave[:,None])/(4*np.sum(wave**2))));h,u=ref.moments(ff);B.append(float(np.sum(u[:,:,0]*wave[:,None])/(4*np.sum(wave**2))));rows.append(dict(omega=k/2,step=j,gate_amplitude=A[-1],float_amplitude=B[-1],theory_amplitude=A[0]*np.exp(-(2/k-.5)/3*k2*j)))
            if j<64:
                z,b=c.step(f);r,rb=ref.integer_step(f,mask,S,k,True);assert np.array_equal(z,r) and b==rb;f=z;bad+=b;ff=ref.float_step(ff,mask,k/2,True)
        def fit(v):return float(-np.polyfit(np.arange(8,49),np.log(np.array(v)[8:49]/v[0]),1)[0]/k2)
        summary.append(dict(omega=k/2,tau=2/k,nu_theoretical_lu=(2/k-.5)/3,nu_gate_fit_lu=fit(A),nu_float_fit_lu=fit(B),rejected_collisions=bad,**c.resources()))
    csvout(OUT/'shear_decay.csv',rows);csvout(OUT/'viscosity.csv',summary);return summary

def pressure():
    S=4096;h=np.linspace(.5,1.5,33);u=np.zeros((len(h),2));f=ref.initialize(h,u,S);c=LBM(np.zeros((2,2),bool),S);z,b=c.oracle.evaluate(f);hd=z.sum(-1)/S;pi=np.einsum('ni,ij,ik->njk',z/S,C,C);p=ref.G*hd**2/2;rows=[dict(depth=float(a),target=float(v),xx=float(m[0,0]),yy=float(m[1,1]),xy=float(m[0,1])) for a,v,m in zip(hd,p,pi)];csvout(OUT/'pressure.csv',rows);return dict(relative_l2=float(np.linalg.norm(pi[:,0,0]-p)/np.linalg.norm(p)),max_abs_error=float(np.max(abs(pi[:,0,0]-p))),rejected=int(b.sum()))

def run():
    OUT.mkdir(exist_ok=True);mask=np.zeros((16,16),bool);mask[5:11,8:10]=True;mask[1:4,12]=True;c=LBM(mask,4096,2);start=time.monotonic();wet=experiment(c,1.25,32,'wet_bed');dry=experiment(c,0.,24,'dry_bed_stress');p=pressure();s=shear();report=dict(model='D2Q9 shallow-water BGK collide-stream, flat bed, g_lu=1/8, conservative integer rounding and reject-invalid limiter',backend='bit-sliced execution of X/CX/CCX gates on computational basis; streaming exact wire permutation; no QPU/full statevector; fresh registers each step',scale=c.S,omega=1.,grid=[16,16],solid_cells=int(mask.sum()),resources=c.resources(),wet_bed=wet,dry_bed_stress=dry,pressure=p,viscosity=s,native_qiskit='not executed; not installed',elapsed_seconds=time.monotonic()-start);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':run()
