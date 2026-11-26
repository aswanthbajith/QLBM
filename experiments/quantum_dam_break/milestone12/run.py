"""Execute M12 gate circuits; compare with independent integer and float models."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone12 import reference as r
from milestone12.collision import CollisionOracle
from milestone11 import reference as old
from milestone11.collision import CollisionOracle as OldOracle
from milestone11.solver import Solver
from milestone11.run import ref_transport,vtk
from milestone10.kinetic import G,ritter,feasible,SETS,equilibrium2d
from milestone5.run_2d import csvout
OUT=Path(__file__).parent/'results'
def step(c,f):
    z,flags=c.step(f);p=np.zeros_like(f);o=c.oracle;omega=getattr(o,'omega_num',4);sym=getattr(o,'symmetry',False)
    p[~c.solid]=[r.collision(a,c.S,o.w,omega,sym)[0] for a in f[~c.solid]]
    assert np.array_equal(z,ref_transport(p,c.solid,c.stride,c.periodic)) and not flags
    return z

def experiment(name,mask,h,oracle,steps,periodic=False,save_vtk=False,initial=None):
    c=Solver(mask,oracle.S,periodic=periodic,oracle=oracle);S=c.S;omega=getattr(oracle,'omega_num',4)/4
    f=r.initialize(h,S,oracle.w) if initial is None else initial.copy();mass=int(f.sum());ff,_=old.float_collision(f/S);ff=f/S # identical starting quantized populations
    folder=OUT/name;folder.mkdir(exist_ok=True);hist=[];rows=[]
    for t in range(steps+1):
        a=f/S;d,q,u=old.moments(a);rh=ff.sum(-1);wet=d>0;ok=feasible(SETS['D2Q25'],d,u[...,0])&feasible(SETS['D2Q25'],d,u[...,1]);row=dict(step=t,mass_ticks=int(f.sum()),wet_cells=int(wet.sum()),infeasible_cells=int((wet&~ok).sum()),max_transverse_velocity=float(abs(u[...,1]).max()),depth_rmse_float=float(np.sqrt(np.mean((d[~mask]-rh[~mask])**2))),obstacle_ticks=int(f[mask].sum()))
        if periodic and t:
            x=(np.arange(c.nx)+.5-c.nx/2)*c.dx;exact,_=ritter(x,t);sel=abs(x)<28;row['ritter_rmse']=float(np.sqrt(np.mean((d[0,sel]-exact[sel])**2)))
        else:row['ritter_rmse']=0.
        rows.append(row);hist.append(d)
        if save_vtk:vtk(folder/f'step_{t:03}.vti',a,mask,rh,c.dx)
        if t<steps:
            f=step(c,f);eq,_=old.float_collision(ff);ff=ref_transport((1-omega)*ff+omega*eq,mask,1,periodic)
        assert f.sum()==mass and not f[mask].any()
    csvout(folder/'diagnostics.csv',rows);np.savez_compressed(folder/'depth.npz',depth=hist,solid=mask,dx=c.dx)
    result=dict(initial=rows[0],final=rows[-1],max_mass_error_ticks=max(abs(z['mass_ticks']-mass) for z in rows),max_transverse_velocity=max(z['max_transverse_velocity'] for z in rows),resources=c.resources(),steps=steps,integer_reference_agreement_steps=steps)
    print(name,result,flush=True);return result

def shear(o,steps=40):
    n=32;mask=np.zeros((n,1),bool);c=Solver(mask,o.S,periodic=True,oracle=o);y=np.arange(n)*c.dx;k=2*np.pi/(n*c.dx);wave=np.sin(k*y);h=np.full(mask.shape,1.5);u=(.08*wave)[:,None];f=r.initialize(h,o.S,o.w,ux=u);ff=f/o.S;mass=f.sum();omega=o.omega_num/4;rows=[]
    for t in range(steps+1):
        d,q,v=old.moments(f/o.S);df,qf,vf=old.moments(ff);A=float(2*np.mean(v[:,0,0]*wave));Af=float(2*np.mean(vf[:,0,0]*wave));rows.append(dict(omega=omega,step=t,amplitude=A,float_amplitude=Af,max_depth_error=float(abs(d-1.5).max()),transverse_velocity=float(abs(v[...,1]).max())))
        if t<steps:
            f=step(c,f);eq,_=old.float_collision(ff);ff=ref_transport((1-omega)*ff+omega*eq,mask,1,True)
            assert f.sum()==mass
    def fit(key):
        a=np.array([r[key] for r in rows]);t=np.arange(len(a));ok=(t>=5)&(a>0);b=np.polyfit(t[ok],np.log(a[ok]),1);pred=np.polyval(b,t[ok]);r2=1-((np.log(a[ok])-pred)**2).sum()/((np.log(a[ok])-np.log(a[ok]).mean())**2).sum() if np.ptp(a[ok]) else 0.;return float(-b[0]/k**2),float(r2)
    v,r2=fit('amplitude');vf,r2f=fit('float_amplitude');report=dict(omega=omega,gate_viscosity=v,gate_fit_r2=r2,float_viscosity=vf,float_fit_r2=r2f,formal_small_k_viscosity=G*1.5/2*(1/omega-.5),initial_amplitude=rows[0]['amplitude'],final_amplitude=rows[-1]['amplitude'],max_depth_error=max(a['max_depth_error'] for a in rows),max_transverse_velocity=max(a['transverse_velocity'] for a in rows));print('shear',report,flush=True);return report,rows

def main():
    OUT.mkdir(exist_ok=True);report={'backend':'bit-sliced X/CX/CCX computational-basis gate simulation and reversible word permutation; no QPU/Qiskit execution','model':'dyadic convex BGK with conservative integer residual repair and conditional local reflection projection','shear':[],'strip':{}}
    mask=np.zeros((16,16),bool);mask[5:11,8:10]=True;mask[1:4,12]=True;h=np.zeros(mask.shape);h[:,:5]=1.5
    o=CollisionOracle(256,omega_num=2);report['obstacle']=experiment('obstacle',mask,h,o,16,save_vtk=True)
    mask=np.zeros((1,256),bool);h=np.zeros(mask.shape);h[:,:128]=1.5
    S=1024;full=CollisionOracle(S,omega_num=4);half=CollisionOracle(S,omega_num=2);base=OldOracle(S);ini=r.initialize(h,S,full.w)
    for name,c in [('baseline',base),('symmetric_full',full),('symmetric_half',half)]:report['strip'][name]=experiment(name,mask,h,c,24,True,initial=ini)
    shearrows=[]
    for n in (1,2,3,4):
        o=half if n==2 else full if n==4 else CollisionOracle(S,omega_num=n);a,b=shear(o);report['shear'].append(a);shearrows+=b
    csvout(OUT/'shear.csv',shearrows);csvout(OUT/'viscosity.csv',report['shear']);(OUT/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
