"""Gate simulation and independent references for the positivity limiter."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from quantum_core import C
from milestone8 import reference as old
from milestone8.solver import LBM
from milestone8.run import vtk
from milestone5.run_2d import csvout
from milestone9.collision import LimitedCollisionOracle
from milestone9 import reference as ref
OUT=Path(__file__).parent/'results'
def bound(f,S):
    h,u=old.moments(f/S);q=f@C/S;ideal=h[...,None]*u*u+old.G*h[...,None]**2/2
    gap=np.maximum(abs(q)-ideal,ideal-h[...,None]);return (gap>1e-10).any(-1),np.maximum(gap,0).max()
def case(o,mask,h,steps,name,periodic=False):
    S=o.scale;c=LBM(mask,S,o.omega_num,periodic,o);f=old.initialize(h,np.zeros((*h.shape,2)),S);g=f.copy();m=f.sum();rows=[];hist=[];base=[];counts=np.zeros(3,int);bad=0;folder=OUT/name;folder.mkdir(exist_ok=True)
    for j in range(steps+1):
        h,u=old.moments(f/S);h0=g.sum(-1)/S;b,gap=bound(f,S)
        rows.append(dict(step=j,total_depth=float(h.sum()),wet_cells=int((h>0).sum()),max_speed=float(np.linalg.norm(u,axis=-1).max()),obstacle_depth=float(h[mask].sum()),full_bgk=int(counts[0]),limited_update=int(counts[1]),blocked=int(counts[2]),m8_rejected=int(bad),pressure_bound_violations=int(b.sum()),max_pressure_gap=float(gap),depth_rmse_vs_m8=float(np.sqrt(np.mean((h[~mask]-h0[~mask])**2)))))
        hist.append(f.copy());base.append(g.copy());vtk(folder/f'step_{j:03}.vti',f/S,mask,h0)
        if j==steps:break
        a,flag=o.evaluate(f[~mask]);changed=np.any(a!=f[~mask],axis=1);counts=np.bincount(np.where(flag,np.where(changed,1,2),0),minlength=3)
        post=np.zeros_like(f);post[~mask]=a
        assert np.array_equal(post.sum(-1),f.sum(-1)) and np.array_equal(post@C,f@C)
        z=c.transport(post);assert np.array_equal(c.transport(z,True),post) and z.sum()==m and not z[mask].any()
        r,rc=ref.step(f,mask,S,o.omega_num,periodic);assert np.array_equal(z,r) and np.array_equal(counts,rc)
        f=z;g,bad=old.integer_step(g,mask,S,o.omega_num,periodic)
        print(name,j+1,counts.tolist(),flush=True)
    a=np.array(hist);b=np.array(base);np.savez_compressed(folder/'history.npz',populations_ticks=a,m8_ticks=b,scale=S,solid=mask);csvout(folder/'diagnostics.csv',rows)
    return dict(steps=steps,initial=rows[0],final=rows[-1],**{k+'_total':sum(r[k] for r in rows) for k in ('full_bgk','limited_update','blocked','m8_rejected')},reference_agreement_steps=steps,inverse_and_cleanup_steps=steps,resources=c.resources(),mass_error_max=max(abs(r['total_depth']-rows[0]['total_depth']) for r in rows)),a,b

def main():
    S=4096;o=LimitedCollisionOracle(S);mask=np.zeros((16,16),bool);mask[5:11,8:10]=True;mask[1:4,12]=True
    h=np.zeros(mask.shape);h[:,:5]=1.5;h[mask]=0;dry,_,_=case(o,mask,h,24,'dry_bed')
    h=np.full(mask.shape,1.25);h[:,:5]=1.5;h[mask]=0;wet,_,_=case(o,mask,h,8,'wet_bed_regression')
    mask=np.zeros((2,128),bool);h=np.zeros(mask.shape);h[:,:64]=1.5;one,a,b=case(o,mask,h,24,'ritter_1d',True)
    x=np.arange(128)+.5-64;sl=abs(x)<32;rows=[]
    for j in (4,8,12,16,24):
        c=np.sqrt(old.G*1.5);xi=x/j;exact=np.where(xi<-c,1.5,np.where(xi>2*c,0,(2*c-xi)**2/(9*old.G)));h=a[j].sum(-1)/S;g=b[j].sum(-1)/S
        rows.append(dict(step=j,rmse_m9=float(np.sqrt(np.mean((h[:,sl]-exact[sl])**2))),rmse_m8=float(np.sqrt(np.mean((g[:,sl]-exact[sl])**2))),exact_front=float(2*c*j),m9_front=float(x[(h[0]>.01)&(x>0)&sl].max())))
    csvout(OUT/'ritter.csv',rows);report=dict(model='D2Q9 BGK with bounded conservative coordinate fallback',backend='bit-sliced explicit X/CX/CCX basis simulation; streaming permutation; no QPU',scale=S,omega=1,g_lattice=old.G,dry_bed=dry,wet_bed_regression=wet,ritter_1d=one,ritter=rows);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
