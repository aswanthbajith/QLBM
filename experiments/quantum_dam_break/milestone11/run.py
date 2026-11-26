"""Run integrated gate-based D2Q25 plus separate integer and float references."""
import sys,json,time
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement,ElementTree
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone11 import reference as r
from milestone11.solver import Solver
from milestone11.collision import CollisionOracle
from milestone10.kinetic import G,ritter,feasible,SETS
from milestone5.run_2d import csvout
OUT=Path(__file__).parent/'results'
def ref_transport(f,solid,stride,periodic):
    ny,nx=solid.shape
    if periodic:return np.stack([np.roll(f[...,i],(int(y)*stride,int(x)*stride),axis=(0,1)) for i,(x,y) in enumerate(r.V)],-1)
    out=np.zeros_like(f)
    for y,x in np.argwhere(~solid):
        for i,(cx,cy) in enumerate(r.V):
            a,b=x+cx*stride,y+cy*stride
            if not(0<=a<nx and 0<=b<ny) or solid[min(y,b):max(y,b)+1,min(x,a):max(x,a)+1].any():out[y,x,24-i]+=f[y,x,i]
            else:out[b,a,i]+=f[y,x,i]
    return out
def reference_step(f,c):
    z=np.zeros_like(f);flags=0
    for idx in zip(*np.nonzero(~c.solid)):
        z[idx],flag=r.collision(f[idx],c.S,c.oracle.w);flags+=flag
    return ref_transport(z,c.solid,c.stride,c.periodic),flags
def vtk(path,f,solid,ref,dx):
    h,q,u=r.moments(f);ny,nx=solid.shape;ext=f'0 {nx} 0 {ny} 0 0';root=Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian');im=SubElement(root,'ImageData',WholeExtent=ext,Origin='0 0 0',Spacing=f'{dx} {dx} 1');p=SubElement(im,'Piece',Extent=ext);SubElement(p,'PointData');d=SubElement(p,'CellData',Scalars='depth_lu')
    vec=lambda a:np.concatenate((a,np.zeros((*h.shape,1))),axis=-1)
    stress=np.einsum('...i,ij,ik->...jk',f,r.V/2,r.V/2)
    fields=[('depth_lu',h,1),('discharge_lu',vec(q),3),('velocity_lu',vec(u),3),('hydrostatic_target_lu',G*h*h/2,1),('actual_kinetic_moment_xx_lu',stress[...,0,0],1),('actual_kinetic_moment_yy_lu',stress[...,1,1],1),('actual_kinetic_moment_xy_lu',stress[...,0,1],1),('wet_mask',h>0,1),('solid_mask',solid,1),('reference_depth_lu',ref,1)]+[(f'f{i:02}_lu',f[...,i],1) for i in range(25)]
    for name,arr,nc in fields:
        node=SubElement(d,'DataArray',type='Float64',Name=name,NumberOfComponents=str(nc),format='ascii');node.text=' '.join(f'{a:.12g}' for a in np.asarray(arr,float).ravel())
    ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
def experiment(c,h0,steps,name,save_vtk=False):
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True);f=r.initialize(h0,c.S,c.oracle.w);ff,_=r.float_collision(np.pad(h0[...,None],((0,0),(0,0),(12,12))));m=int(f.sum());hist=[];refs=[];rows=[];flags=0
    for j in range(steps+1):
        a=f/c.S;h,q,u=r.moments(a);rh=ff.sum(-1);wet=h>0;ok=feasible(SETS['D2Q25'],h,u[...,0])&feasible(SETS['D2Q25'],h,u[...,1]);hist.append(a);refs.append(rh.copy());row=dict(step=j,total_depth_ticks=int(f.sum()),integrated_depth_lu=float(h.sum()*c.dx**2),wet_cells=int(wet.sum()),infeasible_wet_cells=int((wet&~ok).sum()),max_speed=float(np.linalg.norm(u,axis=-1).max()),max_abs_transverse_velocity=float(abs(u[...,1]).max()),depth_rmse_vs_float=float(np.sqrt(np.mean((h[~c.solid]-rh[~c.solid])**2))),overflow_rejected_last_step=flags,obstacle_ticks=int(f[c.solid].sum()))
        if c.periodic and j:
            x=(np.arange(c.nx)+.5-c.nx/2)*c.dx;ex,_=ritter(x,j);sl=abs(x)<28;row['ritter_rmse']=float(np.sqrt(np.mean((h[0,sl]-ex[sl])**2)));row['float_ritter_rmse']=float(np.sqrt(np.mean((rh[0,sl]-ex[sl])**2)))
        elif c.periodic:row.update(ritter_rmse=0.,float_ritter_rmse=0.)
        rows.append(row)
        if save_vtk:vtk(folder/f'step_{j:03}.vti',a,c.solid,rh,c.dx)
        if j<steps:
            z,flags=c.step(f);zz,rf=reference_step(f,c);assert np.array_equal(z,zz) and flags==rf and z.sum()==m;f=z
            eq,_=r.float_collision(ff);ff=ref_transport(eq,c.solid,c.stride,c.periodic)
        if j%8==0 or j==steps:print(name,j,row,flush=True)
    csvout(folder/'diagnostics.csv',rows);np.savez_compressed(folder/'history.npz',populations=np.array(hist),reference_depth=np.array(refs),solid=c.solid,dx=c.dx)
    return dict(initial=rows[0],final=rows[-1],steps=steps,verified_inverse_steps=steps,integer_reference_agreement_steps=steps,max_mass_error_ticks=max(abs(a['total_depth_ticks']-m) for a in rows),overflow_rejections=sum(a['overflow_rejected_last_step'] for a in rows),resources=c.resources())
def local_benchmark(o):
    rng=np.random.default_rng(11);n=1024;H=rng.integers(0,1<<o.w,n);f=np.array([rng.multinomial(int(h),rng.dirichlet(np.ones(25)*.3)) for h in H]);z,flag=o.evaluate(f);assert not flag.any();assert np.array_equal(z,[r.collision(a,o.S,o.w)[0] for a in f]);assert np.array_equal(z.sum(-1),H) and np.array_equal(z@r.V,f@r.V)
    h,q,u=r.moments(z/o.S);stress=np.einsum('ni,ij,ik->njk',z/o.S,r.V/2,r.V/2);target=h[:,None,None]*u[:,:,None]*u[:,None,:]+G*h[:,None,None]**2/2*np.eye(2);feas=feasible(SETS['D2Q25'],h,u[:,0])&feasible(SETS['D2Q25'],h,u[:,1]);err=stress-target
    rows=[dict(depth=float(h[i]),ux=float(u[i,0]),uy=float(u[i,1]),feasible=bool(feas[i]),xx_error=float(err[i,0,0]),yy_error=float(err[i,1,1]),xy_error=float(err[i,0,1])) for i in range(n)]
    csvout(OUT/'local_moments.csv',rows);return dict(tested=n,feasible=int(feas.sum()),max_mass_error_ticks=int(abs(z.sum(-1)-H).max()),max_momentum_error_index_ticks=int(abs(z@r.V-f@r.V).max()),feasible_moment_rmse=float(np.sqrt(np.mean(err[feas]**2))),feasible_cross_moment_rmse=float(np.sqrt(np.mean(err[feas,0,1]**2))))
def main():
    OUT.mkdir(exist_ok=True);t=time.monotonic();mask=np.zeros((16,16),bool);mask[5:11,8:10]=True;mask[1:4,12]=True;h=np.zeros(mask.shape);h[:,:5]=1.5;o=CollisionOracle(256);c=Solver(mask,256,oracle=o);report={'model':'D2Q25 projected product equilibrium, full relaxation omega=1, g=1/8, nonnegative fixed-point words','backend':'X/CX/CCX bit-sliced computational-basis execution, exact word permutation for transport, fresh basis registers per step; classical references independent; no native Qiskit or hardware','local':local_benchmark(o),'obstacle':experiment(c,h,24,'obstacle',True),'strip':[]}
    mask=np.zeros((1,2048),bool);h=np.zeros(mask.shape);h[:,:1024]=1.5
    for S in (64,256,1024):
        c=Solver(mask,S,stride=4,periodic=True,oracle=o if S==256 else None);report['strip'].append(dict(scale=S,**experiment(c,h,24,f'strip_{S}')))
    report['elapsed_seconds']=time.monotonic()-t;(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
