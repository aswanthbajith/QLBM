"""Reproduce local covariance, coherent dilation and sampled fluid trajectories."""
import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone14.twirl import TwirledCollision,GROUP,action,reference_branches,coherent_dilation
from milestone13.reference import collision,initialize
from milestone11.reference import V,moments,float_collision
from milestone11.solver import Solver
from milestone11.run import ref_transport,vtk
from milestone10.kinetic import G,ritter,feasible,SETS
OUT=Path(__file__).parent/'results'

def csvout(path,rows):
    with path.open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def branches(o,f):
    return np.concatenate([o.branches(f[i:i+64])[0] for i in range(0,len(f),64)])

def local():
    rng=np.random.default_rng(13);o=TwirledCollision(256,omega_num=4)
    f=np.array([rng.multinomial(int(h),rng.dirichlet(np.ones(25)*.3)) for h in rng.integers(1,1<<o.w,1024)])
    base,flag=o.base.evaluate(f);assert not flag.any()
    y=branches(o,f);assert np.array_equal(y,np.array([reference_branches(a,o.S,o.w,o.omega_num)[0] for a in f]))
    assert np.all(y.sum(-1)==f.sum(-1)[:,None]) and np.all(y@V==(f@V)[:,None,:]) and y.min()>=0
    h,q,u=moments(f/o.S);v=V/2;target=h[:,None,None]*u[:,:,None]*u[:,None,:]+G*h[:,None,None]**2/2*np.eye(2)
    ok=feasible(SETS['D2Q25'],h,u[:,0])&feasible(SETS['D2Q25'],h,u[:,1]);result={};rows=[]
    def report(name,z):
        e=np.einsum('...i,ij,ik->...jk',z/o.S,v,v)-target.reshape((len(f),)+(1,)*(z.ndim-2)+(2,2))
        result[name]=dict(diagonal_rmse=float(np.sqrt(np.mean(e[ok][...,[0,1],[0,1]]**2))),cross_rmse=float(np.sqrt(np.mean(e[ok][...,0,1]**2))),tensor_rmse=float(np.sqrt(np.mean(e[ok]**2))))
    report('M13',base);report('M14_mean',y.mean(1));report('M14_all_branches',y)
    for p in GROUP[1:]:
        a,_=o.base.evaluate(action(f,p));expect=action(base,p);z=branches(o,action(f,p));ex=action(y,p)
        mismatch=sum(sorted(map(tuple,a))!=sorted(map(tuple,b)) for a,b in zip(z,ex))
        rows.append(dict(permutation=' '.join(map(str,p)),baseline_mismatches=int(np.any(a!=expect,axis=1).sum()),channel_mismatches=mismatch,mean_error_ticks=int(np.max(abs(z.sum(1)-ex.sum(1))))))
        assert mismatch==0
    csvout(OUT/'covariance.csv',rows)
    state,error=coherent_dilation(o,f[:2],np.array([1,1j])/np.sqrt(2));assert len(state)==16
    expected=np.array([1,1j])/4
    phase_error=max(abs(a-expected[0 if k[0]==tuple(f[0]) else 1]) for k,a in state.items())
    assert phase_error<1e-12 and error<1e-12
    ans=dict(samples=len(f),feasible=int(ok.sum()),group_actions=7,baseline_mismatches=sum(r['baseline_mismatches'] for r in rows),channel_mismatches=sum(r['channel_mismatches'] for r in rows),metrics=result,mean_population_std_ticks=float(np.sqrt(np.mean(y.var(1)))),max_population_std_ticks=float(np.sqrt(y.var(1)).max()),coherence=dict(support=len(state),norm=float(sum(abs(a)**2 for a in state.values())),relative_phase_error=float(phase_error),inverse_error=error),resources=dict(base_gates=len(o.base.gates),wrapper_gates=o.wrapper_gates,hadamards=3,base_qubits=o.base.nqubits,total_qubits=o.nqubits))
    print('local',json.dumps(ans),flush=True);return ans

def experiment(name,mask,h,S,omega,steps,periodic,seeds,save=False):
    o=TwirledCollision(S,omega_num=omega);c=Solver(mask,S,periodic=periodic,oracle=o.base);f0=initialize(h,S,o.w);mass=int(f0.sum());rows=[];histories=[]
    folder=OUT/name;folder.mkdir(exist_ok=True)
    ff=f0/S;float_hist=[ff.sum(-1)]
    for _ in range(steps):
        eq,_=float_collision(ff);ff=ref_transport((1-omega/4)*ff+omega/4*eq,mask,1,periodic);float_hist.append(ff.sum(-1))
    for seed in [None]+seeds:
        rng=np.random.default_rng(seed);f=f0.copy();hist=[]
        for t in range(steps+1):
            h1,q,u=moments(f/S);hist.append(h1);ok=feasible(SETS['D2Q25'],h1,u[...,0])&feasible(SETS['D2Q25'],h1,u[...,1]);err=0.
            if periodic and t:
                x=(np.arange(c.nx)+.5-c.nx/2)*c.dx;exact,_=ritter(x,t);sel=abs(x)<28;err=float(np.sqrt(np.mean((h1[0,sel]-exact[sel])**2)))
            rows.append(dict(seed='baseline' if seed is None else seed,step=t,mass_ticks=int(f.sum()),obstacle_ticks=int(f[mask].sum()),wet=int((h1>0).sum()),infeasible=int(((h1>0)&~ok).sum()),depth_rmse_float=float(np.sqrt(np.mean((h1[~mask]-float_hist[t][~mask])**2))),ritter_rmse=err,max_transverse=float(abs(u[...,1]).max())))
            if save and seed==seeds[0]:vtk(folder/f'step_{t:03}.vti',f/S,mask,float_hist[t],c.dx)
            assert f.sum()==mass and not f[mask].any() and f.min()>=0
            if t==steps:break
            a=f[~mask];g=np.zeros(len(a),int) if seed is None else rng.integers(0,8,len(a));b,flag=o.evaluate(a,g);assert not flag.any()
            ref=[]
            for v,k in zip(a,g):
                p=GROUP[k];z,bad=collision(action(v,p),S,o.w,omega);assert not bad;ref.append(action(z,np.argsort(p)))
            assert np.array_equal(b,ref) and np.array_equal(a.sum(1),b.sum(1)) and np.array_equal(a@V,b@V)
            post=np.zeros_like(f);post[~mask]=b;f=c.transport(post);assert np.array_equal(c.transport(f,True),post) and np.array_equal(f,ref_transport(post,mask,1,periodic))
        histories.append(hist);print(name,seed,rows[-1],flush=True)
    csvout(folder/'diagnostics.csv',rows);hist=np.asarray(histories);np.savez_compressed(folder/'depth.npz',depth=hist,solid=mask,float_depth=float_hist,seeds=[-1]+seeds,dx=c.dx)
    finals=[r for r in rows if r['step']==steps];return dict(steps=steps,seeds=seeds,final=finals,ensemble_rmse_float=float(np.sqrt(np.mean((hist[1:,-1].mean(0)[~mask]-float_hist[-1][~mask])**2))),ensemble_std_depth=float(np.sqrt(np.mean(hist[1:,-1].var(0)[~mask]))),mass_error_ticks=max(abs(r['mass_ticks']-mass) for r in rows),independent_reference_agreement_steps=(len(seeds)+1)*steps,resources=dict(base_local_gates=len(o.base.gates),wrapper_local_gates=o.wrapper_gates,hadamards=3,selector_qubits=3,local_qubits=o.nqubits))

def main():
    OUT.mkdir(exist_ok=True);report={'backend':'exact sparse complex statevector for local dilation; sampled branch computational-basis gates and reversible word permutation for grid trajectories; no Qiskit/QPU','local':local()}
    mask=np.zeros((16,16),bool);mask[5:11,8:10]=True;mask[1:4,12]=True;h=np.zeros(mask.shape);h[:,:5]=1.5
    report['obstacle']=experiment('obstacle',mask,h,256,2,16,False,[14,15,16,17],True)
    mask=np.zeros((1,256),bool);h=np.zeros(mask.shape);h[:,:128]=1.5
    report['strip']=experiment('strip',mask,h,1024,4,24,True,[14,15,16],False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
