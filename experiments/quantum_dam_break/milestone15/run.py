from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,csv,time
import numpy as np
from milestone15.coherent import Circuit,reference_histories,retained_output_cleanup_probability
from milestone12.reference import initialize
from milestone11.reference import V
OUT=Path(__file__).parent/'results'
def inputs(S,w):
    f=initialize(np.array([[1.5,.5]]),S,w)[0]
    rng=np.random.default_rng(15)
    g=np.array([rng.multinomial(S+S//2,rng.dirichlet(np.ones(25)*.5)),rng.multinomial(S//2,rng.dirichlet(np.ones(25)*.5))])
    return np.stack([f,g])
def main(S=16, out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();c=Circuit(S=S,omega_num=4,steps=2);f=inputs(c.oracle.S,c.w);a=np.array([1,1j])/np.sqrt(2)
    r=c.execute(f,a);h=r['history'];ref,flags=reference_histories(f,r['labels'],c.steps,c.oracle.S,c.w,c.oracle.omega_num)
    assert np.array_equal(h,ref) and np.array_equal(r['flags'],flags)
    total=h.sum(axis=(2,3));assert np.all(total==total[:,:1]);assert np.all(h>=0);assert not flags.any()
    assert r['inverse_error']<1e-12
    n=1<<(6*c.steps);unique=[len(np.unique(h[i*n:(i+1)*n,-1].reshape(n,-1),axis=0)) for i in range(2)]
    p=retained_output_cleanup_probability(h,r['amplitudes'],c.steps);probs=abs(r['amplitudes'])**2
    rows=[]
    for t in range(c.steps+1):
        for cell in range(2):
            d=h[:,t,cell].sum(-1)/c.oracle.S;q=h[:,t,cell]@V/(2*c.oracle.S)
            rows.append(dict(step=t,cell=cell,mean_depth=float(probs@d),depth_std=float(np.sqrt(probs@((d-probs@d)**2))),mean_qx=float(probs@q[:,0]),mean_qy=float(probs@q[:,1])))
    with (out/'observables.csv').open('w') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'coherent_histories.npz',history=h,flags=r['flags'],labels=r['labels'],amplitudes=r['amplitudes'],inputs=f)
    report=dict(scale=c.oracle.S,population_bits=c.w,omega=1.0,grid=[2,1],steps=c.steps,input_superposition_terms=2,selector_histories_per_input=n,nonzero_support_terms=len(h),reference_history_mismatches=int(np.count_nonzero(h!=ref)),reference_flag_mismatches=int(np.count_nonzero(r['flags']!=flags)),mass_error_ticks=int(np.max(abs(total-total[:,:1]))),overflow_flags=int(flags.sum()),negative_populations=int(np.count_nonzero(h<0)),norm=r['norm'],inverse_amplitude_error=r['inverse_error'],unique_final_outputs_per_input=unique,copy_then_inverse_selector_zero_probability=p,resources=c.resources(),elapsed_seconds=time.perf_counter()-start)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scale',type=int,default=16);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();main(a.scale,a.out)
