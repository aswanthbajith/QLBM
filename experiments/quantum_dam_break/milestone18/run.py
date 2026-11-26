"""Reproducible source-audit benchmarks; run from quantum_dam_break directory."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,csv, numpy as np
from milestone18.transport import *
from milestone18.channel import *
from milestone18.hybrid import init,step
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
def save_csv(name,rows):
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
    rng=np.random.default_rng(20261125);solid=np.zeros((4,4,4),bool);solid[0]=True;solid[1,1,1]=True
    summary={};trans=[];wallrows=[]
    for label,c in [('D3Q19',C19),('D3Q7',C7)]:
        t=LinkTransport(solid,c);trans.append(t);f=np.zeros((len(c),)+solid.shape);f[6,1,2,2]=1
        old=legacy_stream(f,solid,c);new=t.stream(f);v=rng.normal(size=t.dim)+1j*rng.normal(size=t.dim);v[t.invalid]=0;v/=np.linalg.norm(v);a=v.copy();b=v.copy()
        for _ in range(16):a=t.apply(a);b=b[np.argsort(t.perm)]
        err=float(np.max(abs(a-b)));inv=a.copy()
        for _ in range(16):inv=t.apply(inv,True)
        wallrows.append(dict(lattice=label,qubits=t.qubits,fluid_cells=int((~solid).sum()),wall_links=len(t.wall),factorized_mcx=len(t.gates),direct_mcx=len(t.direct_gates),legacy_pulse_mass=float(old.sum()),new_pulse_mass=float(new.sum()),new_reflected_population=float(new[5,1,2,2]),statevector_16step_error=err,inverse_error=float(np.max(abs(inv-v))),solid_padding_probability=float(np.sum(abs(a[t.invalid])**2))))
        assert err==0 and np.array_equal(inv,v)
    save_csv('transport.csv',wallrows);summary['transport']=wallrows
    f,g=init(solid);mass=float(f.sum());phase=float(g.sum());ff,gg=f.copy(),g.copy();fr,gr=f.copy(),g.copy();fl,gl=f.copy(),g.copy();rows=[];history=[]
    for k in range(33):
        err=max(np.max(abs(ff-fr)),np.max(abs(gg-gr)));row=dict(step=k,mass=float(ff.sum()),phase=float(gg.sum()),mass_error=float(abs(ff.sum()-mass)),phase_error=float(abs(gg.sum()-phase)),legacy_fluid_mass=float(fl[:,~solid].sum()),legacy_fluid_phase=float(gl[:,~solid].sum()),legacy_solid_population=float(np.sum(abs(fl[:,solid]))+np.sum(abs(gl[:,solid]))),solid_population=float(np.sum(abs(ff[:,solid]))+np.sum(abs(gg[:,solid]))),reference_max_error=float(err),max_speed=float(np.max(np.linalg.norm(np.einsum('id,izyx->dzyx',C19,ff)/np.maximum(ff.sum(0),1e-10),axis=0))))
        rows.append(row);history.append((ff.copy(),gg.copy(),fl.copy(),gl.copy()));assert err<1e-12 and row['mass_error']<1e-11 and row['phase_error']<1e-11 and row['solid_population']==0
        if k<32:ff,gg=step(ff,gg,solid,'gate',trans);fr,gr=step(fr,gr,solid,'reference');fl,gl=step(fl,gl,solid,'legacy')
    save_csv('hybrid.csv',rows);np.savez_compressed(OUT/'history.npz',f=np.array([h[0] for h in history]),g=np.array([h[1] for h in history]),legacy_f=np.array([h[2] for h in history]),legacy_g=np.array([h[3] for h in history]),solid=solid)
    summary['hybrid']={'steps':32,'initial_mass':mass,'initial_phase':phase,'final':rows[-1],'max_reference_error':max(r['reference_max_error'] for r in rows),'max_mass_error':max(r['mass_error'] for r in rows),'max_phase_error':max(r['phase_error'] for r in rows)}
    # Compare branch bit-flip channel with the paper's actual two-rail amplitude damping.
    # Fixed common scale, no adaptive renormalization, no intermediate measurement.
    ch=[];worst=0.;traceerr=0.
    for lam in [-.9,-.25,0.,.5,.9]:
        for d in [-.4,-.05,0,.05,.4]:
            r=rail(d,.5);s=dual_rail(d,.5)
            for k in range(1,17):
                r=apply(r,lam);s=apply_damping(s,lam);exact=d*lam**k;v1=decode(r,.5);v2=dual_decode(s,.5);worst=max(worst,abs(v1-exact),abs(v2-exact));traceerr=max(traceerr,abs(np.trace(r)-1),abs(np.trace(s)-1));ch.append(dict(lambda_value=lam,initial=d,step=k,expected=exact,bitflip=v1,dual_rail=v2,bitflip_variance=.25-v1*v1,dual_rail_variance=max(0,dual_variance(s,.5))))
    save_csv('channel.csv',ch);shotrows=[]
    for lam,d,k in [(-.25,.4,1),(.5,.4,4),(.9,.05,16),(.5,0,4)]:
        r=rail(d,.5);s=dual_rail(d,.5)
        for _ in range(k):r=apply(r,lam);s=apply_damping(s,lam)
        exact=d*lam**k
        for n in [100,1000,10000]:
            counts=rng.binomial(n,r[0,0].real,256);v1=.5*(2*counts/n-1);cnt=rng.multinomial(n,np.diag(s).real/np.trace(s).real,256);v2=.5*(cnt[:,1]-cnt[:,2])/n
            shotrows.append(dict(lambda_value=lam,initial=d,step=k,shots=n,repeats=256,expected=exact,bitflip_rmse=float(np.sqrt(np.mean((v1-exact)**2))),dual_rail_rmse=float(np.sqrt(np.mean((v2-exact)**2))),bitflip_theory_se=float(np.sqrt((.25-exact**2)/n)),dual_rail_theory_se=float(np.sqrt(max(0,dual_variance(s,.5))/n))))
    save_csv('shots.csv',shotrows);summary['channel']={'updates':len(ch),'max_decoded_error':float(worst),'max_trace_error':float(traceerr),'shot_experiments':len(shotrows),'sample_shots':shotrows[4]}
    assert worst<1e-13 and traceerr<1e-13
    (OUT/'report.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
