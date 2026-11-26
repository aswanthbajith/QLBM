"""Run pressure closure, force and shear benchmarks; gate-level pressure oracle."""
import csv,json,argparse
from pathlib import Path
from fractions import Fraction
import numpy as np
from physics import equilibrium,eos,force,force_kernel,expected_impulse,M,P,STRESS
from collision import collide,stream,permutation,transition,BITS,C
from arithmetic import compile_pressure,basis_apply,pressure_word,statevector_apply,compile_bed_source,apply_controlled,signed


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def shear(shots=2048,nx=8,ny=16,steps=30,omega=.7,seed=731):
    rng=np.random.default_rng(seed); y=np.arange(ny); mode=np.sin(2*np.pi*y/ny); k=2*np.pi/ny
    # local product distributions with small x drift, no y drift
    s=np.empty((shots,ny,nx),np.int16)
    for j in range(ny): s[:,j,:]=rng.choice(512,size=(shots,nx),p=equilibrium(-.6,(.15*mode[j],0)))
    mass0=M[s].sum((1,2)); mom0=P[s].sum((1,2)); p=permutation(nx,ny,np.zeros((ny,nx),bool),True); rows=[]
    for t in range(steps+1):
        # transverse momentum Fourier mode per trajectory; mass-normalized using mean mass per site
        h=M[s].mean(); coeff=2*(P[s][...,0]*mode[None,:,None]).mean((1,2))/h
        rows.append({'step':t,'shear_amplitude_lu':float(coeff.mean()),'standard_error':float(coeff.std(ddof=1)/np.sqrt(shots)),'mass_error':int(abs(M[s].sum((1,2))-mass0).max()),'momentum_error':int(abs(P[s].sum((1,2))-mom0).max())})
        if t<steps: s=stream(collide(s,omega,rng),p)
    # fit predeclared transient-excluded window; report diagnostic, not a certified viscosity
    fit=rows[4:21]; ts=np.array([r['step'] for r in fit]); a=np.array([r['shear_amplitude_lu'] for r in fit]); err=np.array([r['standard_error'] for r in fit])
    if np.any(a<=0): return rows,{'status':'fit unavailable: sign change','k':k}
    log=np.log(a); slope,inter=np.polyfit(ts,log,1); pred=inter+slope*ts
    r2=1-np.sum((log-pred)**2)/np.sum((log-log.mean())**2)
    return rows,{'status':'effective lattice-gas estimate only','k':k,'omega':omega,'fit_steps':[4,20],'nu_eff_lu':float(-slope/k**2),'log_fit_r2':float(r2),'min_amplitude_to_se':float(min(a/err)),'shots':shots,'grid':[nx,ny]}


def run(out,shots=2048):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; densities=np.linspace(.25,9.75,39)
    pressure=np.array([eos(h)[1] for h in densities]); h2=densities**2/2; g=float(h2@pressure/(h2@h2))
    for h,p in zip(densities,pressure):
        w,_,cs2=eos(h); target=g*h*h/2
        rows.append({'mass_lu':h,'kinetic_pressure_lu':p,'best_fit_hydrostatic_flux_lu':target,'relative_error':(p-target)/target,'kinetic_dp_dm_lu':cs2,'target_dp_dm_lu':g*h,'collision_stationarity_l1':abs(transition(.7)@w-w).sum()})
    write_csv(out/'pressure_eos.csv',rows)
    # body force response, uniform equilibrium states, compare exact kernel to sampled circuit-channel trajectories
    k=force_kernel(0,.4,.25); rng=np.random.default_rng(42); force_rows=[]
    for h in [.25,1,2,4,6,8,9.75]:
        w,_,_=eos(h); before=w@P; after=(k@w)@P; s=rng.choice(512,size=100000,p=w); z=force(s,0,.4,.25,rng); impulse=P[z]-P[s]
        force_rows.append({'mass_lu':h,'exact_impulse_x_lu':float(after[0]-before[0]),'effective_acceleration_lu':float((after[0]-before[0])/h),'exact_transverse_impulse_lu':float(after[1]-before[1]),'sampled_impulse_x_lu':float(impulse[:,0].mean()),'sample_standard_error':float(impulse[:,0].std(ddof=1)/np.sqrt(len(s))),'mass_error':int(abs(M[z]-M[s]).max()),'formula_error':float(abs(expected_impulse(np.arange(512),0,.4,.25)@w-(after[0]-before[0])))})
    write_csv(out/'force_response.csv',force_rows)
    sr,smeta=shear(shots=shots); write_csv(out/'shear_decay.csv',sr)
    # nonlinear arithmetic on computational-basis fixed-point registers, g=1 dimensionless
    gates=compile_pressure(); pr=[]
    for h in range(8):
        word=basis_apply(h,gates,3); p=word>>3; exact=float(Fraction(h,4)**2/2)
        pr.append({'h_word':h,'h_lu':h/4,'pressure_word':p,'quantized_pressure_lu':p/16,'exact_pressure_lu':exact,'absolute_error':abs(p/16-exact)})
    write_csv(out/'pressure_oracle.csv',pr)
    state=np.zeros(512,complex); state[:8]=1/np.sqrt(8); v=statevector_apply(state,gates,3); recovered=statevector_apply(v,gates,3)
    summary={'backend':'NumPy exact local channels and basis-controlled-X statevector; no Qiskit/QPU','pressure':{'best_fit_g_lu':g,'relative_l2_error':float(np.linalg.norm(pressure-g*h2)/np.linalg.norm(pressure)),'max_stationarity_l1':float(max(r['collision_stationarity_l1'] for r in rows)),'mass_range':[.25,9.75],'decision':'M2 equilibrium does not implement shallow-water hydrostatic closure'},'force':{'bias':.4,'rate':.25,'acceleration_min':min(r['effective_acceleration_lu'] for r in force_rows),'acceleration_max':max(r['effective_acceleration_lu'] for r in force_rows),'max_mass_error':max(r['mass_error'] for r in force_rows),'decision':'directional forcing, not calibrated gravity'},'shear':smeta,'oracle':{'data_qubits':9,'pattern_controlled_X_gates':len(gates),'max_pressure_error_lu':max(r['absolute_error'] for r in pr),'statevector_norm':float(np.linalg.norm(v)),'uncompute_max_error':float(abs(recovered-state).max()),'decision':'reversible nonlinear pressure primitive; no fluid flux update yet'}}
    bg=compile_bed_source(); br=[]
    for h in range(8):
        q=signed(apply_controlled(h,bg)>>3,5); br.append({'h_lu':h/4,'q_before_lu':0,'q_after_lu':q/16,'exact_source_lu':-h/16,'absolute_error':abs(q/16+h/16)})
    write_csv(out/'bed_source_oracle.csv',br)
    summary['bed_source_oracle']={'data_qubits':8,'pattern_controlled_X_gates':len(bg),'acceleration_times_dt_lu':-.25,'max_error_from_zero_momentum_lu':max(r['absolute_error'] for r in br),'all_basis_roundtrip':all(apply_controlled(apply_controlled(w,bg),bg[::-1])==w for w in range(256)),'decision':'reversible source primitive with modular signed momentum; not a well-balanced fluid update'}
    (out/'report.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(Path(__file__).parent/'results')); p.add_argument('--shots',type=int,default=2048); run(**vars(p.parse_args()))
