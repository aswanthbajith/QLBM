"""Reproduce signed phase-field gate experiments and independent comparisons."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,platform,time
import numpy as np
from milestone16.phase_field import PhaseStepper,CoherentGrid,DT
from milestone16.reference import integer_step,float_step,energy,chemical

OUT=Path(__file__).parent/'results'

def experiment(name,S,initial,steps=256):
    n=len(initial); solver=PhaseStepper(S);p=np.rint(initial*S).astype(np.int64);ref=p/S;h=[p.copy()];r=[ref.copy()];rows=[];start=time.perf_counter()
    x=np.arange(n);wave=np.cos(2*np.pi*6*x/n)
    for t in range(steps+1):
        field=p/S
        rows.append(dict(step=t,time=t*DT,mass_ticks=int(p.sum()),energy=energy(field),reference_energy=energy(ref),rmse=float(np.sqrt(np.mean((field-ref)**2))),cosine_amplitude=float(2*field@wave/n),min_phi=float(field.min()),max_phi=float(field.max())))
        if t==steps:break
        q=solver.step(p); expected=integer_step(p,S)
        if not np.array_equal(q,expected):raise AssertionError('integer reference mismatch')
        p=q;ref=float_step(ref);h.append(p.copy());r.append(ref.copy())
    h=np.asarray(h);r=np.asarray(r);path=OUT/f'{name}_{S}';path.mkdir(exist_ok=True)
    with (path/'diagnostics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    np.savez_compressed(path/'history.npz',ticks=h,scale=S,reference=r,time=np.arange(steps+1)*DT)
    energies=np.array([i['energy'] for i in rows]);err=np.array([i['rmse'] for i in rows]);delta=np.diff(energies)
    result=dict(name=name,scale=S,grid=n,steps=steps,time=steps*DT,initial_phase_mass_ticks=int(h[0].sum()),max_mass_error_ticks=int(np.max(np.abs(h.sum(axis=1)-h[0].sum()))),reference_match_steps=steps,initial_energy=float(energies[0]),final_energy=float(energies[-1]),energy_increase_steps=int(np.sum(delta>1e-12)),max_energy_increase=float(max(0,delta.max())),final_rmse=float(err[-1]),max_rmse=float(err.max()),initial_amplitude=rows[0]['cosine_amplitude'],final_amplitude=rows[-1]['cosine_amplitude'],min_phi=float(h.min()/S),max_phi=float(h.max()/S),changed_cells_final=int(np.sum(h[-1]!=h[0])),resources=solver.resources(n),seconds=time.perf_counter()-start)
    print(json.dumps(result),flush=True);return result

def main():
    OUT.mkdir(exist_ok=True);n=64;x=np.arange(n)
    # A small mode inside the spinodal region; nonlinear saturation is not tested by linear theory.
    spinodal=.05*np.cos(2*np.pi*6*x/n)
    # Two periodic diffuse interfaces, deliberately sharper than equilibrium.
    interface=np.tanh(12*np.sin(2*np.pi*(x+.5)/n))
    results=[experiment(name,S,f) for name,f in [('spinodal',spinodal),('interface',interface)] for S in (256,4096,65536)]
    s=4*np.sin(np.pi*6/n)**2;linear_gain=(1+DT*s*(1-s))**256
    # Exact sparse complex-state verification on a coherent four-cell two-step circuit.
    S=4096;co=CoherentGrid(S,4,2);states=np.array([[S//2,0,-S//2,0],[S//4,-S//4,S//2,-S//2]])
    a=np.array([1,1j],complex)/np.sqrt(2);hist,amps=co.execute(states,a,copy_final=True)
    for i,p in enumerate(states):
        for t in range(2):p=integer_step(p,S);assert np.array_equal(p,hist[i,t+1])
    np.savez_compressed(OUT/'coherent_history.npz',history=hist,amplitudes=amps,scale=S)
    report=dict(model='periodic 1D finite-difference Cahn-Hilliard, u=0; signed fixed point',A=1,kappa=1,mobility=1,dx=1,dt=DT,linear_spinodal_gain=float(linear_gain),experiments=results,coherent=dict(scale=S,cells=4,steps=2,qubits=co.nqubits,gates=sum(map(len,co.schedule)),nonzero_basis_states=len(amps),norm=float(np.vdot(amps,amps).real),phase_ratio=[float((amps[1]/amps[0]).real),float((amps[1]/amps[0]).imag)],reference_match=True,clean_edges_and_scratch=True,inverse_exact=True,copy_then_inverse_verified=True,copy_qubits=co.n*co.w,copy_cnot_gates=co.n*co.w,copy_total_gates=2*sum(map(len,co.schedule))+co.n*co.w,intermediate_measurements=0),environment=dict(python=platform.python_version(),numpy=np.__version__))
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
