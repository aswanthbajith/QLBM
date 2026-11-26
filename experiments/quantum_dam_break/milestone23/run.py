"""Matched M22/M23 experiments; independent population-space collision/transport backend."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone23.solver import FaceDamBreak
from milestone23.screen import spectrum
from milestone22 import run as prev
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)

def save(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 # Reuse exact M22 benchmark harness; change only the corrected class and output location.
 prev.BalancedDamBreak=FaceDamBreak;prev.OUT=OUT
 hydro=[];trace=[]
 for width in [.1,.5,1.]:
  for perturb in [0.,1e-7,1e-5]:
   r,h=prev.hydro_case(width,998/1.2,True,perturb=perturb,steps=64);hydro.append(r);trace+=h
 save('hydrostatic.csv',hydro);save('hydrostatic_history.csv',trace)
 results={};hist={};rows=[];final={}
 for name,mode,g in [('bounded_reference','reference',.0005),('bounded_gate','gate',.0005),('no_gravity','reference',0.),('strong_reference','reference',.003)]:
  # The old harness writes VTK only for this name.
  key='balanced_gate' if name=='bounded_gate' else name
  r,rr,hh,f,h=prev.dam_case(key,mode,background=True,gravity=g,steps=96)
  for row in rr:row['case']=name
  results[name]=r;rows+=rr;hist[name]=hh;final[name]=(f,h)
 a,b=hist['bounded_gate'],hist['bounded_reference'];n=min(len(a),len(b));results['bounded_gate']['reference_comparison_steps']=n-1;results['bounded_gate']['reference_phase_rmse']=float(np.sqrt(np.mean((a[n-1,0]-b[n-1,0])**2)))
 if len(a)==len(b):results['bounded_gate']['final_population_max_error']=float(max(np.max(abs(x-y)) for x,y in zip(final['bounded_gate'],final['bounded_reference'])))
 save('dam_release.csv',rows);np.savez_compressed(OUT/'history.npz',**hist)
 (OUT/'report.json').write_text(json.dumps(dict(hydrostatic=hydro,dam_release=results),indent=2)+'\n')
 # Recalculate full Jacobians at two difference scales for selected model.
 spectra={};sr=[]
 for width in [.1,.5,1.]:
  for eps in [1e-8,1e-9]:
   s=FaceDamBreak(shape=(8,2,2),gravity=1e-5);r,n,e=spectrum(s,width,eps);sr.append(dict(width=width,epsilon=eps,spectral_radius=r,unstable_modes=n));spectra[f'{width}_{eps}']=e;print(sr[-1],flush=True)
 save('stability.csv',sr);np.savez_compressed(OUT/'spectra.npz',**spectra)
if __name__=='__main__':main()
