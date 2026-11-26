from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
p=Path(__file__).parent/'results';old=p.parent.parent/'milestone22'/'results'
with np.load(p/'history.npz') as hist, np.load(old/'history.npz') as base:
 fig,axs=plt.subplots(3,5,figsize=(14,6),layout='constrained')
 for row,(label,h) in enumerate([('M22',base['balanced_gate']),('M23',hist['bounded_gate']),('M23, zero gravity',hist['no_gravity'])]):
  for j,t in enumerate([0,24,48,72,96]):
   im=axs[row,j].imshow(h[t,0,:,2,:],origin='lower',vmin=0,vmax=1,cmap='Blues');axs[row,j].set_title(f'Step {t}' if row==0 else '');axs[row,j].set_xticks([]);axs[row,j].set_yticks([])
   if j==0:axs[row,j].set_ylabel(label)
 fig.colorbar(im,ax=axs,label='Liquid fraction α',shrink=.8);fig.suptitle('Matched dam-release trajectories | 831.67:1 material density ratio');fig.savefig(p/'dam_release.png',dpi=160);plt.close(fig)
with np.load(p/'spectra.npz') as new,np.load(old/'spectra.npz') as prior:
 fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
 for j,w in enumerate([.1,.5,1.]):
  for row,(label,vals) in enumerate([('M22',prior[f'{w}_{998/1.2}_1e-08']),('M23',new[f'{w}_1e-08'])]):
   ax=axs[row,j];ax.scatter(vals.real,vals.imag,s=8,alpha=.65);ax.add_patch(plt.Circle((0,0),1,fill=False,linestyle='--',color='gray'));ax.set_title(f'{label} | width {w:g} | ρ(J)={max(abs(vals)):.4g}');ax.set_xlabel('Real');ax.set_ylabel('Imaginary');ax.set_aspect('equal',adjustable='datalim')
 fig.suptitle('Frozen-interface stability spectra (608 populations)');fig.savefig(p/'stability.png',dpi=160);plt.close(fig)
with (p/'hydrostatic_history.csv').open() as f:rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for w in [.1,.5,1.]:
 r=[x for x in rows if float(x['width'])==w and float(x['perturbation'])==1e-5];ax.semilogy([int(x['step']) for x in r],[float(x['max_speed']) for x in r],label=f'Width {w:g}')
ax.set(xlabel='Timestep',ylabel='Maximum speed (lattice units)',title='Decay of imposed perturbation at density ratio 831.67:1');ax.legend();ax.grid(alpha=.25);fig.savefig(p/'perturbation.png',dpi=160);plt.close(fig)
