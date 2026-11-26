"""Plot and derive physical-admissibility diagnostics (not a density-coupled run)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
OUT=Path(__file__).parent/'results';h=np.load(OUT/'history.npz');p=h['bounded_phase'];t=np.arange(len(p));mid=p.shape[2]//2
cases=[('m19','M19 unbounded'),('bounded','M20 bounded, ideal'),('no_gravity','M20 zero gravity'),('shots_256','M20 256 shots/mode'),('shots_4096','M20 4096 shots/mode')]
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for name,label in cases:
    q=h[f'{name}_phase'];a=(q+1)/2;z,y,x=np.indices(q.shape[1:]);den=a.sum((1,2,3));ax[0,0].plot(t,(a*x).sum((1,2,3))/den,label=label);ax[0,1].plot(t,(a*z).sum((1,2,3))/den,label=label)
    if name!='no_gravity':ax[1,0].semilogy(t,np.maximum(np.sqrt(np.mean((q-h['reference_phase'])**2,axis=(1,2,3))),1e-17),label=label)
    if name in ('m19','bounded'):ax[1,1].plot(t,q.max((1,2,3)),label=label+' max');ax[1,1].plot(t,q.min((1,2,3)),label=label+' min')
ax[1,1].axhline(1,ls='--',c='black',lw=.8);ax[1,1].axhline(-1,ls='--',c='black',lw=.8)
for a,title,y in zip(ax.flat,['Horizontal phase moment','Vertical phase moment','Numerical equivalence and sampling error','Phase bounds'],['Centre x (lu)','Centre z (lu)','Phase RMSE vs bounded reference','Phase order parameter']):
    a.set(title=title,xlabel='Timestep',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=7)
fig.suptitle('M20 | Bound-preserving dam release — small density contrast, hybrid QLBM',fontsize=13);fig.savefig(OUT/'benchmark.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(3,3,figsize=(12,8),constrained_layout=True)
for a,(name,step) in zip(axs.flat,[(n,k) for n in ['m19','bounded','no_gravity'] for k in [0,48,96]]):
    q=h[f'{name}_phase'][step,:,mid,:];im=a.imshow(q,origin='lower',vmin=-1.13,vmax=1.13,cmap='coolwarm',interpolation='nearest');a.contour(q,levels=[0],colors='black',linewidths=.7);a.set_title(f'{name.replace("_"," ")} | t={step}');a.set(xlabel='x (lu)',ylabel='z (lu)')
fig.colorbar(im,ax=axs,label='Signed phase');fig.suptitle('Matched computed cross-sections | not calibrated water–air');fig.savefig(OUT/'dam_release.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);im=ax.imshow(p[0,:,mid,:],origin='lower',vmin=-1,vmax=1,cmap='coolwarm',interpolation='nearest');fig.colorbar(im,ax=ax,label='Signed phase');ax.set(xlabel='x (lu)',ylabel='z (lu)');title=ax.set_title('');
def update(k):
    im.set_data(p[k,:,mid,:]);title.set_text(f'M20 bounded hybrid dam release | t={k:03d} | small density contrast');return im,title
FuncAnimation(fig,update,frames=range(0,len(p),3),interval=140,blit=False).save(OUT/'dam_release.gif',writer=PillowWriter(fps=7));plt.close(fig)
# Illustrative density interpolation only, not an input to the hydrodynamic solver.
res={}
for name,_ in cases:
    q=h[f'{name}_phase'];rho=1.2+(998.-1.2)*(q+1)/2
    res[name]={'min_kg_m3':float(rho.min()),'max_kg_m3':float(rho.max()),'negative_density_cell_steps':int(np.count_nonzero(rho<0)),'outside_material_range_cell_steps':int(np.count_nonzero((rho<1.2-1e-7)|(rho>998+1e-7)))}
r={'warning':'Postprocessing only. Density interpolation is NOT coupled to momentum or pressure. Values are representative endpoints, not calibrated inputs.','rho_liquid_kg_m3':998.,'rho_gas_kg_m3':1.2,'ratio':998/1.2,'results':res,'final_phase_rmse_m20_vs_m19':float(np.sqrt(np.mean((p[-1]-h['m19_phase'][-1])**2)))}
(OUT/'material_admissibility.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
