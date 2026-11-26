"""Generate plots directly from the executed gate simulation history."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path(__file__).parent/'results'
fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for S in (256,4096,65536):
    d=np.load(OUT/f'spinodal_{S}'/'history.npz');h=d['ticks']/S;t=d['time'];mode=np.cos(2*np.pi*6*np.arange(h.shape[1])/h.shape[1]);a=2*h@mode/h.shape[1]
    axs[0,0].plot(t,a,label=f'1/{S}')
    err=np.sqrt(np.mean((h-d['reference'])**2,axis=1));axs[0,1].plot(t,err,label=f'1/{S}')
axs[0,0].plot(t,2*d['reference']@mode/h.shape[1],'k--',label='float reference');axs[0,0].set(title='Spinodal mode: coarse precision pins growth',xlabel='Nondimensional time',ylabel='Mode amplitude');axs[0,0].legend()
axs[0,1].set(title='Fixed-point error vs same discretization',xlabel='Nondimensional time',ylabel='Phase RMSE');axs[0,1].legend()
d=np.load(OUT/'interface_65536'/'history.npz');h=d['ticks']/65536;axs[1,0].plot(h[0],label='Initial');axs[1,0].plot(h[-1],label='Gate-simulated final');axs[1,0].plot(d['reference'][-1],'--',label='Float reference');axs[1,0].set(title='1D diffuse interface relaxation',xlabel='Cell',ylabel='Signed phase');axs[1,0].legend()
d=np.load(OUT/'interface_3d'/'history.npz');h=d['ticks']/65536;energy=np.genfromtxt(OUT/'interface_3d'/'diagnostics.csv',delimiter=',',names=True);axs[1,1].plot(energy['time'],energy['energy'],label='Gate simulation');axs[1,1].plot(energy['time'],energy['reference_energy'],'--',label='Float reference');axs[1,1].set(title='3D droplet: discrete free energy',xlabel='Nondimensional time',ylabel='Free energy');axs[1,1].legend();fig.suptitle('Milestone 16 | Conservative phase-field evolution (not coupled hydrodynamics)',fontsize=13);fig.savefig(OUT/'benchmark.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(11,3.6),constrained_layout=True);z=h.shape[1]//2
for ax,p,title in zip(axs,[h[0,z],h[-1,z],d['reference'][-1,z]],['Initial field','Gate-simulated final','Independent reference']):
    im=ax.imshow(p,origin='lower',vmin=-1.1,vmax=1.1,cmap='coolwarm');ax.set_title(title);ax.set(xlabel='x cell',ylabel='y cell')
fig.colorbar(im,ax=axs,label='Signed order parameter $\\phi$');fig.suptitle('3D periodic interface benchmark | central z slice');fig.savefig(OUT/'droplet_slices.png',dpi=160);plt.close(fig)
