from pathlib import Path
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
OUT=Path(__file__).parent/'results'
h=np.load(OUT/'history.npz');p=h['quantum_exact_phase'];v=h['quantum_exact_velocity'];t=np.arange(len(p));mid=p.shape[2]//2
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for name,label in [('quantum_exact','Quantum circuit (ideal)'),('no_gravity','Zero gravity'),('shots_256','256 shots/mode'),('shots_4096','4096 shots/mode')]:
    q=h[f'{name}_phase'];a=(q+1)/2;z,y,x=np.indices(q.shape[1:]);den=a.sum((1,2,3))
    ax[0,0].plot(t,(a*x).sum((1,2,3))/den,label=label);ax[0,1].plot(t,(a*z).sum((1,2,3))/den,label=label)
    if name!='no_gravity':ax[1,0].semilogy(t,np.maximum(np.sqrt(np.mean((q-h['reference_phase'])**2,axis=(1,2,3))),1e-17),label=label)
ax[1,1].plot(t,p.max((1,2,3)),label='Maximum');ax[1,1].plot(t,p.min((1,2,3)),label='Minimum');ax[1,1].axhline(1,ls='--',c='black',lw=.8);ax[1,1].axhline(-1,ls='--',c='black',lw=.8)
for a,title,y in zip(ax.flat,['Horizontal phase moment','Vertical phase moment','Numerical equivalence and shot error','Phase bounds (failed)'],['Formal centre x (lu)','Formal centre z (lu)','Phase RMSE vs matched reference','Phase order parameter']):
    a.set(title=title,xlabel='Timestep',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=8)
fig.suptitle('M19 | Two-phase dam-release integration — small density contrast, hybrid QLBM',fontsize=13);fig.savefig(OUT/'benchmark.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(13,6),constrained_layout=True)
for a,(name,step) in zip(axs.flat,[(n,t) for n in ['quantum_exact','no_gravity'] for t in [0,48,96]]):
    q=h[f'{name}_phase'][step,:,mid,:];im=a.imshow(q,origin='lower',vmin=-1.15,vmax=1.15,cmap='coolwarm',interpolation='nearest');a.contour(q,levels=[0],colors='black',linewidths=.7);a.set_title(f'{name.replace("_"," ")} | t={step}');a.set(xlabel='x (lu)',ylabel='z (lu)')
fig.colorbar(im,ax=axs,label='Signed phase');fig.suptitle('Computed central cross-sections | not a calibrated water–air model');fig.savefig(OUT/'dam_release.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);im=ax.imshow(p[0,:,mid,:],origin='lower',vmin=-1.15,vmax=1.15,cmap='coolwarm',interpolation='nearest');fig.colorbar(im,ax=ax,label='Signed phase');ax.set(xlabel='x (lu)',ylabel='z (lu)');title=ax.set_title('');
def update(k):
    im.set_data(p[k,:,mid,:]);title.set_text(f'Hybrid two-phase dam-release | t={k:03d} | small density contrast');return im,title
FuncAnimation(fig,update,frames=range(0,len(p),2),interval=130,blit=False).save(OUT/'dam_release.gif',writer=PillowWriter(fps=8));plt.close(fig)
