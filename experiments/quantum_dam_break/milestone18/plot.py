from pathlib import Path
import csv,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent/'results'
def load(n):return np.genfromtxt(OUT/n,delimiter=',',names=True,encoding=None)
a=load('hybrid.csv');c=load('channel.csv');s=load('shots.csv');h=np.load(OUT/'history.npz');mask=h['solid'];
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
ax[0,0].plot(a['step'][:11],a['mass'][:11],label='Link circuit');ax[0,0].plot(a['step'][:11],a['legacy_fluid_mass'][:11],label='Legacy update');ax[0,0].set(xlabel='Timestep',ylabel='Fluid population sum',title='Wall mass loss (first 10 steps)');ax[0,0].legend()
ax[0,1].plot(a['step'],a['phase_error'],label='Phase sum error');ax[0,1].plot(a['step'],a['mass_error'],label='Hydrodynamic mass error');ax[0,1].plot(a['step'],a['reference_max_error'],label='Max reference error');ax[0,1].set(xlabel='Timestep',ylabel='Absolute error',title='Corrected gate transport');ax[0,1].legend()
q=c[(c['lambda_value']==.5)&(c['initial']==.4)];ax[1,0].semilogy(q['step'],np.sqrt(q['bitflip_variance']),label='Single-qubit bit flip');ax[1,0].semilogy(q['step'],np.sqrt(q['dual_rail_variance']),label='Dual-rail damping');ax[1,0].set(xlabel='Repeated relaxation steps',ylabel='Single-shot standard deviation',title='Same mean, different readout variance');ax[1,0].legend()
q=s[(s['lambda_value']==.5)&(s['initial']==.4)];ax[1,1].loglog(q['shots'],q['bitflip_rmse'],'o-',label='Bit flip, sampled');ax[1,1].loglog(q['shots'],q['dual_rail_rmse'],'o-',label='Dual rail, sampled');ax[1,1].loglog(q['shots'],q['bitflip_theory_se'],'--',label='Bit flip, theory');ax[1,1].loglog(q['shots'],q['dual_rail_theory_se'],'--',label='Dual rail, theory');ax[1,1].set(xlabel='Shots per estimate',ylabel='RMSE',title='Finite-shot estimates after four relaxations');ax[1,1].legend()
fig.suptitle('Milestone 18 | Transport correction and open-system collision audit');fig.savefig(OUT/'benchmark.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
for ax,k in zip(axs,[0,8,32]):
    p=h['g'][k].sum(0);im=ax.imshow(np.ma.masked_where(mask[:,2,:],p[:,2,:]),vmin=-1,vmax=1,origin='lower',cmap='coolwarm');ax.set(title=f'Step {k}',xlabel='x',ylabel='z')
fig.colorbar(im,ax=axs,label='Signed phase parameter');fig.suptitle('Hybrid periodic test with solid floor and obstacle (y=2)');fig.savefig(OUT/'phase_slices.png',dpi=160);plt.close(fig)
