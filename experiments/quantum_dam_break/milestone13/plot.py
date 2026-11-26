"""Generate plots and animation from executed M13 result arrays."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone10.kinetic import ritter
OUT=Path(__file__).parent/'results'

def main():
    r=json.loads((OUT/'report.json').read_text());fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');a=ax[0,0];x=np.arange(2)
    for i,key in enumerate(('M12','M13')):a.bar(x+(i-.5)*.35,[r['local'][key]['diagonal_rmse'],r['local'][key]['cross_rmse']],.35,label=key)
    a.set_xticks(x,['Diagonal','Cross']);a.set_ylabel('Moment RMSE (lu)');a.set_title('Feasible local equilibrium states');a.legend();a.set_yscale('log')
    a=ax[0,1]
    for name in ('M12','M13'):
        z=np.load(OUT/f'strip_{name}_4'/'depth.npz');d=z['depth'][-1,0];x=(np.arange(len(d))+.5-len(d)/2)*.5;sel=abs(x)<28;a.plot(x[sel],d[sel],label=name)
    h,_=ritter(x,24);a.plot(x[sel],h[sel],'k--',label='Ritter');a.set_xlabel('Position (lu)');a.set_ylabel('Depth (lu)');a.set_title('Dry-bed strip, t=24, omega=1');a.legend()
    for a,n in zip(ax[1],(32,64)):
        for name in ('M12','M13'):
            q=[v for v in r['shear'] if v['model']==name and v['n']==n];a.plot([v['omega'] for v in q],[v['gate_viscosity'] for v in q],'-o',label=name)
        a.plot([v['omega'] for v in q],[v['float_viscosity'] for v in q],'--',label='Float reference');a.plot([v['omega'] for v in q],[v['formal_small_k_viscosity'] for v in q],':',label='Formal estimate');a.set(xlabel='Relaxation rate',ylabel='Fitted viscosity (lu)',title=f'Shear decay: wavelength {n*.5:g} lu');a.legend(fontsize=8)
    fig.suptitle('Milestone 13 | pressure repair and viscosity benchmarks');fig.savefig(OUT/'benchmark.png',dpi=160);plt.close(fig)
    z=np.load(OUT/'obstacle_M13'/'depth.npz');d=z['depth'];mask=z['solid'];cmap=plt.get_cmap('viridis').copy();cmap.set_bad('black');fig,ax=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for a,t in zip(ax,(0,8,16)):
        im=a.imshow(np.ma.array(d[t],mask=mask),origin='lower',vmin=0,vmax=1.5,cmap=cmap,extent=(0,8,0,8));a.set(title=f'Step {t}',xlabel='x (lu)',ylabel='y (lu)')
    fig.colorbar(im,ax=ax,label='Depth (lu)');fig.suptitle('Executed D2Q25 pressure-repaired obstacle release');fig.savefig(OUT/'obstacle.png',dpi=160);plt.close(fig)
    fig,a=plt.subplots(figsize=(5,4.5));im=a.imshow(np.ma.array(d[0],mask=mask),origin='lower',vmin=0,vmax=1.5,cmap=cmap,extent=(0,8,0,8));fig.colorbar(im,ax=a,label='Depth (lu)');a.set(xlabel='x (lu)',ylabel='y (lu)')
    def update(t):im.set_data(np.ma.array(d[t],mask=mask));a.set_title(f'M13 D2Q25 | Step {t:02d}');return im,
    anim=FuncAnimation(fig,update,frames=len(d),interval=200);anim.save(OUT/'dam_break.gif',writer=PillowWriter(fps=5));plt.close(fig)
if __name__=='__main__':main()
