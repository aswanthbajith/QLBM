"""Plot executed M12 results, with fixed scales across animation frames."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
OUT=Path(__file__).parent/'results'
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def main():
    report=json.loads((OUT/'report.json').read_text());a=np.load(OUT/'obstacle/depth.npz');d=a['depth'];mask=a['solid'];fig,ax=plt.subplots(figsize=(6,5));cm=plt.colormaps['Blues'].copy();cm.set_bad('#333333');im=ax.imshow(np.ma.masked_where(mask,d[0]),origin='lower',vmin=0,vmax=d.max(),cmap=cm);fig.colorbar(im,ax=ax,label='Depth (lu)');ax.set(xlabel='x cell',ylabel='y cell');title=ax.set_title('D2Q25 | step 0 | relaxation 0.5')
    def update(i):im.set_data(np.ma.masked_where(mask,d[i]));title.set_text(f'D2Q25 | step {i} | relaxation 0.5');return im,title
    ani=FuncAnimation(fig,update,frames=len(d),interval=220,blit=False);ani.save(OUT/'dam_break_d2q25_relaxation.gif',writer=PillowWriter(fps=5));plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,i in zip(axs,[0,8,16]):
        im=ax.imshow(np.ma.masked_where(mask,d[i]),origin='lower',vmin=0,vmax=d.max(),cmap=cm);ax.set(title=f'Step {i}',xlabel='x cell',ylabel='y cell')
    fig.colorbar(im,ax=axs,label='Depth (lu)',shrink=.85);fig.suptitle('Executed 16 × 16 D2Q25 gate-simulation benchmark | ω = 0.5');fig.savefig(OUT/'obstacle.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,label in [('baseline','M11 baseline'),('symmetric_full','M12 ω=1'),('symmetric_half','M12 ω=0.5')]:
        rows=read(OUT/name/'diagnostics.csv');t=[int(r['step']) for r in rows];axs[0,0].plot(t[1:],[float(r['ritter_rmse']) for r in rows[1:]],label=label);axs[0,1].plot(t,[float(r['max_transverse_velocity']) for r in rows],label=label)
    axs[0,0].set(title='Dry-bed analytical error',xlabel='Timestep',ylabel='Depth RMSE (lu)');axs[0,1].set(title='Spurious transverse velocity',xlabel='Timestep',ylabel='Max |uy| (lu/time)');axs[0,0].legend();axs[0,1].legend()
    rows=read(OUT/'shear.csv');
    for w in (.25,.5,.75,1.):
        sel=[r for r in rows if float(r['omega'])==w];axs[1,0].plot([int(r['step']) for r in sel],[float(r['amplitude']) for r in sel],label=f'ω={w:g}')
    axs[1,0].set(title='Executed shear-wave decay',xlabel='Timestep',ylabel='Velocity amplitude');axs[1,0].legend();v=report['shear'];w=[r['omega'] for r in v]
    for key,label in [('gate_viscosity','Gate simulation'),('float_viscosity','Floating reference'),('formal_small_k_viscosity','Formal long-wave estimate')]:axs[1,1].plot(w,[r[key] for r in v],marker='o',label=label)
    axs[1,1].set(title='Viscosity calibration: mismatch remains',xlabel='Relaxation rate ω',ylabel='Effective viscosity (lu²/time)');axs[1,1].legend();fig.savefig(OUT/'benchmark.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
