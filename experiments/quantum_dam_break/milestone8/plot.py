"""Plot computed LBM data only; dry-bed run is explicitly a failed validation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
OUT=Path(__file__).parent/'results'

def run():
    a=np.load(OUT/'wet_bed/history.npz');b=np.load(OUT/'dry_bed_stress/history.npz');p=np.genfromtxt(OUT/'pressure.csv',delimiter=',',names=True);s=np.genfromtxt(OUT/'shear_decay.csv',delimiter=',',names=True);mask=a['solid'];h=a['populations'].sum(-1);d=b['populations'].sum(-1)
    fig,ax=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for aa,data,title,vmin,vmax in [(ax[0,0],h[0],'Wet-bed release: initial',1.2,1.5),(ax[0,1],h[-1],'Wet-bed release: step 32',1.2,1.5),(ax[0,2],d[-1],'Dry-bed stress: step 24 (not validated)',0,1.5)]:
        cm=plt.colormaps['viridis'].copy();cm.set_bad('#333333');im=aa.imshow(np.ma.array(data,mask=mask),origin='lower',vmin=vmin,vmax=vmax,cmap=cm);aa.set_title(title,fontsize=11);aa.set_xlabel('x cell');aa.set_ylabel('y cell');fig.colorbar(im,ax=aa,label='Depth (lattice units)',shrink=.8)
    ax[1,0].plot(p['depth'],p['target'],label='Hydrostatic target');ax[1,0].plot(p['depth'],p['xx'],'o',ms=3,label='Decoded equilibrium');ax[1,0].set(xlabel='Depth (lu)',ylabel='Momentum flux (lu)',title='Equilibrium pressure law');ax[1,0].legend(fontsize=8)
    for om in (.5,1.,1.5):
        z=s[s['omega']==om];l,=ax[1,1].plot(z['step'],z['gate_amplitude']/z['gate_amplitude'][0],label=f'ω={om:g} gate');ax[1,1].plot(z['step'],z['theory_amplitude']/z['theory_amplitude'][0],'--',color=l.get_color(),alpha=.65)
    ax[1,1].set(xlabel='Step',ylabel='Normalized shear amplitude',title='Viscosity benchmark (dashed: theory)');ax[1,1].legend(fontsize=8)
    for name,label in [('wet_bed','Wet bed'),('dry_bed_stress','Dry bed')]:
        z=np.genfromtxt(OUT/name/'diagnostics.csv',delimiter=',',names=True);ax[1,2].plot(z['step'],z['rejected_collisions_last_step']/241,label=label)
    ax[1,2].set(xlabel='Step',ylabel='Fraction of active cells',title='Rejected collision fraction');ax[1,2].legend();fig.suptitle('Milestone 8 | Reversible D2Q9 shallow-water LBM',fontsize=16);fig.savefig(OUT/'benchmark.png',dpi=160);plt.close(fig)
    for dat,label,lo,hi,name in [(h,'Wet-bed release — fixed-point D2Q9',1.2,1.5,'wet_bed.gif'),(d,'Dry-bed stress — rejected collisions; not validated',0,1.5,'dry_bed_stress.gif')]:
        fig,ax=plt.subplots(figsize=(5.6,4.5),layout='constrained');cm=plt.colormaps['viridis'].copy();cm.set_bad('#333333');im=ax.imshow(np.ma.array(dat[0],mask=mask),origin='lower',vmin=lo,vmax=hi,cmap=cm);fig.colorbar(im,ax=ax,label='Depth (lattice units)');title=ax.set_title(f'{label}\nStep 0',fontsize=11);ax.set(xlabel='x cell',ylabel='y cell')
        def update(i):im.set_data(np.ma.array(dat[i],mask=mask));title.set_text(f'{label}\nStep {i}');return im,title
        FuncAnimation(fig,update,frames=len(dat),interval=180).save(OUT/name,writer=PillowWriter(fps=6));plt.close(fig)
if __name__=='__main__':run()
