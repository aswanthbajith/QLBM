"""Plot actual archived gate trajectories; no synthetic flow values."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from milestone9.run import bound
from milestone8.reference import G
P=Path(__file__).parent/'results'
def main():
    z=np.load(P/'dry_bed/history.npz');s=int(z['scale']);a=z['populations_ticks'];b=z['m8_ticks'];solid=z['solid'];h=a.sum(-1)/s;g=b.sum(-1)/s
    fig,ax=plt.subplots(2,2,figsize=(10,8),layout='constrained');cmap=plt.colormaps['viridis'].copy();cmap.set_bad('gray')
    for q,field,title in zip(ax.flat[:3],[h[0],g[-1],h[-1]],['Initial reservoir','M8: reject invalid collision','M9: limited collision']):
        im=q.imshow(np.ma.masked_where(solid,field),origin='lower',vmin=0,vmax=1.5,cmap=cmap);q.set_title(title);q.set_xlabel('x (lu)');q.set_ylabel('y (lu)')
    fig.colorbar(im,ax=list(ax.flat[:3]),label='Depth (lu)',shrink=.8)
    bad,_=bound(a[-1],s);ax[1,1].imshow(np.ma.masked_where(solid,bad),origin='lower',vmin=0,vmax=1,cmap='Reds');ax[1,1].set_title(f'Pressure moment infeasible: {bad.sum()} cells');ax[1,1].set_xlabel('x (lu)');ax[1,1].set_ylabel('y (lu)');fig.suptitle('M9 dry-bed experiment — not physically validated');fig.savefig(P/'obstacle_comparison.png',dpi=160);plt.close(fig)
    z=np.load(P/'ritter_1d/history.npz');x=np.arange(128)+.5-64;sl=abs(x)<32;j=24;xi=x/j;c=np.sqrt(G*1.5);exact=np.where(xi<-c,1.5,np.where(xi>2*c,0,(2*c-xi)**2/(9*G)));ha=z['populations_ticks'][j].sum(-1)[0]/s;hb=z['m8_ticks'][j].sum(-1)[0]/s
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.plot(x[sl],exact[sl],label='Ritter analytical');ax.plot(x[sl],hb[sl],label='M8',ls='--');ax.plot(x[sl],ha[sl],label='M9 (overlaps M8)',ls=':',lw=2.5);ax.set(xlabel='Distance from dam (lu)',ylabel='Depth (lu)',title='Dry-front benchmark at step 24 — no accuracy improvement');ax.grid(alpha=.25);ax.legend();fig.savefig(P/'ritter_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.6,5),layout='constrained');im=ax.imshow(np.ma.masked_where(solid,h[0]),origin='lower',vmin=0,vmax=1.5,cmap=cmap);fig.colorbar(im,ax=ax,label='Depth (lu)');ax.set_xlabel('x (lu)');ax.set_ylabel('y (lu)')
    def update(j):
        im.set_data(np.ma.masked_where(solid,h[j]));ax.set_title(f'M9 experimental dry bed | step {j:02d}');return im,
    FuncAnimation(fig,update,frames=len(h),interval=180).save(P/'dry_bed.gif',writer=PillowWriter(fps=5),dpi=95);plt.close(fig)
if __name__=='__main__':main()
