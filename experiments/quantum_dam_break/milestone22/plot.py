"""Plots from executed M22 data only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent/'results'

def main():
    with (OUT/'hydrostatic_history.csv').open() as f:rows=list(csv.DictReader(f))
    with (OUT/'dam_release.csv').open() as f:dam=list(csv.DictReader(f))
    hist=np.load(OUT/'history.npz')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for width in [.1,.5,1.]:
        for b,sty in [('False','--'),('True','-')]:
            for pert,ind in [(0.,0),(1e-7,1)]:
                r=[x for x in rows if float(x['width'])==width and x['balanced']==b and float(x['perturbation'])==pert]
                ax[0,ind].semilogy([int(x['step']) for x in r],np.maximum([float(x['max_speed']) for x in r],1e-23),sty,label=f'w={width:g}, '+('balanced' if b=='True' else 'baseline'))
    for a,title in zip(ax[0],['Exact frozen reference','Perturbed reference (10⁻⁷ lu)']):
        a.axhline(.1,color='gray',ls=':',label='speed guard');a.set(title=title,xlabel='Step',ylabel='Maximum speed (lu)');a.legend(fontsize=7,ncol=2);a.grid(alpha=.2)
    for name in ['baseline','legacy_offset','gauge_offset','balanced_gate','balanced_offset','strong_balanced']:
        r=[x for x in dam if x['case']==name];ax[1,0].plot([int(x['step']) for x in r],[float(x['max_speed']) for x in r],label=name)
        ax[1,1].plot([int(x['step']) for x in r],[float(x['com_z']) for x in r],label=name)
    ax[1,0].axhline(.1,color='gray',ls=':');ax[1,0].set(title='Dynamic dam release',xlabel='Step',ylabel='Maximum speed (lu)');ax[1,1].set(title='Liquid vertical centre of mass',xlabel='Step',ylabel='z (lu)')
    for a in ax[1]:a.legend(fontsize=7);a.grid(alpha=.2)
    fig.suptitle('M22 | Hydrostatic balance, perturbation stability and dam evolution',fontsize=15);fig.savefig(OUT/'benchmark.png',dpi=150);plt.close(fig)
    names=['baseline','balanced_gate','balanced_offset'];steps=[0,24,48,72,96]
    fig,ax=plt.subplots(len(names),len(steps),figsize=(13,5.4),layout='constrained',sharex=True,sharey=True)
    for i,name in enumerate(names):
        for j,t in enumerate(steps):
            v=hist[name][t,0,:,2,:];im=ax[i,j].imshow(v,origin='lower',vmin=0,vmax=1,aspect='equal');ax[i,j].contour(v,levels=[.5],colors='white',linewidths=.6)
            if i==0:ax[i,j].set_title(f'Step {t}')
            if j==0:ax[i,j].set_ylabel(name.replace('_',' ')+'\nz (lu)',fontsize=9)
            if i==len(names)-1:ax[i,j].set_xlabel('x (lu)')
    fig.colorbar(im,ax=ax,label='Liquid fraction',shrink=.8);fig.suptitle('M22 | High-density-ratio dam release (computed central slices)');fig.savefig(OUT/'dam_release.png',dpi=150);plt.close(fig)
    if (OUT/'stability.json').exists():
        rows=json.loads((OUT/'stability.json').read_text());data=np.load(OUT/'spectra.npz');fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for a,width in zip(ax,[.1,.5,1.]):
            r=next(x for x in rows if x['width']==width and x['ratio']>1 and x['epsilon']==1e-8);e=data[f"{width}_{r['ratio']}_{r['epsilon']}"];t=np.linspace(0,2*np.pi,256);a.plot(np.cos(t),np.sin(t),'k:',lw=1);a.scatter(e.real,e.imag,s=7,alpha=.55);a.set_title(f"w={width:g}; ρ(J)={r['spectral_radius']:.3g}");a.set(xlabel='Real eigenvalue',ylabel='Imaginary eigenvalue');a.grid(alpha=.2);a.set_aspect('equal',adjustable='datalim')
        fig.suptitle('M22 | Frozen-interface Jacobian spectra (831.67:1 density ratio)');fig.savefig(OUT/'stability.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
