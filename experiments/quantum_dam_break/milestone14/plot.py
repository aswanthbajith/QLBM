"""Plot measured covariance and trajectory outcomes; no synthetic animation."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
OUT=Path(__file__).parent/'results'

def main():
    r=json.loads((OUT/'report.json').read_text());cov=list(csv.DictReader((OUT/'covariance.csv').open()));d=np.load(OUT/'obstacle/depth.npz');h=d['depth'];solid=d['solid'];ref=d['float_depth'];s=np.load(OUT/'strip/depth.npz');strip=s['depth'];x=(np.arange(strip.shape[-1])+.5-strip.shape[-1]/2)*s['dx']
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    a=ax[0,0];a.bar(np.arange(1,8),[int(c['baseline_mismatches']) for c in cov],label='M13');a.scatter(np.arange(1,8),[int(c['channel_mismatches']) for c in cov],label='M14 channel',marker='x',s=70);a.set(title='D4 covariance: exact distribution test',xlabel='Nonidentity group action',ylabel='Mismatched inputs (of 1024)');a.legend()
    a=ax[0,1];names=['M13','M14_mean','M14_all_branches'];p=np.arange(3);a.bar(p-.17,[r['local']['metrics'][n]['diagonal_rmse'] for n in names],.34,label='Diagonal');a.bar(p+.17,[r['local']['metrics'][n]['cross_rmse'] for n in names],.34,label='Cross');a.set_xticks(p,['M13','M14 mean','M14 branches']);a.set(title='Pressure error on feasible local states',ylabel='RMSE (lu)');a.legend()
    a=ax[1,0];finals=r['obstacle']['final'];a.bar(range(len(finals)),[f['depth_rmse_float'] for f in finals]);a.axhline(r['obstacle']['ensemble_rmse_float'],linestyle='--',label='Sample ensemble mean field');a.set_xticks(range(len(finals)),[str(f['seed']) for f in finals]);a.set(title='2D obstacle: depth error vs float',ylabel='RMSE (lu)',xlabel='Trajectory seed');a.legend()
    a=ax[1,1];a.plot(x,strip[0,-1,0],label='M13');a.plot(x,strip[1:,-1,0].mean(0),label='M14 sampled mean');a.plot(x,s['float_depth'][-1,0],'--',label='Float kinetic');a.set(xlim=(-28,28),title='1D dry-bed profile at t=24',xlabel='x (lu)',ylabel='Depth (lu)');a.legend();fig.suptitle('Milestone 14 | Symmetry-averaged quantum collision',fontsize=15);fig.savefig(OUT/'benchmark.png',dpi=150);plt.close(fig)
    frames=[]
    for t in range(h.shape[1]):
        fig,ax=plt.subplots(1,2,figsize=(8,4),layout='constrained')
        for a,v,title in zip(ax,[h[0,t],h[1,t]],['M13 baseline','M14 sampled trajectory (seed 14)']):
            im=a.imshow(np.ma.masked_where(solid,v),origin='lower',vmin=0,vmax=1.5,cmap='viridis');a.set_title(title,fontsize=10);a.set_xlabel('x cell');a.set_ylabel('y cell')
        fig.colorbar(im,ax=ax,label='Depth (lu)',shrink=.75);fig.suptitle(f'Gate-executed branches | step {t:02d}');buf=BytesIO();fig.savefig(buf,format='png',dpi=100);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB'))
    frames[0].save(OUT/'comparison.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)
if __name__=='__main__':main()
