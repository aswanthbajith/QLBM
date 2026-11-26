"""Plots and GIF from previously executed gate-simulation histories only."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone10.kinetic import ritter
OUT=Path(__file__).parent/'results'
def main():
    a=np.load(OUT/'obstacle/history.npz');h=a['populations'].sum(-1);rh=a['reference_depth'];solid=a['solid'];dx=float(a['dx']);r=json.loads((OUT/'report.json').read_text());extent=(0,h.shape[2]*dx,0,h.shape[1]*dx)
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained');vmax=max(h.max(),rh.max());im=None
    for ax,j in zip(axs[0],(0,8,24)):
        im=ax.imshow(np.ma.array(h[j],mask=solid),origin='lower',extent=extent,vmin=0,vmax=vmax,cmap='Blues');ax.set_title(f'Gate-simulated depth · t={j}');ax.set_xlabel('x (lu)');ax.set_ylabel('y (lu)')
    fig.colorbar(im,ax=axs[0],label='Depth (lu)',shrink=.75)
    ax=axs[1,0];im=ax.imshow(np.ma.array(h[-1]-rh[-1],mask=solid),origin='lower',extent=extent,cmap='coolwarm',vmin=-.2,vmax=.2);ax.set_title('Final depth error vs float');ax.set_xlabel('x (lu)');fig.colorbar(im,ax=ax,shrink=.75)
    ax=axs[1,1];ax.plot(np.arange(len(h)),h.sum((1,2))*dx**2,label='Integrated depth');ax.set_ylim(29.8,30.2);ax.set_xlabel('Timestep');ax.set_title('Exact volume conservation');ax.legend(fontsize=8)
    ax=axs[1,2];ax.plot(np.arange(len(h)),(h>0).sum((1,2)),label='Wet cells');ax.axhline((~solid).sum(),linestyle='--',label='Available cells');ax.set_xlabel('Timestep');ax.set_title('Numerical wetting');ax.legend(fontsize=8)
    fig.suptitle('D2Q25 reversible collision–streaming · 16 × 16 · S=256');fig.savefig(OUT/'benchmark_2d.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained');im=ax.imshow(np.ma.array(h[0],mask=solid),origin='lower',extent=extent,vmin=0,vmax=vmax,cmap='Blues');fig.colorbar(im,ax=ax,label='Depth (lu)');ax.set_xlabel('x (lu)');ax.set_ylabel('y (lu)');frames=[]
    for j in range(len(h)):
        im.set_data(np.ma.array(h[j],mask=solid));ax.set_title(f'D2Q25 experimental dry release · t={j}');fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    frames[0].save(OUT/'dam_break_d2q25.gif',save_all=True,append_images=frames[1:],duration=170,loop=0);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');ax=axs[0];x=(np.arange(2048)+.5-1024)*.125;ex,_=ritter(x,24);sl=abs(x)<28;ax.plot(x[sl],ex[sl],label='Ritter analytical',linewidth=2)
    for s in (64,256,1024):
        data=np.load(OUT/f'strip_{s}/history.npz');hh=data['populations'][-1,0].sum(-1);ax.plot(x[sl],hh[sl],label=f'Gate simulation 1/{s}',alpha=.8)
    ax.plot(x[sl],data['reference_depth'][-1,0,sl],'--',label='Floating D2Q25');ax.set(xlabel='x (lu)',ylabel='Depth (lu)',title='Dry-bed benchmark at t=24');ax.legend(fontsize=8)
    ax=axs[1];sc=[v['scale'] for v in r['strip']];ax.loglog(sc,[v['final']['ritter_rmse'] for v in r['strip']],'o-',label='vs Ritter');ax.loglog(sc,[v['final']['depth_rmse_vs_float'] for v in r['strip']],'s-',label='vs float (full strip)');ax.axhline(r['strip'][0]['final']['float_ritter_rmse'],linestyle='--',label='Float vs Ritter');ax.set(xlabel='Population scale S',ylabel='Depth RMSE (lu)',title='Precision and model-error floor');ax.legend(fontsize=8);fig.savefig(OUT/'ritter_comparison.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
