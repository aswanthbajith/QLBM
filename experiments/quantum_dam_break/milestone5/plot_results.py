"""Plots and GIF made only from computed histories."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
P=Path(__file__).parent/'results'
a=np.load(P/'history.npz');h=a['quantum'];r=a['reference'];solid=a['solid'];b=np.load(P/'ritter.npz')
def img(ax,s,title):
    a=np.ma.masked_where(solid,s[:,:,0]);im=ax.imshow(a,origin='lower',vmin=0,vmax=1.5,cmap='Blues',interpolation='nearest');ax.set_title(title);ax.set_xlabel('x cell');ax.set_ylabel('y cell');return im
fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
for ax,t in zip(axs.flat[:3],[0,16,32]):im=img(ax,h[t],f'Gate-simulated depth: step {t}')
fig.colorbar(im,ax=axs[0,:],label='Depth (dimensionless)',shrink=.8)
im=img(axs[1,0],r[-1],'Floating reference: step 32')
err=np.ma.masked_where(solid,h[-1,:,:,0]-r[-1,:,:,0]);lim=max(abs(err.min()),abs(err.max()));e=axs[1,1].imshow(err,origin='lower',cmap='coolwarm',vmin=-lim,vmax=lim);axs[1,1].set_title('Depth difference at step 32');fig.colorbar(e,ax=axs[1,1],shrink=.8)
ax=axs[1,2];ax.plot(np.arange(len(h))/8,(h[:,:,:,0]>0).sum(axis=(1,2)),label='Gate-simulated');ax.plot(np.arange(len(h))/8,(r[:,:,:,0]>0).sum(axis=(1,2)),label='Float (any positive depth)',ls='--');ax.set(xlabel='Time (dimensionless)',ylabel='Wet cells',title='Wetting over time');ax.legend(fontsize=8)
fig.suptitle('Milestone 5 | Reversible 2D shallow-water prototype',fontsize=16);fig.savefig(P/'benchmark_2d.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.plot(b['x'],b['exact_depth'],label='Ritter analytical solution',lw=2.5);ax.plot(b['x'],b['float_depth'],label='Floating Rusanov',ls='--');
for s in (8,16,32):ax.step(b['x'],b[f'depth_{s}'],where='mid',label=f'Gate simulation: depth quantum 1/{s}',alpha=.85)
ax.set(xlabel='Position (dimensionless)',ylabel='Depth (dimensionless)',title='Dry-bed dam break at t = 1: precision comparison');ax.legend(fontsize=9);fig.savefig(P/'ritter_comparison.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(6.8,5.8),layout='constrained');im=img(ax,h[0],'');fig.colorbar(im,ax=ax,label='Depth (dimensionless)');xx,yy=np.meshgrid(np.arange(16),np.arange(16));q=ax.quiver(xx,yy,np.zeros((16,16)),np.zeros((16,16)),scale=12,color='black',alpha=.7,width=.004)
def update(t):
    s=h[t];d=s[:,:,0];u=np.divide(s[:,:,1:],d[:,:,None],out=np.zeros_like(s[:,:,1:]),where=d[:,:,None]>0);im.set_data(np.ma.masked_where(solid,d));q.set_UVC(np.ma.masked_where(solid,u[:,:,0]),np.ma.masked_where(solid,u[:,:,1]));ax.set_title(f'2D gate-simulated dam break | step {t:02d}, t={t/8:.3f}\nSynthetic obstacles; nondimensional shallow-water model');return im,q
ani=FuncAnimation(fig,update,frames=len(h),interval=150,blit=False);ani.save(P/'dam_break_2d.gif',writer=PillowWriter(fps=6),dpi=100);plt.close(fig)
