from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
out=Path(__file__).parent/'results'
a=np.load(out/'history.npz');h=a['quantum'][:,:,:,0];mask=a['solid'];u=a['undamped'][:,:,:,0]
D=np.genfromtxt(out/'drag_decay.csv',delimiter=',',names=True);S=np.genfromtxt(out/'spatial_reference.csv',delimiter=',',names=True);W=np.genfromtxt(out/'workspace_scaling.csv',delimiter=',',names=True)
fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained');fig.suptitle('Milestone 7 | Reversible transport and linear drag',fontsize=17)
for p,z,title in [(ax[0,0],h[0],'Initial water depth'),(ax[0,1],h[-1],'Damped: step 16'),(ax[0,2],u[-1],'Undamped M6: step 16')]:
    im=p.imshow(np.ma.masked_where(mask,z),origin='lower',vmin=0,vmax=1.5,cmap='viridis');p.set_facecolor('#333333');p.set_title(title);p.set_xlabel('x (m)');p.set_ylabel('y (m)');fig.colorbar(im,ax=p,label='Depth (m)',shrink=.7)
p=ax[1,0];p.plot(D['time_s'][D['scale']==32],D['exact_ratio'][D['scale']==32],'k--',label='Analytic');
for s in (32,128,512):
    ix=D['scale']==s;p.plot(D['time_s'][ix],D['velocity_ratio'][ix],label=f'1/{s} ticks')
p.set(xlabel='Time (s)',ylabel='Velocity ratio',title='Isolated drag gate benchmark');p.legend(fontsize=8)
p=ax[1,1];p.loglog(S['cells'],S['depth_rmse_nd'],'o-');p.set(xlabel='Cells',ylabel='Depth RMSE',title='Ritter refinement (classical only)');p.grid(alpha=.25)
p=ax[1,2];p.plot(W['scale'],W['m6_work_qubits'],'o-',label='M6');p.plot(W['scale'],W['m7_work_qubits'],'o-',label='M7');p.set(xscale='log',xlabel='Fixed-point scale',ylabel='Work qubits',title='Local positive flux oracle');p.legend();p.grid(alpha=.25)
fig.savefig(out/'benchmark.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(5,4.5),layout='constrained');im=ax.imshow(np.ma.masked_where(mask,h[0]),origin='lower',vmin=0,vmax=1.5,cmap='viridis');ax.set_facecolor('#333333');ax.set(xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=ax,label='Water depth (m)');title=ax.set_title('')
def update(i):
    im.set_data(np.ma.masked_where(mask,h[i]));title.set_text(f'M7 linear drag | step {i} | {i*.125/np.sqrt(9.81):.3f} s');return im,title
FuncAnimation(fig,update,frames=len(h),interval=250,blit=False).save(out/'dam_break_damped.gif',writer=PillowWriter(fps=4));plt.close(fig)
