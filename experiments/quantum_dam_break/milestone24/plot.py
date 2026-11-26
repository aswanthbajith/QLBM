from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import csv,io
OUT=Path(__file__).parent/'results'
h=np.load(OUT/'history.npz');steps=h['span_gate_steps'];names=['span_gate','closed_reference','zero_gravity'];labels=['Periodic span · quantum subroutines','Closed sidewalls · reference','No gravity · reference']

def frame(ax,a,title):
    img=ax.imshow(a[:,a.shape[1]//2,:],origin='lower',extent=[0,32,0,16],vmin=0,vmax=1,cmap='Blues',interpolation='nearest',aspect='equal')
    ax.contour(np.arange(32)+.5,np.arange(16)+.5,a[:,a.shape[1]//2,:],levels=[.5],colors='#14375a',linewidths=.9)
    ax.set_title(title,fontsize=10);ax.set_xticks([0,10,20,30]);ax.set_yticks([0,8,16]);ax.set_xlabel('x (cells)');return img
fig,axs=plt.subplots(3,6,figsize=(19,7),layout='constrained',sharex=True,sharey=True)
for row,(name,label) in enumerate(zip(names,labels)):
 for col,i in enumerate([0,4,8,12,16,20]):
    img=frame(axs[row,col],h[name][i,0],f'Step {steps[i]}')
    if col==0:axs[row,col].set_ylabel(label+'\nz (cells)',fontsize=9)
fig.colorbar(img,ax=axs,label='Liquid fraction α',shrink=.85);fig.suptitle('M24 — Actual dam-release fields: collapse and floor spreading',fontsize=17);fig.savefig(OUT/'dam_release.png',dpi=150);plt.close(fig)
images=[]
for i,t in enumerate(steps):
 fig,axs=plt.subplots(2,1,figsize=(8,6),layout='constrained',sharex=True,sharey=True)
 for j,name in enumerate(names[:2]):img=frame(axs[j],h[name][i,0],labels[j]);axs[j].set_ylabel('z (cells)')
 fig.colorbar(img,ax=axs,label='Liquid fraction α');fig.suptitle(f'Computed dam release | step {t}/640 | t*={t*np.sqrt(.00025/10.4):.2f}');b=io.BytesIO();fig.savefig(b,format='png',dpi=105);plt.close(fig);b.seek(0);images.append(Image.open(b).convert('RGB').copy())
images[0].save(OUT/'dam_release.gif',save_all=True,append_images=images[1:],duration=[350]*(len(images)-1)+[1200],loop=0)
rows=list(csv.DictReader((OUT/'trajectory.csv').open()));fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for name,label in zip(names,labels):
 r=[a for a in rows if a['case']==name];t=[int(a['step']) for a in r]
 for ax,key,title in zip(axs,['floor_front','left_height','com_z'],['Floor front (cell index)','Left-column height (cells)','Vertical liquid centre of mass (cells)']):
    ax.plot(t,[float(a[key]) for a in r],label=label);ax.set(xlabel='Timestep',title=title);ax.grid(alpha=.2)
axs[0].legend(fontsize=7);fig.suptitle('Matched gravity and boundary controls');fig.savefig(OUT/'benchmark.png',dpi=150);plt.close(fig)
