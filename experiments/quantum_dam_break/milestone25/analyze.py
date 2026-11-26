"""Analyze recorded solver outputs; no synthetic interpolation of interface shape."""
from pathlib import Path
import sys,json,csv,io
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from milestone25.scaling import nominal,si_mapping
OUT=Path(__file__).parent/'results'
NAMES=['nu.02','nu.01','nu.005','nu.0025','nu.001','gate','refined','slow106','slow212','slow530','slow212_gate']
R={n:json.loads((OUT/f'{n}.json').read_text()) for n in NAMES};data={n:np.load(OUT/f'{n}.npz') for n in NAMES};rows={n:list(csv.DictReader((OUT/f'{n}.csv').open())) for n in NAMES}
for n in NAMES:
    c=R[n]['configuration'];c.setdefault('slow',1);R[n]['nominal']=nominal(c);R[n]['SI_example']=si_mapping(c)
    arrivals=[q for q in rows[n] if float(q['front_H']) >= c['shape'][2]/c['H']-1e-12]
    R[n]['wall_first_contact']=dict(step=int(arrivals[0]['step']),t_star=float(arrivals[0]['t_star'])) if arrivals else None
    R[n]['final_front_thresholds_H']={str(v):float((np.flatnonzero(data[n]['h'].sum(0)[0,c['shape'][1]//2] > v)[-1]+1)/c['H']) if np.any(data[n]['h'].sum(0)[0,c['shape'][1]//2]>v) else 0. for v in [.1,.5,.9]}
for n,b in [('gate','nu.01'),('slow212_gate','slow212')]:
    a,d=data[n],data[b];R[n]['matched_phase_snapshot_rmse']=float(np.sqrt(np.mean((a['alpha']-d['alpha'])**2)));R[n]['matched_final_population_max_error']=float(max(np.max(abs(a[k]-d[k])) for k in ('f','h')))
R['refined']['block_average_phase_snapshot_rmse']=float(np.sqrt(np.mean((data['gate']['alpha']-data['refined']['alpha'].reshape(21,16,2,32,2).mean((2,4)))**2)))
(OUT/'report.json').write_text(json.dumps(R,indent=2)+'\n')
selected=['nu.02','gate','slow212_gate','slow530'];labels=['Re 26.5 · baseline reference','Re 53 · quantum subroutines','Re 212 · quantum subroutines','Re 530 · classical screen']
def field(ax,n,i):
    a=data[n]['alpha'][i];H=R[n]['configuration']['H'];z,x=a.shape
    im=ax.imshow(a,origin='lower',extent=[0,x/H,0,z/H],cmap='Blues',vmin=0,vmax=1,aspect='equal',interpolation='nearest');ax.contour((np.arange(x)+.5)/H,(np.arange(z)+.5)/H,a,levels=[.5],colors='#153858',linewidths=.8);ax.set_xticks([0,1,2,3]);ax.set_yticks([0,1]);return im
fig,axs=plt.subplots(4,5,figsize=(15,8),layout='constrained',sharex=True,sharey=True)
for j,n in enumerate(selected):
    for k,i in enumerate([0,5,10,15,20]):
        im=field(axs[j,k],n,i);axs[j,k].set_title(f't* = {i/20*R[n]["nominal"]["t_star"]:.2f}' if j==0 else '')
        if k==0:axs[j,k].set_ylabel(labels[j]+'\nz/H',fontsize=9)
        if j==3:axs[j,k].set_xlabel('x/H')
fig.colorbar(im,ax=axs,label='Liquid fraction α');fig.suptitle('M25: lower viscosity and smaller timesteps at matched dimensionless time');fig.savefig(OUT/'dam_release.png',dpi=150);plt.close(fig)
frames=[]
for i in range(21):
    fig,axs=plt.subplots(2,1,figsize=(7,5),layout='constrained',sharex=True)
    for ax,n,label in zip(axs,['nu.02','slow212_gate'],[labels[0],labels[2]]):im=field(ax,n,i);ax.set_title(label);ax.set_ylabel('z/H')
    axs[-1].set_xlabel('x/H');fig.colorbar(im,ax=axs,label='Liquid fraction α');fig.suptitle(f'Matched physical time | t* = {i/20*R["slow212_gate"]["nominal"]["t_star"]:.2f}');b=io.BytesIO();fig.savefig(b,format='png',dpi=100);plt.close(fig);b.seek(0);frames.append(Image.open(b).convert('RGB').copy())
frames[0].save(OUT/'dam_release.gif',save_all=True,append_images=frames[1:],duration=[250]*20+[1100],loop=0)
fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
for n in ['nu.02','nu.01','nu.0025','slow212_gate','slow530','refined']:
    rr=rows[n];t=[float(q['t_star']) for q in rr];label=f'{n} (Re {R[n]["Re"]:.0f})'
    for ax,k,title in zip(axs,['front_H','left_height_H','mach'],['Floor front x/H (cell edge)','Left-column height / H','Maximum Mach number']):ax.plot(t,[float(q[k]) for q in rr],label=label);ax.set(xlabel='t√(g/H)',title=title);ax.grid(alpha=.2)
axs[2].axhline(np.sqrt(3)*.1,color='k',ls='--',label='Speed guard');axs[0].legend(fontsize=7);fig.suptitle('M25 computed trajectories: failed speed-guard case retained');fig.savefig(OUT/'benchmark.png',dpi=150);plt.close(fig)
print(json.dumps({n:{'Re':R[n]['Re'],'steps':R[n]['steps'],'status':R[n]['status'],'wall':R[n]['wall_first_contact'],'peak_speed':R[n]['peak_speed']} for n in NAMES},indent=2))
