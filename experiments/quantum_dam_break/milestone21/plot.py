from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
OUT=Path(__file__).parent/'results';H=np.load(OUT/'history.npz')
fig,ax=plt.subplots(2,4,figsize=(12,5),layout='constrained')
for row,name in enumerate(['gate','zero_gravity']):
    for col,t in enumerate([0,32,64,96]):
        a=H[name][t,0,:,2,:];im=ax[row,col].imshow(a,origin='lower',vmin=0,vmax=1,cmap='Blues',interpolation='nearest');ax[row,col].contour(a,[.5],colors='k',linewidths=.6);ax[row,col].set_title(f'{"Gravity" if row==0 else "No gravity"} • t={t}');ax[row,col].set_xlabel('x (lu)');ax[row,col].set_ylabel('z (lu)')
fig.colorbar(im,ax=ax,label='Liquid fraction',shrink=.85);fig.suptitle('M21 • 831.67:1 density ratio • hybrid pressure LBM');fig.savefig(OUT/'dam_release.png',dpi=170);plt.close(fig)
rows=list(csv.DictReader(open(OUT/'dam_release.csv')));hyd=list(csv.DictReader(open(OUT/'hydrostatic.csv')))
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for name,label in [('gate','Gravity (gate)'),('zero_gravity','No gravity (gate)'),('reference','Gravity (reference)')]:
    rr=[q for q in rows if q['case']==name];t=np.array([int(q['step']) for q in rr]);cx=np.array([float(q['com_x']) for q in rr]);cz=np.array([float(q['com_z']) for q in rr]);ax[0,0].plot(t,cx-cx[0],label=label);ax[0,1].plot(t,cz-cz[0],label=label);ax[1,0].plot(t,[float(q['max_speed']) for q in rr],label=label)
for k,lab in [(ax[0,0],'Horizontal liquid COM shift (lu)'),(ax[0,1],'Vertical liquid COM shift (lu)'),(ax[1,0],'Maximum speed (lu)')]:k.set_xlabel('Timestep');k.set_ylabel(lab);k.legend(fontsize=8);k.grid(alpha=.25)
for method in ['paper','product']:
    rr=[q for q in hyd if q['force']==method and q['width']=='1.0' and q['pressure_offset']=='0.0' and q['mode']=='reference'];ax[1,1].loglog([float(q['ratio']) for q in rr],[float(q['peak_speed']) for q in rr],'o-',label=method)
ax[1,1].set(xlabel='Density ratio',ylabel='Hydrostatic peak speed (lu)');ax[1,1].legend();ax[1,1].grid(alpha=.25);fig.suptitle('M21 • pressure and gravity preflight');fig.savefig(OUT/'benchmark.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(11,3.5),layout='constrained');h=H['gate'][-1];names=['Material density (lu)','Physical pressure (lu)','Speed (lu)'];data=[h[1,:,2,:],h[3,:,2,:],np.linalg.norm(h[4:7,:,2,:],axis=0)]
for a,d,n in zip(ax,data,names):im=a.imshow(d,origin='lower');a.set_title(n);a.set_xlabel('x (lu)');a.set_ylabel('z (lu)');fig.colorbar(im,ax=a,shrink=.8)
fig.savefig(OUT/'fields.png',dpi=170);plt.close(fig)
