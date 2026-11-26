"""Use M27's multi-angle scientific views for the actual checkpoint continuation."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from milestone27.render import SPECS,limits,frame,toimage
OUT=Path(__file__).parent/'results'

def main():
    report=json.loads((OUT/'report.json').read_text());d=np.load(OUT/'fields.npz');report['max_speed_all_steps']=float(d['speed'].max())
    for k in SPECS:
        lo,hi=limits(k,d['velocity'][:,1] if k=='uy' else d[k]);norm=TwoSlopeNorm(0,vmin=lo,vmax=hi) if lo<0<hi else Normalize(lo,hi);images=[]
        for i in range(len(d['steps'])):
            fig=frame(d,k,i,norm,report);fig._suptitle.set_text(fig._suptitle.get_text().replace('M27 ·','M28 · reversible macroscopic arithmetic ·'));images.append(toimage(fig))
        images[-1].save(OUT/f'{k}_final.png');images[0].save(OUT/f'{k}_multiangle.gif',save_all=True,append_images=images[1:],duration=350,loop=0,optimize=True)
    rows=list(csv.DictReader((OUT/'history.csv').open()));t=np.array([int(r['step']) for r in rows]);fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for key in ('alpha','rho','mu','P'):ax[0].semilogy([p['bits'] for p in report['precision']],[p['errors'][key] for p in report['precision']],marker='o',label=key)
    ax[0].set(xlabel='Fractional bits',ylabel='Maximum absolute error [lu]',title='Fixed-point precision sweep');ax[0].legend();ax[0].grid(alpha=.25)
    for key in ('phase_rmse','max_density_error','max_velocity_component_error','max_population_error'):ax[1].semilogy(t,np.maximum([float(r[key]) for r in rows],1e-16),label=key.replace('_',' '))
    ax[1].set(xlabel='Simulation step',ylabel='Error [lu]',title='Matched checkpoint continuation');ax[1].legend(fontsize=7);ax[1].grid(alpha=.25)
    ax[2].plot(t,[float(r['phase_relative_error']) for r in rows],marker='o');ax[2].set(xlabel='Simulation step',ylabel='Relative phase mass error',title='Raw population conservation');ax[2].grid(alpha=.25)
    fig.suptitle('M28 · Quantum macroscopic moments and material interpolation',fontweight='bold');fig.savefig(OUT/'benchmark.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');wall=d['wall_pressure'];wet=d['wall_alpha'];ax[0].plot(t,wall.max((1,2)),label='Maximum');ax[0].plot(t,wall.mean((1,2)),label='Mean');ax[0].legend();ax[0].set(xlabel='Step',ylabel='Gauge pressure [lu]',title='Downstream wall-adjacent pressure');ax[1].plot(t,wet.max((1,2)));ax[1].axhline(.5,ls='--',label='Wetting threshold');ax[1].legend();ax[1].set(xlabel='Step',ylabel='Liquid fraction',title='Wall wetting diagnostic');fig.suptitle('Cell-centred pressure proxy, not validated impact load');fig.savefig(OUT/'wall_pressure.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
