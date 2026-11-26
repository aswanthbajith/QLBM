from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=Path(__file__).parent/'results'
a=np.genfromtxt(p/'pressure_eos.csv',delimiter=',',names=True); b=np.genfromtxt(p/'force_response.csv',delimiter=',',names=True); c=np.genfromtxt(p/'shear_decay.csv',delimiter=',',names=True)
fig,ax=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
ax[0].plot(a['mass_lu'],a['kinetic_pressure_lu'],label='Exact lattice-gas pressure'); ax[0].plot(a['mass_lu'],a['best_fit_hydrostatic_flux_lu'],'--',label='Best-fit shallow-water law'); ax[0].set(xlabel='Mean mass (lattice units)',ylabel='Momentum flux (lattice units)',title='Pressure closure: mismatch'); ax[0].legend(fontsize=8)
ax[1].plot(b['mass_lu'],b['effective_acceleration_lu'],'o-'); ax[1].set(xlabel='Mean mass (lattice units)',ylabel='Effective acceleration (lattice units)',title='Fixed bias: density-dependent force')
ax[2].errorbar(c['step'],c['shear_amplitude_lu'],yerr=c['standard_error'],fmt='o',ms=3,label='Trajectory mean ± SE'); x=c['step'][4:21]; y=c['shear_amplitude_lu'][4:21]; slope,inter=np.polyfit(x,np.log(y),1); ax[2].plot(x,np.exp(inter+slope*x),label='Exponential fit (4–20)'); ax[2].set(xlabel='Timestep',ylabel='Transverse Fourier amplitude',title='Shear-wave decay'); ax[2].legend(fontsize=8)
for q in ax: q.grid(alpha=.2)
fig.suptitle('Milestone 3 | Quantum-channel kinetic benchmarks (not calibrated water flow)',fontsize=13)
fig.savefig(p/'benchmark_diagnostics.png',dpi=180)
