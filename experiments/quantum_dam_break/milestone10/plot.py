from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kinetic import ritter
P=Path(__file__).parent/'results'
r=list(csv.DictReader((P/'feasibility.csv').open()));z=np.load(P/'profiles.npz');x=z['x'];sel=abs(x)<28
fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
a=np.arange(len(r));ax[0].bar(a-.18,[100*float(t['sample_feasible_fraction']) for t in r],.36,label='Sampled depth–velocity pairs');ax[0].bar(a+.18,[100*float(t['ritter_fan_feasible_fraction']) for t in r],.36,label='Ritter rarefaction fan');ax[0].set_xticks(a,[t['lattice'] for t in r],rotation=20);ax[0].set_ylabel('Representable marginal states (%)');ax[0].set_ylim(0,105);ax[0].legend(fontsize=9);ax[0].set_title('Nonnegative moment feasibility')
for name in ['D2Q9','D2Q25','D2Q81','D2Q289','D2Q1089']:
    ax[1].plot(x[sel],z[name][-1,sel],label=name,linewidth=1)
ax[1].plot(x[sel],ritter(x[sel],24)[0],'k--',label='Ritter exact',linewidth=2);ax[1].set_xlabel('x (lattice units)');ax[1].set_ylabel('Water depth (lattice units)');ax[1].set_title('Classical kinetic screening, t = 24');ax[1].legend(fontsize=8);fig.suptitle('Milestone 10 | velocity-set comparison (not quantum fluid evolution)');fig.savefig(P/'benchmark.png',dpi=160)
