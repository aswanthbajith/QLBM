from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv,json
out=Path(__file__).parent/'results'
r=json.loads((out/'report.json').read_text());s=list(csv.DictReader((out/'arithmetic_scaling.csv').open()));p=np.load(out/'precision.npz');d=list(csv.DictReader((out/'precision.csv').open()))
fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
ax[0].semilogy([int(x['input_bits']) for x in s],[int(x['arithmetic_gates']) for x in s],'o-',label='Reversible arithmetic gates');ax[0].semilogy([int(x['input_bits']) for x in s],[int(x['lut_input_patterns']) for x in s],'s--',label='Possible LUT control patterns (not gate count)');ax[0].set(xlabel='Local input bits',ylabel='Count (log scale)',title='Arithmetic scaling');ax[0].legend(fontsize=8);ax[0].grid(alpha=.3)
ax[1].plot(p['x'],p['exact_depth'],'k--',label='Ritter exact');ax[1].plot(p['x'],p['float_depth'],label='Floating-point FV');
for S in (32,64,128):ax[1].plot(p['x'],p[f'depth_{S}'],label=f'Arithmetic 1/{S}',alpha=.8)
ax[1].set(xlabel='x (nondimensional)',ylabel='Depth',title='Precision benchmark at t=1');ax[1].legend(fontsize=8);ax[1].grid(alpha=.3)
fig.savefig(out/'benchmark.png',dpi=160);plt.close(fig)
