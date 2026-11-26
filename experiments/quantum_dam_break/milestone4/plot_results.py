from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def plot(out=None):
    out=Path(out or Path(__file__).parent/'results'); a=np.load(out/'history.npz'); q=a['quantum']; f=a['reference']; x=np.arange(q.shape[1])+.5; t=np.arange(len(q))*.25
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i in [0,2,5,8]: ax[0].step(x,q[i,:,0],where='mid',label=f't={t[i]:g}')
    ax[0].set(title='Gate-simulated depth profiles',xlabel='Cell centre (dimensionless)',ylabel='Depth (dimensionless)',ylim=(0,1.65));ax[0].legend()
    ax[1].step(x,q[-1,:,0],where='mid',label='Fixed-point gates');ax[1].plot(x,f[-1,:,0],'o--',label='Floating FV reference');ax[1].set(title='Final depth comparison',xlabel='Cell centre',ylabel='Depth',ylim=(0,1.65));ax[1].legend()
    ax[2].plot(t,np.sqrt(np.mean((q[:,:,0]-f[:,:,0])**2,axis=1)),'o-',label='Depth RMSE');ax[2].plot(t,abs(q[:,:,0].sum(1)-q[0,:,0].sum()),'s--',label='Mass error');ax[2].set(title='Quantization and conservation',xlabel='Time (dimensionless)',ylabel='Error');ax[2].legend()
    fig.suptitle('Milestone 4 | 1D reversible shallow-water proof of concept');fig.savefig(out/'benchmark.png',dpi=160);plt.close(fig)
    frames=[]
    for i,s in enumerate(q):
        fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained');ax.bar(x,s[:,0],width=.98,label='Gate-simulated depth');ax.plot(x,f[i,:,0],'o--',label='Floating FV reference');ax.set(xlim=(0,6),ylim=(0,1.7),xlabel='Position (dimensionless)',ylabel='Water depth (dimensionless)',title=f'1D fixed-point fluid update | step {i} | t={t[i]:.2f}');ax.legend(loc='upper right');b=BytesIO();fig.savefig(b,format='png',dpi=110);plt.close(fig);b.seek(0);frames.append(Image.open(b).convert('RGB'))
    frames[0].save(out/'depth_evolution.gif',save_all=True,append_images=frames[1:],duration=600,loop=0)
if __name__=='__main__':plot()
