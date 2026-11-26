"""Water-body isometric views from thresholded actual liquid cells."""
import json, numpy as np
from render import OUT,tank,toimage,plt

def run():
    r=json.loads((OUT/'report.json').read_text());d=np.load(OUT/'fields.npz');nz,ny,nx=d['alpha'].shape[1:];images=[]
    for i,t in enumerate(d['steps']):
        a=d['alpha'][i];filled=(a>=.5).transpose(2,1,0);ts=t*np.sqrt(r['configuration']['gravity']/r['configuration']['H']);fig=plt.figure(figsize=(10,4.2),dpi=110,layout='constrained')
        for j,(elev,azim,title) in enumerate([(25,-60,'Front-side isometric'),(35,120,'Opposite-side isometric')]):
            ax=fig.add_subplot(1,2,j+1,projection='3d');ax.voxels(filled,facecolors='#168dcc',edgecolor='#0d5a86',linewidth=.25,shade=True,alpha=.86);tank(ax,nx,ny,nz);ax.view_init(elev,azim);ax.set_title(title)
        fig.suptitle(f'Water body · cells with α ≥ 0.5  |  step {t:04d}  |  t*={ts:.3f}\nActual 32 × 4 × 16 geometry · air hidden · no artificial spanwise variation',fontweight='bold',fontsize=12)
        im=toimage(fig);images.append(im)
        if i in (0,len(d['steps'])//2,len(d['steps'])-1):im.save(OUT/f'water_body_step_{t:04d}.png')
    images[0].save(OUT/'water_body_3d.gif',save_all=True,append_images=images[1:],duration=220,loop=0,optimize=True)
if __name__=='__main__':run()
