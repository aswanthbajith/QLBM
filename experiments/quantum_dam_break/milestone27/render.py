"""Render measured fields, with fixed scales and honest cell-centred slice coordinates."""
from pathlib import Path
import argparse,json,io,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.ndimage import map_coordinates
from PIL import Image
ROOT=Path(__file__).parent; OUT=ROOT/'results/gate'
SPECS={
 'uy':('Spanwise velocity',r'$u_y$ [lu]','RdBu_r','signed'),
 'phi':('Phase-field order parameter',r'$\phi=2\alpha-1$','coolwarm','fixed'),
 'speed':('Flow velocity magnitude',r'$\|\mathbf{u}\|$ [lu]','viridis','positive'),
 'jz':('Vertical momentum density',r'$\rho u_z$ [lu]','RdBu_r','signed'),
 'pressure':('Hydrodynamic gauge pressure',r'$p-p_{ref}$ [lu]','RdBu_r','signed'),
 'rho':('Material density',r'$\rho/\rho_l$','viridis','positive'),
 'dynamic_pressure':('Dynamic pressure',r'$\frac{1}{2}\rho\|\mathbf{u}\|^2$ [lu]','magma','positive'),
 'mu':('Dynamic viscosity',r'$\mu$ [lu]','cividis','positive')}

def limits(k,v):
    kind=SPECS[k][3]
    if kind=='fixed':return -1,1
    if kind=='signed':
        a=max(float(np.max(np.abs(v))),1e-16);return -a,a
    return 0,max(float(v.max()),1e-16)

def tank(ax,nx,ny,nz):
    for y in (0,ny):
        for z in (0,nz):ax.plot([0,nx],[y,y],[z,z],color='#52606d',lw=.5,alpha=.65)
    for x in (0,nx):
        for z in (0,nz):ax.plot([x,x],[0,ny],[z,z],color='#52606d',lw=.5,alpha=.65)
        for y in (0,ny):ax.plot([x,x],[y,y],[0,nz],color='#52606d',lw=.5,alpha=.65)
    ax.set(xlim=(0,nx),ylim=(0,ny),zlim=(0,nz),xlabel='X [lu]',ylabel='Y',zlabel='Z [lu]')
    ax.set_box_aspect((nx,ny,nz));ax.view_init(elev=25,azim=-61);ax.set_yticks([0,ny//2,ny]);ax.set_xticks([0,nx//2,nx]);ax.set_zticks([0,nz//2,nz]);ax.tick_params(labelsize=7,pad=0);ax.grid(False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.set_facecolor((.96,.97,.99,0))

def center_slice(v,axis):
    n=v.shape[axis];return np.take(v,n//2,axis=axis) if n%2 else .5*(np.take(v,n//2-1,axis=axis)+np.take(v,n//2,axis=axis))

def contour(ax,x,y,alpha):
    if alpha.min()<.5<alpha.max():ax.contour(x,y,alpha,levels=[.5],colors=['#1c1c1c'],linewidths=.9)

def mesh(a):
    # Surface reconstructed ONLY between cell centres, no extrapolation into walls.
    if not a.min()<.5<a.max():return None
    v,f,_,_=marching_cubes(a,.5,allow_degenerate=False)
    return v+.5,f

def frame(d,k,i,norm,report):
    title,lab,cmap,_=SPECS[k];cm=plt.get_cmap(cmap);v=(d['velocity'][i,1] if k=='uy' else d[k][i]);a=d['alpha'][i];u=d['velocity'][i];nz,ny,nx=a.shape;iy=ny//2;iz=nz//2;step=int(d['steps'][i]);ts=step*np.sqrt(report['configuration']['gravity']/report['configuration']['H']);x=np.arange(nx)+.5;y=np.arange(ny)+.5;z=np.arange(nz)+.5
    fig=plt.figure(figsize=(12,4.6),dpi=100,layout='constrained');gs=fig.add_gridspec(2,3,width_ratios=[1.25,1.1,1.3],height_ratios=[1,1]);ax=fig.add_subplot(gs[:,0]);plan=fig.add_subplot(gs[0,1]);iso=fig.add_subplot(gs[:,2],projection='3d');floor=fig.add_subplot(gs[1,1])
    ax.imshow(center_slice(v,1),origin='lower',extent=(0,nx,0,nz),cmap=cm,norm=norm,interpolation='nearest');contour(ax,x,z,center_slice(a,1));ax.set(title=f'Side elevation · X–Z\ny = {ny/2:g}',xlabel='X [lu]',ylabel='Z [lu]',xlim=(0,nx),ylim=(0,nz));ax.set_aspect('equal')
    if k=='speed':
        ax.quiver(x[::3],z[::2],center_slice(u[0],1)[::2,::3],center_slice(u[2],1)[::2,::3],color='white',edgecolor='#202020',linewidth=.2,angles='xy',scale_units='xy',scale=max(report['max_speed_all_steps']/1.5,1e-10),width=.004,pivot='mid')
    plan.imshow(center_slice(v,0),origin='lower',extent=(0,nx,0,ny),cmap=cm,norm=norm,interpolation='nearest');contour(plan,x,y,center_slice(a,0));plan.set(title=f'Top-down plan · X–Y\nz = {nz/2:g}',xlabel='X [lu]',ylabel='Y [lu]',xlim=(0,nx),ylim=(0,ny));plan.set_aspect('equal');plan.set_yticks([0,ny//2,ny]);floor.imshow(v[0],origin='lower',extent=(0,nx,0,ny),cmap=cm,norm=norm,interpolation='nearest');contour(floor,x,y,a[0]);floor.set(title='Near-floor plan · X–Y\nz = 0.5',xlabel='X [lu]',ylabel='Y [lu]',xlim=(0,nx),ylim=(0,ny));floor.set_aspect('equal');floor.set_yticks([0,ny//2,ny])
    # True 3D liquid/gas isosurface; color by the requested field interpolated at triangle centroids.
    m=mesh(a)
    if m is not None:
        verts,faces=m;centres=verts[faces].mean(1);values=map_coordinates(v, (centres-.5).T,order=1,mode='nearest');pol=Poly3DCollection(verts[faces][:,:,::-1],facecolors=('#1676a1' if k=='phi' else cm(norm(values))),edgecolors='none',alpha=.98);iso.add_collection3d(pol)
    # Orthogonal slices sampled from the actual volume, not a synthetic extrusion.
    xx,zz=np.meshgrid(x,z);xx2,yy=np.meshgrid(x,y);yy3,zz3=np.meshgrid(y,z)
    iso.plot_surface(xx,np.full_like(xx,ny/2),zz,facecolors=cm(norm(center_slice(v,1))),rstride=1,cstride=1,shade=False,alpha=.28,linewidth=0)
    iso.plot_surface(xx2,yy,np.full_like(xx2,nz/2),facecolors=cm(norm(center_slice(v,0))),rstride=1,cstride=1,shade=False,alpha=.25,linewidth=0)
    iso.plot_surface(np.full_like(yy3,nx-.5),yy3,zz3,facecolors=cm(norm(v[:,:,-1])),rstride=1,cstride=1,shade=False,alpha=.28,linewidth=0)
    tank(iso,nx,ny,nz);iso.set_title(('3D isometric · blue φ = 0 surface\n+ translucent field cut planes' if k=='phi' else '3D isometric · field on φ = 0 surface\n+ translucent field cut planes'),fontsize=10)
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cm),ax=[ax,plan,floor,iso],location='bottom',shrink=.65,pad=.02,aspect=40);cb.set_label(lab,fontsize=10)
    fig.suptitle(f'{title}  |  step {step:04d}  |  t* = {ts:.3f}\nM27 · 3D finite-width release · Re 212 · black contour: φ = 0 · fixed scales · hybrid quantum-subroutine run',fontsize=13,fontweight='bold')
    return fig

def toimage(fig):
    b=io.BytesIO();fig.savefig(b,format='png',facecolor='white');b.seek(0);im=Image.open(b).convert('RGB');plt.close(fig);return im

def render_field(k):
    report=json.loads((OUT/'report.json').read_text());d=np.load(OUT/'fields.npz');lo,hi=limits(k,d['velocity'][:,1] if k=='uy' else d[k]);norm=TwoSlopeNorm(0,vmin=lo,vmax=hi) if lo<0<hi else Normalize(lo,hi);frames=[]
    for i in range(len(d['steps'])):
        im=toimage(frame(d,k,i,norm,report));frames.append(im)
        if i in (0,len(d['steps'])//2,len(d['steps'])-1):im.save(OUT/f'{k}_step_{int(d["steps"][i]):04d}.png')
    frames[0].save(OUT/f'{k}_multiangle.gif',save_all=True,append_images=frames[1:],duration=220,loop=0,optimize=True)
    print(k,lo,hi,flush=True)

def impact():
    r=json.loads((OUT/'report.json').read_text());d=np.load(OUT/'fields.npz');w=d['wall_pressure'];a=d['wall_alpha'];ts=np.arange(len(w))*np.sqrt(r['configuration']['gravity']/r['configuration']['H']);nz,ny=w.shape[1:];peak=r['peak_wall_pressure_index'][0];contact=r['first_wall_wetting_step'];t_end=len(w)-1;ids=[0,contact,peak,t_end];ids=sorted(set(i for i in ids if i is not None));vmax=max(abs(w.min()),abs(w.max()));norm=TwoSlopeNorm(0,vmin=-vmax,vmax=vmax);cm='RdBu_r'
    fig=plt.figure(figsize=(12,7),layout='constrained');gs=fig.add_gridspec(2,len(ids),height_ratios=[1,1.2]);axs=[]
    for j,t in enumerate(ids):
        ax=fig.add_subplot(gs[0,j]);axs.append(ax);ax.imshow(w[t],origin='lower',extent=(0,ny,0,nz),cmap=cm,norm=norm,interpolation='nearest');contour(ax,np.arange(ny)+.5,np.arange(nz)+.5,a[t]);ax.set(title=f'Step {t} · t*={ts[t]:.3f}',xlabel='Y [lu]',ylabel='Z [lu]');ax.set_aspect('equal')
    fig.colorbar(ScalarMappable(norm=norm,cmap=cm),ax=axs,shrink=.7,label='Wall-adjacent gauge pressure [lu]')
    ax=fig.add_subplot(gs[1,:]);
    for z in (0,1,3,7):ax.plot(ts,w[:,z,ny//2],label=f'z={z+.5:g}')
    ax.plot(ts,w.max((1,2)),color='black',ls='--',lw=1.3,label='Wall maximum');ax.axvline(ts[peak],color='gray',ls=':',label='Peak gauge pressure')
    if contact is not None:ax.axvline(ts[contact],color='tab:green',ls='--',label='First α ≥ 0.5 at wall-adjacent cell')
    ax.set(xlabel='Dimensionless time t*',ylabel='Gauge pressure [lu]',title='Downstream pressure probes · every timestep');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Downstream wall pressure diagnostic | x = 31.5 lu\nCell-centred pressure proxy, not validated wall traction or impact-sensor load',fontweight='bold');fig.savefig(OUT/'wall_pressure.png',dpi=140);plt.close(fig)
    # All wall snapshots, selected at same stride as volume snapshots, and pressure trace up to the frame.
    images=[]
    for t in d['steps']:
        fig,ax=plt.subplots(1,2,figsize=(9,4),layout='constrained',gridspec_kw={'width_ratios':[1,2.8]});ax[0].imshow(w[t],origin='lower',extent=(0,ny,0,nz),cmap=cm,norm=norm);contour(ax[0],np.arange(ny)+.5,np.arange(nz)+.5,a[t]);ax[0].set(title='Downstream wall-adjacent cells',xlabel='Y [lu]',ylabel='Z [lu]');fig.colorbar(ScalarMappable(norm=norm,cmap=cm),ax=ax[0],shrink=.8)
        for z in (0,1,3,7):ax[1].plot(ts[:t+1],w[:t+1,z,ny//2],label=f'z={z+.5:g}')
        ax[1].set(xlim=(0,ts[-1]),ylim=(-vmax,vmax),xlabel='t*',ylabel='Gauge pressure [lu]',title='Pressure probes (full timestep resolution)');ax[1].grid(alpha=.2);ax[1].legend(fontsize=8);fig.suptitle(f'Wall pressure proxy · step {t} · t*={ts[t]:.3f}');images.append(toimage(fig))
    images[0].save(OUT/'wall_pressure.gif',save_all=True,append_images=images[1:],duration=220,loop=0,optimize=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('field',choices=list(SPECS)+['impact','all']);arg=p.parse_args().field
    if arg=='all':
        for k in SPECS:render_field(k)
        impact()
    elif arg=='impact':impact()
    else:render_field(arg)
