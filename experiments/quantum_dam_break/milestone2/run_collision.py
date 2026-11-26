"""Run a dephased quantum lattice-gas circuit ensemble. No classical fluid solver."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from xml.etree.ElementTree import Element,SubElement,ElementTree
from collision import BITS,M,P,STRESS,collide,stream,permutation,swaps,GROUPS,U,K

def write_vti(path,h,vel,stress,solid):
    ny,nx=h.shape; ext=f'0 {nx-1} 0 {ny-1} 0 0'
    root=Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian')
    image=SubElement(root,'ImageData',WholeExtent=ext,Origin='0 0 0',Spacing='1 1 1')
    data=SubElement(SubElement(image,'Piece',Extent=ext),'PointData',Scalars='mean_occupancy_mass_lu')
    for name,a,n in [('mean_occupancy_mass_lu',h,1),('mean_velocity_lu',np.dstack([vel,np.zeros_like(h)]),3),('kinetic_stress_xx_lu',stress[...,0,0],1),('kinetic_stress_yy_lu',stress[...,1,1],1),('kinetic_stress_xy_lu',stress[...,0,1],1),('solid_mask',solid,1)]:
        e=SubElement(data,'DataArray',Name=name,type='Float64',NumberOfComponents=str(n),format='ascii'); e.text=' '.join(f'{v:.12g}' for v in a.ravel())
    SubElement(image.find('Piece'),'CellData'); ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def run(out,nx=8,ny=4,shots=2048,steps=16,omega=.7,seed=2026):
    if nx<4 or ny<2 or shots<1 or steps<0: raise ValueError('invalid dimensions or counts')
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(seed)
    solid=np.zeros((ny,nx),bool); solid[ny//2,nx//2+1]=True
    p=permutation(nx,ny,solid); state=np.zeros((shots,ny,nx),np.int16)
    # Rest bit, mass 2, zero initial momentum; same initial basis state per trajectory.
    state[:,:,:nx//2]=1; state[:,solid]=0
    initial_mass=M[state].sum((1,2)); rows=[]; frames=[]
    for t in range(steps+1):
        mass=M[state]; h=mass.mean(0); mom=P[state].mean(0)
        vel=np.divide(mom,h[...,None],out=np.zeros_like(mom),where=h[...,None]>0)
        stress=STRESS[state].mean(0); frames.append(h.copy())
        write_vti(out/f'step_{t:03}.vti',h,vel,stress,solid)
        rows.append({'step':t,'mean_total_mass_lu':float(h.sum()),'max_mass_error_per_trajectory':int(abs(mass.sum((1,2))-initial_mass).max()),'solid_mass_lu':float(h[solid].sum()),'right_half_mass_lu':float(h[:,nx//2:].sum()),'max_mean_mass_lu':float(h.max()),'collision_mass_error':0,'collision_momentum_error':0})
        if t<steps:
            new=collide(state,omega,rng); assert not new[:,solid].any()
            rows[-1]['collision_mass_error']=int(abs(M[new]-M[state]).max()); rows[-1]['collision_momentum_error']=int(abs(P[new]-P[state]).max())
            state=stream(new,p)
    with (out/'diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out/'occupancy_history.npz',mass=np.array(frames),solid=solid)
    meta={'model':'D2Q9 occupancy quantum channel; kinetic lattice gas, not shallow-water or BGK','backend':'NumPy exact local unitary Born kernel and sampled global diagonal trajectories; no Qiskit/QPU','grid':[nx,ny],'data_qubits':nx*ny*9,'reusable_ancillas_in_optional_circuit':1,'shots':shots,'steps':steps,'omega':omega,'seed':seed,'local_classes':len(GROUPS),'nontrivial_classes':sum(len(g)>1 for g in GROUPS),'max_class_size':max(map(len,GROUPS)),'stream_swap_count':len(list(swaps(p))),'units':'dimensionless lattice units only'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps({'metadata':meta,'final':rows[-1]},indent=2))
    return rows

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--out',default=str(Path(__file__).parent/'results')); a.add_argument('--nx',type=int,default=8); a.add_argument('--ny',type=int,default=4); a.add_argument('--shots',type=int,default=2048); a.add_argument('--steps',type=int,default=16); a.add_argument('--omega',type=float,default=.7); a.add_argument('--seed',type=int,default=2026); run(**vars(a.parse_args()))
