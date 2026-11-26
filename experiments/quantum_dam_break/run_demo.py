"""Actual generated output from an exact reduced quantum-channel simulation."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree
from quantum_core import Grid, initial, probabilities, decode, stream_permutation, apply_permutation, relaxation_channel, counts

def write_vti(path,g,h,u,p,raw):
    root=Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian')
    ext=f'0 {g.nx-1} 0 {g.ny-1} 0 0'
    image=SubElement(root,'ImageData',WholeExtent=ext,Origin='0 0 0',Spacing=f'{g.dx} {g.dx} 1')
    piece=SubElement(image,'Piece',Extent=ext); data=SubElement(piece,'PointData',Scalars='depth_proxy_m')
    for name,arr,n in [('depth_proxy_m',h,1),('transport_moment_m_s',np.dstack([u,np.zeros_like(h)]),3),('position_probability',p,1),('sampled_position_counts',raw,1),('solid_mask',g.solid.astype(int),1)]:
        e=SubElement(data,'DataArray',type='Float64',Name=name,NumberOfComponents=str(n),format='ascii')
        e.text=' '.join(f'{v:.14g}' for v in arr.ravel())
    SubElement(piece,'CellData'); ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def run(out,omega=0.5,steps=12,shots=4096):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    solid=np.zeros((4,4),bool); solid[1,2]=True
    g=Grid(solid=solid); a,M=initial(g); rho=np.outer(a,a.conj()); perm=stream_permutation(g)
    rows=[]; frames=[]
    for t in range(steps+1):
        p=probabilities(rho); f,h,u=decode(g,p,M); raw=counts(g,rho,shots,100+t)
        _,hs,_=decode(g,raw/shots,M)
        pos=p.reshape(16,g.ny,g.nx).sum(0); rc=raw.reshape(16,g.ny,g.nx).sum(0)
        write_vti(out/f'step_{t:03}.vti',g,h,u,pos,rc)
        rows.append(dict(step=t,trace=float(p.sum()),encoded_volume_m3=float(h.sum()*g.dx**2),max_depth_proxy_m=float(h.max()),sample_depth_rmse_m=float(np.sqrt(np.mean((hs-h)**2))),solid_population=float(h[g.solid].sum())))
        frames.append(h.copy())
        if t<steps: rho=apply_permutation(relaxation_channel(g,rho,omega),perm)
    with (out/'diagnostics.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out/'depth_proxy_history.npz',depth=np.array(frames),solid=g.solid)
    (out/'metadata.json').write_text(json.dumps({'model':'fixed-weight quantum reset transport experiment; NOT hydrodynamic dam break','backend':'NumPy exact reduced quantum channel, no QPU or Qiskit run','nx':4,'ny':4,'base_qubits':g.nq,'optional_collision_ancillas':5,'steps':steps,'omega':omega,'shots':shots,'dx_m':g.dx,'dt_s':g.dt,'initial_encoded_volume_m3':M*g.dx**2},indent=2))
    print(json.dumps(rows[-1],indent=2))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(Path(__file__).parent/'results')); p.add_argument('--omega',type=float,default=.5); p.add_argument('--steps',type=int,default=12); p.add_argument('--shots',type=int,default=4096); a=p.parse_args(); run(a.out,a.omega,a.steps,a.shots)
