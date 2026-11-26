"""Reproduce gate-driven 16x16 experiment, precision benchmark and VTK outputs."""
import csv,json,sys
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement,ElementTree
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone5.solver2d import Circuit2D,oracle_count
from milestone5 import reference2d as ref

def write_vti(path,s,mask,r):
    ny,nx=mask.shape;ext=f'0 {nx} 0 {ny} 0 0';root=Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian')
    image=SubElement(root,'ImageData',WholeExtent=ext,Origin='0 0 0',Spacing='1 1 1');piece=SubElement(image,'Piece',Extent=ext);SubElement(piece,'PointData');d=SubElement(piece,'CellData',Scalars='depth_nd')
    h=s[:,:,0];q=s[:,:,1:];u=np.divide(q,h[:,:,None],out=np.zeros_like(q),where=h[:,:,None]>0)
    vec=lambda a:np.concatenate((a,np.zeros((*h.shape,1))),axis=2)
    for name,arr,nc in [('depth_nd',h,1),('discharge_nd',vec(q),3),('velocity_nd',vec(u),3),('speed_nd',np.linalg.norm(u,axis=2),1),('hydrostatic_flux_nd',h*h/2,1),('wet_mask',h>0,1),('solid_mask',mask,1),('reference_depth_nd',r[:,:,0],1),('depth_error_nd',h-r[:,:,0],1)]:
        node=SubElement(d,'DataArray',type='Float64',Name=name,NumberOfComponents=str(nc),format='ascii');node.text=' '.join(f'{a:.12g}' for a in np.asarray(arr,float).ravel())
    ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def csvout(path,rows):
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def precision(out):
    mask=np.zeros((1,32),bool);x=(np.arange(32)-15.5)*.5;t=1.;initial=np.zeros((1,32,3));initial[0,:16,0]=1.5
    h_exact,u_exact=ref.ritter(x,t);r=initial.copy()
    for _ in range(16):r=ref.step(r,mask)
    data={'x':x,'exact_depth':h_exact,'float_depth':r[0,:,0]};rows=[]
    for S in (8,16,32):
        c=Circuit2D(mask,S);s=(initial*S).astype(int)
        for j in range(16):s=c.step(s,verify_inverse=j in (0,15))
        h=s[0,:,0]/S;data[f'depth_{S}']=h
        rows.append({'scale':S,'depth_quantum':1/S,'rmse_vs_float':float(np.sqrt(np.mean((h-r[0,:,0])**2))),'rmse_vs_ritter_point_samples':float(np.sqrt(np.mean((h-h_exact)**2))),'wet_cells':int(np.count_nonzero(h)),'integrated_depth_dx':float(h.sum()*.5),'logical_qubits':c.nqubits,'logical_pattern_controlled_x':c.gate_count()})
    csvout(out/'precision.csv',rows);np.savez_compressed(out/'ritter.npz',**data)
    return rows,float(np.sqrt(np.mean((r[0,:,0]-h_exact)**2)))

def run(out=None):
    out=Path(out or Path(__file__).parent/'results');out.mkdir(parents=True,exist_ok=True)
    mask=np.zeros((16,16),bool);mask[5:11,8:10]=True;mask[1:4,12]=True
    c=Circuit2D(mask,32);s=np.zeros((16,16,3),int);s[:,:5,0]=48;r=s/32;hist=[];refs=[];rows=[]
    for j in range(33):
        a=s/32;h=a[:,:,0];u=np.divide(a[:,:,1:],h[:,:,None],out=np.zeros_like(a[:,:,1:]),where=h[:,:,None]>0);hist.append(a);refs.append(r.copy())
        rows.append({'step':j,'time_nd':j/8,'integrated_depth_nd':float(h.sum()),'wet_cells':int(np.count_nonzero(h)),'max_transverse_discharge_nd':float(abs(a[:,:,2]).max()),'max_speed_nd':float(np.linalg.norm(u,axis=2).max()),'max_directional_signal_speed':float(np.max(np.max(abs(u),axis=2)+np.sqrt(h))),'depth_rmse_vs_float':float(np.sqrt(np.mean((h[~mask]-r[:,:,0][~mask])**2))),'obstacle_mass_nd':float(h[mask].sum()),'min_depth':float(h.min())})
        write_vti(out/f'step_{j:03}.vti',a,mask,r)
        if j<32:
            z=c.step(s,verify_inverse=True);assert np.array_equal(z,ref.quantized_step(s,mask,32));s=z;r=ref.step(r,mask)
    np.savez_compressed(out/'history.npz',quantum=np.array(hist),reference=np.array(refs),solid=mask);csvout(out/'diagnostics.csv',rows)
    gatecount=c.gate_count();nq=c.nqubits;local_bits=c.f.cell;pairbits=2*local_bits;del c
    pr,baseline=precision(out)
    report={'model':'2D nondimensional shallow-water Rusanov split-flux finite volume, flat bed; not LBM; synthetic geometry','backend':'indexed basis controlled-X gate simulator, not full statevector or QPU; re-preparation after every step','grid':[16,16],'solid_cells':int(mask.sum()),'steps':32,'dt_nd':.125,'dx_nd':1,'dy_nd':1,'g_nd':1,'a_nd':3,'depth_quantum':1/32,'initial':rows[0],'final':rows[-1],'max_mass_error':max(abs(row['integrated_depth_nd']-rows[0]['integrated_depth_nd']) for row in rows),'max_obstacle_mass':max(row['obstacle_mass_nd'] for row in rows),'max_transverse_discharge':max(row['max_transverse_discharge_nd'] for row in rows),'max_signal_speed':max(row['max_directional_signal_speed'] for row in rows),'logical_qubits_per_step':nq,'logical_pattern_controlled_x_per_step':gatecount,'single_cell_oracle_controls':local_bits,'hypothetical_pair_oracle_controls_same_encoding':pairbits,'oracle_control_pattern_count_single':2**local_bits,'hypothetical_pair_pattern_count':2**pairbits,'quantized_reference_agreement':'all 32 steps','inverse_and_scratch':'all 32 steps','ritter':{'domain_cells':32,'dx':.5,'dt':.0625,'time':1.,'float_rmse_vs_ritter_point_samples':baseline,'precision':pr},'native_qiskit':'not executed','physical_viscosity':'not implemented; numerical dissipation only'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=='__main__':run()
