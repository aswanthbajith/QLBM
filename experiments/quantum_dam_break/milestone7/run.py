"""Run compact reversible transport and drag; SI conversion is postprocessing."""
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from milestone7.solver import Damped2D
from milestone7.optimization import CompactFlux
from milestone7.drag import DragOracle
from milestone7 import reference as ref
from milestone5 import reference2d as base
from milestone5.run_2d import csvout
from xml.etree.ElementTree import Element,SubElement,ElementTree
G=9.81; RHO=998.2; U0=np.sqrt(G); DT=.125/U0; BETA=7/8

def write_si(path,s,mask,r):
    ny,nx=mask.shape;ext=f'0 {nx} 0 {ny} 0 0';root=Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian');im=SubElement(root,'ImageData',WholeExtent=ext,Origin='0 0 0',Spacing='1 1 1');p=SubElement(im,'Piece',Extent=ext);SubElement(p,'PointData');d=SubElement(p,'CellData',Scalars='water_depth_m')
    h=s[:,:,0];q=s[:,:,1:]*U0;v=np.divide(q,h[:,:,None],out=np.zeros_like(q),where=h[:,:,None]>0)
    vec=lambda a:np.concatenate((a,np.zeros((*h.shape,1))),axis=2)
    for name,arr,nc in [('water_depth_m',h,1),('discharge_m2_s',vec(q),3),('velocity_m_s',vec(v),3),('speed_m_s',np.linalg.norm(v,axis=2),1),('hydrostatic_bed_pressure_Pa',RHO*G*h,1),('water_mass_kg',RHO*h,1),('water_density_kg_m3',np.where(h>0,RHO,0),1),('wet_mask',h>0,1),('solid_mask',mask,1),('reference_depth_m',r[:,:,0],1)]:
        node=SubElement(d,'DataArray',type='Float64',Name=name,NumberOfComponents=str(nc),format='ascii');node.text=' '.join(f'{a:.12g}' for a in np.asarray(arr,float).ravel())
    ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def run():
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=True);old=np.load(Path(__file__).parents[1]/'milestone6/results/history.npz');mask=old['solid'];S=32;s=(old['quantum'][0]*S).astype(int);r=s/S;mass=int(s[:,:,0].sum());c=Damped2D(mask,S);start=time.monotonic();rows=[];hist=[];refs=[];dragrows=[]
    for j in range(17):
        a=s/S;h=a[:,:,0];q=a[:,:,1:];ke=float(.5*np.sum(np.divide(np.sum(q*q,axis=-1),h,out=np.zeros_like(h),where=h>0)))
        rows.append(dict(step=j,time_s=j*DT,volume_m3=float(h.sum()),mass_kg=float(h.sum()*RHO),wet_cells=int(np.count_nonzero(h)),kinetic_energy_nd=ke,depth_rmse_m=float(np.sqrt(np.mean((h[~mask]-r[:,:,0][~mask])**2))),depth_rmse_vs_undamped_m=float(np.sqrt(np.mean((h[~mask]-old['quantum'][j,:,:,0][~mask])**2))),obstacle_depth_m=float(h[mask].sum())))
        hist.append(a.copy());refs.append(r.copy());write_si(out/f'step_{j:03}.vti',a,mask,r)
        if j<16:
            w=c.encode(s);z=c.apply(w);assert c.apply(z,True)==w
            assert ((z>>c.transport.scratch)&((1<<(c.extra-c.transport.scratch))-1))==0
            pre=c.transport.decode(z);post=c.decode(z);c.encode(post)
            assert np.array_equal(post,ref.quantized_step(s,mask,S));assert post[:,:,0].sum()==mass and not np.any(post[mask]);assert np.all(np.abs(post[:,:,1:])<=np.abs(pre[:,:,1:]))
            dragrows.append(dict(step=j+1,pre_momentum_l1=int(np.abs(pre[:,:,1:]).sum()),post_momentum_l1=int(np.abs(post[:,:,1:]).sum())))
            s=post;r=ref.float_step(r,mask,BETA)
        if j%4==0:print('step',j,'elapsed',round(time.monotonic()-start,1),flush=True)
    csvout(out/'diagnostics.csv',rows);csvout(out/'drag_diagnostics.csv',dragrows);np.savez_compressed(out/'history.npz',quantum=np.array(hist),reference=np.array(refs),solid=mask,undamped=old['quantum'][:17])
    decay=[]
    for scale in (32,128,512):
        oracle=DragOracle(scale.bit_length()+1);q=scale
        for j in range(25):
            decay.append(dict(scale=scale,step=j,time_s=j*DT,velocity_ratio=q/scale,exact_ratio=BETA**j));q=oracle.evaluate(q)
    csvout(out/'drag_decay.csv',decay)
    spatial=[]
    for n in (32,64,128,256,512):
        dx=16/n;steps=round(8/dx);u=ref.planar_reference(n,steps);x=(np.arange(n)+.5)*dx-8;exact,_=base.ritter(x,1);err=float(np.sqrt(np.mean((u[:,0]-exact)**2)))
        spatial.append(dict(cells=n,dx=dx,steps=steps,depth_rmse_nd=err,mass_error_nd=float(abs(u[:,0].sum()*dx-12)),observed_order=None if not spatial else float(np.log2(spatial[-1]['depth_rmse_nd']/err))))
    csvout(out/'spatial_reference.csv',spatial)
    scaling=[]
    for scale in (2,4,8,16,32,64,128):
        a=CompactFlux(scale,1);scaling.append(dict(scale=scale,m6_work_qubits=a.original_ancillas,m7_work_qubits=a.ancillas,m6_gates=a.original_gates,m7_gates=len(a.gates)))
    csvout(out/'workspace_scaling.csv',scaling)
    m6=json.loads((Path(__file__).parents[1]/'milestone6/results/report.json').read_text());report=dict(model='flat-bed shallow-water finite volume with linear drag; not LBM or molecular viscosity',backend='computational basis reversible gate simulator with cached local gate outputs; no statevector or QPU',grid=[16,16],steps=16,depth_scale_m=1,length_scale_m=1,velocity_scale_m_s=U0,dt_s=DT,g_m_s2=G,water_density_kg_m3=RHO,beta=BETA,linear_drag_rate_s_inv=-np.log(BETA)/DT,drag_rate_selection='illustrative, not experimentally calibrated',initial=rows[0],final=rows[-1],max_mass_error_kg=max(abs(x['mass_kg']-rows[0]['mass_kg']) for x in rows),max_obstacle_depth_m=max(x['obstacle_depth_m'] for x in rows),quantized_reference_agreement='16/16',inverse_and_clean_workspace='16/16',m6_qubits=m6['new_logical_qubits'],compact_transport_qubits=c.transport.nqubits,compact_transport_gates=c.transport.gate_count(),m6_gates=m6['new_logical_gates'],damped_qubits=c.nqubits,damped_gates=c.gate_count(),m6_local_work_qubits=max(a.original_ancillas for a in c.transport.oracles.values()),m7_local_work_qubits=max(a.ancillas for a in c.transport.oracles.values()),drag_local_gates=len(c.drag.gates),drag_local_work_qubits=c.drag.ancillas,spatial_reference=spatial,elapsed_seconds=time.monotonic()-start,native_qiskit='not available')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
