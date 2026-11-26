"""Run the gate-simulated benchmark and an independent floating reference."""
import csv,json
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement,ElementTree
import numpy as np
from solver import StepCircuit, A
import reference

def write_vti(path,s,ref):
    n=len(s); root=Element('VTKFile',type='ImageData',version='0.1',byte_order='LittleEndian'); ext=f'0 {n} 0 1 0 0'
    image=SubElement(root,'ImageData',WholeExtent=ext,Origin='0 0 0',Spacing='1 1 1'); piece=SubElement(image,'Piece',Extent=ext)
    SubElement(piece,'PointData'); data=SubElement(piece,'CellData',Scalars='depth_nd')
    h,q=s.T; u=np.divide(q,h,out=np.zeros_like(q),where=h>0)
    for name,arr,nc in [('depth_nd',h,1),('discharge_nd',q,1),('velocity_nd',np.column_stack((u,np.zeros((n,2)))),3),('hydrostatic_flux_nd',h*h/2,1),('wet_mask',h>0,1),('reference_depth_nd',ref[:,0],1)]:
        node=SubElement(data,'DataArray',type='Float64',Name=name,NumberOfComponents=str(nc),format='ascii'); node.text=' '.join(f'{x:.12g}' for x in np.asarray(arr,float).ravel())
    ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def run(out=None):
    out=Path(out or Path(__file__).parent/'results');out.mkdir(parents=True,exist_ok=True)
    c=StepCircuit(6); words=[(6,0)]*3+[(0,0)]*3; f=np.array(words,float)/4; history=[]; refs=[]; rows=[]
    for t in range(9):
        s=np.array(words,float)/4; h,q=s.T; u=np.divide(q,h,out=np.zeros_like(q),where=h>0); err=np.sqrt(np.mean((h-f[:,0])**2)); history.append(s.copy()); refs.append(f.copy())
        rows.append(dict(step=t,time_nd=t*.25,mass_nd=float(h.sum()),wet_cells=int((h>0).sum()),depth_rmse_vs_float=float(err),max_signal_speed=float(np.max(abs(u)+np.sqrt(h))),max_depth=float(h.max())))
        write_vti(out/f'step_{t:03}.vti',s,f)
        if t<8:
            w=c.encode(words); z=c.apply(w); assert c.apply(z,True)==w and z>>c.scratch==0
            next_words=c.step(words); assert np.allclose(np.array(next_words)/4,reference.step(s,True)); words=next_words; f=reference.step(f)
    with (out/'diagnostics.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'history.npz',quantum=np.array(history),reference=np.array(refs))
    report={'model':'1D flat-bed nondimensional shallow-water finite volume; coarse fixed point; table-compiled reversible gate prototype, NOT D2Q9 QLBM', 'backend':'indexed controlled-X basis gate simulator; no Qiskit or QPU; per-step classical basis readout and re-preparation', 'cells':6,'steps':8,'qubits_per_step':c.nqubits,'pattern_controlled_X_per_step':c.gate_count(),'gravity_nd':1,'dx_nd':1,'dt_nd':.25,'rusanov_a':float(A),'depth_quantum':.25,'initial_mass':rows[0]['mass_nd'],'final_mass':rows[-1]['mass_nd'],'max_mass_error':max(abs(r['mass_nd']-rows[0]['mass_nd']) for r in rows),'initial_wet_cells':rows[0]['wet_cells'],'final_wet_cells':rows[-1]['wet_cells'],'final_depth_rmse_vs_float':rows[-1]['depth_rmse_vs_float'],'max_depth_rmse_vs_float':max(r['depth_rmse_vs_float'] for r in rows),'max_signal_speed':max(r['max_signal_speed'] for r in rows),'quantized_reference_agreement':'all eight updates','scratch_uncomputation':'passed all eight updates','reversible_roundtrip':'passed all eight updates'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':run()
