"""Cell-centred VTK ImageData and PVD timeseries; no external VTK dependency."""
from pathlib import Path
import numpy as np

def write_series(out,data,steps,c):
    folder=Path(out)/'vtk';folder.mkdir(exist_ok=True);nz,ny,nx=c['shape'];extent=f'0 {nx} 0 {ny} 0 {nz}'; entries=[]
    names={'alpha':'liquid_fraction','phi':'phase_order_parameter','rho':'material_density_lu','mu':'dynamic_viscosity_lu','pressure':'gauge_pressure_lu','velocity':'velocity_lu','speed':'speed_lu','jz':'vertical_momentum_density_lu','dynamic_pressure':'dynamic_pressure_lu'}
    for i,t in enumerate(steps):
        arrays=[]
        for k,name in names.items():
            a=data[k][i];ncomp=3 if k=='velocity' else 1
            if ncomp==3:a=np.moveaxis(a,0,-1)
            arrays.append(f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">'+' '.join(format(v,'.17g') for v in a.ravel())+'</DataArray>')
        file=f'step_{t:04d}.vti';(folder/file).write_text(f'<?xml version="1.0"?><VTKFile type="ImageData" version="1.0" byte_order="LittleEndian"><ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1"><Piece Extent="{extent}"><CellData Vectors="velocity_lu" Scalars="phase_order_parameter">'+''.join(arrays)+'</CellData></Piece></ImageData></VTKFile>\n')
        entries.append(f'<DataSet timestep="{t*np.sqrt(c["gravity"]/c["H"]):.17g}" group="" part="0" file="vtk/{file}"/>')
    (Path(out)/'dam_break.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="1.0" byte_order="LittleEndian"><Collection>'+''.join(entries)+'</Collection></VTKFile>\n')
