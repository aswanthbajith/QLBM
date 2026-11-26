from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone18.transport import C19
OUT=Path(__file__).parent/'results'
h=np.load(OUT/'history.npz');solid=h['solid'];nz,ny,nx=solid.shape;extent=f'0 {nx} 0 {ny} 0 {nz}'
for k in [0,8,16,24,32]:
    f=h['f'][k];rho=f.sum(0);p=h['g'][k].sum(0);j=np.einsum('id,izyx->dzyx',C19,f);u=np.divide(j,rho[None],out=np.zeros_like(j),where=rho[None]>1e-12)
    fields=[('density_lu',rho,1),('phase_order_parameter',p,1),('velocity_raw_moment_lu',u.transpose(1,2,3,0),3),('solid_mask',solid.astype(float),1)]
    data='\n'.join(f'<DataArray type="Float64" Name="{n}" NumberOfComponents="{nc}" format="ascii">'+ ' '.join(f'{v:.17g}' for v in arr.ravel())+'</DataArray>' for n,arr,nc in fields)
    (OUT/f'step_{k:03}.vti').write_text(f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian"><ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1"><Piece Extent="{extent}"><CellData>{data}</CellData></Piece></ImageData></VTKFile>\n')
