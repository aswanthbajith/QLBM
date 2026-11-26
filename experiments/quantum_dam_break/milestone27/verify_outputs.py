from pathlib import Path
from PIL import Image
from xml.etree import ElementTree as ET
import numpy as np
import json
p=Path(__file__).parent/'results/gate';d=np.load(p/'fields.npz');items=ET.parse(p/'dam_break.pvd').findall('.//DataSet');assert len(items)==len(d['steps'])==21
for i,v in enumerate(items):
    a=ET.parse(p/v.attrib['file']);assert a.find('.//ImageData').attrib['WholeExtent']=='0 32 0 16 0 16';arr={n.attrib['Name']:n for n in a.findall('.//CellData/DataArray')};assert len(arr)==9
    np.testing.assert_array_equal(np.fromstring(arr['velocity_lu'].text,sep=' ').reshape(16,16,32,3),np.moveaxis(d['velocity'][i],0,-1))
frames={}
for f in sorted(p.glob('*.gif')):
    with Image.open(f) as im:
        assert im.n_frames==21,(f,im.n_frames);frames[f.name]=im.n_frames
assert len(frames)==10
r=dict(vtk_snapshots=len(items),animations=frames,total_frames=sum(frames.values()))
(p.parent/'visualization_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
