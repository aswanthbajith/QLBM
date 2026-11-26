from pathlib import Path
import sys,xml.etree.ElementTree as ET,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from capture import derive,wall_metrics,OUT,BASE
from render import center_slice,mesh,limits

def test_derived_fields():
    a=np.array([[[0.,.5,1.]]]);rho=1+a;mu=.01*rho;p=a*.1;u=np.stack([a,2*a,-3*a]);d=derive(a,rho,mu,p,u)
    np.testing.assert_allclose(d['phi'],2*a-1)
    np.testing.assert_allclose(d['speed'],np.sqrt(14)*a)
    np.testing.assert_allclose(d['jz'],-3*rho*a)
    np.testing.assert_allclose(d['dynamic_pressure'],7*rho*a*a)

def test_exact_center_plane():
    v=np.arange(4*6*8.).reshape(4,6,8)
    np.testing.assert_array_equal(center_slice(v,0),(v[1]+v[2])/2)
    np.testing.assert_array_equal(center_slice(v,1),(v[:,2]+v[:,3])/2)

def test_surface_coordinates():
    a=np.broadcast_to(np.arange(4)[:,None,None]/3,(4,3,5));v,f=mesh(a)
    np.testing.assert_allclose(v[:,0],2.)
    assert v[:,1].min()==.5 and v[:,2].max()==4.5 and len(f)>0

def test_wall_metrics():
    a=np.zeros((2,2,3));a[0,0,-1]=.8;p=np.arange(12.).reshape(2,2,3);r=wall_metrics(a,p,np.zeros((2,2)))
    assert r['wall_wet_fraction']==.25 and r['wall_pressure_max']==11 and r['wall_pressure_sum']==26

def test_saved_trajectory():
    d=np.load(OUT/'fields.npz');old=np.load(BASE/'slow212_gate.npz');idx=[np.flatnonzero(old['steps']==t)[0] for t in d['steps']]
    np.testing.assert_array_equal(d['alpha'][:,:,2,:],old['alpha'][idx]);np.testing.assert_array_equal(d['phi'],2*d['alpha']-1)
    np.testing.assert_allclose(d['jz'],d['rho']*d['velocity'][:,2]);np.testing.assert_allclose(d['speed'],np.linalg.norm(d['velocity'],axis=1));np.testing.assert_allclose(d['dynamic_pressure'],.5*d['rho']*d['speed']**2)
    np.testing.assert_array_equal(d['pressure'][:,:,:,-1],d['wall_pressure'][d['steps']]);assert len(d['wall_pressure'])==1281
    r=json.loads((OUT/'report.json').read_text());assert max(r['comparison_max_abs'].values())==0
    assert r['first_wall_wetting_step']==np.flatnonzero(d['wall_alpha'].max((1,2))>=.5)[0]

def test_export_vtk():
    d=np.load(OUT/'fields.npz');pvd=ET.parse(OUT/'dam_break.pvd');items=pvd.findall('.//DataSet');assert len(items)==len(d['steps'])==21
    for i,item in enumerate(items):
        vtk=ET.parse(OUT/item.attrib['file']);assert vtk.find('.//ImageData').attrib['WholeExtent']=='0 32 0 4 0 16'
        arrays={a.attrib['Name']:a for a in vtk.findall('.//CellData/DataArray')};assert len(arrays)==9
        np.testing.assert_array_equal(np.fromstring(arrays['vertical_momentum_density_lu'].text,sep=' ').reshape(16,4,32),d['jz'][i]);np.testing.assert_array_equal(np.fromstring(arrays['velocity_lu'].text,sep=' ').reshape(16,4,32,3),np.moveaxis(d['velocity'][i],0,-1))

def test_fixed_signed_color_scale():
    a=np.array([-2.,.5]);assert limits('jz',a)==(-2,2);assert limits('phi',a)==(-1,1)
