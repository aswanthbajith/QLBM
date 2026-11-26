import numpy as np
import pytest
from milestone25.run import configuration,initial_fraction
from milestone25.scaling import nominal,si_mapping
from milestone24.solver import LongDamBreak,hydrostatic

@pytest.mark.parametrize('r,slow',[(1,2),(1,4),(2,1),(2,2)])
def test_matched_dimensionless_numbers(r,slow):
    c=configuration(.005,r,slow);b=configuration(.005)
    for k in ('Re','Bo','t_star','Cahn'):assert nominal(c)[k]==pytest.approx(nominal(b)[k],rel=1e-14)
    assert nominal(c)['characteristic_Mach']==pytest.approx(nominal(b)['characteristic_Mach']/slow)
    m,n=si_mapping(c),si_mapping(b)
    for k in ('duration_s','modeled_nu_m2_s','modeled_sigma_N_m','target_Re','target_Bo'):assert m[k]==pytest.approx(n[k],rel=1e-14)

@pytest.mark.parametrize('r',[1,2])
def test_initial_and_scaling(r):
    c=configuration(.01,r);a=initial_fraction(c)
    assert a.shape==c['shape'] and np.all((a>=0)&(a<=1))
    assert np.max(np.ptp(a,axis=1))==0
    m=si_mapping(c);assert m['target_tau_liquid']>.5 and m['viscosity_excess']>1
    assert m['duration_s']==pytest.approx(c['steps']*m['dt_s'])

def test_smaller_timestep_coupled_gate_reference():
    c=configuration(.0025,1,2);c['shape']=(8,2,16)
    s=LongDamBreak(**{k:c[k] for k in ('shape','gravity','nu_liquid','sigma','xi','tau_phase')});a=initial_fraction(c);f,h=s.initialize(a,hydrostatic(a,s.mat(a)[0],s.gravity));s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),s.shape));f0,h0=f.copy(),h.copy();m=h.sum()
    for _ in range(12):f,h=s.step(f,h,'gate');f0,h0=s.step(f0,h0,'reference')
    assert np.max(abs(f-f0))<1e-13 and np.max(abs(h-h0))<1e-13
    assert abs(h.sum()-m)<1e-10 and h.sum(0).min()>=-1e-12 and h.sum(0).max()<=1+1e-12

def test_invalid_inputs():
    with pytest.raises(ValueError):configuration(-1)
    with pytest.raises(ValueError):configuration(.01,3)
    with pytest.raises(ValueError):si_mapping(configuration(),H_m=0)
