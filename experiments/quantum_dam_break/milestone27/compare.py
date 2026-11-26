"""Comparison of finite-width run with matched reference and gravity control."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent/'results'
def run():
    cases={k:json.loads((OUT/k/'report.json').read_text()) for k in ('gate','reference','zero_gravity')}
    g,r=[np.load(OUT/k/'fields.npz') for k in ('gate','reference')]
    assert np.array_equal(g['steps'],r['steps'])
    errors={k:float(np.max(abs(g[k]-r[k]))) for k in ('alpha','velocity','pressure','rho')}
    rmse=float(np.sqrt(np.mean((g['alpha'][-1]-r['alpha'][-1])**2)))
    gp,rp=[np.load(OUT/k/'final_populations.npz') for k in ('gate','reference')]
    pop={k:float(np.max(abs(gp[k]-rp[k]))) for k in ('f','h')}
    assert max(errors.values())<1e-10 and max(pop.values())<1e-10
    assert all(c['status']=='completed' for c in cases.values())
    assert cases['gate']['min_alpha']>=-1e-10 and cases['gate']['max_alpha']<=1+1e-10
    assert cases['gate']['final']['floor_width']>cases['gate']['initial']['floor_width'] and cases['gate']['max_abs_uy']>1e-3
    report={'cases':cases,'max_absolute_field_error':errors,'final_alpha_rmse':rmse,'final_population_error':pop,'reference_scope':'Independent collision algebra and slice-based transport; shared equations and vector FCT. Scalar FCT cross-check in tests.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    labels={'gate':'Gate subroutines','reference':'Classical reference','zero_gravity':'Zero gravity control'}
    for k in cases:
        with (OUT/k/'history.csv').open() as f:rows=list(csv.DictReader(f))
        ts=np.array([float(q['step']) for q in rows])*np.sqrt(cases['gate']['configuration']['gravity']/cases['gate']['configuration']['H'])
        for ax,field,title in zip(axs.flat,['front','floor_width','com_z','liquid_weighted_abs_uy'],['Front position [cells]','Wetted floor span [cells]','Liquid centre height [lu]','Liquid-weighted |uy| [lu]']):
            ax.plot(ts,[float(q[field]) for q in rows],label=labels[k],ls={'gate':'-','reference':'--','zero_gravity':':'}[k]);ax.set(title=title,xlabel='Matched reference-gravity time t*');ax.grid(alpha=.25)
    axs[0,0].legend();fig.suptitle('M27 finite-width 3D dam release · matched numerical comparison',fontweight='bold');fig.savefig(OUT/'benchmark.png',dpi=140);plt.close(fig)
    print('Comparison',errors,rmse,pop,flush=True)
if __name__=='__main__':run()
