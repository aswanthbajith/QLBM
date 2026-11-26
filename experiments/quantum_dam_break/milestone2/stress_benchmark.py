"""Exact local kinetic stress relaxation from the quantum channel."""
from pathlib import Path
import csv
import numpy as np
from collision import transition, STRESS

def run(out):
    out=Path(out); out.mkdir(exist_ok=True,parents=True)
    stress=STRESS[:,0,0]-STRESS[:,1,1]; rows=[]
    for omega in [0,.2,.5,.7,1]:
        p=np.zeros(512); p[10]=1; T=transition(omega)
        for t in range(9):
            rows.append({'omega':omega,'collision_step':t,'measured_expected_stress_difference_lu':float(stress@p),'analytic_stress_difference_lu':2*(1-omega)**t})
            p=T@p
    with (out/'stress_relaxation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    return max(abs(r['measured_expected_stress_difference_lu']-r['analytic_stress_difference_lu']) for r in rows)
if __name__=='__main__': print('max error:',run(Path(__file__).parent/'results'))
