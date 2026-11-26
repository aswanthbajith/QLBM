"""Finite-difference Jacobian audit of frozen-interface hydrodynamic map.
Classical diagnostic only; not quantum circuit execution or a nonlinear proof.
The fixed residual is independent of f, so its derivative is zero.
"""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone22.solver import BalancedDamBreak
OUT=Path(__file__).parent/'results'

def jacobian(width,ratio,eps=1e-8):
    s=BalancedDamBreak(shape=(8,2,2),ratio=ratio,gravity=1e-5)
    z,y,x=np.indices(s.shape);a=.5*(1-np.tanh((z+.5-4)/width));f,h=s.set_hydrostatic_background(a)
    n=f.size;J=np.empty((n,n));v=f.ravel().copy()
    # Differentiate the actual corrected reference step at its fixed point.
    for j in range(n):
        p=v.copy();m=v.copy();p[j]+=eps;m[j]-=eps
        fp=s.step(p.reshape(f.shape),h,'reference',freeze=True)[0]
        fm=s.step(m.reshape(f.shape),h,'reference',freeze=True)[0]
        J[:,j]=(fp-fm).ravel()/(2*eps)
    eig=np.linalg.eigvals(J);order=np.argsort(-abs(eig));eig=eig[order]
    # The global pressure-population conservation gives a neutral eigenvalue.
    # Count modes above 1+1e-5 to avoid labelling finite-difference noise unstable.
    return dict(width=width,ratio=ratio,epsilon=eps,dimension=n,spectral_radius=float(abs(eig[0])),unstable_modes=int(np.count_nonzero(abs(eig)>1+1e-5)),dominant_real=float(eig[0].real),dominant_imag=float(eig[0].imag),conservation_jacobian_error=float(np.max(abs(J.sum(0)-1)))),eig

def main():
    rows=[];spectra={}
    for width,ratio in [(1.,1.),(.1,998/1.2),(.5,998/1.2),(1.,998/1.2)]:
        for eps in [1e-8,1e-9]:
            r,e=jacobian(width,ratio,eps);rows.append(r);spectra[f'{width}_{ratio}_{eps}']=e;print(r,flush=True)
    with (OUT/'stability.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'spectra.npz',**spectra)
    (OUT/'stability.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
