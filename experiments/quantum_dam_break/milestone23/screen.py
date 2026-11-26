import numpy as np
from milestone23.solver import FaceDamBreak
from milestone22.solver import BalancedDamBreak

def spectrum(s,width,eps=1e-8):
 z,y,x=np.indices(s.shape);a=.5*(1-np.tanh((z+.5-4)/width));f,h=s.set_hydrostatic_background(a);v=f.ravel();J=np.empty((v.size,v.size))
 for j in range(v.size):
  p=v.copy();m=v.copy();p[j]+=eps;m[j]-=eps
  J[:,j]=(s.step(p.reshape(f.shape),h,'reference',freeze=True)[0]-s.step(m.reshape(f.shape),h,'reference',freeze=True)[0]).ravel()/(2*eps)
 e=np.linalg.eigvals(J);return float(max(abs(e))),int(sum(abs(e)>1+1e-5)),e
if __name__=='__main__':
 for width in [.1,.5,1.]:
  for fp,visc in [(False,'original'),(True,'original'),(False,'face'),(True,'face'),(True,'none')]:
   s=FaceDamBreak(shape=(8,2,2),gravity=1e-5,face_pressure=fp,viscous=visc);r,n,_=spectrum(s,width);print(width,fp,visc,r,n,flush=True)
