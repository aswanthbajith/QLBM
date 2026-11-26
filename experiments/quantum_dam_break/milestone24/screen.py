from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from milestone24.solver import LongDamBreak,hydrostatic
from milestone19.solver import initial

def trial(g,span,steps=640,shape=(16,4,32),background=True):
 s=LongDamBreak(shape=shape,gravity=g,sigma=1e-5,span_periodic=span);_,h=initial(shape);a=(1+h.sum(0))/2;rho=s.mat(a)[0];f,h=s.initialize(a,hydrostatic(a,rho,g))
 if background:s.set_hydrostatic_background(np.broadcast_to(a.mean((1,2),keepdims=True),shape))
 z,y,x=np.indices(shape);mass=a.sum();status='completed';rows=[]
 for t in range(steps+1):
  try:
   if t:f,h=s.step(f,h,'fast')
   a,rho,mu,nu,p,u,*_=s.decode(f,h);speed=float(np.linalg.norm(u,axis=0).max());front=int(np.where(a>.5)[2].max());rows.append([t,speed,float((a*x).sum()/mass),float((a*z).sum()/mass),front])
   if t%96==0:print(g,span,background,rows[-1],flush=True)
   if not np.isfinite(speed) or speed>.1:status='speed guard';break
  except (ValueError,RuntimeError) as e:status=str(e);break
 print('END',g,span,background,rows[-1],status,flush=True);return dict(gravity=g,span_periodic=span,background=background,status=status,rows=rows)
if __name__=='__main__':
 out=[]
 for g,span,bg in [(.0005,True,True),(.00025,True,True),(.0005,False,True),(.00025,True,False),(.0001,True,True)]:
  out.append(trial(g,span,background=bg));Path(__file__).with_name('results').joinpath('screen.json').write_text(json.dumps(out))
