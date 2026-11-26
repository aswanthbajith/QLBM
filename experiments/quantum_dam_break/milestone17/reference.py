"""Independent integer and floating point implementations of reduced CH/shear model."""
import numpy as np
from milestone16.reference import energy3
DT=1/256

def mode(shape):return np.broadcast_to(np.array([1,1,-1,-1])[None,:,None],shape)
def lap(p):return sum(np.roll(p,d,axis=i) for i in range(3) for d in (-1,1))-6*p

def rr(x,d):
    # exact nearest, ties away from zero; int64 safe on tested physical range
    return np.sign(x)*((2*np.abs(x)+d)//(2*d))

def integer_step(p,a,S,feedback=True):
    m=rr(p**3-S*S*(p+lap(p)),S*S);s=mode(p.shape);q=p.copy()
    for i in range(3):
        right=np.roll(p,-1,axis=i);diff=m-np.roll(m,-1,axis=i)
        adv=(p+right)*a*s if i==2 else 0
        j=rr(2*S*diff+adv,512*S);q+=np.roll(j,1,axis=i)-j
    F=int(np.sum(s*(p+np.roll(p,-1,axis=2))*(m-np.roll(m,-1,axis=2))))
    # Python integer arithmetic for scalar update.
    v=(F if feedback else 0)-p.size*S*int(a)//2;d=512*p.size*S
    a1=int(a)+(1 if v>=0 else -1)*((2*abs(v)+d)//(2*d))
    return q,a1,F

def float_rhs(p,a,feedback=True):
    m=p**3-p-lap(p);s=mode(p.shape);r=lap(m);face=(p+np.roll(p,-1,axis=2))/2
    flux=face*a*s;r+=np.roll(flux,1,axis=2)-flux
    F=face*(m-np.roll(m,-1,axis=2))*s
    return r,float(F.mean() if feedback else 0)-a/4

def float_step(p,a,feedback=True):
    r,b=float_rhs(p,a,feedback);return p+DT*r,a+DT*b

def balance(p,a):
    m=p**3-p-lap(p);s=mode(p.shape);face=(p+np.roll(p,-1,axis=2))/2
    flux=face*a*s;adv=np.roll(flux,1,axis=2)-flux
    cap=float(np.sum(a*s*face*(m-np.roll(m,-1,axis=2))))
    work=float(np.sum(m*adv));diss=float(sum(np.sum((np.roll(m,-1,axis=i)-m)**2) for i in range(3)))
    r,b=float_rhs(p,a);total=float(np.sum(m*r)+p.size*a*b)
    return dict(advective_work=work,capillary_work=cap,exchange_residual=work+cap,dissipation=diss,viscous_dissipation=p.size*a*a/4,energy_rate_residual=total+diss+p.size*a*a/4)

def total_energy(p,a):return energy3(p)+p.size*a*a/2
