"""Independent comparison, SI scaling, and spatial verification (classical only)."""
import numpy as np
from milestone5 import reference2d as base

def drag_ticks(q,numerator=7,shift=3):
    d=1<<shift
    return np.sign(q)*((np.abs(q)*numerator+d//2)//d)

def quantized_step(s,mask,scale,numerator=7,shift=3):
    z=base.quantized_step(s,mask,scale)
    z[:,:,1:]=drag_ticks(z[:,:,1:],numerator,shift);return z

def float_step(s,mask,beta):
    z=base.step(s,mask);z[:,:,1:]*=beta;return z

def planar_reference(n,steps):
    """Vectorized independent planar Rusanov reference, closed walls."""
    s=np.zeros((n,2));s[:n//2,0]=1.5
    for _ in range(steps):
        ghost=np.vstack((s[0]*[1,-1],s,s[-1]*[1,-1]));h,q=ghost.T
        v=np.divide(q,h,out=np.zeros_like(q),where=h>0)
        f=np.stack((q,q*v+.5*h*h),axis=-1)
        flux=(f[:-1]+f[1:]-3*(ghost[1:]-ghost[:-1]))/16
        s-=flux[1:]-flux[:-1]
    return s
