"""Moment realizability and constructive positive equilibrium, floating research reference.
Not a quantum timestep. Speeds are in common physical lattice units, dt=1.
"""
import numpy as np
G=0.125
SETS={'D2Q9':np.arange(-1,1.01,1.),'D2Q25':np.arange(-1,1.01,.5),'D2Q81':np.arange(-1,1.01,.25),'D2Q289':np.arange(-2,2.01,.25),'D2Q1089':np.arange(-2,2.01,.125)}

def bounds(v,u):
    v=np.asarray(v,float);u=np.asarray(u,float)
    k=np.clip(np.searchsorted(v,u,side='right')-1,0,len(v)-2);a,b=v[k],v[k+1]
    return (a+b)*u-a*b,(v[0]+v[-1])*u-v[0]*v[-1]

def feasible(v,h,u,g=G):
    lo,hi=bounds(v,u);target=u*u+g*h/2
    return (h>0)&(u>=v[0]-1e-12)&(u<=v[-1]+1e-12)&(target>=lo-1e-12)&(target<=hi+1e-12)

def marginal(v,h,u,g=G,project=False):
    """Mixture of adjacent-speed and endpoint distributions. Exact moments if feasible.
    If project=True, clip variance to representable range (explicit model error).
    """
    v=np.asarray(v,float);h,u=np.broadcast_arrays(np.asarray(h,float),np.asarray(u,float));shape=h.shape
    if np.any(h<0) or np.any((h>0)&((u<v[0]-1e-10)|(u>v[-1]+1e-10))):raise ValueError('outside velocity support')
    u=np.clip(u,v[0],v[-1]);k=np.clip(np.searchsorted(v,u,side='right')-1,0,len(v)-2);a,b=v[k],v[k+1]
    lo,hi=bounds(v,u);t=u*u+g*h/2;bad=(h>0)&~feasible(v,h,u,g)
    if np.any(bad) and not project:raise ValueError('unrealizable moments')
    alpha=np.divide(np.clip(t,lo,hi)-lo,hi-lo,out=np.zeros_like(h),where=hi-lo>1e-14)
    p=np.zeros((h.size,len(v)));r=np.arange(h.size);p[r,k.ravel()]=(1-alpha).ravel()*(b-u).ravel()/(b-a).ravel();p[r,(k+1).ravel()]=(1-alpha).ravel()*(u-a).ravel()/(b-a).ravel()
    p[:,0]+=alpha.ravel()*(v[-1]-u).ravel()/(v[-1]-v[0]);p[:,-1]+=alpha.ravel()*(u-v[0]).ravel()/(v[-1]-v[0]);return p.reshape(shape+(len(v),))*h[...,None],bad

def equilibrium2d(v,h,ux,uy,g=G,project=False):
    px,bx=marginal(v,h,ux,g,project);py,by=marginal(v,h,uy,g,project)
    f=np.divide(px[..., :,None]*py[...,None,:],np.asarray(h)[...,None,None],out=np.zeros_like(px[..., :,None]*py[...,None,:]),where=np.asarray(h)[...,None,None]>0)
    return f,bx|by

def ritter(x,t,h0=1.5,g=G):
    c=np.sqrt(g*h0);xi=x/t;h=np.where(xi<-c,h0,np.where(xi>2*c,0,(2*c-xi)**2/(9*g)));u=np.where((xi>=-c)&(xi<=2*c),2*(c+xi)/3,0);return h,u

def kinetic_strip(v,steps=24,dx=.125,n=2048,omega=1.,signed=False):
    """Classical floating-point kinetic candidate screening, NOT circuit execution.
    Periodic distant interfaces; positivity projection only when target unrealizable.
    Signed D1Q3 equilibrium is unique; stop when depth negative/nonfinite.
    """
    x=(np.arange(n)+.5-n//2)*dx;h=np.where(x<0,1.5,0.);f,_=marginal(v,h,np.zeros_like(h),project=True);shifts=np.rint(v/dx).astype(int);assert np.allclose(shifts*dx,v)
    hist=[h.copy()];rows=[];m=f.sum()*dx
    for j in range(1,steps+1):
        h=f.sum(-1);q=f@v;u=np.divide(q,h,out=np.zeros_like(h),where=h>1e-14)
        if signed:
            p=h*u*u+G*h*h/2;eq=np.stack(((p-q)/2,h-p,(p+q)/2),-1);bad=(eq< -1e-12).any(-1)
        else: eq,bad=marginal(v,h,u,project=True)
        f=(1-omega)*f+omega*eq;f=np.stack([np.roll(f[:,k],s) for k,s in enumerate(shifts)],-1);h=f.sum(-1);sl=abs(x)<28;exact,_=ritter(x,j)
        valid=bool(np.isfinite(f).all() and h.min()>=-1e-10)
        rows.append(dict(step=j,rmse=float(np.sqrt(np.mean((h[sl]-exact[sl])**2))),mass_error=float(f.sum()*dx-m),min_depth=float(h.min()),min_population=float(f.min()),infeasible_cells=int(bad.sum()),valid=valid))
        hist.append(h.copy())
        if not valid:break
    return x,np.array(hist),rows
