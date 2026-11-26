"""Conservative FCT of D3Q7 phase fluxes, stationary closed rectangular tank.
Classical coupling guard, applied BEFORE the quantum transport permutation.
No pointwise phase clipping; low-order diffusion + limited antisymmetric corrections.
"""
import numpy as np


def links(shape):
    for d,axis in enumerate((2,1,0)):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(0,-1);b[axis]=slice(1,None)
        yield 1+2*d,tuple(a),tuple(b)


def limit(g, diffusion=0.125):
    g=np.asarray(g,dtype=float)
    if g.ndim!=4 or g.shape[0]!=7 or not np.isfinite(g).all():raise ValueError('finite D3Q7 population field required')
    if not 0<=diffusion<=1/6:raise ValueError('diffusion outside convexity bound')
    a=(1+g.sum(0))/2
    if a.min()<-1e-11 or a.max()>1+1e-11:raise ValueError('input phase outside bounds')
    low=a.copy();high=a.copy();entries=[]
    for i,l,r in links(a.shape):
        j=(g[i][l]-g[i+1][r])/2
        jl=diffusion*(a[l]-a[r]);low[l]-=jl;low[r]+=jl;high[l]-=j;high[r]+=j
        entries.append((i,l,r,j,jl,j-jl))
    # Do not alter a step that already meets its physical acceptance criterion.
    if high.min()>=-1e-14 and high.max()<=1+1e-14:
        return g.copy(),high,dict(limited_faces=0,flux_l1=0.,max_flux_change=0.,unlimited_min=float(high.min()),unlimited_max=float(high.max()))
    pos=np.zeros_like(a);neg=np.zeros_like(a)
    for i,l,r,j,jl,d in entries:
        pos[l]+=np.maximum(-d,0);neg[l]+=np.maximum(d,0)
        pos[r]+=np.maximum(d,0);neg[r]+=np.maximum(-d,0)
    rp=np.divide(np.maximum(1-low,0),pos,out=np.ones_like(a),where=pos>0).clip(0,1)
    rm=np.divide(np.maximum(low,0),neg,out=np.ones_like(a),where=neg>0).clip(0,1)
    out=g.copy();pred=low.copy();n=0;l1=0.;mx=0.
    for i,l,r,j,jl,d in entries:
        factor=np.where(d>=0,np.minimum(rm[l],rp[r]),np.minimum(rp[l],rm[r]))
        corr=factor*d;pred[l]-=corr;pred[r]+=corr
        delta=jl+corr-j
        # g is signed phi population, not alpha. Changing both opposite
        # populations by +/-delta changes alpha flux by delta. Rest compensation
        # maintains each cell's pre-stream collision phase sum.
        out[i][l]+=delta;out[i+1][r]-=delta;out[0][l]-=delta;out[0][r]+=delta
        n+=int(np.count_nonzero(abs(delta)>1e-14));l1+=float(abs(delta).sum());mx=max(mx,float(abs(delta).max(initial=0)))
    return out,pred,dict(limited_faces=n,flux_l1=l1,max_flux_change=mx,unlimited_min=float(high.min()),unlimited_max=float(high.max()))


def scalar_reference(g,diffusion=.125):
    """Independent scalar graph implementation (flat indices, explicit edges)."""
    a=(1+g.sum(0).ravel())/2;shape=g.shape[1:];v=g.reshape(7,-1);o=v.copy();edges=[]
    lo=a.copy();hi=a.copy()
    for k in range(a.size):
        loc=list(np.unravel_index(k,shape))
        for d,axis in enumerate((2,1,0)):
            if loc[axis]+1>=shape[axis]:continue
            dst=loc.copy();dst[axis]+=1;l=np.ravel_multi_index(tuple(dst),shape);i=1+2*d
            h=(v[i,k]-v[i+1,l])/2;b=diffusion*(a[k]-a[l]);edges.append((i,k,l,h,b,h-b))
            lo[k]-=b;lo[l]+=b;hi[k]-=h;hi[l]+=h
    if hi.min()>=-1e-14 and hi.max()<=1+1e-14:return g.copy(),hi.reshape(shape)
    plus=np.zeros_like(a);minus=np.zeros_like(a)
    for _,k,l,_,_,d in edges:
        if d>=0:minus[k]+=d;plus[l]+=d
        else:plus[k]-=d;minus[l]-=d
    for i,k,l,h,b,d in edges:
        donor,receiver=(k,l) if d>=0 else (l,k)
        fac=min(1,max(0,lo[donor])/minus[donor] if minus[donor] else 1,max(0,1-lo[receiver])/plus[receiver] if plus[receiver] else 1)
        delta=b+fac*d-h;o[i,k]+=delta;o[i+1,l]-=delta;o[0,k]-=delta;o[0,l]+=delta
    return o.reshape(g.shape),None
