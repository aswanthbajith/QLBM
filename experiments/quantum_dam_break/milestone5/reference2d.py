"""Independent classical face-flux reference and exact Ritter solution.
Neither is called by the gate simulator.
"""
import numpy as np

def face(l,r,axis):
    def flux(v):
        h,x,y=v
        if h<=0:return np.zeros(3)
        q=v[axis+1]; f=v*q/h; f[axis+1]+=.5*h*h;return f
    return (flux(l)+flux(r)-3*(r-l))/16

def step(s,solid):
    s=np.asarray(s,float); z=s.copy(); ny,nx=solid.shape
    for y,x in np.argwhere(~solid):
        for axis in (0,1):
            for sign in (-1,1):
                xx=x+sign*(axis==0);yy=y+sign*(axis==1);v=s[y,x];w=v.copy()
                if 0<=xx<nx and 0<=yy<ny and not solid[yy,xx]:w=s[yy,xx]
                else:w[axis+1]*=-1
                z[y,x]+=face(w,v,axis) if sign==-1 else -face(v,w,axis)
    return z

def ritter(x,t,h0=1.5):
    x=np.asarray(x,float); c=np.sqrt(h0)
    if t==0:return np.where(x<0,h0,0),np.zeros_like(x)
    xi=x/t; fan=(xi>=-c)&(xi<2*c)
    h=np.where(xi<-c,h0,np.where(fan,(2*c-xi)**2/9,0))
    u=np.where(fan,2*(c+xi)/3,0)
    return h,u

def quantized_step(s,solid,scale):
    """Independent scalar reference for the quantized directional transfer."""
    s=np.asarray(s,dtype=np.int64); z=s.copy();ny,nx=solid.shape
    def nearest(a):return int(np.sign(a)*np.floor(abs(a)+.5+1e-12))
    for y,x in np.argwhere(~solid):
        h=s[y,x,0]
        if h==0:continue
        for axis in (0,1):
            for sign in (-1,1):
                v=s[y,x].astype(float)/scale; n=sign*v[axis+1]; t=v[2-axis]; d=np.zeros(3,dtype=np.int64)
                d[0]=int(np.floor((3*v[0]+n)*scale/16+1e-12))
                if d[0]==0:continue
                d[axis+1]=sign*nearest((3*n+n*n/v[0]+v[0]**2/2)*scale/16)
                d[2-axis]=nearest((3*t+n*t/v[0])*scale/16)
                z[y,x]-=d; xx=x+sign*(axis==0);yy=y+sign*(axis==1)
                if 0<=xx<nx and 0<=yy<ny and not solid[yy,xx]:z[yy,xx]+=d
                else:d[axis+1]*=-1;z[y,x]+=d
    return z
