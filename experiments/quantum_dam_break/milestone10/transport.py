"""Multispeed streaming permutation; conservative whole-link reflection.
A link is blocked if its symmetric bounding rectangle intersects a solid.
Overconservative near corners; no wall intersection timing modeled.
"""
import numpy as np
from quantum_core import transpositions,gate_pairs,apply_gates

def permutation(solid,speeds):
    solid=np.asarray(solid,bool);ny,nx=solid.shape;n=nx*ny;vel=[(int(x),int(y)) for y in speeds for x in speeds];opp=[vel.index((-x,-y)) for x,y in vel];p=np.arange(n*len(vel));
    for i,(cx,cy) in enumerate(vel):
        for y,x in np.argwhere(~solid):
            a,b=x+cx,y+cy;k=i*n+y*nx+x
            blocked=not(0<=a<nx and 0<=b<ny)
            if not blocked:blocked=solid[min(y,b):max(y,b)+1,min(x,a):max(x,a)+1].any()
            p[k]=opp[i]*n+y*nx+x if blocked else i*n+b*nx+a
    assert np.array_equal(np.sort(p),np.arange(len(p)));return p

def apply_word_swaps(f,p):
    z=np.asarray(f).copy()
    for a,b in transpositions(p):z[a],z[b]=z[b],z[a]
    return z
