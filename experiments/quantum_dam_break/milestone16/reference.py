"""Independent integer and floating-point validation, not used by gate evolution."""
import numpy as np

def round_ratio(n,d):
    n=int(n);return (1 if n>=0 else -1)*((2*abs(n)+d)//(2*d))

def integer_step(p,S):
    p=list(map(int,p));N=len(p)
    mu=[x**3-S*S*x-S*S*(p[(i-1)%N]+p[(i+1)%N]-2*x) for i,x in enumerate(p)]
    J=[round_ratio(mu[i]-mu[(i+1)%N],64*S*S) for i in range(N)]
    return np.array([x+J[(i-1)%N]-J[i] for i,x in enumerate(p)],np.int64)

def lap(p):return np.roll(p,-1)+np.roll(p,1)-2*p

def chemical(p):return p**3-p-lap(p)

def rhs(p):return lap(chemical(p))

def float_step(p):return p+rhs(p)/64

def energy(p):return float(np.sum((p*p-1)**2/4+(np.roll(p,-1)-p)**2/2))

def lap3(p):return sum(np.roll(p,sign,axis=a) for a in range(3) for sign in (-1,1))-6*p

def chemical3(p):return p**3-p-lap3(p)

def float_step3(p):return p+lap3(chemical3(p))/256

def integer_step3(p,S):
    # Independent vectorized reference. For selected widths and physical states
    # the integer intermediates fit in int64; no call to circuit implementation.
    mu=p**3-S*S*(p+lap3(p));out=p.copy();den=256*S*S
    for a in range(3):
        v=mu-np.roll(mu,-1,axis=a);j=np.sign(v)*((2*np.abs(v)+den)//(2*den));out+=np.roll(j,1,axis=a)-j
    return out

def energy3(p):return float(np.sum((p*p-1)**2/4+sum((np.roll(p,-1,axis=a)-p)**2/2 for a in range(3))))
