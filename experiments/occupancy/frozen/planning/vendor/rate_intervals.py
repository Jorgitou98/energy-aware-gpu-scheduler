"""Direct interval compiler extracted unchanged from the validated model study.

The original study executes a report at import time. This standalone extraction
reuses its numerical functions without running that study or touching archives.
Profiles use B and points=[[rate, execution_seconds, execution_scv], ...].
Piecewise-linear L and SCV produce an SLO polynomial of degree at most five.
No extrapolation is allowed. Source: work/model_current_20261019/study.py.
"""
import numpy as np
from numpy.polynomial import Polynomial

SOURCE_SHA256 = 'b7047e5f052adbba53799405c8d88444d72658f763e254b1720e1fd20c978ccb'

def response_values(profile,rates,clamp=False):
    x=np.asarray(rates,dtype=float); pts=np.asarray(profile['points']); b=profile['B']
    if not clamp:assert np.all((x>=pts[0,0]-1e-10)&(x<=pts[-1,0]+1e-10))
    L=np.interp(x,pts[:,0],pts[:,1]); C=np.interp(x,pts[:,0],pts[:,2]);rho=x*L/b
    with np.errstate(divide='ignore',invalid='ignore'):
        R=(b-1)/(2*x)+L+(1/b+C)*L*rho/(2*(1-rho))
    return np.where((rho<1)&(x>0),R,np.inf),L,rho

def roots_unit(poly):
    result=[]
    for root in poly.roots():
        if abs(root.imag)<=1e-8 and -1e-10<=root.real<=1+1e-10:
            v=min(1.,max(0.,float(root.real)))
            if not any(abs(v-z)<1e-9 for z in result):result.append(v)
    return sorted(result)

def merge_parts(parts):
    merged=[]
    for a,b in sorted(parts):
        if merged and a<=merged[-1][1]+1e-9*max(1,abs(a)):
            merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged

def direct_intervals(profile,D):
    b=profile['B']; parts=[]; segments=[]
    for left,right in zip(profile['points'],profile['points'][1:]):
        a,L0,C0=left; z,L1,C1=right
        T=Polynomial([a,z-a]); L=Polynomial([L0,L1-L0]); C=Polynomial([C0,C1-C0])
        stable=b-T*L
        F=(b-1)*stable+2*T*(L-D)*stable+(Polynomial([1/b])+C)*T*T*L*L
        slo_roots=roots_unit(F); stability_roots=roots_unit(stable)
        boundaries=sorted(set([0.,1.]+slo_roots+stability_roots))
        admissible=[]
        for lo,hi in zip(boundaries,boundaries[1:]):
            if hi-lo<1e-12:continue
            midpoint=float(T((lo+hi)/2));R,_,rho=response_values(profile,[midpoint])
            if rho[0]<1 and R[0]<=D:
                admissible.append([float(T(lo)),float(T(hi))])
        # Preserve an isolated feasible tangent point if it occurs.
        for u in boundaries:
            rate=float(T(u));R,_,rho=response_values(profile,[rate])
            if rho[0]<1 and abs(R[0]-D)<1e-10 and not any(lo-1e-8<=rate<=hi+1e-8 for lo,hi in admissible):
                admissible.append([rate,rate])
        parts.extend(admissible)
        valid_roots=[]
        for u in slo_roots:
            rate=float(T(u));R,_,rho=response_values(profile,[rate])
            if rho[0]<1:
                assert abs(float(R[0])-D)<1e-7,(profile['configuration'],D,rate,R[0])
                valid_roots.append(dict(rate=rate,response_residual_s=float(R[0])-D))
        segments.append(dict(domain=[a,z],polynomial_degree=int(F.degree()),
                             slo_roots=valid_roots,stability_roots=[float(T(u)) for u in stability_roots],
                             feasible_parts=admissible))
    return merge_parts(parts),segments

def member(rates,intervals,tolerance=1e-9):
    rates=np.asarray(rates);result=np.zeros(rates.shape,dtype=bool)
    for a,b in intervals:result|=(rates>=a-tolerance)&(rates<=b+tolerance)
    return result
