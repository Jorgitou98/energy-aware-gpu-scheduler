"""Compile a finite-window q95 admission curve into ordinary rate intervals.

The full measured response model remains a separate requirement. The offline
q95/Rnom factor is piecewise linear in rate; within each grid segment the
interpolated batch mean and SCV are also linear. All roots on that segment can
therefore be recovered from a degree-at-most-six polynomial. No numeric grid
of MILP decision variables, no runtime queue evaluation, and no extrapolation.
"""
from functools import lru_cache
import csv
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from rate_intervals import direct_intervals, merge_parts, response_values, roots_unit

GRID = Path(__file__).resolve().parents[1] / 'grid_nodes_unified.csv'


@lru_cache(maxsize=1)
def _nodes():
    groups = {}
    with GRID.open(newline='') as stream:
        for row in csv.DictReader(stream):
            rate = float(row['rate'])
            nominal = float(row['nominal'])
            if nominal <= 0 or not np.isfinite(nominal):
                raise ValueError('Invalid nominal response at CPU-grid node')
            groups.setdefault(row['config'], []).append((rate, float(row['q95_1024'])/nominal))
    for config, pairs in groups.items():
        pairs.sort()
        if len(pairs) < 2 or any(right[0] <= left[0] for left,right in zip(pairs,pairs[1:])):
            raise ValueError(f'Duplicate or unordered grid rates for {config}')
    return groups


def _intersect(first, second):
    return merge_parts([[max(a,c),min(b,d)] for a,b in first for c,d in second
                        if max(a,c) <= min(b,d) + 1e-10])


@lru_cache(maxsize=8192)
def _guard_intervals(config, batch, deadline, profile_points):
    nodes = _nodes()[config]
    domain = [float(profile_points[0][0]), float(profile_points[-1][0])]
    if nodes[0][0] > domain[0]+1e-9:
        raise ValueError(f'CPU grid omits calibrated lower rate: {config}')
    segments=[]
    parts=[]
    for (left,k0),(right,k1) in zip(nodes,nodes[1:]):
        if right <= left:
            continue
        # The grid includes every original profiling knot, so L and C2 are
        # affine on each [left,right] segment.
        index = max(i for i,point in enumerate(profile_points[:-1]) if point[0]<=left+1e-9)
        a,L0,C0 = profile_points[index]
        z,L1,C1 = profile_points[index+1]
        if right > z+1e-8:
            raise ValueError(f'CPU grid crosses a profiling knot: {config}')
        T=Polynomial([left,right-left]); L=Polynomial([L0+(L1-L0)*(left-a)/(z-a),
                                                       (L1-L0)*(right-left)/(z-a)])
        C=Polynomial([C0+(C1-C0)*(left-a)/(z-a),
                      (C1-C0)*(right-left)/(z-a)])
        K=Polynomial([k0,k1-k0]); stable=Polynomial([batch])-T*L
        numerator=((batch-1)*stable+2*T*L*stable+
                   (Polynomial([1/batch])+C)*T*T*L*L)
        crossing=K*numerator-2*T*deadline*stable
        boundaries=sorted(set([0.,1.]+roots_unit(crossing)+roots_unit(stable)))
        for lo,hi in zip(boundaries,boundaries[1:]):
            if hi-lo < 1e-12:
                continue
            u=(lo+hi)/2
            r=float(T(u)); latency=float(L(u)); c2=float(C(u));factor=float(K(u))
            rho=r*latency/batch
            if rho < 1 and factor*float(response_values(
                    {'B':batch,'points':profile_points},[r])[0][0]) <= deadline+1e-10:
                parts.append([float(T(lo)),float(T(hi))])
        segments.append(dict(domain=[left,right], polynomial_degree=int(crossing.degree()),
                             crossing_roots=[float(T(v)) for v in roots_unit(crossing)],
                             stability_roots=[float(T(v)) for v in roots_unit(stable)]))
    return merge_parts(parts),segments


def guarded_intervals(profile, deadline):
    """Original stability/mean SLO intersected with offline window q95 <= D."""
    original,original_segments=direct_intervals(profile,deadline)
    if not original:
        return [],dict(original=original_segments,guard=[])
    config=profile['physical_config_id']
    batch=int(profile['B'])
    points=tuple(tuple(map(float,point)) for point in profile['points'])
    guarded,segments=_guard_intervals(config,batch,float(deadline),points)
    # The finite-window grid stops at 95% of measured busy capacity. Nominal
    # intervals may extend farther, but rates without window simulations are
    # conservatively excluded rather than extrapolated or treated as errors.
    return _intersect(original,guarded),dict(
        original=original_segments,guard=segments,
        simulation_domain=[_nodes()[config][0][0],_nodes()[config][-1][0]])
