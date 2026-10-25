"""R2-5: hydraulic events in the trajectories behind the validation and application cases.

For each case the library solves the hydraulics once and the binary hydraulics file is read (hydfile.py). Counted over
consecutive hydraulic periods:
- flow reversals: a link whose flow changes sign with |Q| above the stagnant threshold in both periods,
- status changes: a link whose EPANET status code changes (pump switching, valve and check-valve action, controls),
- tank fill/drain switches: a sign change of a tank's level change between periods,
- junction mixing: junction-periods with two or more inflowing links,
- stagnant link-periods: |Q| at or below the stagnant threshold (0.005 gpm) of the quality solver.
Usage (from research/water_quality_2026_09_12): python -m diffwq.hyd_events --lib LIB --out hyd_events.json
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import numpy as np

from . import epanet_api as api
from .hydfile import read_hydfile
from .verify_m0 import CORE, ROOT
from .calib_data import prepare_inp

QSTAG = 0.005 / 448.831                     # 0.005 gpm in cfs, the internal flow unit of the hydraulics file
CASES = [('Net1', lambda wd: ROOT / CORE['Net1'], None, None), ('Net2', lambda wd: ROOT / CORE['Net2'], None, None),
         ('Net3', lambda wd: ROOT / CORE['Net3'], None, None), ('Jilin', lambda wd: ROOT / CORE['Jilin'], None, None),
         ('Net3 168 h (Exp1)', lambda wd: ROOT / CORE['Net3'], 168, None),
         ('Fairfield', lambda wd: prepare_inp('Fairfield', wd), None, None), ('Bellingham', lambda wd: prepare_inp('Bellingham', wd), None, None),
         ('North Penn', lambda wd: prepare_inp('NorthPenn', wd), None, None),
         ('L-Town 48 h', lambda wd: ROOT / 'networks/L-Town/L-TOWN.inp', 48, None),
         ('BWSN2 48 h', lambda wd: ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp', 48, 10),
         ('BWSN2 168 h', lambda wd: ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp', 168, 10)]


def events(lib, name, inp, hours, unbalanced, wd):
    with api.Project(lib, inp, Path(wd) / 'ev.rpt') as pr:
        if unbalanced:
            pr.set_option(api.EN_UNBALANCED, unbalanced)
        if hours:
            pr.set_time_param(api.EN_DURATION, hours * 3600)
        nn, nl = pr.count(api.EN_NODECOUNT), pr.count(api.EN_LINKCOUNT)
        ntype = np.array([0] + [pr.node_type(i) for i in range(1, nn + 1)])
        ltype = np.array([0] + [pr.link_type(k) for k in range(1, nl + 1)])
        ends = np.array([(0, 0)] + [pr.link_nodes(k) for k in range(1, nl + 1)])
        hyd = Path(wd) / 'ev.hyd'
        pr.solve_hydraulics(hyd)
        dur = pr.time_param(api.EN_DURATION)
    tr = read_hydfile(hyd)
    P = tr.periods
    Q = np.array([p.flow[1:] for p in P]); S = np.array([p.status[1:] for p in P]); H = np.array([p.head[1:] for p in P])
    moving = np.abs(Q) > QSTAG
    sgn = np.sign(Q) * moving
    rev = (sgn[1:] * sgn[:-1]) < 0
    stat = S[1:] != S[:-1]
    tanks = np.where(ntype[1:] == api.EN_TANK)[0]
    dh = np.diff(H[:, tanks], axis=0) if tanks.size else np.zeros((0, 0))
    dsg = np.sign(np.where(np.abs(dh) > 1e-6, dh, 0.0))
    tank_sw = int(((dsg[1:] * dsg[:-1]) < 0).sum()) if dsg.size else 0
    junc = np.where(ntype[1:] == api.EN_JUNCTION)[0] + 1
    inflow = np.zeros((len(P), nn + 1), dtype=np.int64)          # number of inflowing links per node and period
    up, dn = ends[1:, 0], ends[1:, 1]
    for t in range(len(P)):
        into = np.where(Q[t] > QSTAG, dn, np.where(Q[t] < -QSTAG, up, 0))
        np.add.at(inflow[t], into, 1)
    mix = inflow[:, junc] >= 2
    by = lambda m, code: int(m[:, ltype[1:] == code].sum())
    return {'case': name, 'nodes': nn, 'links': nl, 'pipes': int((ltype[1:] <= api.EN_PIPE).sum()), 'pumps': int((ltype[1:] == api.EN_PUMP).sum()),
            'valves': int((ltype[1:] > api.EN_PUMP).sum()), 'tanks': int(tanks.size), 'duration_h': dur / 3600, 'hydraulic_periods': len(P),
            'flow_reversals': int(rev.sum()), 'links_reversing': int(rev.any(axis=0).sum()),
            'status_changes': int(stat.sum()), 'status_changes_pumps': by(stat, api.EN_PUMP),
            'status_changes_valves': int(stat[:, ltype[1:] > api.EN_PUMP].sum()), 'status_changes_pipes': int(stat[:, ltype[1:] <= api.EN_PIPE].sum()),
            'tank_fill_drain_switches': tank_sw, 'junction_mixing_share': float(mix.mean()) if mix.size else 0.0,
            'max_inflows_at_a_junction': int(inflow[:, junc].max()) if junc.size else 0, 'stagnant_link_share': float((~moving).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lib = api.Library(a.lib)
    wd = tempfile.mkdtemp(prefix='diffwq_ev_', dir=os.environ.get('TMPDIR'))
    res = []
    for name, inp, hours, unb in CASES:
        r = events(lib, name, inp(wd), hours, unb, wd)
        res.append(r)
        print(json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps({'engine_version': lib.version(), 'stagnant_threshold_cfs': QSTAG, 'cases': res}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
