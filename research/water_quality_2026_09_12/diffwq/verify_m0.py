"""G1 gate: Python replicator (mode M0) versus the EPANET 2.3 engine, node by node,
time by time. Usage (from research/water_quality_2026_09_12):

    python -m diffwq.verify_m0 --set core --mode nextQ
    python -m diffwq.verify_m0 --set corpus --qstep 60 --ctol 0.001 --out g1_corpus.json
    python -m diffwq.verify_m0 --set variants --mode stepQ

'variants' re-configures the example networks through the API only (original INP
files are never modified) to reach the branches the shipped corpus does not exercise:
MIX2/FIFO/LIFO tanks, overflow, non-first-order and growth kinetics, zero-order wall
reaction, all four source types, negative junction demand, Ctol = 0.
"""
import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path
import numpy as np

from . import epanet_api as api
from .hydfile import read_hydfile
from .network import extract
from .replicate import Replicator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORE = {
    'Net1': 'EPANET-dev/example-networks/Net1.inp',
    'Net2': 'EPANET-dev/example-networks/Net2.inp',
    'Net3': 'EPANET-dev/example-networks/Net3.inp',
    'Jilin': 'networks/asce-tf-wdst/Jilin including water quality/Jilin including water quality.inp',
}


# ---- API-level re-configuration helpers for the 'variants' set ----
def tank_mix(tank_id, model, frac=None):
    def f(pr):
        i = pr.node_index(tank_id)
        pr.set_node_value(i, api.EN_MIXMODEL, model)
        if frac is not None:
            pr.set_node_value(i, api.EN_MIXFRACTION, frac)
    return f


def tank_overflow(tank_id, max_level):
    def f(pr):
        i = pr.node_index(tank_id)
        pr.set_node_value(i, api.EN_MAXLEVEL, max_level)
        pr.set_node_value(i, api.EN_CANOVERFLOW, 1)
    return f


def options(**kw):
    def f(pr):
        for key, val in kw.items():
            pr.set_option(getattr(api, 'EN_' + key), val)
    return f


def source(node_id, stype, strength, pat=0):
    def f(pr):
        i = pr.node_index(node_id)
        pr.set_node_value(i, api.EN_SOURCETYPE, stype)
        pr.set_node_value(i, api.EN_SOURCEQUAL, strength)
        pr.set_node_value(i, api.EN_SOURCEPAT, pat)
    return f


def base_demand(node_id, value):
    return lambda pr: pr.set_node_value(pr.node_index(node_id), api.EN_BASEDEMAND, value)


def all_links(kb=None, kw=None):
    def f(pr):
        for k in range(1, pr.count(api.EN_LINKCOUNT) + 1):
            if pr.link_type(k) > api.EN_PIPE:
                continue
            if kb is not None:
                pr.set_link_value(k, api.EN_KBULK, kb)
            if kw is not None:
                pr.set_link_value(k, api.EN_KWALL, kw)
    return f


def all_tanks_kb(kb):
    def f(pr):
        for i in range(1, pr.count(api.EN_NODECOUNT) + 1):
            if pr.node_type(i) == api.EN_TANK:
                pr.set_node_value(i, api.EN_TANK_KBULK, kb)
    return f


def init_qual(value, reservoirs=None):
    def f(pr):
        for i in range(1, pr.count(api.EN_NODECOUNT) + 1):
            v = reservoirs if (reservoirs is not None and pr.node_type(i) == api.EN_RESERVOIR) else value
            pr.set_node_value(i, api.EN_INITQUAL, v)
    return f


def qual_type(qt, trace_node=''):
    return lambda pr: pr.set_qual_type(qt, 'Chlorine', 'mg/L', trace_node)


def inp_replace(*pairs):
    """EN_setoption rejects negative values (error 213), so a negative (Michaelis-Menten)
    reaction order can only enter through an INP file: a temporary copy is written to the
    scratch directory with the given (old line, new line) substitutions."""
    def f(inp, workdir, name):
        text = Path(inp).read_text(encoding='utf-8', errors='replace')
        for old, new in pairs:
            if old not in text:
                raise ValueError(f'{Path(inp).name}: line {old!r} not found')
            text = text.replace(old, new)
        tmp = Path(workdir) / (name + '.inp')
        tmp.write_text(text, encoding='utf-8')
        return tmp
    return f


NET3_TANKS = [tank_mix('1', api.EN_MIX2, 0.5), tank_mix('2', api.EN_FIFO), tank_mix('3', api.EN_LIFO)]
# (name, base network, setup callables, {qstep, ctol, inp} overrides)
VARIANTS = [
    ('Net1_MIX2', 'Net1', [tank_mix('2', api.EN_MIX2, 0.35)], {}),
    ('Net1_FIFO', 'Net1', [tank_mix('2', api.EN_FIFO)], {}),
    ('Net1_LIFO', 'Net1', [tank_mix('2', api.EN_LIFO)], {}),
    ('Net1_MIX1_overflow', 'Net1', [tank_overflow('2', 122.0)], {}),
    ('Net1_MIX2_overflow', 'Net1', [tank_overflow('2', 122.0), tank_mix('2', api.EN_MIX2, 0.5)], {}),
    ('Net1_FIFO_overflow', 'Net1', [tank_overflow('2', 122.0), tank_mix('2', api.EN_FIFO)], {}),
    ('Net1_LIFO_overflow', 'Net1', [tank_overflow('2', 122.0), tank_mix('2', api.EN_LIFO)], {}),
    ('Net1_order2', 'Net1', [options(BULKORDER=2.0, TANKORDER=2.0)], {}),
    ('Net1_order1p5', 'Net1', [options(BULKORDER=1.5, TANKORDER=1.5)], {}),
    ('Net1_michaelis_menten', 'Net1', [options(CONCENLIMIT=0.3)],
     {'inp': inp_replace((' Order Bulk            \t1', ' Order Bulk            \t-1'),
                         (' Order Tank            \t1', ' Order Tank            \t-1'))}),
    ('Net1_growth_climit', 'Net1', [all_links(kb=0.5), all_tanks_kb(0.5), options(CONCENLIMIT=1.5)], {}),
    ('Net1_wall_order0', 'Net1', [options(WALLORDER=0.0), all_links(kw=-0.05)], {}),
    ('Net1_src_mass', 'Net1', [source('10', api.EN_MASS, 500.0)], {}),
    ('Net1_src_setpoint', 'Net1', [source('10', api.EN_SETPOINT, 1.5)], {}),
    ('Net1_src_flowpaced', 'Net1', [source('10', api.EN_FLOWPACED, 0.5)], {}),
    ('Net1_src_concen_reservoir_pat', 'Net1', [source('9', api.EN_CONCEN, 1.2, pat=1)], {}),
    ('Net1_src_concen_junction_inflow', 'Net1', [base_demand('32', -100.0), source('32', api.EN_CONCEN, 3.0)], {}),
    ('Net1_src_mass_tank', 'Net1', [source('2', api.EN_MASS, 200.0)], {}),
    ('Net1_ctol0', 'Net1', [], {'ctol': 0.0}),
    ('Net1_LIFO_ctol0_fine', 'Net1', [tank_mix('2', api.EN_LIFO)], {'ctol': 0.0, 'qstep': 60}),
    ('Net2_MIX2', 'Net2', [tank_mix('26', api.EN_MIX2, 0.25)], {}),
    ('Net2_FIFO', 'Net2', [tank_mix('26', api.EN_FIFO)], {}),
    ('Net2_LIFO', 'Net2', [tank_mix('26', api.EN_LIFO)], {}),
    ('Net2_ctol0_fine', 'Net2', [], {'ctol': 0.0, 'qstep': 60}),
    ('Net3_CHEM_mixed_tanks', 'Net3', [qual_type(api.EN_CHEM), init_qual(0.5, reservoirs=1.0),
                                       all_links(kb=-0.3), all_tanks_kb(-0.1)] + NET3_TANKS, {}),
    ('Net3_AGE_mixed_tanks', 'Net3', [qual_type(api.EN_AGE)] + NET3_TANKS, {}),
    ('Net3_TRACE_River_mixed_tanks', 'Net3', [qual_type(api.EN_TRACE, 'River')] + NET3_TANKS, {}),
]


def compare_one(lib, name, inp, mode, qstep, ctol, workdir, setup=()):
    with api.Project(lib, inp, Path(workdir) / (name + '.rpt')) as pr:
        for f in setup:
            f(pr)
        if qstep is not None:
            pr.set_time_param(api.EN_QUALSTEP, qstep)
        if ctol is not None:
            pr.set_option(api.EN_TOLERANCE, ctol)
        qualflag = pr.qual_info()[0]
        if qualflag == api.EN_NONE:
            return {'network': name, 'skipped': 'Quality NONE'}
        hyd = Path(workdir) / (name + '.hyd')
        pr.solve_hydraulics(hyd)
        net = extract(pr)
        trace = read_hydfile(hyd)
        t0 = time.perf_counter()
        t_ref, q_ref, ratio_ref = pr.run_quality_reference(mode)
        t_engine = time.perf_counter() - t0
    rep = Replicator(net, trace)
    t0 = time.perf_counter()
    t_rep, q_rep, ratio_rep = rep.run(mode)
    t_python = time.perf_counter() - t0
    q_rep_user = q_rep * net.ucf['QUALITY']
    if t_ref.shape != t_rep.shape or not np.array_equal(t_ref, t_rep):
        return {'network': name, 'error': 'time grids differ', 'engine_times': int(t_ref.size),
                'replicator_times': int(t_rep.size)}
    diff = np.abs(q_rep_user - q_ref)
    scale = max(float(np.max(np.abs(q_ref))), 1e-300)
    imax = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return {
        'network': name, 'path': str(inp), 'qualflag': int(qualflag), 'mode': mode,
        'nodes': net.nnodes, 'links': net.nlinks, 'tanks': int((~net.tank_is_reservoir[1:]).sum()),
        'mix_models': sorted(set(int(m) for m, r in zip(net.tank_mix[1:], net.tank_is_reservoir[1:]) if not r)),
        'reactflag': int(rep.reactflag), 'bulkorder': net.bulkorder, 'wallorder': net.wallorder,
        'qstep': rep.qstep, 'ctol_user': net.ctol * net.ucf['QUALITY'], 'duration': net.duration,
        'hydraulic_periods': len(trace.periods), 'output_times': int(t_ref.size),
        'max_abs_diff': float(diff.max()), 'max_ref': scale,
        'max_rel_diff': float(diff.max() / scale),
        'worst_time': int(t_ref[imax[0]]), 'worst_node': net.node_ids[imax[1]],
        'mass_balance_engine': ratio_ref, 'mass_balance_replicator': ratio_rep,
        'mass_balance_diff': abs(ratio_ref - ratio_rep),
        'segments_created': rep.next_id,
        'seconds_engine': t_engine, 'seconds_python': t_python,
        'source_types': sorted(set(int(s[0]) for s in net.sources.values())),
        'tank_final_volume_at_vmax': [bool(net.tank_v_hyd_final[i] >= net.tank_vmax[i])
                                      for i in range(1, net.ntanks + 1) if not net.tank_is_reservoir[i]],
        'branch_hits': {str(k): v for k, v in sorted(rep.hits.items(), key=lambda kv: str(kv[0]))},
        'climit_user': net.climit * net.ucf['QUALITY'], 'tankorder': net.tankorder,
    }


def scan_corpus(lib, workdir):
    """All INP files under networks/ whose QUALITY option is not NONE (extension matched case-insensitively)."""
    found = []
    for inp in sorted(p for p in (ROOT / 'networks').rglob('*') if p.is_file() and p.suffix.lower() == '.inp'):
        try:
            with api.Project(lib, inp, Path(workdir) / 'scan.rpt') as pr:
                if pr.qual_info()[0] != api.EN_NONE:
                    found.append((inp.stem, inp))
        except api.EpanetError:
            continue
    return found


RARE = ('mix1_overflow', 'mix2_', 'fifo', 'lifo', 'source', "('source'", 'junction_negative_demand',
        'noflowqual', 'react_clamp0')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='core', choices=['core', 'corpus', 'variants', 'all'])
    ap.add_argument('--mode', default='nextQ', choices=['nextQ', 'stepQ'])
    ap.add_argument('--qstep', type=int, default=None)
    ap.add_argument('--ctol', type=float, default=None)
    ap.add_argument('--exclude', default='', help='comma-separated network names to skip')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    workdir = tempfile.mkdtemp(prefix='diffwq_', dir=os.environ.get('TMPDIR'))
    # case = (name, inp, setup callables, per-case overrides)
    cases = [(k, ROOT / v, (), {}) for k, v in CORE.items()] if args.set in ('core', 'all') else []
    if args.set in ('corpus', 'all'):
        cases += [(n, p, (), {}) for n, p in scan_corpus(lib, workdir)]
    if args.set in ('variants', 'all'):
        cases += [(n, ROOT / CORE[b], s, o) for n, b, s, o in VARIANTS]
    excluded = set(filter(None, args.exclude.split(',')))
    results = []
    for name, inp, setup, over in cases:
        if name in excluded:
            continue
        qstep = over.get('qstep', args.qstep)
        ctol = over.get('ctol', args.ctol)
        try:
            if 'inp' in over:
                inp = over['inp'](inp, workdir, name)
            r = compare_one(lib, name, inp, args.mode, qstep, ctol, workdir, setup)
        except Exception as exc:          # keep the batch going; the report records the failure
            r = {'network': name, 'path': str(inp), 'error': f'{type(exc).__name__}: {exc}'}
        results.append(r)
        if 'max_rel_diff' in r:
            print(f"{name:32s} rel={r['max_rel_diff']:.2e} abs={r['max_abs_diff']:.2e} "
                  f"mb|d|={r['mass_balance_diff']:.1e} segs={r['segments_created']} "
                  f"engine={r['seconds_engine']:.3f}s python={r['seconds_python']:.2f}s", flush=True)
            if setup or over:
                rare = {k: v for k, v in r['branch_hits'].items() if k.startswith(RARE)}
                print(f"{'':32s} hits={rare} vmax_at_end={r['tank_final_volume_at_vmax']}", flush=True)
        else:
            print(f"{name:32s} {r.get('skipped') or r.get('error')}", flush=True)
    report = {'engine_version': lib.version(), 'library': str(lib.path), 'platform': platform.platform(),
              'mode': args.mode, 'qstep_override': args.qstep, 'ctol_override': args.ctol, 'results': results}
    out = Path(args.out) if args.out else HERE.parent / f'g1_{args.set}_{args.mode}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('report:', out)


if __name__ == '__main__':
    main()
