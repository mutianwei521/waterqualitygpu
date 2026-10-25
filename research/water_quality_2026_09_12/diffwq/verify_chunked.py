"""Checks of the time-chunked replay (chunked.py) against the fused engine on the same recorded schedule: forward snapshots
and gradients with respect to kb, kw, tank_kb, c0 and src (one parameter set and a batch of three) for several chunk
counts, and the rows of the source map (rows_src, adjoint only) against autograd rows of the fused engine.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.verify_chunked --device cpu --out chunked_check.json
"""
import argparse
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

from . import epanet_api as api
from .chunked import ChunkedWQ
from .hydfile import read_hydfile
from .network import extract, SECperDAY
from .replicate import Replicator
from .torch_wq import TorchWQ
from .verify_m0 import CORE, ROOT, VARIANTS, NET3_TANKS, all_links, all_tanks_kb, init_qual, qual_type
from .verify_torch import open_case

PICK = ('Net1_MIX2_overflow', 'Net1_LIFO', 'Net1_src_setpoint', 'Net1_src_mass_tank', 'Net1_ctol0', 'Net2_LIFO',
        'Net3_CHEM_mixed_tanks', 'Net3_AGE_mixed_tanks', 'Net3_TRACE_River_mixed_tanks')
CASES = [(k, k, [], {}) for k in ('Net1', 'Net2', 'Net3')] + [v for v in VARIANTS if v[0] in PICK] + [
    ('Net2_react', 'Net2', [all_links(kb=-0.5, kw=-0.2), all_tanks_kb(-0.5)], {}),
    ('Net3_CHEM_kw_mixed_tanks', 'Net3', [qual_type(api.EN_CHEM), init_qual(0.5, reservoirs=1.0),
                                          all_links(kb=-0.3, kw=-0.2), all_tanks_kb(-0.1)] + NET3_TANKS, {})]
CHUNKS = (1, 2, 5, 17)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = max(float(np.abs(a).max(initial=0.0)), float(np.abs(b).max(initial=0.0)), 1e-300)
    return float(np.abs(a - b).max(initial=0.0) / den)


def theta_of(net, dev, batch):
    t = lambda v: torch.tensor(np.asarray(v, dtype=np.float64), device=dev)
    th = {'kb': t(net.kb * SECperDAY), 'kw': t(net.kw * SECperDAY), 'tank_kb': t(net.tank_kb * SECperDAY),
          'c0': t(net.c0 * net.ucf['QUALITY']), 'src': t([net.sources.get(i, (0, 0.0, 0))[1] for i in range(net.nnodes + 1)])}
    if batch > 1:                                                   # perturbed parameter sets (kb, c0) in a batch
        g = np.random.default_rng(1)
        th['kb'] = th['kb'].unsqueeze(0) * t(1.0 + 0.2 * g.standard_normal((batch, 1)))
        th['c0'] = th['c0'].unsqueeze(0) * t(1.0 + 0.2 * g.uniform(size=(batch, 1)))
    return {k: v.clone().requires_grad_(True) for k, v in th.items()}


def fwd_bwd(eng, th, w):
    th = {k: v.detach().clone().requires_grad_(True) for k, v in th.items()}
    out = eng.forward(**th)
    (out * w).sum().backward()
    return out.detach().cpu().numpy(), {k: (v.grad.cpu().numpy() if v.grad is not None else np.zeros(tuple(v.shape))) for k, v in th.items()}


def check_case(lib, name, base, setup, over, args, wd):
    inp = str(ROOT / CORE[base])
    with open_case(lib, name, inp, setup, over.get('qstep'), over.get('ctol'), wd) as pr:
        if pr.qual_info()[0] == api.EN_NONE:
            return {'case': name, 'skipped': 'Quality NONE'}
        hyd = Path(wd) / (name + '.hyd')
        pr.solve_hydraulics(hyd)
        net, trace = extract(pr), read_hydfile(hyd)
    rep = Replicator(net, trace, record=True)
    rep.run('nextQ')
    dev = torch.device(args.device)
    ref = TorchWQ(net, rep.schedule, rep.reactflag, device=dev, dtype=torch.float64, engine='fused', spmm_min_batch=0)
    R = {'case': name, 'terms': int(ref.T_total), 'state_slots': int(ref.fsch.M), 'runs': []}
    for batch in (1, 3):
        th = theta_of(net, dev, batch)
        n_out = ref.forward(**{k: v.detach() for k, v in th.items()}).shape[1]
        w = torch.tensor(np.random.default_rng(0).uniform(0.5, 1.5, size=(batch, n_out, net.nnodes + 1)), device=dev)
        o_ref, g_ref = fwd_bwd(ref, th, w)
        for nc in CHUNKS:
            eng = ChunkedWQ(net, rep.schedule, rep.reactflag, device=dev, dtype=torch.float64, n_chunks=nc)
            o, g = fwd_bwd(eng, th, w)
            R['runs'].append({'batch': batch, **eng.chunk_info(), 'forward': rel(o, o_ref),
                              **{f'grad_{k}': rel(g[k], g_ref[k]) for k in g}})
    R['max_rel'] = max(v for r in R['runs'] for k, v in r.items() if k == 'forward' or k.startswith('grad_'))
    return R


def check_rows(lib, args, wd, base='Net3', m=24):
    """Exp2 setting on a small network: conservative CHEM, virtual MASS sources at every junction, K source slots."""
    inp = str(ROOT / CORE[base])
    setup = [qual_type(api.EN_CHEM), init_qual(0.0, reservoirs=0.0), all_links(kb=0.0, kw=0.0), all_tanks_kb(0.0)]
    with open_case(lib, base + '_rows', inp, setup, None, 0.0, wd) as pr:
        hyd = Path(wd) / (base + '_rows.hyd')
        pr.solve_hydraulics(hyd)
        net, trace = extract(pr), read_hydfile(hyd)
    K = int(net.duration // net.pstep)
    junc = [i for i in range(1, net.nnodes + 1) if net.node_type[i] == api.EN_JUNCTION]
    rep = Replicator(net, trace, record=True, src_slots=K, virtual_sources={j: api.EN_MASS for j in junc})
    rep.run('nextQ')
    dev = torch.device(args.device)
    z = lambda n: torch.zeros(n, dtype=torch.float64)
    ref = TorchWQ(net, rep.schedule, rep.reactflag, device=dev, dtype=torch.float64, engine='fused', spmm_min_batch=0)
    ref.set_fixed_reaction(kb=z(net.nlinks + 1), kw=z(net.nlinks + 1), tank_kb=z(net.ntanks + 1))
    N1 = net.nnodes + 1
    g = np.random.default_rng(2)
    n_out = ref.forward(src=torch.zeros(1, K, N1, dtype=torch.float64, device=dev)).shape[1]
    out = []
    for lo, hi in ((0, n_out), (0, n_out // 3), (n_out // 2, n_out)):   # full range, early rows, late rows
        snap = torch.as_tensor(g.integers(lo, hi, m), device=dev)
        node = torch.as_tensor(g.choice(junc, m), device=dev)
        u0 = torch.zeros(m, K, N1, dtype=torch.float64, device=dev, requires_grad=True)
        y = ref.forward(src=u0)
        y[torch.arange(m, device=dev), snap, node].sum().backward()
        J_ref = u0.grad.cpu().numpy()
        for nc in CHUNKS:
            eng = ChunkedWQ(net, rep.schedule, rep.reactflag, device=dev, dtype=torch.float64, n_chunks=nc)
            eng.set_fixed_reaction(kb=z(net.nlinks + 1), kw=z(net.nlinks + 1), tank_kb=z(net.ntanks + 1))
            J = eng.rows_src(snap, node).cpu().numpy()
            out.append({'snapshots': [lo, hi], 'n_chunks': len(eng.chunk_ops), 'rows': m, 'max_rel': rel(J, J_ref),
                        'nonzero': int((np.abs(J_ref) > 0).sum())})
    return {'network': base, 'slots': K, 'junctions': len(junc), 'checks': out, 'max_rel': max(c['max_rel'] for c in out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default='chunked_check.json')
    args = ap.parse_args()
    lib = api.Library(args.lib)
    t0 = time.perf_counter()
    res = {'device': args.device, 'torch': torch.__version__, 'cases': [], 'rows': None}
    with tempfile.TemporaryDirectory(prefix='chk_') as wd:
        for name, base, setup, over in CASES:
            r = check_case(lib, name, base, setup, over, args, wd)
            res['cases'].append(r)
            print(name, r.get('skipped') or f"terms={r['terms']} max rel diff {r['max_rel']:.2e}", flush=True)
        res['rows'] = check_rows(lib, args, wd)
        print('rows_src', res['rows']['network'], f"max rel diff {res['rows']['max_rel']:.2e}", flush=True)
    res['max_rel_all'] = max([c['max_rel'] for c in res['cases'] if 'max_rel' in c] + [res['rows']['max_rel']])
    res['seconds'] = time.perf_counter() - t0
    Path(args.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('max rel diff over all checks', f"{res['max_rel_all']:.2e}", f"({res['seconds']:.0f} s)")


if __name__ == '__main__':
    main()
