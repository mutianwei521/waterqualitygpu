"""Gate G3: measured cost of the differentiable replay vs. the EPANET engine.

For each case: record the schedule once, then time forward and forward+backward for batches of
B parameter sets (kb, kw, tank_kb perturbed +-10 %) on the chosen device, for each replay engine
('adjoint' = per-kind ops, 'fused_nf' = one op per (step, level), 'fused' = one op per step with the
per-op cuSPARSE path (B >= spmm_min_batch, terms x B >= spmm_min_work), 'fused_spmm' / 'fused_gather' =
that path forced on / off).
--shared-theta: one parameter point (frozen reaction factors), B initial-quality / source columns,
gradients w.r.t. c0 and src (the regime of the booster, source-identification and Jacobian-row
workloads). --jac-rows also times jac_rows_kw with `chunk` = B unit cotangents per adjoint sweep.
Every engine after the first is checked against the first (max relative difference of outputs,
gradients and Jacobian rows on identical inputs).
Reference: the DLL's own quality run (EN_openQ..EN_closeQ) and the finite-difference cost
(N_theta + 1) x t_dll.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.bench --device cuda --batches 1,16,128,1024 --engines adjoint,fused [--graph]
    python -m diffwq.bench --device cuda --shared-theta --jac-rows --engines fused_gather,fused_spmm --graph
"""
import argparse
import gc
import json
import os
import platform
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

from . import epanet_api as api
from .hydfile import read_hydfile
from .network import extract, SECperDAY
from .replicate import Replicator
from .torch_wq import TorchWQ
from .verify_torch import CASES, open_case
from .verify_m0 import CORE, ROOT, HERE, all_links, all_tanks_kb, init_qual, qual_type

DEFAULT_CASES = ['Net1', 'Net2_react', 'Jilin_kw', 'Net3_CHEM_kw_mixed_tanks']
ENGINES = {'adjoint': dict(engine='adjoint'), 'fused_nf': dict(engine='fused', flatten=False),
           'fused': dict(engine='fused'), 'fused_spmm': dict(engine='fused', spmm_min_batch=1, spmm_min_work=0),
           'fused_gather': dict(engine='fused', spmm_min_batch=0)}
# the experiment networks (chlorine, 48 h, Ctol = 0 as in exp1-3): 5-min and hourly hydraulics
_chem = lambda kb, kw: [qual_type(api.EN_CHEM), init_qual(0.5, reservoirs=1.0), all_links(kb=kb, kw=kw), all_tanks_kb(kb),
                        lambda pr: pr.set_time_param(api.EN_DURATION, 48 * 3600)]
EXTRA_CASES = {
    'LTown_kw': ('LTown_kw', ROOT / 'networks/L-Town/L-TOWN.inp', _chem(-0.3, -0.1), {'ctol': 0.0}),
    'BWSN2_kw': ('BWSN2_kw', ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp',
                 _chem(-0.3, -0.3) + [lambda pr: pr.set_option(api.EN_UNBALANCED, 10)], {'ctol': 0.0}),
}


def timed(fn, sync, repeat):
    best = float('inf')
    for _ in range(repeat):
        sync(); t0 = time.perf_counter(); fn(); sync()
        best = min(best, time.perf_counter() - t0)
    return best


def rel_err(a, b):
    den = max(float(a.abs().max()), float(b.abs().max()), 1e-300)
    return float((a - b).abs().max()) / den


def run_case(lib, name, inp, setup, over, args, workdir):
    qstep, ctol = over.get('qstep', args.qstep), over.get('ctol', args.ctol)
    with open_case(lib, name, inp, setup, qstep, ctol, workdir) as pr:
        hyd = Path(workdir) / (name + '.hyd')
        pr.solve_hydraulics(hyd)
        net = extract(pr)
        trace = read_hydfile(hyd)
        t_dll = timed(lambda: pr.run_quality_reference(args.mode), lambda: None, args.repeat)
    rep = Replicator(net, trace, record=True)
    t0 = time.perf_counter(); rep.run(args.mode); t_rec = time.perf_counter() - t0
    sch = rep.schedule
    dev = torch.device(args.device)
    sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
    pipes = int(np.sum(net.link_type[1:] == api.EN_PIPE)); tanks = int(np.sum(~net.tank_is_reservoir[1:]))
    n_theta = 2 * pipes + tanks + net.nnodes + len(net.sources)
    engines, ref = {}, {}
    for ename in args.engines:
        t0 = time.perf_counter()
        eng = TorchWQ(net, sch, rep.reactflag, device=dev, dtype=torch.float64, **ENGINES[ename])
        sync(); t_build = time.perf_counter() - t0
        info = {'build_s': t_build}
        if eng.engine == 'fused':
            info.update(ops=len(eng.fsch.ops), terms=eng.T_total, max_op_terms=eng.max_op_terms,
                        spmm_min_batch=eng.spmm_min_batch, spmm_min_work=eng.spmm_min_work)
        print(f"  engine={ename} build={t_build:.2f}s" + (f" ops={info['ops']} terms={info['terms']}" if 'ops' in info else ''), flush=True)
        info['batches'] = bench_engine(eng, net, sch, dev, sync, args, ref)
        if eng.engine == 'fused' and eng._spmm is not None:
            info['spmm_struct_MB'] = spmm_struct_mb(eng)
        engines[ename] = info
        del eng; gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    return {'network': name, 'device': str(dev), 'nodes': net.nnodes, 'links': net.nlinks, 'pipes': pipes,
            'steps': sch.nsteps, 'versions': sch.nver, 'ops': len(sch.ops), 'n_theta': n_theta,
            'shared_theta': bool(args.shared_theta), 'dll_quality_run_s': t_dll,
            'fd_gradient_cost_s': (n_theta + 1) * t_dll, 'record_s': t_rec, 'engines': engines}


def spmm_struct_mb(eng):
    S_ = eng._spmm
    ts = [S_['col_f'], S_['perm_t'], S_['col_t'], *S_['crow_f'], *S_['crow_t'], *S_['U']]
    if S_['chains'] is not None:
        ts += [S_['chains'][2], S_['chains'][3], *S_['chains'][0], *S_['chains'][1]]
    return sum(t.numel() * t.element_size() for t in ts) / 2**20


def bench_engine(eng, net, sch, dev, sync, args, ref):
    """ref: per-B (output, gradients, Jacobian rows) of the first engine, shared by the engines of one case
    (identical random draws: the generator is re-seeded here)."""
    rng = np.random.default_rng(0)
    rows = []
    L1, T1, N1 = net.nlinks + 1, net.ntanks + 1, net.nnodes + 1
    for B in args.batches:
        def batch(v, size, nb):
            base = torch.as_tensor(np.asarray(v), dtype=torch.float64, device=dev).view(1, -1)
            pert = torch.as_tensor(rng.uniform(0.9, 1.1, size=(nb, size)), dtype=torch.float64, device=dev)
            return (base * pert).requires_grad_(True)
        Bt = 1 if args.shared_theta else B
        kb = batch(net.kb * SECperDAY, L1, Bt)
        kw = batch(net.kw * SECperDAY, L1, Bt)
        tkb = batch(net.tank_kb * SECperDAY, T1, Bt)
        c0 = batch(net.c0 * net.ucf['QUALITY'], N1, B)
        src = batch([net.sources.get(i, (0, 0.0, 0))[1] for i in range(N1)], N1, B)
        w = torch.as_tensor(rng.random((1, len(sch.snapshots), N1)), dtype=torch.float64, device=dev)
        if args.shared_theta:                                  # frozen reaction: gradients w.r.t. c0, src only
            th = tuple(v.detach() for v in (kb, kw, tkb))
            if eng.engine == 'fused':
                eng.set_fixed_reaction(*th)
                gfn = lambda a, b: eng.forward(c0=a, src=b)
            else:
                gfn = lambda a, b: eng.forward(*th, a, b)
            params = (c0, src)
        else:
            gfn = lambda a, b, c, d, e: eng.forward(a, b, c, d, e)
            params = (kb, kw, tkb, c0, src)
        fwd = lambda: gfn(*params)
        fb = lambda: (fwd() * w).sum().backward()
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        try:
            fwd(); fb()                                        # warm-up (builds the SpMM structures when used)
            t_f = timed(lambda: (torch.no_grad()(fwd))(), sync, args.repeat)
            t_fb = timed(fb, sync, args.repeat)
        except RuntimeError as exc:                            # OOM at large B: record and continue
            rows.append({'B': B, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'})
            print(f"  B={B:5d} {rows[-1]['error']}", flush=True)
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
            continue
        row = {'B': B, 'forward_s': t_f, 'forward_backward_s': t_fb,
               'forward_per_sample_s': t_f / B, 'forward_backward_per_sample_s': t_fb / B}
        if dev.type == 'cuda':
            row['peak_mem_MB'] = torch.cuda.max_memory_allocated() / 2**20
        # exactness across engines: output and gradients on identical inputs
        for p in params:
            p.grad = None
        with torch.no_grad():
            out = fwd()
        fb()
        cur = {'out': out, 'grad': [p.grad.detach().clone() for p in params]}
        if args.graph and dev.type == 'cuda':
            g = gfb = None
            torch.cuda.empty_cache()                           # the capture gets its own private pool
            try:
                g = torch.cuda.make_graphed_callables(gfn, params, num_warmup_iters=3)
                gfb = lambda: (g(*params) * w).sum().backward()
                gfb()
                row['graph_forward_backward_s'] = timed(gfb, sync, args.repeat)
                with torch.no_grad():
                    gout = g(*params)
                row['graph_forward_rel_err'] = rel_err(gout, out)
                row['graph_peak_mem_MB'] = torch.cuda.max_memory_allocated() / 2**20
            except Exception as exc:
                row['graph_error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            finally:                                           # release the graph pool before the next B
                g = gfb = gout = None
                gc.collect(); torch.cuda.empty_cache()
        if args.jac_rows and eng.engine == 'fused' and eng.chem:
            n_rows = min(4 * B, len(sch.snapshots) * N1)
            jrows = torch.arange(n_rows, device=dev)
            jfn = lambda: eng.jac_rows_kw(jrows, kb[:1].detach(), kw[:1].detach(), tkb[:1].detach(), chunk=B)
            try:
                J = jfn()
                t_j = timed(jfn, sync, args.repeat)
                row['jac_rows'] = {'n_rows': n_rows, 'chunk': B, 'seconds': t_j, 'rows_per_s': n_rows / t_j}
                cur['jac'] = J
            except RuntimeError as exc:
                row['jac_rows'] = {'n_rows': n_rows, 'chunk': B, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
        if B in ref:
            r0 = ref[B]
            row['vs_first_engine'] = {'out_rel_err': rel_err(cur['out'], r0['out']),
                                      'grad_rel_err': max(rel_err(a, b) for a, b in zip(cur['grad'], r0['grad']))}
            if 'jac' in cur and 'jac' in r0:
                row['vs_first_engine']['jac_rel_err'] = rel_err(cur['jac'], r0['jac'])
        else:
            ref[B] = cur
        rows.append(row)
        print(f"    B={B:5d} fwd={t_f:.3f}s ({t_f / B * 1e3:.2f} ms/sample) fwd+bwd={t_fb:.3f}s "
              f"({t_fb / B * 1e3:.2f} ms/sample)" + (f" graph fwd+bwd={row['graph_forward_backward_s']:.3f}s"
              if 'graph_forward_backward_s' in row else '') +
              (f" mem={row['peak_mem_MB']:.0f}MB" if 'peak_mem_MB' in row else '') +
              (f" jac {row['jac_rows']['rows_per_s']:.0f} rows/s" if 'rows_per_s' in row.get('jac_rows', {}) else '') +
              (f" vs first: out {row['vs_first_engine']['out_rel_err']:.1e} grad {row['vs_first_engine']['grad_rel_err']:.1e}"
               + (f" jac {row['vs_first_engine']['jac_rel_err']:.1e}" if 'jac_rel_err' in row['vs_first_engine'] else '')
               if 'vs_first_engine' in row else ''), flush=True)
    if args.shared_theta and eng.engine == 'fused':
        eng.set_fixed_reaction()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=','.join(DEFAULT_CASES), help='verify_torch cases or ' + ', '.join(EXTRA_CASES))
    ap.add_argument('--mode', default='nextQ', choices=['nextQ', 'stepQ'])
    ap.add_argument('--qstep', type=int, default=None)
    ap.add_argument('--ctol', type=float, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batches', default='1,16,128,1024')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--graph', action='store_true', help='also time a CUDA-graph capture of forward+backward')
    ap.add_argument('--engines', default='adjoint', help='comma-separated: ' + ', '.join(ENGINES))
    ap.add_argument('--shared-theta', action='store_true',
                    help='one parameter point (frozen reaction factors), B c0 / src columns, gradients w.r.t. c0 and src')
    ap.add_argument('--jac-rows', action='store_true', help='also time jac_rows_kw (4 B rows, chunk = B) on the fused engines')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    args.batches = [int(b) for b in args.batches.split(',')]
    args.engines = args.engines.split(',')
    lib = api.Library(args.lib)
    workdir = tempfile.mkdtemp(prefix='diffwq_bench_', dir=os.environ.get('TMPDIR'))
    want = args.cases.split(',')
    cases = {c[0]: (ROOT / CORE[c[1]], c[2], c[3]) for c in CASES}
    cases.update({k: (v[1], v[2], v[3]) for k, v in EXTRA_CASES.items()})
    results = []
    for name in want:
        inp, setup, over = cases[name]
        print(f"{name} [{args.device}]", flush=True)
        r = run_case(lib, name, inp, setup, over, args, workdir)
        print(f"  DLL quality run={r['dll_quality_run_s'] * 1e3:.2f} ms, N_theta={r['n_theta']}, "
              f"FD gradient={r['fd_gradient_cost_s']:.3f}s, record={r['record_s']:.2f}s "
              f"ops={r['ops']} versions={r['versions']}", flush=True)
        results.append(r)
    dev_name = torch.cuda.get_device_name(0) if args.device.startswith('cuda') and torch.cuda.is_available() else platform.processor()
    report = {'engine_version': lib.version(), 'platform': platform.platform(), 'torch': torch.__version__,
              'device': args.device, 'device_name': dev_name, 'mode': args.mode, 'ctol_override': args.ctol,
              'repeat': args.repeat, 'engines': args.engines, 'shared_theta': args.shared_theta, 'jac_rows': args.jac_rows,
              'results': results}
    tag = f"{args.device.replace(':', '')}_{args.mode}_ctol{args.ctol if args.ctol is not None else 'inp'}" + \
          ('_shared' if args.shared_theta else '')
    out = Path(args.out) if args.out else HERE.parent / f'g3_bench_{tag}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('report:', out)


if __name__ == '__main__':
    main()
