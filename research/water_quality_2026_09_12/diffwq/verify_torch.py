"""Gate G2: PyTorch replay vs. replicator (forward) and autograd vs. central differences
(torch map, and the EPANET DLL itself). Usage (from research/water_quality_2026_09_12):

    python -m diffwq.verify_torch --device cuda
    python -m diffwq.verify_torch --ctol 0 --device cpu --out g2_ctol0.json
"""
import argparse
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
from .verify_m0 import CORE, ROOT, HERE, VARIANTS, NET3_TANKS, all_links, all_tanks_kb, init_qual, qual_type

# first-order / AGE / TRACE cases only (closed-form replay)
CASES = [(k, k, [], {}) for k in CORE] + [v for v in VARIANTS if v[0] in (
    'Net1_MIX2', 'Net1_FIFO', 'Net1_LIFO', 'Net1_MIX1_overflow', 'Net1_FIFO_overflow', 'Net1_LIFO_overflow',
    'Net1_src_mass', 'Net1_src_setpoint', 'Net1_src_flowpaced', 'Net1_src_concen_reservoir_pat',
    'Net1_src_concen_junction_inflow', 'Net1_src_mass_tank', 'Net1_ctol0', 'Net2_MIX2', 'Net2_LIFO',
    'Net3_CHEM_mixed_tanks', 'Net3_AGE_mixed_tanks', 'Net3_TRACE_River_mixed_tanks')] + [
    # kw != 0 on the larger networks: the DLL central difference is only O(h) at kw = 0 (|kw| kink)
    ('Net2_react', 'Net2', [all_links(kb=-0.5, kw=-0.2), all_tanks_kb(-0.5)], {}),
    ('Jilin_kw', 'Jilin', [all_links(kw=-0.1)], {}),
    ('Net3_CHEM_kw_mixed_tanks', 'Net3', [qual_type(api.EN_CHEM), init_qual(0.5, reservoirs=1.0),
                                          all_links(kb=-0.3, kw=-0.2), all_tanks_kb(-0.1)] + NET3_TANKS, {}),
]


def open_case(lib, name, inp, setup, qstep, ctol, workdir):
    pr = api.Project(lib, inp, Path(workdir) / (name + '.rpt'))
    for f in setup:
        f(pr)
    if qstep is not None:
        pr.set_time_param(api.EN_QUALSTEP, qstep)
    if ctol is not None:
        pr.set_option(api.EN_TOLERANCE, ctol)
    return pr


def dll_loss(lib, name, inp, setup, qstep, ctol, workdir, mode, weights, perturb):
    """Loss of the EPANET DLL output under a parameter perturbation (callable applied to the project)."""
    with open_case(lib, name, inp, setup, qstep, ctol, workdir) as pr:
        perturb(pr)
        pr.solve_hydraulics(Path(workdir) / (name + '_fd.hyd'))
        _, q, _ = pr.run_quality_reference(mode)
    return float((weights * q).sum())


def run_case(lib, name, inp, setup, over, args, workdir):
    qstep, ctol = over.get('qstep', args.qstep), over.get('ctol', args.ctol)
    mode = args.mode
    with open_case(lib, name, inp, setup, qstep, ctol, workdir) as pr:
        if pr.qual_info()[0] == api.EN_NONE:
            return {'network': name, 'skipped': 'Quality NONE'}
        hyd = Path(workdir) / (name + '.hyd')
        pr.solve_hydraulics(hyd)
        net = extract(pr)
        trace = read_hydfile(hyd)
        _, q_ref, _ = pr.run_quality_reference(mode)
    rep = Replicator(net, trace, record=True)
    t0 = time.perf_counter(); _, q_rep, _ = rep.run(mode); t_rec = time.perf_counter() - t0
    sch = rep.schedule
    dev, dtype = args.device, torch.float64
    fused = args.engine in ('fused', 'fused_nf', 'fused_spmm')
    engine = 'fused' if fused else ('adjoint' if args.engine == 'both' else args.engine)
    t0 = time.perf_counter()
    eng = TorchWQ(net, sch, rep.reactflag, device=dev, dtype=dtype, engine=engine, flatten=(args.engine != 'fused_nf'),
                  **({'spmm_min_batch': 1, 'spmm_min_work': 0} if args.engine == 'fused_spmm' else {}))
    t_build = time.perf_counter() - t0
    theta = {
        'kb': torch.tensor(net.kb * SECperDAY, dtype=dtype, device=dev, requires_grad=True),
        'kw': torch.tensor(net.kw * SECperDAY, dtype=dtype, device=dev, requires_grad=True),
        'tank_kb': torch.tensor(net.tank_kb * SECperDAY, dtype=dtype, device=dev, requires_grad=True),
        'c0': torch.tensor(net.c0 * net.ucf['QUALITY'], dtype=dtype, device=dev, requires_grad=True),
        'src': torch.tensor([net.sources.get(i, (0, 0.0, 0))[1] for i in range(net.nnodes + 1)],
                            dtype=dtype, device=dev, requires_grad=True),
    }
    t0 = time.perf_counter(); out = eng.forward(**theta)[0]; t_fwd = time.perf_counter() - t0
    out_np = out.detach().cpu().numpy()
    ref = q_rep * net.ucf['QUALITY']
    scale = max(float(np.abs(ref).max()), 1e-300)
    fwd_rel = float(np.abs(out_np - ref).max() / scale)
    fwd_rel_dll = float(np.abs(out_np - q_ref).max() / scale)
    # --- G2a: autograd vs central differences of the torch map (random-weighted loss) ---
    rng = np.random.default_rng(0)
    w_np = rng.uniform(0.5, 1.5, size=out_np.shape)
    w = torch.tensor(w_np, dtype=dtype, device=dev)
    def loss_fn(**th):
        return (eng.forward(**th)[0] * w).sum()
    sync = (lambda: torch.cuda.synchronize()) if torch.device(dev).type == 'cuda' else (lambda: None)
    t0 = time.perf_counter(); loss = loss_fn(**theta); loss.backward(); sync(); t_bwd = time.perf_counter() - t0
    grads = {k: (v.grad.detach().cpu().numpy().copy() if v.grad is not None else np.zeros(v.shape[0]))
             for k, v in theta.items()}          # grad None = parameter unused by this network
    adj_vs_ad = None
    if args.engine == 'both' or fused:           # cross-check on the same schedule: adjoint vs autograd / fused vs adjoint
        eng2 = TorchWQ(net, sch, rep.reactflag, device=dev, dtype=dtype, engine='adjoint' if fused else 'autograd')
        theta2 = {k: v.detach().clone().requires_grad_(True) for k, v in theta.items()}
        t0 = time.perf_counter(); out2 = eng2.forward(**theta2)[0]; (out2 * w).sum().backward(); sync(); t_bwd2 = time.perf_counter() - t0
        adj_vs_ad = {'reference': eng2.engine, 'max_rel_err': 0.0, 'seconds_forward_backward_autograd': t_bwd2,
                     'forward_max_rel_err': float((out2.detach() - out.detach()).abs().max().item() / scale)}
        for k, v in theta2.items():
            g2 = v.grad.detach().cpu().numpy() if v.grad is not None else np.zeros(v.shape[0])
            den = max(float(np.abs(grads[k]).max()), float(np.abs(g2).max()), 1e-300)
            adj_vs_ad['max_rel_err'] = max(adj_vs_ad['max_rel_err'], float(np.abs(g2 - grads[k]).max() / den))
    fd = {}
    for k, v in theta.items():
        g = grads[k]
        idx = np.flatnonzero(np.abs(g) > 1e-12 * max(np.abs(g).max(), 1e-300))
        if idx.size == 0:
            continue
        idx = rng.choice(idx, size=min(args.n_fd, idx.size), replace=False)
        errs = []
        for i in idx:
            base = v.detach().clone(); h = 1e-6 * max(abs(float(base[i])), 1.0)
            th = {kk: vv.detach() for kk, vv in theta.items()}
            p, m = base.clone(), base.clone(); p[i] += h; m[i] -= h
            th[k] = p; lp = float(loss_fn(**th)); th[k] = m; lm = float(loss_fn(**th))
            fdv = (lp - lm) / (2 * h)
            errs.append(abs(fdv - g[i]) / max(abs(g[i]), abs(fdv), 1e-300))
        fd[k] = {'n': int(idx.size), 'max_rel_err': float(max(errs)), 'grad_max_abs': float(np.abs(g).max())}
    # --- G2b: autograd directional derivatives vs central differences of the EPANET DLL ---
    # direction = unit step on every listed parameter; items = [(autograd grad, api index, base user value)]
    dll = {}
    def direction(items, prop, setter, nonneg=False):
        h = args.h_dll * max(1.0, max((abs(b) for _, _, b in items), default=1.0))
        gdir = float(sum(g for g, _, _ in items))
        if not items or gdir == 0.0:
            return None
        # EN_setnodevalue rejects negative values (209); the map is affine in c0/src (first-order
        # kinetics, non-setpoint sources), so a central difference about b + h has the same slope
        shift = h if (nonneg and min(b for _, _, b in items) < h) else 0.0
        def pert(sign):
            def f(pr):
                for _, i, b in items:
                    setter(pr, i, prop, b + shift + sign * h)
            return f
        lp = dll_loss(lib, name, inp, setup, qstep, ctol, workdir, mode, w_np, pert(+1))
        lm = dll_loss(lib, name, inp, setup, qstep, ctol, workdir, mode, w_np, pert(-1))
        fdv = (lp - lm) / (2 * h)
        return {'autograd': gdir, 'dll_fd': fdv, 'rel_err': abs(fdv - gdir) / max(abs(gdir), abs(fdv), 1e-300),
                'h': h, 'shift': shift, 'n_params': len(items)}
    set_link = lambda pr, i, p, v: pr.set_link_value(i, p, v)
    set_node = lambda pr, i, p, v: pr.set_node_value(i, p, v)
    pipes = [k for k in range(1, net.nlinks + 1) if net.link_type[k] == api.EN_PIPE]
    real_tanks = [i for i in range(1, net.ntanks + 1) if not net.tank_is_reservoir[i]]
    c0_user = theta['c0'].detach().cpu().numpy(); src_user = theta['src'].detach().cpu().numpy()
    if net.qualflag == api.EN_CHEM:
        dll['kb_all_pipes'] = direction([(grads['kb'][k], k, net.kb[k] * SECperDAY) for k in pipes], api.EN_KBULK, set_link)
        dll['kw_all_pipes'] = direction([(grads['kw'][k], k, net.kw[k] * SECperDAY) for k in pipes], api.EN_KWALL, set_link)
        dll['tank_kb_all'] = direction([(grads['tank_kb'][i], int(net.tank_node[i]), net.tank_kb[i] * SECperDAY)
                                        for i in real_tanks], api.EN_TANK_KBULK, set_node)
        dll['src_all'] = direction([(grads['src'][i], i, float(src_user[i])) for i in net.sources],
                                   api.EN_SOURCEQUAL, set_node, nonneg=True)
    if net.qualflag != api.EN_TRACE:
        dll['c0_all_nodes'] = direction([(grads['c0'][i], i, float(c0_user[i])) for i in range(1, net.nnodes + 1)],
                                        api.EN_INITQUAL, set_node, nonneg=True)
    dll = {k: v for k, v in dll.items() if v is not None}
    # reactflag False (all kb = kw = tank kb = 0): EPANET then runs the non-reactive path (findnodequal
    # skips noflowqual), so a DLL difference across kb = 0 measures a structurally different program
    fused_stats = {'fused_ops': len(eng.fsch.ops), 'fused_terms': eng.T_total, 'flatten': eng.fsch.flatten,
                   'spmm_min_batch': eng.spmm_min_batch, 'spmm_min_work': eng.spmm_min_work,
                   'spmm_used': eng._spmm is not None} if fused else {}
    return {'network': name, 'mode': mode, 'qualflag': int(net.qualflag), 'ctol_user': net.ctol * net.ucf['QUALITY'],
            'reactflag': bool(rep.reactflag), 'engine': args.engine, **fused_stats,
            'steps': sch.nsteps, 'versions': sch.nver, 'ops': len(sch.ops), 'terms': int(sum(len(t[0]) for op in sch.ops for t in op.terms.values())),
            'max_level': int(max(op.level for op in sch.ops)), 'forward_rel_vs_replicator': fwd_rel,
            'forward_rel_vs_dll': fwd_rel_dll, 'fd_torch': fd, 'fd_dll': dll, 'adjoint_vs_autograd': adj_vs_ad,
            'device': dev, 'seconds': {'record': t_rec, 'build': t_build, 'forward': t_fwd, 'forward_backward': t_bwd}}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='nextQ', choices=['nextQ', 'stepQ'])
    ap.add_argument('--qstep', type=int, default=None)
    ap.add_argument('--ctol', type=float, default=None, help='override Ctol (0 = mode M1, no merging)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n-fd', type=int, default=8, help='parameters per group checked by torch central differences')
    ap.add_argument('--h-dll', type=float, default=1e-4, help='DLL central-difference step (x max|param|)')
    ap.add_argument('--engine', default='adjoint', choices=['adjoint', 'autograd', 'both', 'fused', 'fused_nf', 'fused_spmm'],
                    help='both = adjoint engine, cross-checked against pure autograd; fused / fused_nf = fused engine '
                         'with / without step flattening, cross-checked against the adjoint engine; fused_spmm = fused '
                         'engine with the cuSPARSE path forced at every batch size (CUDA)')
    ap.add_argument('--only', default='', help='comma-separated case names')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    workdir = tempfile.mkdtemp(prefix='diffwq_', dir=os.environ.get('TMPDIR'))
    only = set(filter(None, args.only.split(',')))
    results = []
    for name, base, setup, over in CASES:
        if only and name not in only:
            continue
        inp = ROOT / CORE[base]
        try:
            r = run_case(lib, name, inp, setup, over, args, workdir)
        except Exception as exc:
            r = {'network': name, 'error': f'{type(exc).__name__}: {exc}'}
        results.append(r)
        if 'forward_rel_vs_replicator' in r:
            fdt = max((v['max_rel_err'] for v in r['fd_torch'].values()), default=float('nan'))
            fdd = {k: f"{v['rel_err']:.1e}" for k, v in r['fd_dll'].items()}
            s = r['seconds']
            flag = '' if r['reactflag'] or r['qualflag'] != api.EN_CHEM else ' [Reactflag off]'
            a = r['adjoint_vs_autograd']
            adj = '' if a is None else (f" grad|{a['reference']}={a['max_rel_err']:.1e} fwd|{a['reference']}="
                                        f"{a['forward_max_rel_err']:.1e} ({a['reference']} fwd+bwd {a['seconds_forward_backward_autograd']:.2f}s)")
            fo = f" fused_ops={r['fused_ops']} fused_terms={r['fused_terms']}" if 'fused_ops' in r else ''
            print(f"{name:32s} fwd|rep={r['forward_rel_vs_replicator']:.1e} fwd|dll={r['forward_rel_vs_dll']:.1e} "
                  f"AD|torchFD={fdt:.1e} AD|dllFD={fdd}{flag}{adj} lvl={r['max_level']} ops={r['ops']}{fo} "
                  f"t(rec/build/fwd/fwd+bwd)={s['record']:.2f}/{s['build']:.2f}/{s['forward']:.2f}/{s['forward_backward']:.2f}s",
                  flush=True)
        else:
            print(f"{name:32s} {r.get('skipped') or r.get('error')}", flush=True)
    report = {'engine_version': lib.version(), 'platform': platform.platform(), 'torch': torch.__version__,
              'device': args.device, 'mode': args.mode, 'qstep_override': args.qstep, 'ctol_override': args.ctol,
              'h_dll': args.h_dll, 'engine': args.engine, 'results': results}
    tag = f"{args.mode}_ctol{args.ctol if args.ctol is not None else 'inp'}"
    out = Path(args.out) if args.out else HERE.parent / f'g2_{tag}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('report:', out)


if __name__ == '__main__':
    main()
