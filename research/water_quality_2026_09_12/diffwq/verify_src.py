"""Fidelity of time-slot and virtual sources: the schedule is recorded WITHOUT the sources (zero-strength
virtual MASS dosers at candidate junctions / tanks, K slots on the pattern grid) and replayed with slot
strengths s_j * p_j[k]; the reference is the EPANET DLL run WITH those sources installed through the
API (EN_addpattern / EN_setnodevalue). Ctol = 0 (mode M1) so the schedule does not depend on the
strengths. Checks: forward vs DLL, zero-strength replay vs source-free DLL run, adjoint slot gradients
vs DLL central differences (two directions), fixed-reaction path and CUDA-graph replay.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.verify_src --cases Net1,Net2_react,Jilin_kw,Net3_CHEM_kw_mixed_tanks --device cuda
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
from .network import extract
from .replicate import Replicator
from .torch_wq import TorchWQ, default_src
from .verify_torch import CASES, open_case
from .verify_m0 import CORE, ROOT, HERE


def dll_run(lib, name, inp, setup, qstep, workdir, mode, install=None):
    with open_case(lib, name, inp, setup, qstep, 0.0, workdir) as pr:
        if install is not None:
            install(pr)
        pr.solve_hydraulics(Path(workdir) / (name + '_src.hyd'))
        _, q, _ = pr.run_quality_reference(mode)
    return q


def run_case(lib, name, inp, setup, over, args, workdir):
    qstep, mode, K = over.get('qstep', args.qstep), args.mode, args.slots
    with open_case(lib, name, inp, setup, qstep, 0.0, workdir) as pr:
        if pr.qual_info()[0] != api.EN_CHEM:
            return {'network': name, 'skipped': 'not CHEM'}
        hyd = Path(workdir) / (name + '.hyd')
        pr.solve_hydraulics(hyd)
        net = extract(pr)
        trace = read_hydfile(hyd)
        _, q_base, _ = pr.run_quality_reference(mode)
    rng = np.random.default_rng(1)
    junc = [i for i in range(1, net.nnodes + 1) if net.node_type[i] == api.EN_JUNCTION and i not in net.sources
            and len(net.adj[i]) >= 2]
    cand = sorted(rng.choice(junc, size=min(args.n_cand, len(junc)), replace=False).tolist())
    tanks = [int(net.tank_node[t]) for t in range(1, net.ntanks + 1) if not net.tank_is_reservoir[t]
             and int(net.tank_node[t]) not in net.sources]
    if tanks:
        cand.append(int(rng.choice(tanks)))
    for i, (_, _, p) in net.sources.items():                       # slots must cover existing source patterns
        if p != 0 and net.node_type[i] != api.EN_RESERVOIR:
            K = int(np.lcm(K, len(net.patterns[p])))
    strength = {j: float(rng.uniform(50.0, 500.0)) for j in cand}                      # mg/min (MASS)
    pat = {j: np.where(rng.uniform(size=K) < 0.3, 0.0, rng.uniform(0.2, 2.0, size=K)) for j in cand}
    S = default_src(net, K)                                                            # existing sources per slot
    for j in cand:
        S[:, j] = strength[j] * pat[j]
    # --- schedule without the sources, virtual dosers at the candidates ---
    rep = Replicator(net, trace, record=True, src_slots=K, virtual_sources={j: api.EN_MASS for j in cand})
    t0 = time.perf_counter(); _, q_rep, _ = rep.run(mode); t_rec = time.perf_counter() - t0
    dev, dt = args.device, torch.float64
    eng = TorchWQ(net, rep.schedule, rep.reactflag, device=dev, dtype=dt, engine='fused')
    eng1 = TorchWQ(net, rep.schedule, rep.reactflag, device=dev, dtype=dt, engine='adjoint')
    scale0 = max(float(np.abs(q_base).max()), 1e-300)
    zero_rel = float(np.abs(eng.forward()[0].cpu().numpy() - q_base).max() / scale0)   # default src = 0 here
    # --- DLL with the sources installed ---
    def install(pr):
        for j in cand:
            pidx = pr.add_pattern(f'vsrc{j}', pat[j])
            pr.set_source(j, api.EN_MASS, strength[j], pidx)
    q_dll = dll_run(lib, name, inp, setup, qstep, workdir, mode, install)
    scale = max(float(np.abs(q_dll).max()), 1e-300)
    St = torch.tensor(S, dtype=dt, device=dev, requires_grad=True)
    out = eng.forward(src=St)[0]
    fwd_rel = float(np.abs(out.detach().cpu().numpy() - q_dll).max() / scale)
    out1 = eng1.forward(src=St)[0]
    fwd_rel_v1 = float((out1.detach() - out.detach()).abs().max().item() / scale)
    # --- adjoint slot gradients vs DLL central differences ---
    w_np = rng.uniform(0.5, 1.5, size=q_dll.shape); w = torch.tensor(w_np, dtype=dt, device=dev)
    (out * w).sum().backward(); g = St.grad.detach().cpu().numpy()
    g1 = torch.autograd.grad((out1 * w).sum(), St)[0].cpu().numpy()
    grad_rel_v1 = float(np.abs(g1 - g).max() / max(np.abs(g).max(), 1e-300))
    def dll_loss(scale_s, slot_shift):
        def inst(pr):
            for j in cand:
                p = pat[j].copy(); p[slot_shift[0]] += slot_shift[1]
                pidx = pr.add_pattern(f'vsrc{j}', p)
                pr.set_source(j, api.EN_MASS, strength[j] * scale_s, pidx)
        return float((w_np * dll_run(lib, name, inp, setup, qstep, workdir, mode, inst)).sum())
    fd = {}
    h = 1e-4                                                           # all strengths x (1 +- h)
    fdv = (dll_loss(1 + h, (0, 0.0)) - dll_loss(1 - h, (0, 0.0))) / (2 * h)
    gdir = float(sum(strength[j] * pat[j][k] * g[k, j] for j in cand for k in range(K)))
    fd['all_strengths'] = {'adjoint': gdir, 'dll_fd': fdv, 'rel_err': abs(fdv - gdir) / max(abs(gdir), abs(fdv), 1e-300)}
    k0 = int(np.argmax(np.abs(g[:, cand]).sum(1)))                    # most sensitive slot: pattern value +- h
    fdv = (dll_loss(1.0, (k0, h)) - dll_loss(1.0, (k0, -h))) / (2 * h)
    gdir = float(sum(strength[j] * g[k0, j] for j in cand))
    fd['slot_%d' % k0] = {'adjoint': gdir, 'dll_fd': fdv, 'rel_err': abs(fdv - gdir) / max(abs(gdir), abs(fdv), 1e-300)}
    # --- fixed reaction (affine map in src) and CUDA graph of it ---
    eng.set_fixed_reaction(kb=torch.tensor(net.kb * 86400.0), kw=torch.tensor(net.kw * 86400.0),
                           tank_kb=torch.tensor(net.tank_kb * 86400.0))
    S2 = St.detach().clone().requires_grad_(True)
    out2 = eng.forward(src=S2)[0]; (out2 * w).sum().backward()
    fixed = {'forward_rel': float((out2.detach() - out.detach()).abs().max().item() / scale),
             'grad_rel': float(np.abs(S2.grad.cpu().numpy() - g).max() / max(np.abs(g).max(), 1e-300))}
    if torch.device(dev).type == 'cuda':
        S3 = St.detach().clone().requires_grad_(True)
        gph = torch.cuda.make_graphed_callables(lambda s: eng.forward(src=s), (S3,), num_warmup_iters=3)
        out3 = gph(S3)[0]; (out3 * w).sum().backward()
        fixed['graph_forward_rel'] = float((out3.detach() - out.detach()).abs().max().item() / scale)
        fixed['graph_grad_rel'] = float(np.abs(S3.grad.cpu().numpy() - g).max() / max(np.abs(g).max(), 1e-300))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10):
            (gph(S3)[0] * w).sum().backward()
        torch.cuda.synchronize(); fixed['graph_fwd_bwd_s'] = (time.perf_counter() - t0) / 10
    return {'network': name, 'mode': mode, 'slots': K, 'candidates': cand, 'n_virtual_terms': int(rep.hits['virtual_source']),
            'zero_strength_vs_dll_base': zero_rel, 'forward_rel_vs_dll': fwd_rel, 'forward_rel_fused_vs_adjoint': fwd_rel_v1,
            'grad_rel_fused_vs_adjoint': grad_rel_v1, 'fd_dll': fd, 'fixed_reaction': fixed,
            'fused_ops': len(eng.fsch.ops), 'fused_terms': eng.T_total, 'record_s': t_rec, 'device': dev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='Net1,Net2_react,Jilin_kw,Net3_CHEM_kw_mixed_tanks')
    ap.add_argument('--mode', default='nextQ', choices=['nextQ', 'stepQ'])
    ap.add_argument('--qstep', type=int, default=None)
    ap.add_argument('--slots', type=int, default=24)
    ap.add_argument('--n-cand', type=int, default=5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    workdir = tempfile.mkdtemp(prefix='diffwq_src_', dir=os.environ.get('TMPDIR'))
    cases = {c[0]: c for c in CASES}
    results = []
    for name in args.cases.split(','):
        _, base, setup, over = cases[name]
        r = run_case(lib, name, ROOT / CORE[base], setup, over, args, workdir)
        results.append(r)
        if 'skipped' in r:
            print(f"{name:28s} {r['skipped']}", flush=True); continue
        fdd = {k: f"{v['rel_err']:.1e}" for k, v in r['fd_dll'].items()}
        fx = r['fixed_reaction']
        print(f"{name:28s} cand={r['candidates']} vterms={r['n_virtual_terms']} zero|dll={r['zero_strength_vs_dll_base']:.1e} "
              f"fwd|dll={r['forward_rel_vs_dll']:.1e} fwd|v1={r['forward_rel_fused_vs_adjoint']:.1e} grad|v1={r['grad_rel_fused_vs_adjoint']:.1e} "
              f"AD|dllFD={fdd} fixed(fwd/grad)={fx['forward_rel']:.1e}/{fx['grad_rel']:.1e}"
              + (f" graph(fwd/grad)={fx['graph_forward_rel']:.1e}/{fx['graph_grad_rel']:.1e} {fx['graph_fwd_bwd_s'] * 1e3:.1f}ms" if 'graph_grad_rel' in fx else ''),
              flush=True)
    report = {'engine_version': lib.version(), 'platform': platform.platform(), 'torch': torch.__version__,
              'device': args.device, 'mode': args.mode, 'slots': args.slots, 'results': results}
    out = Path(args.out) if args.out else HERE.parent / f'verify_src_{args.mode}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('report:', out)


if __name__ == '__main__':
    main()
