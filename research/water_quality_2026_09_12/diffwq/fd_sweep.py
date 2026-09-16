"""Step-size sweep of EPANET central differences vs. the replay gradient (direction: +1 on
every pipe's kb, kw or every node's c0). Run at the INP Ctol (mode M0) and at Ctol = 0 (M1).
Usage: python -m diffwq.fd_sweep --case Net3_CHEM_kw_mixed_tanks --param kb
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch

from . import epanet_api as api
from .hydfile import read_hydfile
from .network import extract, SECperDAY
from .replicate import Replicator
from .torch_wq import TorchWQ
from .verify_torch import CASES, open_case, dll_loss
from .verify_m0 import CORE, ROOT, HERE

PROP = {'kb': (api.EN_KBULK, 'link'), 'kw': (api.EN_KWALL, 'link'), 'c0': (api.EN_INITQUAL, 'node')}


def sweep(lib, name, inp, setup, over, param, ctol, hs, workdir, mode='nextQ'):
    qstep = over.get('qstep')
    with open_case(lib, name, inp, setup, qstep, ctol, workdir) as pr:
        hyd = Path(workdir) / (name + '.hyd')
        pr.solve_hydraulics(hyd)
        net = extract(pr)
        trace = read_hydfile(hyd)
    rep = Replicator(net, trace, record=True)
    rep.run(mode)
    eng = TorchWQ(net, rep.schedule, rep.reactflag, device='cpu')
    dt = torch.float64
    theta = {'kb': torch.tensor(net.kb * SECperDAY, dtype=dt, requires_grad=True),
             'kw': torch.tensor(net.kw * SECperDAY, dtype=dt, requires_grad=True),
             'c0': torch.tensor(net.c0 * net.ucf['QUALITY'], dtype=dt, requires_grad=True)}
    out = eng.forward(kb=theta['kb'], kw=theta['kw'], c0=theta['c0'])[0]
    w_np = np.random.default_rng(0).uniform(0.5, 1.5, size=tuple(out.shape))
    (out * torch.tensor(w_np, dtype=dt)).sum().backward()
    prop, kind = PROP[param]
    vals = theta[param].detach().numpy()
    if kind == 'link':
        ids = [k for k in range(1, net.nlinks + 1) if net.link_type[k] == api.EN_PIPE]
        setter = lambda pr, i, v: pr.set_link_value(i, prop, v)
    else:
        ids = list(range(1, net.nnodes + 1))
        setter = lambda pr, i, v: pr.set_node_value(i, prop, v)
    base = {i: float(vals[i]) for i in ids}
    g = theta[param].grad.numpy()
    gdir = float(sum(g[i] for i in ids))
    rows = []
    for h in hs:
        # EN_INITQUAL must stay >= 0: centre the difference at c0 + h (the map is affine in c0 on each
        # Ctol piece, so the slope is unchanged unless the shift itself crosses a merge decision)
        shift = h if (kind == 'node' and min(base.values()) < h) else 0.0
        def pert(sign):
            return lambda pr: [setter(pr, i, base[i] + shift + sign * h) for i in ids]
        lp = dll_loss(lib, name, inp, setup, qstep, ctol, workdir, mode, w_np, pert(+1))
        lm = dll_loss(lib, name, inp, setup, qstep, ctol, workdir, mode, w_np, pert(-1))
        fd = (lp - lm) / (2 * h)
        rows.append({'h': h, 'shift': shift, 'dll_fd': fd,
                     'rel_err': abs(fd - gdir) / max(abs(gdir), abs(fd), 1e-300)})
    return {'ctol_user': net.ctol * net.ucf['QUALITY'], 'autograd': gdir, 'n_params': len(ids), 'sweep': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='Net3_CHEM_kw_mixed_tanks')
    ap.add_argument('--param', default='kb', choices=list(PROP))
    ap.add_argument('--hs', default='1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    hs = [float(h) for h in args.hs.split(',')]
    lib = api.Library(args.lib)
    workdir = tempfile.mkdtemp(prefix='diffwq_fd_', dir=os.environ.get('TMPDIR'))
    name, base, setup, over = {c[0]: c for c in CASES}[args.case]
    report = {'case': name, 'param': args.param, 'engine_version': lib.version(), 'runs': {}}
    for label, ctol in (('M0_inp_ctol', over.get('ctol')), ('M1_ctol0', 0.0)):
        r = sweep(lib, name, ROOT / CORE[base], setup, over, args.param, ctol, hs, workdir)
        report['runs'][label] = r
        print(f"{name} {args.param} {label}: Ctol={r['ctol_user']:g} autograd={r['autograd']:.10g} (n={r['n_params']})")
        for row in r['sweep']:
            print(f"    h={row['h']:.0e}  DLL central diff={row['dll_fd']:.10g}  rel err={row['rel_err']:.2e}"
                  + (f"  (centre shifted by {row['shift']:g})" if row['shift'] else ''))
    out = Path(args.out) if args.out else HERE.parent / f'g2_fd_sweep_{name}_{args.param}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('report:', out)


if __name__ == '__main__':
    main()
