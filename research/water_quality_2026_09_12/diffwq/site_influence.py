"""G4 / H1 probe: influence of one sampling site on the grouped calibration.

The grouped model (calib.py parameters) is fitted with the exact adjoint gradient on all sites and
again with one site removed; the change of theta* and of the predictions at the removed site show
whether a group coefficient is carried by that single site.

Usage (from research/water_quality_2026_09_12):
    python -m diffwq.site_influence --cases Bellingham:150,NorthPenn:192,Fairfield:178 --out g4_site_influence.json
"""
import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize

from . import epanet_api as api
from .calib import CalibProblem

OPTS = {'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 200}


def fit(prob, x0, mask=None):
    m_t = None if mask is None else torch.as_tensor(mask, dtype=torch.float64, device=prob.dev)

    def fg(x):
        th = torch.as_tensor(np.asarray(x, dtype=np.float64), device=prob.dev).view(1, -1).requires_grad_(True)
        r = prob.simulate_torch(th)[0] - prob.y
        loss = (r * r).sum() if m_t is None else (m_t * r * r).sum()
        loss.backward()
        return loss.item(), th.grad[0].cpu().numpy().copy()
    res = minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=prob.bounds, options=OPTS)
    with torch.no_grad():
        sim = prob.simulate_torch(torch.as_tensor(res.x, device=prob.dev).view(1, -1))[0].cpu().numpy()
    return res.x, float(res.fun), sim, int(res.nfev)


def run_case(name, site_id, lib, workdir, args):
    prob = CalibProblem(name, lib, workdir, device=args.device, verbose=False)
    try:
        obs = prob.obs
        s = obs.site_node_id.index(site_id)
        held = obs.site == s
        x_all, sse_all, sim_all, n_all = fit(prob, prob.theta_nominal)
        x_out, sse_out, sim_out, n_out = fit(prob, x_all, mask=(~held).astype(np.float64))
        rmse = lambda e: float(np.sqrt(np.mean(e * e)))
        out = {'dataset': name, 'site': site_id, 'n_site': int(held.sum()), 'n_obs': len(obs), 'param_names': prob.param_names,
               'y_site_min': float(obs.y[held].min()), 'y_site_max': float(obs.y[held].max()), 'y_site_mean': float(obs.y[held].mean()),
               'all_sites': {'theta': x_all.tolist(), 'sse': sse_all, 'n_grad': n_all, 'site_rmse': rmse(sim_all[held] - obs.y[held]),
                             'other_rmse': rmse(sim_all[~held] - obs.y[~held]), 'site_sim_mean': float(sim_all[held].mean())},
               'without_site': {'theta': x_out.tolist(), 'sse_train': sse_out, 'n_grad': n_out, 'site_rmse': rmse(sim_out[held] - obs.y[held]),
                                'other_rmse': rmse(sim_out[~held] - obs.y[~held]), 'site_sim_mean': float(sim_out[held].mean())}}
        a, w = out['all_sites'], out['without_site']
        print(f"[{name}] site {site_id}: {out['n_site']}/{len(obs)} samples, {out['y_site_min']:.2f}-{out['y_site_max']:.2f} mg/L", flush=True)
        print(f"  all sites    theta*={np.round(a['theta'], 3).tolist()} site rmse {a['site_rmse']:.3f} (sim mean {a['site_sim_mean']:.3f} vs obs {out['y_site_mean']:.3f}), others {a['other_rmse']:.4f}")
        print(f"  without site theta*={np.round(w['theta'], 3).tolist()} site rmse {w['site_rmse']:.3f} (sim mean {w['site_sim_mean']:.3f}), others {w['other_rmse']:.4f}", flush=True)
        return out
    finally:
        prob.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='Bellingham:150,NorthPenn:192,Fairfield:178', help='comma list of dataset:site_node_id')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    torch.set_num_threads(args.threads)
    results = {'platform': platform.platform(), 'torch': torch.__version__, 'epanet': lib.version(), 'args': vars(args), 'cases': []}
    with tempfile.TemporaryDirectory(prefix='g4si_') as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for case in args.cases.split(','):
            name, site = case.split(':')
            results['cases'].append(run_case(name, site, lib, workdir, args))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1))
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
