"""G4 / H1 probe: identifiability of per-pipe wall coefficients from the field data.

At the grouped optimum theta* (Kb + Kw per diameter class, from calib.py's ADJ-LBFGSB) the
residual Jacobian with respect to [Kb, Kw_1 .. Kw_P] (one Kw per pipe) is obtained in ONE batched
adjoint sweep (B = M observations, one-hot output weights). Its singular spectrum gives the number
of parameter directions the sampling campaign actually constrains; zero columns are pipes no
sample ever "sees". A Tikhonov-regularised per-pipe fit (penalty towards the group value) and a
leave-one-site-out cross-validation of grouped vs per-pipe models show whether the extra freedom
buys predictive skill or only fits noise.

Usage (from research/water_quality_2026_09_12):
    python -m diffwq.perpipe --datasets Fairfield,Bellingham,NorthPenn --out g4_perpipe.json
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
from scipy.optimize import minimize

from . import epanet_api as api
from .calib import CalibProblem, run_adj_lbfgsb


class PerPipe:
    def __init__(self, prob: CalibProblem):
        self.p = prob
        self.pipes = np.flatnonzero(prob.is_pipe)                     # link indices of pipes
        self.P = len(self.pipes)
        self.pipes_t = torch.as_tensor(self.pipes, device=prob.dev)
        self.group_of_pipe = prob.link_group[self.pipes]

    def theta_from_grouped(self, theta_g):
        return np.concatenate(([theta_g[0]], theta_g[1 + self.group_of_pipe]))

    def simulate(self, theta):                                        # theta [B, 1 + P]
        p, B = self.p, theta.shape[0]
        L, T = p.net.nlinks, p.net.ntanks
        kb = theta[:, 0:1].expand(B, L + 1)
        kw = torch.zeros(B, L + 1, dtype=theta.dtype, device=p.dev)
        kw[:, self.pipes_t] = theta[:, 1:]
        tkb = theta[:, 0:1].expand(B, T + 1)
        out = p.eng.forward(kb=kb, kw=kw, tank_kb=tkb).reshape(B, -1)
        return out[:, p.obs_i0] * (1.0 - p.obs_w) + out[:, p.obs_i1] * p.obs_w

    def jacobian(self, theta):
        M = len(self.p.obs)
        th = torch.as_tensor(np.asarray(theta, dtype=np.float64), device=self.p.dev).repeat(M, 1).requires_grad_(True)
        self.simulate(th).diagonal().sum().backward()
        return th.grad.cpu().numpy().copy()

    def fit(self, x0, prior, lam, mask=None, maxiter=300):
        """min_theta  sum_m mask_m (sim_m - y_m)^2 + lam * sum_k (Kw_k - prior_k)^2, bounds as in prob."""
        p = self.p
        m_t = torch.as_tensor(np.ones(len(p.obs)) if mask is None else mask, dtype=torch.float64, device=p.dev)
        prior_t = torch.as_tensor(prior, device=p.dev)
        n = [0]

        def fg(x):
            n[0] += 1
            th = torch.as_tensor(x, device=p.dev).view(1, -1).requires_grad_(True)
            r = self.simulate(th)[0] - p.y
            d = th[0, 1:] - prior_t
            loss = (m_t * r * r).sum() + lam * (d * d).sum()
            loss.backward()
            return loss.item(), th.grad[0].cpu().numpy().copy()
        bounds = [p.bounds[0]] + [p.bounds[1]] * self.P
        t0 = time.perf_counter()
        res = minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9, 'maxcor': 20})
        return res.x, n[0], time.perf_counter() - t0

    def predict(self, theta):
        with torch.no_grad():
            return self.simulate(torch.as_tensor(np.asarray(theta, dtype=np.float64), device=self.p.dev).view(1, -1))[0].cpu().numpy()


def loso_cv(prob, pp, theta_g_star, lam):
    """Leave-one-site-out: grouped model (calib.py parameters) vs per-pipe model with Tikhonov lam."""
    obs = prob.obs
    S = len(obs.site_node_id)
    err_g, err_p = np.zeros(len(obs)), np.zeros(len(obs))
    n_g = n_p = 0
    for s in range(S):
        mask = (obs.site != s).astype(np.float64)
        # grouped model on the training sites (exact gradient, warm start at theta*)
        m_t = torch.as_tensor(mask, device=prob.dev)
        def fg(x):
            th = torch.as_tensor(x, device=prob.dev).view(1, -1).requires_grad_(True)
            r = prob.simulate_torch(th)[0] - prob.y
            loss = (m_t * r * r).sum(); loss.backward()
            return loss.item(), th.grad[0].cpu().numpy().copy()
        res = minimize(fg, theta_g_star, jac=True, method='L-BFGS-B', bounds=prob.bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
        n_g += res.nfev
        with torch.no_grad():
            sim_g = prob.simulate_torch(torch.as_tensor(res.x, device=prob.dev).view(1, -1))[0].cpu().numpy()
        # per-pipe model: start and prior from the training-site grouped fit (held-out site unseen)
        x0 = pp.theta_from_grouped(res.x)
        xp, nfev, _ = pp.fit(x0, x0[1:], lam, mask=mask)
        n_p += nfev
        sim_p = pp.predict(xp)
        held = obs.site == s
        err_g[held] = sim_g[held] - obs.y[held]
        err_p[held] = sim_p[held] - obs.y[held]
    return {'lam': lam, 'cv_rmse_grouped': float(np.sqrt(np.mean(err_g ** 2))),
            'cv_rmse_perpipe': float(np.sqrt(np.mean(err_p ** 2))), 'n_grad_grouped': int(n_g), 'n_grad_perpipe': int(n_p),
            'cv_site_rmse_grouped': {obs.site_node_id[s]: float(np.sqrt(np.mean(err_g[obs.site == s] ** 2))) for s in range(S)},
            'cv_site_rmse_perpipe': {obs.site_node_id[s]: float(np.sqrt(np.mean(err_p[obs.site == s] ** 2))) for s in range(S)}}


def run_dataset(name, lib, workdir, args):
    prob = CalibProblem(name, lib, workdir, device=args.device)
    try:
        pp = PerPipe(prob)
        # grouped optimum (exact gradient from the INP values)
        rec = run_adj_lbfgsb(prob, prob.theta_nominal)
        th_g = np.asarray(rec['theta'])
        out = {'dataset': name, 'n_obs': len(prob.obs), 'n_pipes': pp.P, 'param_names_grouped': prob.param_names,
               'grouped': {k: rec[k] for k in ('theta', 'rmse', 'sse', 'n_eval', 'wall_s', 'jtj_eigvals', 'ci95_halfwidth', 'correlation')}}
        print(f"[{name}] grouped theta*={np.round(th_g, 4).tolist()} rmse={rec['rmse']:.4f}", flush=True)
        # per-pipe Jacobian at theta*, one batched adjoint sweep
        x_star = pp.theta_from_grouped(th_g)
        t0 = time.perf_counter(); J = pp.jacobian(x_star); t_j = time.perf_counter() - t0
        sv = np.linalg.svd(J, compute_uv=False)
        col = np.linalg.norm(J[:, 1:], axis=0)
        order = np.argsort(-col)
        sigma = rec['rmse']
        out['perpipe_jacobian'] = {
            'shape': list(J.shape), 'wall_s': t_j, 'singular_values': sv.tolist(),
            'rank_rel_1e-2': int(np.sum(sv > 1e-2 * sv[0])), 'rank_rel_1e-3': int(np.sum(sv > 1e-3 * sv[0])),
            # directions whose 1-sigma parameter uncertainty (sigma / s_i) is below 1 ft/day
            'n_directions_sd_below_1': int(np.sum(sigma / np.maximum(sv, 1e-300) < 1.0)),
            'n_pipes_zero_sensitivity': int(np.sum(col == 0.0)),
            'n_pipes_sensitivity_above_1pct_of_max': int(np.sum(col > 0.01 * col.max())),
            'top10_pipes': [(prob.net.link_ids[pp.pipes[i]], float(col[i])) for i in order[:10]],
            'kb_column_norm': float(np.linalg.norm(J[:, 0])), 'kw_column_norm_max': float(col.max())}
        pj = out['perpipe_jacobian']
        print(f"  per-pipe Jacobian {J.shape} in {t_j:.1f}s: rank(1e-2)={pj['rank_rel_1e-2']} rank(1e-3)={pj['rank_rel_1e-3']} "
              f"directions with sd<1 ft/day: {pj['n_directions_sd_below_1']}; pipes never seen: {pj['n_pipes_zero_sensitivity']}/{pp.P}",
              flush=True)
        # Tikhonov per-pipe fits on all data
        prior = x_star[1:]
        out['perpipe_fits'] = []
        for lam in args.lams:
            xp, nfev, t_fit = pp.fit(x_star, prior, lam)
            r = pp.predict(xp) - prob.obs.y
            out['perpipe_fits'].append({'lam': lam, 'rmse': float(np.sqrt(np.mean(r * r))), 'n_grad': nfev, 'wall_s': t_fit,
                                        'kw_min': float(xp[1:].min()), 'kw_max': float(xp[1:].max()), 'kb': float(xp[0]),
                                        'n_kw_moved_gt_0.1': int(np.sum(np.abs(xp[1:] - prior) > 0.1))})
            print(f"  per-pipe fit lam={lam}: rmse={out['perpipe_fits'][-1]['rmse']:.4f} ({nfev} gradients, {t_fit:.0f}s)", flush=True)
        # leave-one-site-out CV
        out['loso'] = []
        for lam in args.cv_lams:
            t0 = time.perf_counter()
            cv = loso_cv(prob, pp, th_g, lam)
            cv['wall_s'] = time.perf_counter() - t0
            out['loso'].append(cv)
            print(f"  LOSO lam={lam}: CV-RMSE grouped {cv['cv_rmse_grouped']:.4f} vs per-pipe {cv['cv_rmse_perpipe']:.4f} "
                  f"({cv['n_grad_grouped']}+{cv['n_grad_perpipe']} gradients, {cv['wall_s']:.0f}s)", flush=True)
        return out
    finally:
        prob.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='Fairfield,Bellingham,NorthPenn')
    ap.add_argument('--lams', default='0.01,0.1,1')
    ap.add_argument('--cv-lams', default='0.1')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.lams = [float(v) for v in args.lams.split(',')]
    args.cv_lams = [float(v) for v in args.cv_lams.split(',')]
    lib = api.Library(args.lib)
    torch.set_num_threads(args.threads or max(1, os.cpu_count() // 2))
    results = {'platform': platform.platform(), 'torch': torch.__version__, 'epanet': lib.version(), 'args': vars(args), 'datasets': []}
    with tempfile.TemporaryDirectory(prefix='g4p_') as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in args.datasets.split(','):
            results['datasets'].append(run_dataset(name, lib, workdir, args))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1))
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
