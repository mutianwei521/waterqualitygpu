"""R1-6: leave-one-site-out test with a site-level error term, next to the independent-error model of uq.py.

Error model y_ij = c_ij(theta) + b_i + e_ij with site offsets b_i ~ N(0, tau2) and e_ij ~ N(0, s2), all independent.
For each held-out site the grouped coefficients are refitted on the other sites exactly as in uq.loso (same engine,
same start, same mask); tau2 and s2 are the one-way random-effects ANOVA (method of moments) estimates from the
training residuals; the covariance of the free parameters is the least-squares sandwich
(JtJ)^-1 Jt V J (JtJ)^-1 with V = s2 I + tau2 Z Zt; the predictive variance of a held-out observation is
J Sigma Jt + tau2 + s2. The independent-error band of uq.py is recomputed in the same run for comparison.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.uq_site --datasets Fairfield,Bellingham,NorthPenn --device cuda --out uq_site.json
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from scipy.stats import norm, t as student_t

from . import epanet_api as api
from .uq import Engine, linearised, CalibProblem


def anova_components(r, site):
    """Method-of-moments (one-way random effects) estimates of the site variance tau2 and the within-site variance s2."""
    labels = np.unique(site)
    k, N = len(labels), len(r)
    n = np.array([(site == s).sum() for s in labels])
    m = np.array([r[site == s].mean() for s in labels])
    msw = sum(((r[site == s] - r[site == s].mean()) ** 2).sum() for s in labels) / max(N - k, 1)
    msb = float((n * (m - r.mean()) ** 2).sum() / max(k - 1, 1))
    n0 = (N - (n ** 2).sum() / N) / max(k - 1, 1)
    return max((msb - msw) / n0, 0.0), float(msw)


def run(name, lib, wd, args):
    prob = CalibProblem(name, lib, wd, device='cpu', fix_kb=False)
    eng = Engine(prob, args.device, graph=True)
    obs, y = prob.obs, prob.obs.y
    M, S = len(obs), len(obs.site_node_id)
    out = {k: np.full(M, np.nan) for k in ('resid', 'hw_iid', 'hw_site', 'var_par_iid', 'var_par_site')}
    folds = []
    for s in range(S):
        held, train = obs.site == s, obs.site != s
        t0 = time.perf_counter()
        th, n_grad, _ = eng.fit(prob.theta_nominal, train.astype(np.float64))
        r = y - eng.predict(th)
        J = eng.jacobian(th)
        lin = linearised(J[train], r[train], th, prob.bounds)              # independent errors, as in uq.loso
        vp_iid = np.einsum('mi,ij,mj->m', J[held], lin['cov'], J[held])
        hw_iid = student_t.ppf(0.975, lin['dof']) * np.sqrt(lin['sigma2'] + vp_iid)
        tau2, s2 = anova_components(r[train], obs.site[train])
        free = lin['free']
        cov = np.zeros((len(th), len(th)))
        if free.any():
            Jf = J[train][:, free]
            A = np.linalg.pinv(Jf.T @ Jf, rcond=1e-12, hermitian=True)
            meat = s2 * (Jf.T @ Jf)
            for q in np.unique(obs.site[train]):
                g = Jf[obs.site[train] == q].sum(axis=0)
                meat += tau2 * np.outer(g, g)
            cov[np.ix_(free, free)] = A @ meat @ A
        vp_site = np.einsum('mi,ij,mj->m', J[held], cov, J[held])
        hw_site = norm.ppf(0.975) * np.sqrt(tau2 + s2 + vp_site)
        out['resid'][held], out['hw_iid'][held], out['hw_site'][held] = r[held], hw_iid, hw_site
        out['var_par_iid'][held], out['var_par_site'][held] = vp_iid, vp_site
        folds.append({'site': obs.site_node_id[s], 'n_held': int(held.sum()), 'theta': th.tolist(), 'n_grad': n_grad,
                      'sigma2_iid': lin['sigma2'], 'tau2': tau2, 's2': s2,
                      'cover_iid': float(np.mean(np.abs(r[held]) <= hw_iid)), 'cover_site': float(np.mean(np.abs(r[held]) <= hw_site)),
                      'wall_s': time.perf_counter() - t0})
        f = folds[-1]
        print(f"  [{name}] site {f['site']:>6s}: tau2={tau2:.4f} s2={s2:.4f} sigma2_iid={lin['sigma2']:.4f} "
              f"cover iid {f['cover_iid']:.2f} site {f['cover_site']:.2f}", flush=True)
    ok = ~np.isnan(out['resid'])
    res = {'dataset': name, 'n_obs': int(ok.sum()), 'n_sites': S,
           'coverage95_iid': float(np.mean(np.abs(out['resid'][ok]) <= out['hw_iid'][ok])),
           'coverage95_site': float(np.mean(np.abs(out['resid'][ok]) <= out['hw_site'][ok])),
           'mean_halfwidth_iid': float(np.mean(out['hw_iid'][ok])), 'mean_halfwidth_site': float(np.mean(out['hw_site'][ok])),
           'median_tau2': float(np.median([f['tau2'] for f in folds])), 'median_s2': float(np.median([f['s2'] for f in folds])),
           'sites_below_095_iid': int(sum(f['cover_iid'] < 0.95 for f in folds)), 'sites_below_095_site': int(sum(f['cover_site'] < 0.95 for f in folds)),
           'folds': folds}
    print(f"[{name}] 95% coverage: independent errors {res['coverage95_iid']:.3f} (mean half-width {res['mean_halfwidth_iid']:.3f}), "
          f"site term {res['coverage95_site']:.3f} ({res['mean_halfwidth_site']:.3f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='Fairfield,Bellingham,NorthPenn')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lib = api.Library(a.lib)
    wd = tempfile.mkdtemp(prefix='diffwq_uqsite_', dir=os.environ.get('TMPDIR'))
    R = [run(n, lib, wd, a) for n in a.datasets.split(',')]
    Path(a.out).write_text(json.dumps({'engine_version': lib.version(), 'datasets': R}, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
