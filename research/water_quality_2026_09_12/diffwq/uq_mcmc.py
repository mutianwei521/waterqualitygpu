"""R2-8: MCMC posterior of the grouped chlorine map (Kb + Kw per diameter class) as a check on the
local (Laplace) uncertainty of uq.py.

Posterior: uniform prior on the box of the calibration, Jeffreys prior on sigma^2 integrated out,
    p(theta | y) ~ SSE(theta)^(-n/2)  on the box,
so parameters resting on a bound get a proper distribution instead of no interval. Predictions add
noise with sigma^2 | theta, y ~ SSE / chi2_n.

Sampler: HMC with reflection at the box walls (exact for a uniform prior). The full data and every
leave-one-site-out fold are groups of C chains in ONE batched replay (rows are independent, so the
gradient of the summed log density is the per-row gradient); step size (dual averaging) and diagonal
mass matrix are adapted per group in the warm-up; jittered number of leapfrog steps; optional CUDA
graph of the batched replay. Diagnostics: split R-hat and bulk ESS per parameter and group.

Outputs:
  full  posterior marginals, mass near the bounds, correlations, predictive bands (cf. uq.py);
  loso  posterior predictive coverage of each held-out site from the other sites.

Usage (from research/water_quality_2026_09_12):
    python -m diffwq.uq_mcmc --datasets NorthPenn --device cuda --chains 16 --out mcmc.json
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

from . import epanet_api as api
from .calib import CalibProblem
from .uq import Engine, LEVELS, linearised, _box

NEAR = 0.01     # "at the bound": within this distance of a wall (units of the coefficients)


# ----------------------------------------------------------------------------- target
class Target:
    """Batched log posterior and gradient; row r uses the observations of masks[r // C]."""

    def __init__(self, eng, masks, C, graph=False):
        self.eng, self.G, self.C = eng, masks.shape[0], C
        self.m = torch.as_tensor(np.repeat(masks, C, 0), dtype=torch.float64, device=eng.dev)   # [G*C, M]
        self.n = self.m.sum(1)
        lo, hi = _box(eng.p.bounds)
        self.lo = torch.as_tensor(lo, device=eng.dev)
        self.hi = torch.as_tensor(hi, device=eng.dev)
        self.calls = 0
        self.sim, self.graph_error, self.graphed = eng.simulate, None, False
        if graph and eng.dev.type == 'cuda':
            try:
                th = torch.as_tensor(np.tile(eng.p.theta_nominal, (self.G * C, 1)), device=eng.dev).requires_grad_(True)
                self.sim = torch.cuda.make_graphed_callables(eng.simulate, (th,), num_warmup_iters=3); self.graphed = True
            except Exception as exc:
                self.graph_error = f'{type(exc).__name__}: {str(exc)[:300]}'
                self.sim = eng.simulate

    def logp_grad(self, x):
        self.calls += 1
        x = x.detach().requires_grad_(True)
        r = self.sim(x) - self.eng.y
        lp = -0.5 * self.n * torch.log((self.m * r * r).sum(1))
        g, = torch.autograd.grad(lp.sum(), x)
        return lp.detach(), g


def reflect(x, p, lo, hi):
    """Reflect positions into [lo, hi] and flip the momenta of the reflected coordinates."""
    for _ in range(8):
        below, above = x < lo, x > hi
        if not (below.any() or above.any()):
            break
        x = torch.where(below, 2 * lo - x, x)
        x = torch.where(above, 2 * hi - x, x)
        p = torch.where(below | above, -p, p)
    return x.clamp(lo, hi), p


def hmc(tgt, x0, warmup, draws, rng, target_accept=0.8, L_mean=10):
    """Batched HMC on x0 [G*C, p], adapted per group. Returns draws [draws, G*C, p] and per-group info."""
    dev = x0.device
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(2 ** 31)))
    G, C = tgt.G, tgt.C
    R, P = x0.shape
    x = x0.clone()
    lp, g = tgt.logp_grad(x)
    minv = np.full((G, P), 0.05 ** 2)       # inverse mass = rough posterior variance
    log_eps = np.full(G, np.log(0.1)); log_eps_bar = np.zeros(G); H = np.zeros(G); mu = np.zeros(G)
    gamma, t0, kappa = 0.05, 10.0, 0.75
    win, m_it = [], 0
    out = np.zeros((draws, R, P), dtype=np.float32)
    acc = np.zeros(G)
    rep = lambda a: torch.as_tensor(np.repeat(a, C, 0), device=dev)
    eps_r, minv_r = rep(np.exp(log_eps))[:, None], rep(minv)
    for it in range(warmup + draws):
        L = int(rng.integers(max(1, L_mean // 2), L_mean * 3 // 2 + 1))
        p0 = torch.randn(R, P, dtype=torch.float64, device=dev, generator=gen) / minv_r.sqrt()
        xn, pn, gn = x, p0 + 0.5 * eps_r * g, g
        for k in range(L):
            xn = xn + eps_r * minv_r * pn
            xn, pn = reflect(xn, pn, tgt.lo, tgt.hi)
            lpn, gn = tgt.logp_grad(xn)
            if k < L - 1:
                pn = pn + eps_r * gn
        pn = pn + 0.5 * eps_r * gn
        H0 = lp - 0.5 * (p0 * p0 * minv_r).sum(1)
        H1 = lpn - 0.5 * (pn * pn * minv_r).sum(1)
        a = torch.exp(torch.clamp(H1 - H0, max=0.0))
        a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
        keep = torch.rand(R, dtype=torch.float64, device=dev, generator=gen) < a
        x = torch.where(keep[:, None], xn, x)
        lp = torch.where(keep, lpn, lp)
        g = torch.where(keep[:, None], gn, g)
        abar = a.view(G, C).mean(1).cpu().numpy()
        if it < warmup:
            m_it += 1
            H = (1 - 1 / (m_it + t0)) * H + (target_accept - abar) / (m_it + t0)
            log_eps = mu - np.sqrt(m_it) / gamma * H
            eta = m_it ** -kappa
            log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
            if it >= warmup // 4:
                win.append(x.view(G, C, P).cpu().numpy())
            if it + 1 in (warmup // 2, warmup * 3 // 4) and len(win) > 5:
                minv = np.maximum(np.concatenate(win, 1).var(1), 1e-8)
                win, m_it = [], 0
                mu, H, log_eps_bar = np.log(10 * np.exp(log_eps)), np.zeros(G), np.zeros(G)
            if it == warmup - 1:
                log_eps = log_eps_bar
            eps_r, minv_r = rep(np.exp(log_eps))[:, None], rep(minv)
        else:
            out[it - warmup] = x.cpu().numpy()
            acc += abar
    info = [{'accept': float(acc[gi] / max(draws, 1)), 'step': float(np.exp(log_eps[gi])),
             'minv': minv[gi].tolist(), 'L_mean': L_mean} for gi in range(G)]
    return out, info


# ----------------------------------------------------------------------------- diagnostics
def split_rhat(ch):
    """ch [n, C] -> split R-hat."""
    n = ch.shape[0] // 2
    s = np.concatenate([ch[:n], ch[n:2 * n]], 1)
    W = s.var(0, ddof=1).mean()
    B = n * s.mean(0).var(ddof=1)
    if W <= 0:
        return 1.0
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def ess(ch):
    """ch [n, C] -> ESS (Geyer initial positive sequence on the chain-averaged autocorrelation)."""
    n, C = ch.shape
    x = ch - ch.mean(0)
    v = (x * x).mean()
    if v <= 0:
        return float(n * C)
    f = np.fft.rfft(x, 2 * n, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:n].real.mean(1) / (n * v)
    tau, k = -1.0, 0
    while k + 1 < n:
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
        k += 2
    return float(n * C / max(tau, 1e-12))


def diagnostics(d):
    return {'rhat': [split_rhat(d[:, :, j]) for j in range(d.shape[2])], 'ess': [ess(d[:, :, j]) for j in range(d.shape[2])]}


# ----------------------------------------------------------------------------- summaries
def summarise(flat, prob, theta_star, lin):
    lo, hi = _box(prob.bounds)
    q = np.quantile(flat, [0.025, 0.5, 0.975], axis=0)
    return {'names': prob.param_names, 'theta_star': np.asarray(theta_star).tolist(),
            'active_laplace': lin['active'].tolist(), 'sd_laplace_free': lin['sd'].tolist(),
            'mean': flat.mean(0).tolist(), 'sd': flat.std(0).tolist(),
            'q025': q[0].tolist(), 'q50': q[1].tolist(), 'q975': q[2].tolist(),
            'p_near_lower': (flat - lo <= NEAR).mean(0).tolist(), 'p_near_upper': (hi - flat <= NEAR).mean(0).tolist(),
            'corr': np.corrcoef(flat.T).tolist(), 'corr_laplace_free': lin['corr'].tolist()}


def bands(pred):
    return {lv: np.quantile(pred, [0.5 - lv / 2, 0.5 + lv / 2], axis=0) for lv in LEVELS}


# ----------------------------------------------------------------------------- dataset
def run_dataset(name, lib, workdir, args, rng):
    prob = CalibProblem(name, lib, workdir, device='cpu', fix_kb=args.fix_kb)
    eng = Engine(prob, args.device, graph=False)
    fit_eng = Engine(prob, 'cpu')
    y = prob.obs.y
    M, S = len(prob.obs), len(prob.obs.site_node_id)
    nf = 0 if args.no_loso else (S if args.max_folds is None else min(S, args.max_folds))
    lo, hi = _box(prob.bounds)
    out = {'dataset': name, 'n_obs': M, 'n_sites': S, 'param_names': prob.param_names, 'bounds': prob.bounds}
    # groups: full data, then one per held-out site; chains start near the group's L-BFGS-B optimum
    masks = np.array([np.ones(M)] + [(prob.obs.site != si).astype(np.float64) for si in range(nf)])
    stars, lins = [], []
    for gi, mk in enumerate(masks):
        th, _, _ = fit_eng.fit(prob.theta_nominal, None if gi == 0 else mk)
        r = y - eng.predict(th)
        J = eng.jacobian(th)
        stars.append(th)
        lins.append(linearised(J[mk > 0], r[mk > 0], th, prob.bounds))
    G, C = len(masks), args.chains
    tgt = Target(eng, masks, C, graph=args.graph)
    x0 = np.concatenate([np.clip(st + rng.uniform(-args.init_jitter, args.init_jitter, (C, len(st))), lo, hi) for st in stars])
    t0 = time.perf_counter()
    d, info = hmc(tgt, torch.as_tensor(x0, device=eng.dev), args.warmup, args.draws, rng, L_mean=args.leapfrog)
    if eng.dev.type == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    out['sampler'] = {'groups': G, 'chains_per_group': C, 'batch': G * C, 'warmup': args.warmup, 'draws': args.draws,
                      'batched_grads': tgt.calls, 'wall_s': wall, 's_per_batched_grad': wall / max(tgt.calls, 1),
                      'graph': tgt.graphed, 'graph_error': tgt.graph_error, 'device': str(eng.dev)}
    print(f"[{name}] {G} groups x {C} chains: {tgt.calls} batched grads in {wall:.0f}s "
          f"({wall / max(tgt.calls, 1):.3f} s each, graph={out['sampler']['graph']})", flush=True)
    lo_b = {lv: np.full(M, np.nan) for lv in LEVELS}
    hi_b = {lv: np.full(M, np.nan) for lv in LEVELS}
    folds = []
    for gi in range(G):
        dg = d[:, gi * C:(gi + 1) * C].astype(np.float64)
        diag = diagnostics(dg)
        flat = dg.reshape(-1, dg.shape[2])
        th = flat[rng.choice(flat.shape[0], size=min(args.pred_samples, flat.shape[0]), replace=False)]
        mu, _ = eng.sample_predict(th, args.chunk)
        mk = masks[gi]
        sse = ((mu - y) ** 2 * mk).sum(1)
        s2 = sse / rng.chisquare(mk.sum(), size=len(sse))
        pred = mu + rng.standard_normal(mu.shape) * np.sqrt(s2)[:, None]
        rec = dict(info[gi], rhat=diag['rhat'], ess=diag['ess'])
        if gi == 0:
            b, bp = bands(pred), bands(mu)
            rf = y - eng.predict(stars[0])
            out['full'] = {'fit_rmse': float(np.sqrt(np.mean(rf ** 2))), 'sigma_laplace': float(np.sqrt(lins[0]['sigma2'])),
                           'sampler': rec, 'posterior': summarise(flat, prob, stars[0], lins[0]),
                           'sigma_posterior_median': float(np.sqrt(np.median(s2))),
                           'in_sample_coverage': {str(lv): float(np.mean((y >= b[lv][0]) & (y <= b[lv][1]))) for lv in LEVELS},
                           'pred_halfwidth_mean': {str(lv): float(np.mean(b[lv][1] - b[lv][0]) / 2) for lv in LEVELS},
                           'param_only_halfwidth_mean': {str(lv): float(np.mean(bp[lv][1] - bp[lv][0]) / 2) for lv in LEVELS},
                           'param_share_of_pred_var': float(mu.var(0).mean() / pred.var(0).mean())}
            s = out['full']
            print(f"  full: rhat max {max(diag['rhat']):.3f} ess min {min(diag['ess']):.0f} accept {rec['accept']:.2f} step {rec['step']:.3g}", flush=True)
            print(f"  q50 {np.round(s['posterior']['q50'], 3).tolist()} q025 {np.round(s['posterior']['q025'], 3).tolist()} "
                  f"q975 {np.round(s['posterior']['q975'], 3).tolist()} theta* {np.round(stars[0], 3).tolist()}", flush=True)
            print(f"  near lower {np.round(s['posterior']['p_near_lower'], 2).tolist()} near upper "
                  f"{np.round(s['posterior']['p_near_upper'], 2).tolist()} sd_laplace {np.round(lins[0]['sd'], 3).tolist()}", flush=True)
            print(f"  in-sample cov95 {s['in_sample_coverage']['0.95']:.3f} hw95 {s['pred_halfwidth_mean']['0.95']:.4f} "
                  f"param share {s['param_share_of_pred_var']:.3f}", flush=True)
        else:
            held = mk == 0
            bf = bands(pred[:, held])
            for lv in LEVELS:
                lo_b[lv][held], hi_b[lv][held] = bf[lv][0], bf[lv][1]
            cov95 = float(np.mean((y[held] >= bf[0.95][0]) & (y[held] <= bf[0.95][1])))
            folds.append({'site': prob.obs.site_node_id[gi - 1], 'n_held': int(held.sum()), 'theta_star': stars[gi].tolist(),
                          'active_laplace': lins[gi]['active'].tolist(), 'rhat_max': max(diag['rhat']), 'ess_min': min(diag['ess']),
                          'accept': rec['accept'], 'coverage95': cov95,
                          'p_near_bound': ((flat - lo <= NEAR) | (hi - flat <= NEAR)).mean(0).tolist()})
            f = folds[-1]
            print(f"  LOSO {f['site']:>8s} ({f['n_held']:3d}): cov95 {cov95:.3f} rhat max {f['rhat_max']:.3f} "
                  f"ess min {f['ess_min']:.0f} accept {f['accept']:.2f}", flush=True)
    if folds:
        ok = ~np.isnan(lo_b[0.95])
        out['loso'] = {'n_heldout': int(ok.sum()), 'n_folds': len(folds),
                       'coverage': {str(lv): float(np.mean((y[ok] >= lo_b[lv][ok]) & (y[ok] <= hi_b[lv][ok]))) for lv in LEVELS},
                       'halfwidth_mean': {str(lv): float(np.mean(hi_b[lv][ok] - lo_b[lv][ok]) / 2) for lv in LEVELS},
                       'folds': folds}
        print(f"  LOSO pooled coverage {out['loso']['coverage']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='Fairfield,Bellingham,NorthPenn')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--chains', type=int, default=16, help='chains per group (full data and each fold)')
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--leapfrog', type=int, default=10)
    ap.add_argument('--init-jitter', type=float, default=0.2)
    ap.add_argument('--pred-samples', type=int, default=4096)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--no-loso', action='store_true')
    ap.add_argument('--no-graph', dest='graph', action='store_false')
    ap.add_argument('--max-folds', type=int, default=None)
    ap.add_argument('--fix-kb', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    torch.set_num_threads(args.threads or int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or max(1, os.cpu_count() // 2))
    results = {'torch': torch.__version__, 'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu',
               'epanet': lib.version(), 'args': vars(args), 'levels': list(LEVELS), 'datasets': []}
    rng = np.random.default_rng(args.seed)
    with tempfile.TemporaryDirectory(prefix='mcmc_', ignore_cleanup_errors=True) as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in args.datasets.split(','):
            results['datasets'].append(run_dataset(name, lib, workdir, args, rng))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1))
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
