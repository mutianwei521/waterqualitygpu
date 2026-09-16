"""G4 / H2 probe: uncertainty of the calibrated chlorine map, and whether that uncertainty is honest.

At the grouped optimum theta* (Kb + Kw per diameter class) of a field data set, the exact residual
Jacobian J (one batched adjoint sweep, B = M one-hot outputs) gives the linearised (Gauss-Newton /
Laplace) covariance  Sigma = sigma^2 (J_F^T J_F)^-1  over the parameters F not resting on a bound,
sigma^2 = SSE / (M - |F|). From it:
  * linearised confidence band (parameters only) and prediction interval (+ sigma^2) at every sample;
  * Laplace sampling: S draws from N(theta*, Sigma) kept inside the box (rejection, else clipping)
    pushed through the exact forward map in GPU batches -> non-linear bands at the samples and the
    sd of the chlorine field at every node x quality step (the "map with a certificate");
  * leave-one-site-out coverage test: refit on the other sites, build the intervals for the held-out
    site from training data only, count how often its measurements fall inside at nominal levels
    50/80/90/95 %, also for the naive interval that ignores parameter uncertainty.
The B = 1 fits (L-BFGS-B, exact gradient, same settings as calib.py) run through a CUDA-graph
capture of the replay when the device is CUDA (calib.py has the same capture behind its method
adjg; its method adj runs eagerly); the batched parts run eagerly with B = chunk.

Usage (from research/water_quality_2026_09_12):
    python -m diffwq.uq --datasets Fairfield,Bellingham,NorthPenn --device cuda --out h2_uq.json
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
from scipy.stats import t as student_t

from . import epanet_api as api
from .calib import CalibProblem
from .torch_wq import TorchWQ

LEVELS = (0.5, 0.8, 0.9, 0.95)
LBFGSB = {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9, 'maxcor': 20}


class Engine:
    """Forward map of one CalibProblem on a chosen device: theta [B, p] -> samples [B, M] or the full
    node x step field; the B = 1 loss + gradient optionally through a CUDA graph."""

    def __init__(self, prob, device, graph=False):
        self.p, self.dev = prob, torch.device(device)
        same = self.dev.type == prob.dev.type and (self.dev.index or 0) == (prob.dev.index or 0)
        self.eng = prob.eng if same else TorchWQ(prob.net, prob.sch, prob.reactflag, device=self.dev)
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        n = prob.net
        self.i0, self.i1, self.w = t(prob._i0), t(prob._i1), t(prob._w, torch.float64)
        self.y = t(prob.obs.y, torch.float64)
        self.lg = t(prob.link_group)
        # c0 / src cached on the device: forward() would otherwise copy them host->device on every
        # call, which is not allowed inside a CUDA-graph capture
        self.c0 = t(n.c0 * n.ucf['QUALITY'], torch.float64).view(1, -1)
        self.src = t([n.sources.get(i, (0, 0.0, 0))[1] for i in range(n.nnodes + 1)], torch.float64).view(1, -1)
        self.graphed, self.graph_error = None, None
        if graph and self.dev.type == 'cuda':
            try:
                th = torch.as_tensor(prob.theta_nominal, device=self.dev).view(1, -1).requires_grad_(True)
                self.graphed = torch.cuda.make_graphed_callables(self.simulate, (th,), num_warmup_iters=3,
                                                                 allow_unused_input=True)
            except Exception as exc:
                self.graph_error = f'{type(exc).__name__}: {str(exc)[:300]}'

    def expand(self, theta):
        p, B = self.p, theta.shape[0]
        L, T = p.net.nlinks, p.net.ntanks
        kb_col = torch.full_like(theta[:, 0:1], p.kb_fixed) if p.fix_kb else theta[:, 0:1]
        kw = theta[:, p.kw_off + self.lg]
        tkb = (theta[:, -1:] if p.free_tank_kb else kb_col).expand(B, T + 1)
        return kb_col.expand(B, L + 1), kw, tkb

    def field(self, theta):
        """[B, n_snap, N+1] node quality at every quality step (user units)."""
        kb, kw, tkb = self.expand(theta)
        B = theta.shape[0]
        return self.eng.forward(kb=kb, kw=kw, tank_kb=tkb, c0=self.c0.expand(B, -1), src=self.src.expand(B, -1))

    def simulate(self, theta):
        """[B, p] -> simulated observations [B, M] (linear interpolation between snapshots)."""
        out = self.field(theta).reshape(theta.shape[0], -1)
        return out[:, self.i0] * (1.0 - self.w) + out[:, self.i1] * self.w

    def loss_grad(self, x, weights, use_graph=True):
        """Weighted SSE and its gradient at one theta (numpy) from one replay (graphed if captured)."""
        th = torch.tensor(np.asarray(x, dtype=np.float64), device=self.dev).view(1, -1).requires_grad_(True)
        sim = self.graphed(th) if (use_graph and self.graphed is not None) else self.simulate(th)
        r = sim[0] - self.y
        loss = (weights * r * r).sum()
        loss.backward()
        return loss.item(), th.grad[0].cpu().numpy().copy()

    def fit(self, x0, mask=None):
        """L-BFGS-B (calib.run_adj_lbfgsb settings) on the masked SSE; returns (theta, n_grad, wall_s)."""
        w = torch.as_tensor(np.ones(len(self.p.obs)) if mask is None else mask, dtype=torch.float64, device=self.dev)
        n = [0]

        def fg(x):
            n[0] += 1
            return self.loss_grad(x, w)
        t0 = time.perf_counter()
        res = minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=self.p.bounds, options=LBFGSB)
        return res.x, n[0], time.perf_counter() - t0

    def jacobian(self, theta):
        """Residual Jacobian [M, p] from one batched adjoint sweep (B = M one-hot outputs)."""
        M = len(self.p.obs)
        th = torch.tensor(np.asarray(theta, dtype=np.float64), device=self.dev).repeat(M, 1).requires_grad_(True)
        self.simulate(th).diagonal().sum().backward()
        return th.grad.cpu().numpy().copy()

    def predict(self, theta):
        with torch.no_grad():
            th = torch.tensor(np.asarray(theta, dtype=np.float64), device=self.dev).view(1, -1)
            return self.simulate(th)[0].cpu().numpy()

    def _chunks(self, thetas, chunk, fn):
        """fn(theta chunk) over the rows of thetas [S, p]; halves the chunk on CUDA OOM. Returns the chunk used."""
        th_all = torch.tensor(np.asarray(thetas, dtype=np.float64), device=self.dev)
        i = 0
        with torch.no_grad():
            while i < th_all.shape[0]:
                try:
                    fn(th_all[i:i + chunk]); i += chunk
                except torch.cuda.OutOfMemoryError:
                    if chunk == 1:
                        raise
                    chunk //= 2
                    torch.cuda.empty_cache()
        return chunk

    def sample_predict(self, thetas, chunk):
        """[S, M] simulated observations for S parameter draws."""
        outs = []
        used = self._chunks(thetas, chunk, lambda th: outs.append(self.simulate(th).cpu().numpy()))
        return np.concatenate(outs), used

    def sample_field(self, thetas, chunk, ref):
        """Mean and sd over S draws of the node x step field, [n_snap, N+1] each. Sums of deviations
        from the field at ref (theta*) avoid the cancellation of sum / sum-of-squares at nodes whose
        concentration hardly varies."""
        acc = {}
        with torch.no_grad():
            f_ref = self.field(torch.tensor(np.asarray(ref, dtype=np.float64), device=self.dev).view(1, -1))

        def fn(th):
            d = self.field(th) - f_ref
            s, q = d.sum(0), (d * d).sum(0)
            acc['s'] = acc.get('s', 0) + s; acc['q'] = acc.get('q', 0) + q
        used = self._chunks(thetas, chunk, fn)
        S = thetas.shape[0]
        dm = acc['s'] / S
        var = (acc['q'] / S - dm * dm).clamp_min(0.0)
        return (f_ref[0] + dm).cpu().numpy(), var.sqrt().cpu().numpy(), used


# ----------------------------------------------------------------------------- linearised statistics
def _box(bounds):
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def linearised(J, r, theta, bounds, tol=1e-6):
    """Gauss-Newton covariance over the parameters not on a bound (the others are held fixed).
    Also the full-p version (calib.fit_stats convention) for comparison. r = y - yhat."""
    M, p = J.shape
    lo, hi = _box(bounds)
    act = (theta - lo <= tol) | (hi - theta <= tol)
    free = ~act
    dof = max(M - int(free.sum()), 1)
    sigma2 = float(r @ r) / dof
    cov = np.zeros((p, p))
    ev = np.zeros(0)
    if free.any():
        JtJ = J[:, free].T @ J[:, free]
        ev = np.linalg.eigvalsh(JtJ)[::-1]
        cov[np.ix_(free, free)] = sigma2 * np.linalg.pinv(JtJ, rcond=1e-12, hermitian=True)
    sd = np.sqrt(np.diag(cov))
    outer = np.outer(sd, sd)
    cov_all = float(r @ r) / max(M - p, 1) * np.linalg.pinv(J.T @ J, rcond=1e-12, hermitian=True)
    return {'active': act, 'free': free, 'dof': dof, 'sigma2': sigma2, 'cov': cov, 'sd': sd,
            'corr': np.divide(cov, outer, out=np.zeros_like(cov), where=outer > 0),
            'eig_free': ev, 'cond_free': float(ev[0] / ev[-1]) if ev.size and ev[-1] > 0 else float('inf'),
            'sd_all': np.sqrt(np.maximum(np.diag(cov_all), 0.0)),
            'var_param': np.einsum('mi,ij,mj->m', J, cov, J)}      # parameter part of the prediction variance


def draw_theta(theta, lin, bounds, S, rng, oversample=8):
    """S draws from N(theta*, Sigma) restricted to the box: rejection when the acceptance rate allows
    it, otherwise clipping (recorded). Parameters on a bound stay fixed."""
    lo, hi = _box(bounds)
    free = lin['free']
    out = np.tile(np.asarray(theta, dtype=np.float64), (S, 1))
    if not free.any():
        return out, {'mode': 'no-free-parameter', 'frac_in_box': 1.0}
    w, V = np.linalg.eigh(lin['cov'][np.ix_(free, free)])
    A = V * np.sqrt(np.maximum(w, 0.0))
    cand = theta[free] + rng.standard_normal((oversample * S, int(free.sum()))) @ A.T
    inside = np.all((cand >= lo[free]) & (cand <= hi[free]), axis=1)
    if inside.sum() >= S:
        out[:, free] = cand[inside][:S]; mode = 'rejection'
    else:
        out[:, free] = np.clip(cand[:S], lo[free], hi[free]); mode = 'clipped'
    return out, {'mode': mode, 'frac_in_box': float(inside.mean())}


def t_quantiles(dof):
    return {lv: float(student_t.ppf(0.5 + lv / 2, dof)) for lv in LEVELS}


def lin_halfwidths(sigma2, var_param, dof):
    """Half-widths of the linearised intervals per level: prediction (sigma^2 + parameter), parameter
    only (confidence band on the mean), naive (sigma^2 only)."""
    tq = t_quantiles(dof)
    return {'pred': {lv: tq[lv] * np.sqrt(sigma2 + var_param) for lv in LEVELS},
            'param_only': {lv: tq[lv] * np.sqrt(var_param) for lv in LEVELS},
            'naive': {lv: tq[lv] * np.sqrt(np.full_like(var_param, sigma2)) for lv in LEVELS}}


def sampled_bounds(preds, sigma2, rng):
    """Interval bounds per level from parameter draws preds [S, m]: predictive (+ Gaussian noise) and
    parameter only."""
    noise = rng.standard_normal(preds.shape) * np.sqrt(sigma2)
    out = {}
    for kind, arr in (('pred', preds + noise), ('param_only', preds)):
        out[kind] = {lv: np.quantile(arr, [0.5 - lv / 2, 0.5 + lv / 2], axis=0) for lv in LEVELS}
    return out


def coverage_table(y, resid, hw, sb, ok):
    """Empirical coverage and mean half-width per interval type and level over the points ok."""
    tab = {'linearised': {}, 'sampled': {}}
    for kind, d in hw.items():
        tab['linearised'][kind] = {str(lv): {'coverage': float(np.mean(np.abs(resid[ok]) <= d[lv][ok])),
                                             'mean_halfwidth': float(np.mean(d[lv][ok]))} for lv in LEVELS}
    for kind, d in sb.items():
        tab['sampled'][kind] = {str(lv): {'coverage': float(np.mean((y[ok] >= d[lv][0][ok]) & (y[ok] <= d[lv][1][ok]))),
                                          'mean_halfwidth': float(np.mean(d[lv][1][ok] - d[lv][0][ok]) / 2)} for lv in LEVELS}
    return tab


def time_grad(eng, use_graph, reps=5):
    w = torch.ones(len(eng.p.obs), dtype=torch.float64, device=eng.dev)
    sync = torch.cuda.synchronize if eng.dev.type == 'cuda' else (lambda: None)
    eng.loss_grad(eng.p.theta_nominal, w, use_graph); sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        eng.loss_grad(eng.p.theta_nominal, w, use_graph)
    sync()
    return (time.perf_counter() - t0) / reps


# ----------------------------------------------------------------------------- probes
def loso(prob, fit_eng, gpu, args, rng):
    """Leave-one-site-out: refit on the other sites (from the INP start), intervals for the held-out
    site from training data only, pooled coverage over all held-out samples."""
    obs, y = prob.obs, prob.obs.y
    S, M = len(obs.site_node_id), len(obs)
    nan = lambda: np.full(M, np.nan)
    hw = {k: {lv: nan() for lv in LEVELS} for k in ('pred', 'param_only', 'naive')}
    sb = {k: {lv: (nan(), nan()) for lv in LEVELS} for k in ('pred', 'param_only')}
    resid, z, vpar, sig2 = nan(), nan(), nan(), nan()
    folds = []
    for s in range(S if args.max_folds is None else min(S, args.max_folds)):
        held = obs.site == s
        t0 = time.perf_counter()
        th, n_grad, t_fit = fit_eng.fit(prob.theta_nominal, (~held).astype(np.float64))
        r_all = y - gpu.predict(th)
        J = gpu.jacobian(th)
        lin = linearised(J[~held], r_all[~held], th, prob.bounds)
        vp = np.einsum('mi,ij,mj->m', J[held], lin['cov'], J[held])
        resid[held], vpar[held], sig2[held] = r_all[held], vp, lin['sigma2']
        z[held] = r_all[held] / np.sqrt(lin['sigma2'] + vp)
        for k, d in lin_halfwidths(lin['sigma2'], vp, lin['dof']).items():
            for lv in LEVELS:
                hw[k][lv][held] = d[lv]
        thetas, draw = draw_theta(th, lin, prob.bounds, args.loso_samples, rng)
        preds, _ = gpu.sample_predict(thetas, args.chunk)
        for k, d in sampled_bounds(preds[:, held], lin['sigma2'], rng).items():
            for lv in LEVELS:
                sb[k][lv][0][held], sb[k][lv][1][held] = d[lv][0], d[lv][1]
        folds.append({'site': obs.site_node_id[s], 'n_held': int(held.sum()), 'theta': th.tolist(),
                      'active': lin['active'].tolist(), 'sigma': float(np.sqrt(lin['sigma2'])), 'dof': lin['dof'],
                      'sd': lin['sd'].tolist(), 'held_rmse': float(np.sqrt(np.mean(r_all[held] ** 2))),
                      'held_sd_param_mean': float(np.sqrt(vp).mean()), 'n_grad': n_grad, 'fit_wall_s': t_fit,
                      'draw': draw, 'wall_s': time.perf_counter() - t0})
        f = folds[-1]
        print(f"  LOSO site {f['site']:>8s} ({f['n_held']:3d} held): theta={np.round(th, 3).tolist()} sigma={f['sigma']:.4f} "
              f"held RMSE={f['held_rmse']:.4f} sd_param={f['held_sd_param_mean']:.4f} ({n_grad} grads, {f['wall_s']:.1f}s)", flush=True)
    ok = ~np.isnan(resid)
    tab = coverage_table(y, resid, hw, sb, ok)
    dof_med = float(np.median([f['dof'] for f in folds]))
    q95 = float(np.quantile(np.abs(z[ok]), 0.95))
    tab.update({
        'n_heldout': int(ok.sum()), 'n_folds': len(folds), 'cv_rmse': float(np.sqrt(np.mean(resid[ok] ** 2))),
        'param_share_of_pred_var': float(np.mean(vpar[ok]) / np.mean(vpar[ok] + sig2[ok])),
        'abs_z_quantiles': {str(lv): float(np.quantile(np.abs(z[ok]), lv)) for lv in LEVELS},
        'inflation_95': q95 / float(student_t.ppf(0.975, dof_med)),
        'site_coverage95': {f['site']: {'n': f['n_held'],
                                        'pred_lin': float(np.mean(np.abs(resid[obs.site == i]) <= hw['pred'][0.95][obs.site == i])),
                                        'pred_sampled': float(np.mean((y[obs.site == i] >= sb['pred'][0.95][0][obs.site == i]) &
                                                                      (y[obs.site == i] <= sb['pred'][0.95][1][obs.site == i]))),
                                        'param_only_lin': float(np.mean(np.abs(resid[obs.site == i]) <= hw['param_only'][0.95][obs.site == i]))}
                            for i, f in enumerate(folds)},
        'folds': folds,
        'per_obs': {'site': obs.site.tolist(), 'y': y.tolist(), 'resid': resid.tolist(), 'sd_param': np.sqrt(vpar).tolist(),
                    'sigma_train': np.sqrt(sig2).tolist(), 'hw95_pred_lin': hw['pred'][0.95].tolist(),
                    'lo95_pred_sampled': sb['pred'][0.95][0].tolist(), 'hi95_pred_sampled': sb['pred'][0.95][1].tolist()}})
    return tab


def field_summary(prob, mean, sd, sigma):
    """Per-node summary of the sampled sd of the chlorine field (parameters only) [n_snap, N+1]."""
    n = prob.net
    sd_n, mean_n = sd[:, 1:], mean[:, 1:]                             # drop the unused index 0
    node_mean_sd, node_max_sd = sd_n.mean(0), sd_n.max(0)
    is_site = np.zeros(n.nnodes, dtype=bool)
    is_site[np.unique(prob.obs.node) - 1] = True
    hw95 = 1.96 * sd_n
    order = np.argsort(-node_max_sd)
    return {'n_snap': int(sd.shape[0]), 'n_nodes': int(n.nnodes),
            'node_mean_sd_quantiles': {str(q): float(np.quantile(node_mean_sd, q)) for q in (0.5, 0.9, 1.0)},
            'node_max_sd_quantiles': {str(q): float(np.quantile(node_max_sd, q)) for q in (0.5, 0.9, 1.0)},
            'mean_sd_sampled_sites': float(node_mean_sd[is_site].mean()),
            'mean_sd_unsampled_nodes': float(node_mean_sd[~is_site].mean()) if (~is_site).any() else None,
            'frac_node_steps_hw95_gt_0.05': float(np.mean(hw95 > 0.05)), 'frac_node_steps_hw95_gt_0.1': float(np.mean(hw95 > 0.1)),
            'frac_node_steps_hw95_gt_sigma': float(np.mean(hw95 > sigma)),
            'field_mean_conc': float(mean_n.mean()),
            'top10_nodes_max_sd': [(n.node_ids[i + 1], float(node_max_sd[i]), bool(is_site[i])) for i in order[:10]],
            'node_ids': n.node_ids[1:], 'node_is_site': is_site.tolist(),
            'node_mean_sd': node_mean_sd.tolist(), 'node_max_sd': node_max_sd.tolist(),
            'node_mean_conc': mean_n.mean(0).tolist()}


def run_dataset(name, lib, workdir, args, rng):
    prob = CalibProblem(name, lib, workdir, device='cpu', fix_kb=args.fix_kb)
    try:
        cpu = Engine(prob, 'cpu')
        gpu = Engine(prob, args.device, graph=args.graph) if args.device != 'cpu' else cpu
        out = {'dataset': name, 'n_obs': len(prob.obs), 'n_sites': len(prob.obs.site_node_id), 'param_names': prob.param_names,
               'bounds': prob.bounds, 'fix_kb': args.fix_kb, 'theta_nominal': prob.theta_nominal.tolist(),
               'schedule': {'steps': prob.sch.nsteps, 'versions': prob.sch.nver, 'ops': len(prob.sch.ops),
                            'n_snap': int(len(prob.times)), 'record_s': prob.t_record, 'build_s': prob.t_build}}
        # ---- B = 1 gradient cost per device / mode; the fits use the fastest (or the requested) one
        timing = {'cpu_eager_s': time_grad(cpu, False)}
        if gpu is not cpu:
            timing['cuda_eager_s'] = time_grad(gpu, False)
            if gpu.graphed is not None:
                timing['cuda_graph_s'] = time_grad(gpu, True)
            else:
                timing['cuda_graph_error'] = gpu.graph_error
        cand = {'cpu': timing['cpu_eager_s']}
        if gpu is not cpu:
            cand['cuda'] = timing.get('cuda_graph_s', timing['cuda_eager_s'])
        fit_dev = min(cand, key=cand.get) if args.fit_device == 'auto' else args.fit_device
        fit_eng = cpu if fit_dev == 'cpu' else gpu
        out['timing_grad_B1'] = timing
        out['fit_device'] = fit_dev + (' (graph)' if fit_eng is gpu and gpu.graphed is not None else ' (eager)')
        print(f"[{name}] B=1 gradient: {', '.join(f'{k}={v:.3f}s' for k, v in timing.items() if isinstance(v, float))}"
              f"{'; graph capture failed: ' + gpu.graph_error if gpu.graph_error else ''}; fits on {out['fit_device']}", flush=True)
        # ---- full-data fit, exact Jacobian, linearised covariance
        th, n_grad, t_fit = fit_eng.fit(prob.theta_nominal)
        sim = gpu.predict(th)
        r = prob.obs.y - sim
        t0 = time.perf_counter(); J = gpu.jacobian(th); t_jac = time.perf_counter() - t0
        lin = linearised(J, r, th, prob.bounds)
        sse_dll = prob.sse_dll(th)
        out['fit'] = {'theta': th.tolist(), 'n_grad': n_grad, 'wall_s': t_fit, 'sse': float(r @ r), 'sse_dll_check': sse_dll,
                      'rmse': float(np.sqrt(np.mean(r * r))), 'sigma': float(np.sqrt(lin['sigma2'])), 'dof': lin['dof'],
                      'active': lin['active'].tolist(), 'sd_free': lin['sd'].tolist(), 'sd_full_p': lin['sd_all'].tolist(),
                      'ci95_halfwidth_free': (student_t.ppf(0.975, lin['dof']) * lin['sd']).tolist(),
                      'corr_free': lin['corr'].tolist(), 'jtj_eig_free': lin['eig_free'].tolist(), 'cond_free': lin['cond_free'],
                      'jacobian_wall_s': t_jac, 'sd_param_at_obs': np.sqrt(lin['var_param']).tolist(),
                      'param_share_of_pred_var': float(np.mean(lin['var_param']) / np.mean(lin['var_param'] + lin['sigma2']))}
        hw = lin_halfwidths(lin['sigma2'], lin['var_param'], lin['dof'])
        print(f"  fit: theta*={np.round(th, 4).tolist()} rmse={out['fit']['rmse']:.4f} sigma={out['fit']['sigma']:.4f} "
              f"|SSE-SSE_dll|={abs(out['fit']['sse'] - sse_dll):.1e} active={lin['active'].astype(int).tolist()} "
              f"sd_free={np.round(lin['sd'], 3).tolist()} sd_full_p={np.round(lin['sd_all'], 3).tolist()} "
              f"param share of pred var={out['fit']['param_share_of_pred_var']:.3f} ({n_grad} grads {t_fit:.1f}s; J {t_jac:.2f}s)", flush=True)
        # ---- Laplace sampling through the exact map: samples and the whole field
        thetas, draw = draw_theta(th, lin, prob.bounds, args.samples, rng)
        sync = torch.cuda.synchronize if gpu.dev.type == 'cuda' else (lambda: None)
        sync(); t0 = time.perf_counter(); preds, chunk_used = gpu.sample_predict(thetas, args.chunk); sync(); t_pred = time.perf_counter() - t0
        sb = sampled_bounds(preds, lin['sigma2'], rng)
        ok = np.ones(len(r), dtype=bool)
        out['in_sample'] = coverage_table(prob.obs.y, r, hw, sb, ok)
        out['in_sample']['note'] = 'intervals from the full-data fit evaluated on the same samples (optimistic reference)'
        sd_lin95 = hw['param_only'][0.95]
        sd_smp95 = (sb['param_only'][0.95][1] - sb['param_only'][0.95][0]) / 2
        out['sampling'] = {'S': args.samples, 'draw': draw, 'chunk': chunk_used, 'wall_obs_s': t_pred,
                           'theta_sample_mean': thetas.mean(0).tolist(), 'theta_sample_sd': thetas.std(0).tolist(),
                           'halfwidth95_param_only_lin_mean': float(sd_lin95.mean()), 'halfwidth95_param_only_sampled_mean': float(sd_smp95.mean()),
                           'nonlinearity_ratio_sampled_over_lin': float(sd_smp95.mean() / max(sd_lin95.mean(), 1e-300))}
        if gpu.dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter(); mean, sd, chunk_f = gpu.sample_field(thetas, args.chunk, th); sync(); t_field = time.perf_counter() - t0
        out['field'] = field_summary(prob, mean, sd, np.sqrt(lin['sigma2']))
        out['field'].update({'wall_s': t_field, 'chunk': chunk_f, 'device': str(gpu.dev),
                             'peak_mem_MB': torch.cuda.max_memory_allocated() / 2 ** 20 if gpu.dev.type == 'cuda' else None})
        if args.cpu_reference and gpu is not cpu:
            t0 = time.perf_counter(); mean_c, sd_c, _ = cpu.sample_field(thetas, args.cpu_chunk, th); t_cpu = time.perf_counter() - t0
            out['field']['cpu_reference'] = {'wall_s': t_cpu, 'chunk': args.cpu_chunk, 'threads': torch.get_num_threads(),
                                             'max_abs_diff_mean': float(np.abs(mean_c - mean).max()), 'max_abs_diff_sd': float(np.abs(sd_c - sd).max())}
        fs = out['field']
        print(f"  sampling S={args.samples} ({draw['mode']}, {draw['frac_in_box']:.2f} in box): obs {t_pred:.1f}s, field {t_field:.1f}s "
              f"[{gpu.dev}, chunk {chunk_f}]" + (f", cpu {fs['cpu_reference']['wall_s']:.1f}s (max diff {fs['cpu_reference']['max_abs_diff_sd']:.1e})"
              if 'cpu_reference' in fs else '') + f"; 95% param-only half-width lin {sd_lin95.mean():.4f} vs sampled {sd_smp95.mean():.4f}; "
              f"field sd: node-mean median {fs['node_mean_sd_quantiles']['0.5']:.4f} max {fs['node_max_sd_quantiles']['1.0']:.4f}, "
              f"{fs['frac_node_steps_hw95_gt_0.05']:.1%} of node-steps with 95% half-width > 0.05 mg/L", flush=True)
        # ---- leave-one-site-out coverage
        if not args.no_loso:
            t0 = time.perf_counter()
            out['loso'] = loso(prob, fit_eng, gpu, args, rng)
            out['loso']['wall_s'] = time.perf_counter() - t0
            lo = out['loso']
            row = lambda tab, k: ' '.join(f"{lv}:{tab[k][str(lv)]['coverage']:.2f}" for lv in LEVELS)
            print(f"  LOSO ({lo['n_folds']} folds, {lo['n_heldout']} held-out, {lo['wall_s']:.0f}s): CV-RMSE {lo['cv_rmse']:.4f} vs in-sample sigma "
                  f"{out['fit']['sigma']:.4f}; param share of pred var {lo['param_share_of_pred_var']:.3f}; inflation(95%) {lo['inflation_95']:.2f}\n"
                  f"    coverage nominal->empirical  pred lin [{row(lo['linearised'], 'pred')}]  pred sampled [{row(lo['sampled'], 'pred')}]\n"
                  f"                                 naive    [{row(lo['linearised'], 'naive')}]  param-only lin [{row(lo['linearised'], 'param_only')}]",
                  flush=True)
        return out
    finally:
        prob.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='Fairfield,Bellingham,NorthPenn')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device of the batched parts')
    ap.add_argument('--fit-device', default='auto', choices=['auto', 'cpu', 'cuda'], help='device of the B = 1 L-BFGS-B fits')
    ap.add_argument('--no-graph', dest='graph', action='store_false', help='do not capture the B = 1 replay in a CUDA graph')
    ap.add_argument('--samples', type=int, default=4096, help='Laplace draws for the full-data map')
    ap.add_argument('--loso-samples', type=int, default=1024, help='Laplace draws per LOSO fold')
    ap.add_argument('--chunk', type=int, default=1024, help='batch size of the sampled forward passes (halved on OOM)')
    ap.add_argument('--cpu-chunk', type=int, default=256)
    ap.add_argument('--no-cpu-reference', dest='cpu_reference', action='store_false')
    ap.add_argument('--no-loso', action='store_true')
    ap.add_argument('--max-folds', type=int, default=None)
    ap.add_argument('--fix-kb', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    torch.set_num_threads(args.threads or int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or max(1, os.cpu_count() // 2))
    cpu_name = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu_name = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), cpu_name)
    results = {'platform': platform.platform(), 'cpu': cpu_name, 'threads': torch.get_num_threads(), 'torch': torch.__version__,
               'device': args.device, 'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else cpu_name,
               'epanet': lib.version(), 'args': vars(args), 'levels': list(LEVELS), 'datasets': []}
    rng = np.random.default_rng(args.seed)
    with tempfile.TemporaryDirectory(prefix='h2_') as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in args.datasets.split(','):
            results['datasets'].append(run_dataset(name, lib, workdir, args, rng))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1))
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
