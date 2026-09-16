"""Exp3: pipe-by-pipe wall-decay calibration, Laplace identifiability audit and D-optimal sensor prescription.

Chlorine (first-order bulk kb, first-order wall kw with EPANET's mass-transfer limitation) replayed on the recorded
hydraulic schedule: the sensor series c(kw) and the Jacobian J = dc/dkw [n_obs, P] (one adjoint pass per `chunk`
one-hot cotangents) are those of EPANET itself for every kw field (DLL check at the heterogeneous truth).
Synthetic studies (L-Town, BWSN2): truth kw_p = -exp(log|m_g(p)| + s eps_p) around diameter-class medians m_g
(log-s.d. s), kb known; n_s random-junction sensors sampled every `sample_min` minutes over 48 h with N(0, sigma^2)
errors. Estimators (L-BFGS-B on exact adjoint gradients): (i) grouped kw, one value per diameter class, flat prior
-> m_hat; (ii) pipe-level MAP with the hierarchical prior kw_p ~ N(m_hat_g(p), tau_p^2), tau_p = rel_sd |m_hat_g|.
Laplace posterior Sigma = (J^T J / sigma^2 + diag(tau^-2))^-1 at the MAP: a pipe is identifiable when its posterior
s.d. is below half its prior s.d.; p_eff = sum (1 - sd^2 / tau^2) is the effective number of identified pipes.
Sensor prescription over candidate junctions from their exact Jacobian rows (Woodbury updates): greedy Bayesian
D-optimal augmentation (log det of the posterior precision) and greedy identifiability-count augmentation, against
a random augmentation of the same size; every sensor set is re-estimated and audited against the truth (kw error,
coverage, hold-out concentration error at all junctions). Field data sets (Fairfield, Bellingham, North Penn): the
same audit and prescriptions linearised at the G4 grouped estimate theta* with a fixed prior s.d. `field_tau`
(ft/d) for the pipe-level deviations and sigma = the residual RMSE of theta*.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp3_pipe_kw --network LTown --device cuda --out exp3_LTown.json
    python -m diffwq.exp3_pipe_kw --field Fairfield,Bellingham,NorthPenn --theta-dir g4_graph_5090 --out exp3_field.json
"""
import argparse
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize

from . import epanet_api as api
from .calib import CalibProblem
from .hydfile import read_hydfile
from .network import extract
from .replicate import Replicator
from .torch_wq import TorchWQ
from .verify_m0 import ROOT, all_links, all_tanks_kb, init_qual, qual_type

CONFIGS = {
    # BattLeDIM L-Town (SI, kw in m/d): 5-min hydraulics, 905 pipes of 63-225 mm
    'LTown': {'inp': ROOT / 'networks/L-Town/L-TOWN.inp', 'duration_h': 48, 'unbalanced': None, 'n_sensors': 10, 'n_add': 5,
              'edges': (100.0, 160.0), 'groups': ('<=100mm', '110-160mm', '>160mm'), 'kw_median': (-0.12, -0.08, -0.05),
              'kw_unit': 'm/d', 'candidates': None},
    # BWSN Network 2 (US, kw in ft/d): hourly hydraulics, 14,822 pipes of 0.8-48 in; Unbalanced = Continue 10 as in Exp2
    'BWSN2': {'inp': ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp', 'duration_h': 48,
              'unbalanced': 10, 'n_sensors': 20, 'n_add': 10, 'edges': (6.0, 8.0, 12.0), 'groups': ('<=6in', '8in', '10-12in', '>=16in'),
              'kw_median': (-0.6, -0.4, -0.25, -0.15), 'kw_unit': 'ft/d', 'candidates': 256},
}
KW_BOUNDS = (-5.0, 0.0)


def timed(fn, repeat=3):
    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def rms(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x * x))) if x.size else None


def laplace(J, sigma, tau):
    """Gaussian (Laplace) posterior of the parameters from the residual Jacobian J [n, P], noise s.d. sigma and prior
    s.d. tau [P]: precision Lam = J^T J / sigma^2 + diag(tau^-2). Returns (covariance, posterior s.d., log det Lam)."""
    Lam = J.T @ J / sigma ** 2 + torch.diag(1.0 / tau ** 2)
    Lc = torch.linalg.cholesky(Lam)
    Sigma = torch.cholesky_inverse(Lc)
    return Sigma, Sigma.diagonal().clamp_min(0.0).sqrt(), float(2.0 * Lc.diagonal().log().sum())


def p_eff(sd, tau):
    """Effective number of identified parameters sum_p (1 - sd_p^2 / tau_p^2) = P - tr(Sigma Lambda_prior)."""
    r = np.asarray(sd, dtype=np.float64) / np.asarray(tau, dtype=np.float64)
    return float(np.clip(1.0 - r * r, 0.0, None).sum())


def greedy_design(Jc, Sigma0, sigma, n_add, tau, exclude=(), chunk=64, criterion='logdet'):
    """Greedy Bayesian augmentation. Jc [n_c, n_t, P]: Jacobian rows of every candidate sensor (its n_t readings),
    Sigma0 [P, P]: the current posterior covariance. Adding candidate v gives Sigma - M^T S^-1 M (Woodbury) with
    M = J_v Sigma and S = J_v Sigma J_v^T + sigma^2 I; its D-gain log det(S / sigma^2) is the increase of log det of
    the posterior precision, its identifiability count the number of parameters whose posterior s.d. would fall
    below tau / 2. Each step adds the candidate maximising `criterion` ('logdet', or 'count' with D-gain as the
    tie-breaker). Returns chosen indices, per-step D-gains, per-step counts and the final covariance."""
    n_c, n_t, _ = Jc.shape
    Sig = Sigma0.clone()
    I = torch.eye(n_t, dtype=Jc.dtype, device=Jc.device)
    half = (0.5 * tau) ** 2
    chosen, gains, counts = [], [], []
    banned = torch.zeros(n_c, dtype=torch.bool, device=Jc.device)
    banned[list(exclude)] = True
    for _ in range(n_add):
        g = torch.empty(n_c, dtype=Jc.dtype, device=Jc.device)
        cnt = torch.zeros(n_c, dtype=torch.long, device=Jc.device)
        dS = Sig.diagonal()
        for c0 in range(0, n_c, chunk):
            Jb = Jc[c0:c0 + chunk]
            M = Jb @ Sig                                                                       # [m, n_t, P]
            Lc = torch.linalg.cholesky(M @ Jb.transpose(1, 2) + sigma ** 2 * I)
            g[c0:c0 + chunk] = 2.0 * Lc.diagonal(dim1=1, dim2=2).log().sum(1) - n_t * math.log(sigma ** 2)
            Y = torch.linalg.solve_triangular(Lc, M, upper=False)
            cnt[c0:c0 + chunk] = ((dS - (Y * Y).sum(1)) < half).sum(1)
        score = cnt.to(g.dtype) * 1e6 + g if criterion == 'count' else g
        score[banned] = -float('inf')
        v = int(score.argmax())
        chosen.append(v); gains.append(float(g[v])); counts.append(int(cnt[v])); banned[v] = True
        MJ = Jc[v] @ Sig                                                                       # [n_t, P]
        S = MJ @ Jc[v].T + sigma ** 2 * I
        Sig = Sig - MJ.T @ torch.linalg.solve(S, MJ)
        Sig = 0.5 * (Sig + Sig.T)
    return chosen, gains, counts, Sig


def precision_add(J, sigma, tau, Jc_rows):
    """Posterior s.d. after adding the rows Jc_rows [k, P] to the data of J [n, P]."""
    Lam = J.T @ J / sigma ** 2 + torch.diag(1.0 / tau ** 2) + Jc_rows.T @ Jc_rows / sigma ** 2
    return torch.cholesky_inverse(torch.linalg.cholesky(Lam)).diagonal().clamp_min(0.0).sqrt()



class PipeKwProblem:
    """Recorded chlorine network (kb fixed, kw a free per-pipe field) with sensors at junctions."""

    def __init__(self, name, lib, workdir, args):
        cfg = CONFIGS[name]
        self.name, self.cfg, self.args = name, cfg, args
        self.dev, self.dt = torch.device(args.device), torch.float64
        self.sync = torch.cuda.synchronize if self.dev.type == 'cuda' else (lambda: None)
        self.workdir = Path(workdir)
        t0 = time.perf_counter()
        self.pr = pr = api.Project(lib, cfg['inp'], self.workdir / f'{name}.rpt')
        qual_type(api.EN_CHEM)(pr); init_qual(args.c_init, reservoirs=args.c_source)(pr)
        all_links(kb=args.kb, kw=0.0)(pr); all_tanks_kb(args.kb)(pr)
        pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
        pr.set_option(api.EN_TOLERANCE, 0.0)
        if cfg['unbalanced'] is not None:
            pr.set_option(api.EN_UNBALANCED, cfg['unbalanced'])
        hyd = self.workdir / f'{name}.hyd'
        pr.solve_hydraulics(hyd)
        self.net = net = extract(pr)
        self.trace = read_hydfile(hyd)
        # stepQ: one snapshot per quality step (k * qstep), so the sampling instants lie on the quality grid for every
        # network (nextQ records hydraulic event times only: 49 instants at 15 min on BWSN2's hourly hydraulics)
        self.mode = getattr(args, 'mode', 'stepQ')
        self.t_dll = timed(lambda: pr.run_quality_reference(self.mode))
        rep = Replicator(net, self.trace, record=True)
        t1 = time.perf_counter(); self.times, _, _ = rep.run(self.mode); self.t_record = time.perf_counter() - t1
        t1 = time.perf_counter()
        self.eng = TorchWQ(net, rep.schedule, rep.reactflag, device=self.dev, dtype=self.dt, engine='fused')
        self.t_build = time.perf_counter() - t1
        self.sch_info = {'versions': int(rep.schedule.nver), 'steps': int(rep.schedule.nsteps), 'fused_ops': len(self.eng.fsch.ops),
                         'fused_terms': int(self.eng.T_total), 'snapshots': len(self.times), 'hydraulic_periods': len(self.trace.periods)}
        L, T, N1 = net.nlinks, net.ntanks, net.nnodes + 1
        self.N1, self.n_out = N1, len(self.times)
        self.pipes = np.array([k for k in range(1, L + 1) if net.link_type[k] <= api.EN_PIPE])
        self.P = len(self.pipes)
        self.diam = np.array([pr.link_value(int(k), api.EN_DIAMETER) for k in self.pipes])
        self.group = (self.diam[:, None] > np.asarray(cfg['edges'], dtype=np.float64)).sum(1)
        self.G = len(cfg['groups'])
        self.junc = np.array([i for i in range(1, N1) if net.node_type[i] == api.EN_JUNCTION])
        self.pipes_t = torch.as_tensor(self.pipes, device=self.dev)
        self.group_t = torch.as_tensor(self.group, device=self.dev)
        self.kb_t = torch.full((1, L + 1), float(args.kb), dtype=self.dt, device=self.dev)
        self.tkb_t = torch.full((1, T + 1), float(args.kb), dtype=self.dt, device=self.dev)
        times = np.asarray(self.times)
        first = np.array([int(np.flatnonzero(times == t)[0]) for t in np.unique(times)])      # one snapshot per event time
        self.t_idx = np.array([i for i in first if times[i] % int(round(args.sample_min * 60)) == 0])
        self.t_idx_t = torch.as_tensor(self.t_idx, device=self.dev)
        self.n_t = len(self.t_idx)
        self.n_grad, self.t_grad, self.n_pass = 0, 0.0, 0
        self.t_setup = time.perf_counter() - t0

    def close(self):
        self.pr.__exit__(None, None, None)

    # ---- the map kw (per pipe, user units) -> node quality ----
    def kw_full(self, kw_p):
        """[B, P] -> [B, L+1] (non-pipe links 0)."""
        kw = torch.zeros(kw_p.shape[0], self.net.nlinks + 1, dtype=self.dt, device=self.dev)
        return kw.index_copy(1, self.pipes_t, kw_p)

    def forward(self, kw_p):
        """[B, P] -> [B, n_out, N+1] mg/L (recorded initial quality and reservoir quality)."""
        return self.eng.forward(kb=self.kb_t, kw=self.kw_full(kw_p), tank_kb=self.tkb_t)

    def sim(self, kw_p, sens_t):
        """[B, P] -> [B, n_t, n_s] sensor readings at the sampling instants."""
        return self.forward(kw_p)[:, self.t_idx_t][:, :, sens_t]

    def rows_of(self, nodes):
        """Flat indices into [n_out, N+1] of (every sampling instant) x (the given nodes), node-major."""
        nodes = torch.as_tensor(np.asarray(nodes), device=self.dev)
        return (self.t_idx_t.view(1, -1) * self.N1 + nodes.view(-1, 1)).reshape(-1)

    def truth_field(self, rng):
        med = np.asarray(self.cfg['kw_median'], dtype=np.float64)[self.group]
        return -np.exp(np.log(-med) + self.args.log_sd * rng.standard_normal(self.P))

    # ---- objective and exact adjoint gradient (B = 1) ----
    def loss_grad(self, x, d, sens_t, sigma, mu=None, tau=None, grouped=False):
        """x: kw per pipe [P] (or per diameter class [G] when grouped); 1/2 ||c(x) - d||^2 / sigma^2 plus, with a
        prior, 1/2 sum ((kw_p - mu_p) / tau_p)^2. Returns (f, df/dx) from one forward + one adjoint sweep."""
        t0 = time.perf_counter()
        th = torch.as_tensor(np.asarray(x, dtype=np.float64), device=self.dev).requires_grad_(True)
        kw_p = th[self.group_t] if grouped else th
        r = self.sim(kw_p.view(1, -1), sens_t)[0] - d
        f = 0.5 * (r * r).sum() / sigma ** 2
        if mu is not None:
            f = f + 0.5 * (((kw_p - mu) / tau) ** 2).sum()
        f.backward()
        self.sync(); self.n_grad += 1; self.t_grad += time.perf_counter() - t0
        return f.detach().item(), th.grad.detach().cpu().numpy()

    def fit(self, x0, d, sens_t, sigma, mu=None, tau=None, grouped=False, maxiter=500):
        hist = []

        def fg(x):
            f, g = self.loss_grad(x, d, sens_t, sigma, mu, tau, grouped)
            hist.append(f)
            return f, g

        t0 = time.perf_counter()
        res = minimize(fg, np.asarray(x0, dtype=np.float64), jac=True, method='L-BFGS-B', bounds=[KW_BOUNDS] * len(x0),
                       options={'maxiter': maxiter, 'maxfun': 4 * maxiter, 'ftol': 1e-13, 'gtol': 1e-9, 'maxcor': 30})
        return res.x, {'f': float(res.fun), 'n_eval': len(hist), 'n_iter': int(res.nit), 'wall_s': time.perf_counter() - t0,
                       'message': str(res.message), 'f_hist': [float(v) for v in hist[::max(1, len(hist) // 100)]]}

    # ---- exact Jacobian rows: one shared forward, `chunk` unit cotangents per adjoint sweep ----
    def jac_rows(self, kw_p, rows, chunk):
        """d out[flat row] / d kw_pipes at kw_p [P] for the flat indices `rows` into [n_out, N+1]: [n_rows, P]."""
        kw = self.kw_full(torch.as_tensor(np.asarray(kw_p, dtype=np.float64), device=self.dev).view(1, -1))
        J = self.eng.jac_rows_kw(rows, kb=self.kb_t, kw=kw, tank_kb=self.tkb_t, chunk=chunk)[:, self.pipes_t]
        self.n_pass += int(np.ceil(len(rows) / chunk))
        self.sync()
        return J

    def jac_rows_autograd(self, kw_p, rows, chunk):
        """Reference: the same rows from a batch of `chunk` replicated forwards and one-hot autograd cotangents."""
        n = len(rows)
        J = torch.empty(n, self.P, dtype=self.dt, device=self.dev)
        base = torch.as_tensor(np.asarray(kw_p, dtype=np.float64), device=self.dev).view(1, -1)
        for r0 in range(0, n, chunk):
            m = min(chunk, n - r0)
            th = base.expand(m, -1).clone().requires_grad_(True)
            out = self.forward(th).reshape(m, -1)
            out[torch.arange(m, device=self.dev), rows[r0:r0 + m]].sum().backward()
            J[r0:r0 + m] = th.grad
        self.sync()
        return J

    def jac_check(self, kw_p, rows, chunk):
        """Row Jacobian by the shared-forward adjoint vs the autograd batch (and its speed) on the given rows."""
        for fn in (self.jac_rows, self.jac_rows_autograd):
            fn(kw_p, rows[:min(8, len(rows))], chunk)
        t0 = time.perf_counter(); J1 = self.jac_rows(kw_p, rows, chunk); t1 = time.perf_counter() - t0
        t0 = time.perf_counter(); J0 = self.jac_rows_autograd(kw_p, rows, chunk); t2 = time.perf_counter() - t0
        return {'n_rows': int(len(rows)), 'chunk': int(chunk), 'max_abs_diff': float((J1 - J0).abs().max()), 'max_abs_J': float(J0.abs().max()),
                'wall_shared_forward_s': t1, 'wall_autograd_batch_s': t2, 'rows_per_s_shared_forward': len(rows) / t1}

    def dll_check(self, kw_p):
        """EPANET DLL at the heterogeneous kw field vs the replay: max abs error over all nodes and event times."""
        pr = self.pr
        for k, v in zip(self.pipes, kw_p):
            pr.set_link_value(int(k), api.EN_KWALL, float(v))
        try:
            _, q, _ = pr.run_quality_reference(self.mode)
        finally:
            for k in self.pipes:
                pr.set_link_value(int(k), api.EN_KWALL, 0.0)
        with torch.no_grad():
            c = self.forward(torch.as_tensor(np.asarray(kw_p), device=self.dev).view(1, -1))[0].cpu().numpy()
        return {'max_abs_err_mgL': float(np.abs(q - c).max()), 'max_c_mgL': float(np.abs(q).max()),
                'max_rel_err': float(np.abs(q - c).max() / max(np.abs(q).max(), 1e-300)), 'n_snapshots_dll': int(q.shape[0])}



def kw_metrics(kw_hat, kw_true, sd=None, ident=None):
    e = np.asarray(kw_hat) - np.asarray(kw_true)
    out = {'rmse': rms(e), 'mae': float(np.abs(e).mean()), 'bias': float(e.mean()),
           'median_rel_abs_err': float(np.median(np.abs(e) / np.abs(kw_true)))}
    if ident is not None:
        out.update({'rmse_identifiable': rms(e[ident]), 'rmse_unidentifiable': rms(e[~ident]), 'n_identifiable': int(ident.sum()),
                    'median_rel_abs_err_identifiable': float(np.median(np.abs(e[ident]) / np.abs(kw_true)[ident])) if ident.any() else None})
    if sd is not None:
        out['coverage_2sd'] = float(np.mean(np.abs(e) < 2.0 * np.asarray(sd)))
    return out


def estimate(P, sens, d_all, c_true, kw_true, args, label):
    """Grouped fit -> hierarchical prior -> pipe-level MAP -> Laplace audit for the sensor set `sens` (node indices)."""
    T, dev = time.perf_counter, P.dev
    sens = np.asarray(sens)
    sens_t = torch.as_tensor(sens, device=dev)
    d = d_all[:, sens_t]                                                                  # [n_t, n_s]
    sigma = args.noise
    out = {'label': label, 'sensors': [P.net.node_ids[int(j)] for j in sens], 'n_sensors': len(sens), 'n_obs': int(d.numel())}
    # (i) grouped kw, flat prior
    P.n_grad, P.t_grad = 0, 0.0
    m_hat, info_g = P.fit(np.full(P.G, -0.3), d, sens_t, sigma, grouped=True, maxiter=args.maxiter)
    out['grouped'] = {'kw': m_hat.tolist(), **info_g, 'grad_s_mean': P.t_grad / max(P.n_grad, 1)}
    mu = m_hat[P.group]
    tau = np.maximum(args.rel_sd * np.abs(mu), args.tau_floor)
    mu_t, tau_t = torch.as_tensor(mu, device=dev), torch.as_tensor(tau, device=dev)
    # (ii) pipe-level MAP around the grouped estimate
    P.n_grad, P.t_grad = 0, 0.0
    kw_hat, info_p = P.fit(mu, d, sens_t, sigma, mu=mu_t, tau=tau_t, maxiter=args.maxiter)
    out['map'] = {**info_p, 'grad_s_mean': P.t_grad / max(P.n_grad, 1), 'n_at_bounds': int(np.sum((kw_hat <= KW_BOUNDS[0] + 1e-9) | (kw_hat >= KW_BOUNDS[1] - 1e-9)))}
    # (iii) Laplace audit at the MAP
    t0 = T()
    J = P.jac_rows(kw_hat, P.rows_of(sens), args.chunk)                                  # [n_t * n_s, P] node-major rows
    t_jac = T() - t0; t0 = T()
    Sigma, sd, logdet = laplace(J, sigma, tau_t)
    P.sync(); t_lap = T() - t0
    sd_np = sd.cpu().numpy()
    ident = sd_np < 0.5 * tau
    colnorm = J.norm(dim=0).cpu().numpy()
    out['audit'] = {'n_obs': int(J.shape[0]), 'jacobian_passes': int(np.ceil(J.shape[0] / args.chunk)), 'wall_jacobian_s': t_jac,
                    'wall_laplace_s': t_lap, 'logdet_precision': logdet, 'n_identifiable': int(ident.sum()),
                    'frac_identifiable': float(ident.mean()), 'p_eff': p_eff(sd_np, tau),
                    'n_sensitive': int((colnorm > 1e-9 * colnorm.max()).sum()), 'median_sd_over_tau': float(np.median(sd_np / tau)),
                    'identifiable_by_group': {g: int(ident[P.group == gi].sum()) for gi, g in enumerate(P.cfg['groups'])},
                    'pipes_by_group': {g: int((P.group == gi).sum()) for gi, g in enumerate(P.cfg['groups'])},
                    'sd_over_tau_quantiles': np.quantile(sd_np / tau, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()}
    # metrics against the truth
    out['kw_error'] = {'grouped_prior_mean': kw_metrics(mu, kw_true, ident=ident),
                       'map': kw_metrics(kw_hat, kw_true, sd=sd_np, ident=ident)}
    with torch.no_grad():
        c_map = P.forward(torch.as_tensor(kw_hat, device=dev).view(1, -1))[0]
        c_grp = P.forward(torch.as_tensor(mu, device=dev).view(1, -1))[0]
        jt = torch.as_tensor(P.junc, device=dev)
        hold = lambda c: rms((c - c_true)[:, jt].cpu().numpy())
        out['holdout_rmse_mgL'] = {'grouped': hold(c_grp), 'map': hold(c_map),
                                   'grouped_nonsensor': rms((c_grp - c_true)[:, jt][:, ~torch.isin(jt, sens_t)].cpu().numpy()),
                                   'map_nonsensor': rms((c_map - c_true)[:, jt][:, ~torch.isin(jt, sens_t)].cpu().numpy())}
        out['sensor_fit_rmse_mgL'] = {'grouped': rms((c_grp[P.t_idx_t][:, sens_t] - d).cpu().numpy()),
                                      'map': rms((c_map[P.t_idx_t][:, sens_t] - d).cpu().numpy()), 'noise_sd': sigma}
    print(f'  [{label}] n_s={len(sens)} grouped kw={np.round(m_hat, 4).tolist()} ({info_g["n_eval"]} grads, {info_g["wall_s"]:.1f}s) | '
          f'MAP {info_p["n_eval"]} grads {info_p["wall_s"]:.1f}s | J {J.shape[0]}x{P.P} in {t_jac:.1f}s, Laplace {t_lap:.1f}s | '
          f'identifiable {int(ident.sum())}/{P.P} | kw RMSE grouped {out["kw_error"]["grouped_prior_mean"]["rmse"]:.4f} '
          f'MAP {out["kw_error"]["map"]["rmse"]:.4f} (ident. {out["kw_error"]["map"]["rmse_identifiable"]}) coverage '
          f'{out["kw_error"]["map"]["coverage_2sd"]:.3f} | hold-out mg/L grouped {out["holdout_rmse_mgL"]["grouped"]:.4f} '
          f'MAP {out["holdout_rmse_mgL"]["map"]:.4f}', flush=True)
    return out, kw_hat, Sigma, sd_np, tau, J


def run_synthetic(name, lib, workdir, args):
    T = time.perf_counter
    t_start = T()
    P = PipeKwProblem(name, lib, workdir, args)
    try:
        return _run_synthetic(P, args, t_start)
    finally:
        P.close()



def _run_synthetic(P, args, t_start):
    T, net, dev, cfg = time.perf_counter, P.net, P.dev, P.cfg
    n_s = args.n_sensors or cfg['n_sensors']
    n_add = args.n_add or cfg['n_add']
    n_cand = args.candidates or cfg['candidates']
    print(f'[{P.name}] nodes={net.nnodes} junctions={len(P.junc)} links={net.nlinks} pipes={P.P} groups={dict(zip(cfg["groups"], np.bincount(P.group, minlength=P.G).tolist()))} '
          f'dur={net.duration // 3600}h snapshots={P.n_out} sampling={args.sample_min:g}min x {P.n_t} | record={P.t_record:.1f}s '
          f'build={P.t_build:.1f}s setup={P.t_setup:.1f}s dll_run={P.t_dll * 1e3:.0f}ms | {P.sch_info}', flush=True)
    rng = np.random.default_rng(args.seed)
    R = {'network': P.name, 'sizes': {'nodes': net.nnodes, 'junctions': len(P.junc), 'links': net.nlinks, 'pipes': P.P, 'snapshots': P.n_out,
                                      'n_t': P.n_t, 'duration_h': net.duration / 3600, 'kw_unit': cfg['kw_unit'], **P.sch_info},
         'settings': {k: getattr(args, k) for k in ('kb', 'c_init', 'c_source', 'log_sd', 'rel_sd', 'tau_floor', 'noise', 'sample_min', 'mode', 'chunk',
                                                    'maxiter', 'seed')}, 'groups': dict(zip(cfg['groups'], np.bincount(P.group, minlength=P.G).tolist())),
         'kw_median_true': list(cfg['kw_median']), 'n_sensors': n_s, 'n_add': n_add}
    tm = {'record_s': P.t_record, 'build_s': P.t_build, 'setup_s': P.t_setup, 'dll_quality_run_s': P.t_dll}
    # ---- truth, fidelity, cost of one gradient ----
    kw_true = P.truth_field(rng)
    R['truth'] = {'kw': kw_true.round(6).tolist(), 'pipe_ids': [net.link_ids[int(k)] for k in P.pipes], 'group': P.group.tolist(),
                  'diameter': P.diam.tolist()}
    fd_b = 32                                                                                  # FD baseline: batches of 32 forwards
    with torch.no_grad():
        kt = torch.as_tensor(kw_true, device=dev).view(1, -1)
        c_true = P.forward(kt)[0]                                                              # [n_out, N+1]
        t_fwd1 = timed(lambda: (P.forward(kt), P.sync()))
        tb = kt.expand(fd_b, -1).contiguous()
        t_fwd_b = timed(lambda: (P.forward(tb), P.sync()), repeat=2)
        del tb
    t0 = T(); R['dll_check'] = P.dll_check(kw_true); tm['dll_check_s'] = T() - t0
    st = torch.as_tensor(P.junc, device=dev)
    sens0 = torch.as_tensor(rng.choice(P.junc, n_s, replace=False), device=dev)
    t0 = T(); R['jac_check'] = P.jac_check(kw_true, P.rows_of(sens0.cpu().numpy())[:64], 32); tm['jac_check_s'] = T() - t0
    dummy = torch.zeros(P.n_t, n_s, dtype=P.dt, device=dev)
    P.n_grad, P.t_grad = 0, 0.0
    for _ in range(3):
        P.loss_grad(kw_true, dummy, sens0, args.noise)
    t_grad1 = P.t_grad / P.n_grad
    n_fd = int(np.ceil((P.P + 1) / fd_b))
    R['cost'] = {'n_params': P.P, 'dll_forward_s': P.t_dll, 'engine_forward_B1_s': t_fwd1, 'fd_batch': fd_b, 'engine_forward_Bfd_s': t_fwd_b,
                 'adjoint_grad_B1_s': t_grad1, 'fd_grad_dll_projected_s': (P.P + 1) * P.t_dll, 'fd_grad_engine_B1_projected_s': (P.P + 1) * t_fwd1,
                 'fd_grad_engine_batched_projected_s': n_fd * t_fwd_b, 'speedup_vs_fd_dll': (P.P + 1) * P.t_dll / t_grad1,
                 'speedup_vs_fd_engine_batched': n_fd * t_fwd_b / t_grad1}
    print(f'  truth kw: median {np.median(kw_true):.4f} range [{kw_true.min():.4f}, {kw_true.max():.4f}] {cfg["kw_unit"]} | c_true max {float(c_true.max()):.3f} mg/L, '
          f'mean at sampling instants {float(c_true[P.t_idx_t][:, st].mean()):.3f} | DLL check max abs err {R["dll_check"]["max_abs_err_mgL"]:.2e} mg/L | '
          f'row-Jacobian check max abs diff {R["jac_check"]["max_abs_diff"]:.2e} ({R["jac_check"]["rows_per_s_shared_forward"]:.0f} rows/s vs autograd batch '
          f'{R["jac_check"]["n_rows"] / R["jac_check"]["wall_autograd_batch_s"]:.0f} rows/s at chunk 32) | one adjoint gradient {t_grad1 * 1e3:.0f} ms vs FD '
          f'{P.P + 1} DLL runs {(P.P + 1) * P.t_dll:.1f}s / batched engine {R["cost"]["fd_grad_engine_batched_projected_s"]:.1f}s', flush=True)
    # ---- data of every junction (a sensor set is a column subset) ----
    eps = torch.as_tensor(rng.normal(0.0, args.noise, size=(P.n_t, P.N1)), dtype=P.dt, device=dev)
    d_all = c_true[P.t_idx_t] + eps                                                             # [n_t, N+1]
    # ---- sensor sets: random baseline; D-optimal, identifiability-count and random augmentations of the same size ----
    S_R = sens0.cpu().numpy()
    t0 = T()
    est_R, kw_R, Sigma_R, sd_R, tau_R, J_R = estimate(P, S_R, d_all, c_true, kw_true, args, 'random')
    tm['estimate_random_s'] = T() - t0
    pool = np.setdiff1d(P.junc, S_R)
    cands = pool if n_cand is None or n_cand >= len(pool) else np.sort(rng.choice(pool, n_cand, replace=False))
    t0 = T()
    Jc = P.jac_rows(kw_R, P.rows_of(cands), args.chunk).view(len(cands), P.n_t, P.P)         # [n_c, n_t, P]
    P.sync(); tm['candidate_jacobian_s'] = T() - t0
    tau_t = torch.as_tensor(tau_R, device=dev)
    sets, pred, design = {'random': S_R}, {}, {'n_candidates': len(cands), 'candidate_rows': int(Jc.shape[0] * Jc.shape[1]),
                                                'candidate_passes': int(np.ceil(Jc.shape[0] * Jc.shape[1] / args.chunk)),
                                                'wall_candidate_jacobian_s': tm['candidate_jacobian_s'],
                                                'rows_per_s': Jc.shape[0] * Jc.shape[1] / tm['candidate_jacobian_s'], 'greedy': {}}
    summary = lambda sd: {'n_identifiable': int((sd < 0.5 * tau_R).sum()), 'p_eff': p_eff(sd, tau_R), 'median_sd_over_tau': float(np.median(sd / tau_R))}
    pred['random'] = summary(sd_R)
    for crit, label in (('logdet', 'random+dopt'), ('count', 'random+count')):
        t0 = T()
        chosen, gains, counts, Sigma_A = greedy_design(Jc, Sigma_R, args.noise, n_add, tau_t, chunk=args.design_chunk, criterion=crit)
        P.sync(); tm[f'design_{crit}_s'] = T() - t0
        sets[label] = np.concatenate([S_R, cands[chosen]])
        pred[label] = summary(Sigma_A.diagonal().clamp_min(0.0).sqrt().cpu().numpy())
        design['greedy'][label] = {'criterion': crit, 'wall_s': tm[f'design_{crit}_s'], 'added': [net.node_ids[int(j)] for j in cands[chosen]],
                                   'gain_logdet': gains, 'count_after_step': counts}
        print(f'  design [{crit}]: {len(cands)} candidates x {P.n_t} rows in {tm["candidate_jacobian_s"]:.1f}s, greedy {n_add} in {tm[f"design_{crit}_s"]:.1f}s -> '
              f'{design["greedy"][label]["added"]} D-gains {np.round(gains, 2).tolist()} counts {counts} | predicted {pred[label]}', flush=True)
        del Sigma_A
    S_RR = np.concatenate([S_R, rng.choice(pool, n_add, replace=False)])
    sets['random+random'] = S_RR
    rows_rr = (Jc[np.searchsorted(cands, S_RR[n_s:])].reshape(-1, P.P) if np.all(np.isin(S_RR[n_s:], cands))
               else P.jac_rows(kw_R, P.rows_of(S_RR[n_s:]), args.chunk))
    pred['random+random'] = summary(precision_add(J_R, args.noise, tau_t, rows_rr).cpu().numpy())
    design['random_added'] = [net.node_ids[int(j)] for j in S_RR[n_s:]]
    design['predicted'] = pred
    R['design'] = design
    del Jc, rows_rr, Sigma_R, J_R
    R['sets'] = {'random': est_R}
    for label in ('random+dopt', 'random+count', 'random+random'):
        t0 = T()
        R['sets'][label] = estimate(P, sets[label], d_all, c_true, kw_true, args, label)[0]
        tm[f'estimate_{label}_s'] = T() - t0
    R['map_kw'] = {'random': kw_R.round(6).tolist()}
    R['posterior_sd'] = {'random': sd_R.round(6).tolist(), 'prior_sd': tau_R.round(6).tolist()}
    tm['adjoint_passes_total'] = P.n_pass
    tm['total_s'] = T() - t_start
    R['timing'] = tm
    col = lambda f, nd: ' / '.join(f'{f(R["sets"][l]):.{nd}f}' for l in ('random', 'random+dopt', 'random+count', 'random+random'))
    print(f'  random / +dopt / +count / +random (of {P.P} pipes): identifiable {col(lambda s: s["audit"]["n_identifiable"], 0)} | '
          f'p_eff {col(lambda s: s["audit"]["p_eff"], 1)} | logdet {col(lambda s: s["audit"]["logdet_precision"], 1)} | '
          f'MAP kw RMSE {col(lambda s: s["kw_error"]["map"]["rmse"], 4)} | hold-out {col(lambda s: s["holdout_rmse_mgL"]["map"], 4)} mg/L | '
          f'total {tm["total_s"]:.0f}s', flush=True)
    return R



# ---- field data sets: audit and prescription linearised at the G4 grouped estimate ----
def load_theta(theta_dir, name):
    d = json.loads((Path(theta_dir) / f'g4_graph_{name}_cuda.json').read_text(encoding='utf-8'))
    dss = d['datasets'] if isinstance(d, dict) and 'datasets' in d else d
    for ds in dss:
        if isinstance(ds, dict) and ds.get('dataset') == name and not ds.get('fix_kb'):
            for meth in ('ADJ-LBFGSB-graph', 'ADJ-LBFGSB', 'FD-LBFGSB'):
                runs = [r for r in ds['runs'] if r['method'] == meth]
                if runs:
                    return np.asarray(runs[0]['theta'], dtype=np.float64), meth, ds['param_names'], runs[0]['rmse']
    raise ValueError(f'no grouped estimate of {name} in {theta_dir}')


def run_field(name, lib, workdir, args):
    t_start = time.perf_counter()
    prob = CalibProblem(name, lib, workdir, device=args.device, fix_kb=False, engine='fused', verbose=True)
    try:
        return _run_field(prob, name, args, t_start)
    finally:
        prob.close()


def _run_field(prob, name, args, t_start):
    T, dev, dt = time.perf_counter, prob.dev, torch.float64
    sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
    theta, meth, pnames, rmse_g4 = load_theta(args.theta_dir, name)
    kb, kw_link, tkb = prob.expand_np(theta)
    kb_t, kw_t, tkb_t = (torch.as_tensor(v, dtype=dt, device=dev).view(1, -1) for v in (kb, kw_link, tkb))
    net, N1, n_out = prob.net, prob.net.nnodes + 1, len(prob.times)
    pipes = np.flatnonzero(prob.is_pipe)
    P, pipes_t = len(pipes), torch.as_tensor(np.flatnonzero(prob.is_pipe), device=dev)
    grp = prob.link_group[pipes]

    def forward(delta):
        """delta [B, P] (kw deviations, ft/d) -> node quality [B, n_out, N+1] at theta* + delta."""
        B = delta.shape[0]
        kw = kw_t.expand(B, -1).index_add(1, pipes_t, delta)
        return prob.eng.forward(kb=kb_t, kw=kw, tank_kb=tkb_t, c0=prob.c0_t.expand(B, -1), src=prob.src_t.expand(B, -1))

    def sim_obs(delta):
        out = forward(delta).reshape(delta.shape[0], -1)
        return out[:, prob.obs_i0] * (1.0 - prob.obs_w) + out[:, prob.obs_i1] * prob.obs_w

    def rows_jac(rows):
        """d out[rows] / d delta at delta = 0 (theta*): [n_rows, P], one shared forward."""
        Jm = prob.eng.jac_rows_kw(rows, kb=kb_t, kw=kw_t, tank_kb=tkb_t, c0=prob.c0_t, src=prob.src_t, chunk=args.chunk)[:, pipes_t]
        sync()
        return Jm

    def rows_jac_autograd(rows):
        m = len(rows)
        th = torch.zeros(m, P, dtype=dt, device=dev, requires_grad=True)
        forward(th).reshape(m, -1)[torch.arange(m, device=dev), rows].sum().backward()
        sync()
        return th.grad

    M = len(prob.obs)
    with torch.no_grad():
        res0 = (sim_obs(torch.zeros(1, P, dtype=dt, device=dev))[0] - prob.y).cpu().numpy()
    sigma = rms(res0)
    t0 = T(); J2 = rows_jac(torch.cat([prob.obs_i0, prob.obs_i1])); t_jac = T() - t0
    J = J2[:M] * (1.0 - prob.obs_w).view(-1, 1) + J2[M:] * prob.obs_w.view(-1, 1)             # interpolated observations
    chk = prob.obs_i0[:32]
    jac_check = {'n_rows': int(len(chk)), 'max_abs_diff': float((rows_jac(chk) - rows_jac_autograd(chk)).abs().max()), 'max_abs_J': float(J2.abs().max())}
    tau_v = float(args.field_tau)
    tau = torch.full((P,), tau_v, dtype=dt, device=dev)
    Sigma, sd, logdet = laplace(J, sigma, tau)
    sd_np = sd.cpu().numpy()
    ident = sd_np < 0.5 * tau_v
    colnorm = J.norm(dim=0).cpu().numpy()
    # grouped Jacobian at theta*: standard errors of the diameter-class kw without a prior (compare with G4's CI)
    onehot = torch.zeros(P, prob.G, dtype=dt, device=dev)
    onehot[torch.arange(P, device=dev), torch.as_tensor(grp, device=dev)] = 1.0
    Jg = J @ onehot
    eig = torch.linalg.eigvalsh(Jg.T @ Jg).cpu().numpy()
    se_g = (torch.linalg.pinv(Jg.T @ Jg).diagonal().clamp_min(0.0).sqrt() * sigma).cpu().numpy()
    # prescription: every junction at the campaign's sampling instants (nearest quality step)
    t_obs = np.unique(np.clip(np.round(prob.obs.t / prob.qstep).astype(np.int64), 0, n_out - 1))
    junc = np.array([i for i in range(1, N1) if net.node_type[i] == api.EN_JUNCTION])
    rows = (torch.as_tensor(t_obs, device=dev).view(1, -1) * N1 + torch.as_tensor(junc, device=dev).view(-1, 1)).reshape(-1)
    t0 = T(); Jc = rows_jac(rows).view(len(junc), len(t_obs), P); t_cj = T() - t0
    existing = [int(np.flatnonzero(junc == n)[0]) for n in np.unique(prob.obs.node) if n in junc]
    by_group = lambda mask: {g: int(mask[grp == gi].sum()) for gi, g in enumerate(prob.group_names)}
    presc = {'n_candidates': len(junc), 'sampling_instants': len(t_obs), 'candidate_rows': int(len(rows)),
             'candidate_passes': int(np.ceil(len(rows) / args.chunk)), 'wall_candidate_jacobian_s': t_cj, 'rows_per_s': len(rows) / t_cj, 'greedy': {}}
    for crit in ('logdet', 'count'):
        t0 = T()
        chosen, gains, counts, Sigma_A = greedy_design(Jc, Sigma, sigma, args.field_add, tau, exclude=existing, chunk=args.design_chunk, criterion=crit)
        sync(); t_des = T() - t0
        sd_after = Sigma_A.diagonal().clamp_min(0.0).sqrt().cpu().numpy()
        presc['greedy'][crit] = {'wall_greedy_s': t_des, 'added_sites': [net.node_ids[int(junc[v])] for v in chosen], 'gain_logdet': gains,
                                 'count_after_step': counts, 'n_identifiable_after': int((sd_after < 0.5 * tau_v).sum()),
                                 'identifiable_after_by_group': by_group(sd_after < 0.5 * tau_v), 'p_eff_after': p_eff(sd_after, tau_v),
                                 'median_sd_over_tau_after': float(np.median(sd_after / tau_v)), 'pipe_sd_over_tau_after': (sd_after / tau_v).round(5).tolist()}
        del Sigma_A
    R = {'dataset': name, 'theta_star': theta.tolist(), 'theta_method': meth, 'param_names': pnames, 'rmse_g4': rmse_g4, 'n_obs': M,
         'sites': prob.obs.site_node_id, 'n_pipes': P, 'groups': dict(zip(prob.group_names, prob.group_count)),
         'sigma_residual_rmse': sigma, 'field_tau_ft_d': tau_v, 'jac_check': jac_check,
         'audit': {'n_identifiable': int(ident.sum()), 'frac_identifiable': float(ident.mean()), 'identifiable_by_group': by_group(ident),
                   'pipes_by_group': by_group(np.ones(P, dtype=bool)), 'p_eff': p_eff(sd_np, tau_v), 'n_sensitive': int((colnorm > 1e-9 * colnorm.max()).sum()),
                   'median_sd_over_tau': float(np.median(sd_np / tau_v)), 'sd_over_tau_quantiles': np.quantile(sd_np / tau_v, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                   'logdet_precision': logdet, 'jacobian_passes': int(np.ceil(M / args.chunk)), 'wall_jacobian_s': t_jac},
         'group_se_kw_only_kb_fixed_ft_d': dict(zip(prob.group_names, se_g.tolist())), 'group_jtj_eigvals': eig.tolist(),
         'prescription': presc,
         'pipe_ids': [net.link_ids[int(k)] for k in pipes], 'pipe_group': grp.tolist(), 'pipe_sd_over_tau': (sd_np / tau_v).round(5).tolist(),
         'pipe_sensitivity': colnorm.round(6).tolist(),
         'timing': {'record_s': prob.t_record, 'build_s': prob.t_build, 'total_s': T() - t_start}}
    gd = presc['greedy']
    print(f'  [{name}] theta*={np.round(theta, 4).tolist()} ({meth}) sigma={sigma:.4f} mg/L | J {M}x{P} in {t_jac:.1f}s (row check {jac_check["max_abs_diff"]:.1e}) | '
          f'identifiable pipes {int(ident.sum())}/{P} (tau {tau_v} ft/d) by group {by_group(ident)} p_eff {R["audit"]["p_eff"]:.1f} | group s.e. (kw only) {np.round(se_g, 3).tolist()} | '
          f'candidates {len(junc)}x{len(t_obs)} rows in {t_cj:.1f}s ({len(rows) / t_cj:.0f} rows/s) | +{args.field_add} sites D-opt {gd["logdet"]["added_sites"]} gains '
          f'{np.round(gd["logdet"]["gain_logdet"], 2).tolist()} -> identifiable {gd["logdet"]["n_identifiable_after"]} p_eff {gd["logdet"]["p_eff_after"]:.1f} | '
          f'count-greedy {gd["count"]["added_sites"]} -> identifiable {gd["count"]["n_identifiable_after"]} p_eff {gd["count"]["p_eff_after"]:.1f}', flush=True)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='', help='comma-separated synthetic studies: ' + ','.join(CONFIGS))
    ap.add_argument('--field', default='', help='comma-separated field data sets (calib_data.DATASETS) audited at the G4 theta*')
    ap.add_argument('--theta-dir', default='g4_graph_5090')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kb', type=float, default=-0.5, help='bulk decay (1/d) of pipes and tanks, known')
    ap.add_argument('--c-init', type=float, default=0.5, help='initial chlorine at junctions and tanks (mg/L)')
    ap.add_argument('--c-source', type=float, default=1.0, help='constant reservoir chlorine (mg/L)')
    ap.add_argument('--log-sd', type=float, default=0.5, help='log-normal spread of the true kw around its class median')
    ap.add_argument('--rel-sd', type=float, default=0.5, help='prior s.d. of kw_p relative to the fitted class value')
    ap.add_argument('--tau-floor', type=float, default=0.02, help='minimum prior s.d. (kw units)')
    ap.add_argument('--noise', type=float, default=0.02, help='sensor noise s.d. (mg/L)')
    ap.add_argument('--sample-min', type=float, default=15.0, help='sampling interval of the sensors (min)')
    ap.add_argument('--mode', default='stepQ', choices=['nextQ', 'stepQ'], help='quality stepping / snapshot grid of the synthetic studies')
    ap.add_argument('--n-sensors', type=int, default=None)
    ap.add_argument('--n-add', type=int, default=None)
    ap.add_argument('--candidates', type=int, default=None, help='candidate junctions of the prescription (default: config; None = all)')
    ap.add_argument('--chunk', type=int, default=256, help='unit cotangents per adjoint sweep (Jacobian rows)')
    ap.add_argument('--design-chunk', type=int, default=32)
    ap.add_argument('--maxiter', type=int, default=500)
    ap.add_argument('--field-tau', type=float, default=0.3, help='prior s.d. (ft/d) of the pipe-level kw deviations in the field audit')
    ap.add_argument('--field-add', type=int, default=5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    results = {'platform': platform.platform(), 'torch': torch.__version__, 'epanet': lib.version(), 'args': vars(args),
               'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else platform.processor(),
               'networks': [], 'field': []}
    dump = lambda: Path(args.out).write_text(json.dumps(results, indent=1, default=float), encoding='utf-8') if args.out else None
    with tempfile.TemporaryDirectory(prefix='exp3_', dir=os.environ.get('TMPDIR')) as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in filter(None, args.network.split(',')):
            results['networks'].append(run_synthetic(name, lib, workdir, args)); dump()
        for name in filter(None, args.field.split(',')):
            results['field'].append(run_field(name, lib, workdir, args)); dump()
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
