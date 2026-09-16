"""Exp1: joint booster-station siting and dosing schedule as one convex program on the exact EPANET map.

With Ctol = 0 the node concentrations are an affine function of the source strengths, c(u) = c0 + A u
(verify_src.py). Decision u[j, k] >= 0: mass rate (mg/min) of a virtual MASS doser at candidate node j
in pattern slot k (K hourly slots; periodic over a multi-day horizon when the patterns repeat). Program

    min_{u >= 0}  1/2 sum_o w_o [ (c_lo - c_o(u))_+^2 + (c_o(u) - c_hi)_+^2 ]  +  mu M(u)  +  lam sum_j ||u_j||_2

over the demand-weighted node-hours o of the evaluation window; M = daily dosed mass (kg/d); the group
penalty selects stations. Solved matrix-free by FISTA (batched over lam / mu / supports) with the fused
engine's adjoint (one forward + one backward per iteration, CUDA-graph replay), never forming A.
Baselines on the same inner solver: plant-only, greedy forward selection (best re-optimized station per
stage), constant-rate dosing on the group-lasso supports. Checks: dense response matrix (one-hot
forwards) solved with the same optimizer; DLL run with the optimal sources installed through the API.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp1_booster --network Net3 --device cuda --out exp1_Net3.json
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
from .calib import DIAM_EDGES, GROUP_NAMES
from .calib_data import prepare_inp
from .hydfile import read_hydfile
from .network import extract, SECperDAY
from .replicate import Replicator
from .torch_wq import TorchWQ, default_src
from .verify_m0 import ROOT, HERE, all_links, all_tanks_kb, init_qual, qual_type, inp_replace

CONFIGS = {
    # EPANET example 3 as a chlorine system: plants (River, Lake) at 1.0 mg/L, first-order bulk / wall decay
    # (kb = -1.0 /d, kw = -1.0 ft/d: warm-water, unlined-pipe kinetics), 7-day horizon (one-shot pump controls
    # made daily) with the last day as the design window (plant-only day 6 vs day 7 agree to < 0.01 mg/L).
    'Net3': {'inp': ROOT / 'EPANET-dev/example-networks/Net3.inp',
             'inp_edit': inp_replace(('Link 10 OPEN AT TIME 1', 'Link 10 OPEN AT CLOCKTIME 1:00 AM'),
                                     ('Link 10 CLOSED AT TIME 15', 'Link 10 CLOSED AT CLOCKTIME 3:00 PM')),
             'setup': [qual_type(api.EN_CHEM), init_qual(0.5, reservoirs=1.0), all_links(kb=-1.0, kw=-1.0),
                       all_tanks_kb(-1.0)],
             'duration_h': 168, 'eval_h': (144, 168), 'slots': 24, 'periodic': True},
    # North Penn (Lansdale, PA) total-chlorine field model: 36 h test period, 11 existing CONCEN sources,
    # one-shot 36-slot schedule (transient, not periodic), evaluation on the last 24 h.
    'NorthPenn': {'dataset': 'NorthPenn', 'setup': [], 'duration_h': None, 'eval_h': (12, 36), 'slots': 36,
                  'periodic': False},
}


def apply_g4_theta(pr, path):
    """Install the G4-calibrated (Kb, Kw by diameter class) of a calib.py result JSON."""
    R = json.load(open(path))
    d = R['datasets'][0]
    runs = [r for r in d['runs'] if str(r.get('method', '')).startswith('ADJ') and 'theta' in r]
    best = min(runs, key=lambda r: r['sse'])
    theta = dict(zip(d['param_names'], best['theta']))
    for k in range(1, pr.count(api.EN_LINKCOUNT) + 1):
        if pr.link_type(k) > api.EN_PIPE:
            continue
        g = GROUP_NAMES[int(np.digitize(pr.link_value(k, api.EN_DIAMETER), DIAM_EDGES))]
        if 'Kb' in theta:
            pr.set_link_value(k, api.EN_KBULK, theta['Kb'])
        pr.set_link_value(k, api.EN_KWALL, theta.get(f'Kw[{g}]', 0.0))
    if 'Kb' in theta:
        for i in range(1, pr.count(api.EN_NODECOUNT) + 1):
            if pr.node_type(i) == api.EN_TANK:
                pr.set_node_value(i, api.EN_TANK_KBULK, theta.get('Kb_tank', theta['Kb']))
    return {'source': str(path), 'method': best['method'], 'sse': best['sse'], 'theta': theta}


def timed(fn, sync, repeat=3):
    best = float('inf')
    for _ in range(repeat):
        sync(); t0 = time.perf_counter(); fn(); sync()
        best = min(best, time.perf_counter() - t0)
    return best


class BoosterProblem:
    """Recorded network with virtual MASS dosers at all candidate nodes; exact affine map u -> c."""

    def __init__(self, name, lib, workdir, args):
        cfg = CONFIGS[name]
        self.name, self.cfg, self.args, self.lib = name, cfg, args, lib
        self.dev, self.dt = torch.device(args.device), torch.float64
        self.sync = torch.cuda.synchronize if self.dev.type == 'cuda' else (lambda: None)
        self.workdir = Path(workdir)
        self.theta_info = None
        t0 = time.perf_counter()
        self.pr = pr = self.open()
        hyd = self.workdir / f'{name}.hyd'
        pr.solve_hydraulics(hyd)
        self.net = net = extract(pr)
        self.trace = read_hydfile(hyd)
        self.t_dll = timed(lambda: pr.run_quality_reference('nextQ'), lambda: None)
        self.K = K = cfg['slots']
        N1 = net.nnodes + 1
        junc = [i for i in range(1, N1) if net.node_type[i] == api.EN_JUNCTION and i not in net.sources]
        tanks = [int(net.tank_node[t]) for t in range(1, net.ntanks + 1)
                 if not net.tank_is_reservoir[t] and int(net.tank_node[t]) not in net.sources]
        self.cand = junc + (tanks if not args.no_tanks else [])
        self.J = len(self.cand)
        rep = Replicator(net, self.trace, record=True, src_slots=K, virtual_sources={j: api.EN_MASS for j in self.cand})
        t1 = time.perf_counter(); self.times, _, _ = rep.run('nextQ'); self.t_record = time.perf_counter() - t1
        self.n_virtual = int(rep.hits['virtual_source'])
        t1 = time.perf_counter()
        self.eng = TorchWQ(net, rep.schedule, rep.reactflag, device=self.dev, dtype=self.dt, engine='fused')
        self.eng.set_fixed_reaction(kb=torch.tensor(net.kb * SECperDAY), kw=torch.tensor(net.kw * SECperDAY),
                                    tank_kb=torch.tensor(net.tank_kb * SECperDAY))
        self.t_build = time.perf_counter() - t1
        self.sch_info = {'versions': int(rep.schedule.nver), 'steps': int(rep.schedule.nsteps),
                         'fused_ops': len(self.eng.fsch.ops), 'fused_terms': int(self.eng.T_total),
                         'virtual_terms': self.n_virtual, 'snapshots': len(self.times)}
        # ---- demand-weighted node-hours of the evaluation window: w[s, i] = max(D, 0) * hydstep ----
        by_t = {int(p.t): p for p in self.trace.periods}
        W = np.zeros((len(self.times), N1))
        lo_s, hi_s = cfg['eval_h'][0] * 3600, cfg['eval_h'][1] * 3600
        is_junc = net.node_type == api.EN_JUNCTION
        for s, t in enumerate(self.times):
            p = by_t.get(int(t))
            if p is None or not (lo_s <= t < hi_s):
                continue
            W[s, is_junc] = np.maximum(p.demand[is_junc], 0.0) * p.hydstep
        self.W_np = W / W.sum()
        self.obs = np.flatnonzero(self.W_np.reshape(-1) > 0)                          # flat (snapshot, node)
        self.obs_t = torch.as_tensor(self.obs, device=self.dev)
        self.w = torch.as_tensor(self.W_np.reshape(-1)[self.obs], dtype=self.dt, device=self.dev)   # sums to 1
        self.S0 = torch.as_tensor(default_src(net, K), dtype=self.dt, device=self.dev)               # [K, N1]
        self.cand_t = torch.as_tensor(self.cand, device=self.dev)
        slot_h = net.pstep / 3600.0
        self.mass_coef = 60.0 * slot_h * 1e-6 / (K * slot_h / 24.0)      # kg/d per unit sum of u (mg/min)
        self.lo_rep, self.lo, self.hi = args.lo, args.lo + args.margin, args.hi   # compliance / design thresholds
        self._graphs = {}
        with torch.no_grad():
            self.c_plant = self.eng.forward()[0]                                                     # [n_out, N1]
        # mass weight made scale-free: mu = mu_rel * V0 / M_ref, V0 = plant-only violation, M_ref = mass (kg/d)
        # that would raise the whole supply of the window by 0.1 mg/L (hyd demands are internal cfs)
        Q = np.array([np.maximum(p.demand[is_junc], 0.0).sum() for p in self.trace.periods
                      if lo_s <= p.t < hi_s]).mean() * 28.316846592 * 60.0                          # L/min
        self.M_ref = Q * 0.1 * 1440.0 * 1e-6
        self.V0 = float(self.viol(self.c_plant.reshape(1, -1)[:, self.obs_t])[0])
        self.mu = args.mu_rel * self.V0 / self.M_ref
        self.t_setup = time.perf_counter() - t0
        self.L = None

    def open(self, tag=''):
        """A project configured like the recorded one (also used for the DLL check of a design)."""
        cfg, args = self.cfg, self.args
        inp = prepare_inp(cfg['dataset'], self.workdir) if 'dataset' in cfg else cfg['inp']
        if cfg.get('inp_edit'):
            inp = cfg['inp_edit'](inp, self.workdir, self.name)
        pr = api.Project(self.lib, inp, self.workdir / f'{self.name}{tag}.rpt')
        for f in cfg['setup']:
            f(pr)
        if args.theta_json:
            self.theta_info = apply_g4_theta(pr, args.theta_json)
        if cfg['duration_h']:
            pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
        pr.set_option(api.EN_TOLERANCE, 0.0)
        return pr

    # ---- the affine map and its evaluation-window restriction ----
    def src_of(self, u):
        """u [B, J, K] mg/min -> src [B, K, N+1] (existing sources kept)."""
        B = u.shape[0]
        full = torch.zeros(B, self.K, self.net.nnodes + 1, dtype=self.dt, device=self.dev)
        full.index_add_(2, self.cand_t, u.transpose(1, 2).contiguous())
        return full + self.S0

    def c_full(self, u):
        """[B, n_out, N+1] user units; CUDA-graph replay per batch size."""
        B = u.shape[0]
        if self.dev.type == 'cuda' and not self.args.no_graph:
            if B not in self._graphs:
                with torch.enable_grad():                           # capture forward AND backward graphs
                    ex = torch.zeros(B, self.J, self.K, dtype=self.dt, device=self.dev, requires_grad=True)
                    t0 = time.perf_counter()
                    gph = torch.cuda.make_graphed_callables(lambda x: self.eng.forward(src=self.src_of(x)), (ex,),
                                                            num_warmup_iters=3)
                    self._graphs[B] = (gph, time.perf_counter() - t0)
            return self._graphs[B][0](u)
        return self.eng.forward(src=self.src_of(u))

    def free_graphs(self):
        t = sum(v[1] for v in self._graphs.values())
        self._graphs = {}
        gc.collect()
        if self.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return t

    def c_obs(self, u):
        """[B, n_obs] concentrations at the weighted node-hours."""
        return self.c_full(u).reshape(u.shape[0], -1)[:, self.obs_t]

    def viol(self, c):
        """1/2 sum w [(lo - c)_+^2 + (c - hi)_+^2] per sample."""
        return 0.5 * (self.w * (torch.relu(self.lo - c) ** 2 + torch.relu(c - self.hi) ** 2)).sum(1)

    def metrics(self, c, u=None):
        """Compliance statistics of c [n_obs] (torch or numpy) at the reporting threshold lo_rep (coverage,
        shortfall), at the design threshold lo (coverage_design, viol = the program's violation), and of u [J, K]."""
        c = torch.as_tensor(c, dtype=self.dt, device=self.dev).reshape(-1)
        w = self.w
        short, excess = torch.relu(self.lo_rep - c), torch.relu(c - self.hi)
        out = {'coverage': float((w * ((c >= self.lo_rep) & (c <= self.hi)).to(self.dt)).sum()),
               'coverage_design': float((w * ((c >= self.lo) & (c <= self.hi)).to(self.dt)).sum()),
               'mean_short_mgL': float((w * short).sum()), 'max_short_mgL': float(short.max()),
               'mean_excess_mgL': float((w * excess).sum()), 'max_c_mgL': float(c.max()), 'min_c_mgL': float(c.min()),
               'viol': float(self.viol(c.view(1, -1))[0])}
        if u is not None:
            u = torch.as_tensor(u, dtype=self.dt, device=self.dev).reshape(self.J, self.K)
            act = (u.norm(dim=1) > 0)
            out.update({'n_stations': int(act.sum()), 'mass_kg_d': float(u.sum() * self.mass_coef),
                        'stations': [self.net.node_ids[self.cand[j]] for j in torch.nonzero(act).flatten().tolist()],
                        'station_mass_kg_d': [float(u[j].sum() * self.mass_coef) for j in torch.nonzero(act).flatten().tolist()]})
        return out

    # ---- constants of the program: per-station scaling u_j = D_j v_j and the Lipschitz constant in v ----
    def group_scale(self, m=64):
        """D_j = 1 / ||W^1/2 A_{:, j.}||_F, the column-block norms estimated matrix-free with m Gaussian
        probes: E[(A^T W^1/2 xi)_{jk}^2] = ||W^1/2 A_{:, jk}||^2 (one batched adjoint pass, eager engine)."""
        g = torch.Generator(device='cpu').manual_seed(1)
        xi = torch.randn(m, len(self.obs), dtype=self.dt, generator=g).to(self.dev) * self.w.sqrt()
        u0 = torch.zeros(m, self.J, self.K, dtype=self.dt, device=self.dev, requires_grad=True)
        c = self.eng.forward(src=self.src_of(u0)).reshape(m, -1)[:, self.obs_t]
        (c * xi).sum().backward()
        n2 = (u0.grad ** 2).sum(dim=(0, 2)) / m                                                    # [J]
        self.D = torch.where(n2 > 0, 1.0 / n2.clamp_min(1e-300).sqrt(), torch.zeros_like(n2))
        return {'probes': m, 'n_inactive_stations': int((n2 == 0).sum()),
                'block_norm_range': [float(n2[n2 > 0].min().sqrt()), float(n2.max().sqrt())]}

    def lipschitz(self, n_iter=60):
        """Largest eigenvalue of D A^T W A D (power iteration on the forward / adjoint), x1.2 safety."""
        if not hasattr(self, 'D'):
            self.group_scale()
        D = self.D.view(1, self.J, 1)
        g = torch.Generator(device='cpu').manual_seed(0)
        v = torch.randn(1, self.J, self.K, dtype=self.dt, generator=g).to(self.dev)
        v /= v.norm()
        with torch.no_grad():                  # graph outputs are static buffers: clone what is kept
            c0 = self.c_obs(torch.zeros(1, self.J, self.K, dtype=self.dt, device=self.dev)).clone()
        lam, hist = 0.0, []
        for _ in range(n_iter):
            with torch.no_grad():
                Av = self.c_obs(D * v) - c0
            u0 = torch.zeros(1, self.J, self.K, dtype=self.dt, device=self.dev, requires_grad=True)
            (self.c_obs(u0) * (self.w * Av)).sum().backward()
            wv = D * u0.grad
            lam = float(wv.norm())
            hist.append(lam)
            v = wv / max(lam, 1e-300)
        self.L = 1.2 * lam
        return {'lambda_max_scaled': lam, 'L': self.L, 'power_history': hist[-5:]}

    def grad_f(self, u, mu):
        """(f(u) per sample, grad f [B, J, K]) with f = viol + mu * mass; u [B, J, K]."""
        u = u.detach().requires_grad_(True)
        c = self.c_obs(u)
        v = self.viol(c)
        v.sum().backward()
        g = u.grad + (mu * self.mass_coef).view(-1, 1, 1)
        f = v.detach() + mu * self.mass_coef * u.detach().sum((1, 2))
        return f, g, c.detach().clone()

    def f_only(self, u, mu):
        with torch.no_grad():
            return self.viol(self.c_obs(u)) + mu * self.mass_coef * u.sum((1, 2))

    def lambda_zero(self, mu):
        """Smallest lam for which u = 0 is optimal: max_j ||(-grad_j f(0))_+||_2."""
        _, g, _ = self.grad_f(torch.zeros(1, self.J, self.K, dtype=self.dt, device=self.dev), mu.view(1))
        return float(torch.relu(-g[0]).norm(dim=1).max())


    # ---- batched FISTA in the scaled variables (adaptive restart, Lipschitz safeguard) ----
    def solve(self, lam, mu, mask=None, const_rate=False, u0=None, n_iter=None, tol=None, label=''):
        """min f(u) + lam ||u||_{2,1} + I(u >= 0) for B independent problems sharing the map: lam, mu [B];
        u_j = 0 outside mask [B, J]; u_j = D_j v_j (v_j = z_j 1_K if const_rate); warm start u0 [B, J, K].
        Returns u [B, J, K], per-sample stats and the objective history (every 50 iterations)."""
        args = self.args
        n_iter, tol = n_iter or args.iters, tol or args.tol
        B, J, K = lam.shape[0], self.J, self.K
        if self.L is None:
            self.lipschitz()
        D = self.D.view(1, J, 1)
        L = self.L * (K if const_rate else 1.0)
        scale = K ** 0.5 if const_rate else 1.0                       # ||z 1_K||_2 = sqrt(K) |z|
        mask = torch.ones(B, J, 1, dtype=self.dt, device=self.dev) if mask is None else mask.view(B, J, 1).to(self.dt)
        lam3, mu = lam.view(B, 1, 1).to(self.dt), mu.to(self.dt)
        thr = lam3 * D * scale                                          # lam ||u_j|| = lam D_j ||v_j||
        expand = (lambda z: D * z.expand(B, J, K) * mask) if const_rate else (lambda z: D * z * mask)
        z = torch.zeros(B, J, 1 if const_rate else K, dtype=self.dt, device=self.dev)
        if u0 is not None:
            v0 = torch.where(D > 0, u0.to(self.dt) / D.clamp_min(1e-300), torch.zeros_like(u0))
            z = v0.mean(2, keepdim=True) if const_rate else v0.clone()
        y, t = z.clone(), torch.ones(B, dtype=self.dt, device=self.dev)
        it_done = torch.full((B,), -1, dtype=torch.long, device=self.dev)
        quiet = torch.zeros(B, dtype=torch.long, device=self.dev)
        bumps, it, hist, t0 = 0, 0, [], time.perf_counter()
        while it < n_iter:
            f_y, g_u, _ = self.grad_f(expand(y), mu)
            g = D * g_u * mask
            if const_rate:
                g = g.sum(2, keepdim=True)
            p = torch.relu(y - g / L)
            if bool((lam > 0).any()):
                nrm = p.norm(dim=2, keepdim=True)
                p = p * torch.clamp(1.0 - thr / (L * nrm.clamp_min(1e-300)), min=0.0)
            z_new = p * mask
            if it % 25 == 0:                                            # f(z_new) <= f(y) + <g, d> + L/2 ||d||^2 ?
                d = z_new - y
                q = f_y + (g * d).sum((1, 2)) + 0.5 * L * (d * d).sum((1, 2))
                if bool((self.f_only(expand(z_new), mu) > q + 1e-12 * (1.0 + q.abs())).any()):
                    L *= 2.0; bumps += 1
                    continue
            if it % 50 == 0:
                u_ = expand(z_new)
                hist.append([it + 1] + (self.f_only(u_, mu) + lam * u_.norm(dim=2).sum(1)).tolist())
            restart = ((y - z_new) * (z_new - z)).sum((1, 2)) > 0
            t_new = torch.where(restart, torch.ones_like(t), 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t)))
            beta = torch.where(restart, torch.zeros_like(t), (t - 1.0) / t_new)
            y = z_new + beta.view(B, 1, 1) * (z_new - z)
            rel = (z_new - z).flatten(1).norm(dim=1) / z_new.flatten(1).norm(dim=1).clamp_min(1e-12)
            quiet = torch.where(rel < tol, quiet + 1, torch.zeros_like(quiet))
            it_done = torch.where((it_done < 0) & (quiet >= 5), torch.full_like(it_done, it + 1), it_done)
            z, t, it = z_new, t_new, it + 1
            if bool((it_done >= 0).all()):
                break
        self.sync()
        wall = time.perf_counter() - t0
        u = expand(z).detach()
        f, _, c = self.grad_f(u, mu)
        obj = f + lam * u.norm(dim=2).sum(1)
        hist.append([it] + obj.tolist())
        stats = [dict(self.metrics(c[b], u[b]), obj=float(obj[b]), f=float(f[b]), lam=float(lam[b]), mu=float(mu[b]),
                      iters=int(it_done[b]) if int(it_done[b]) > 0 else it, converged=bool(it_done[b] > 0))
                 for b in range(B)]
        if label:
            print(f'  {label:34s} B={B} iters={it} conv={sum(s["converged"] for s in stats)}/{B} L_bumps={bumps} wall={wall:.1f}s '
                  f'stations={[s["n_stations"] for s in stats][:12]} cover={[round(s["coverage"], 4) for s in stats][:12]}', flush=True)
        return {'u': u, 'c': c, 'stats': stats, 'iters': it, 'wall_s': wall, 'L_bumps': bumps, 'L_final': L, 'B': B, 'history': hist}

    # ---- checks ----
    def response_matrix(self, chunk):
        """Dense A_eval [n_obs, J*K] by one-hot forwards (eager engine, no graph), and c0 [n_obs]."""
        J, K = self.J, self.K
        with torch.no_grad():
            c0 = self.eng.forward(src=self.src_of(torch.zeros(1, J, K, dtype=self.dt, device=self.dev)))
            c0 = c0.reshape(1, -1)[0, self.obs_t].clone()
            cols = []
            for s in range(0, J * K, chunk):
                idx = torch.arange(s, min(s + chunk, J * K), device=self.dev)
                U = torch.zeros(len(idx), J * K, dtype=self.dt, device=self.dev)
                U[torch.arange(len(idx), device=self.dev), idx] = 1.0
                C = self.eng.forward(src=self.src_of(U.view(-1, J, K))).reshape(len(idx), -1)[:, self.obs_t]
                cols.append((C - c0).t().clone())
        return torch.cat(cols, 1), c0

    def solve_dense(self, A, c0, lam, mu, n_iter=50000, tol=1e-11):
        """The same program with the explicit matrix (reference solution): exact block scaling D from the
        columns of W^1/2 A, exact Lipschitz constant, FISTA with restart to a tight tolerance."""
        B, J, K = lam.shape[0], self.J, self.K
        w = self.w
        M = A * w.sqrt().view(-1, 1)
        n2 = (M * M).sum(0).view(J, K).sum(1)
        D = torch.where(n2 > 0, 1.0 / n2.clamp_min(1e-300).sqrt(), torch.zeros_like(n2))            # [J]
        Dc = D.view(J, 1).expand(J, K).reshape(-1)
        AD = A * Dc.view(1, -1)
        L = float(torch.linalg.eigvalsh((AD * w.view(-1, 1)).t() @ AD).max()) * 1.000001
        mc = (mu.view(B, 1) * self.mass_coef) * Dc.view(1, -1)
        thr = lam.view(B, 1, 1) * D.view(1, J, 1)
        z = torch.zeros(B, J * K, dtype=self.dt, device=self.dev); y, t = z.clone(), torch.ones(B, dtype=self.dt, device=self.dev)
        for it in range(n_iter):
            c = c0 + y @ AD.t()
            r = -torch.relu(self.lo - c) + torch.relu(c - self.hi)
            g = (r * w) @ AD + mc
            p = torch.relu(y - g / L).view(B, J, K)
            nrm = p.norm(dim=2, keepdim=True)
            z_new = (p * torch.clamp(1.0 - thr / (L * nrm.clamp_min(1e-300)), min=0.0)).view(B, -1)
            restart = ((y - z_new) * (z_new - z)).sum(1) > 0
            t_new = torch.where(restart, torch.ones_like(t), 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t)))
            y = z_new + torch.where(restart, torch.zeros_like(t), (t - 1.0) / t_new).view(B, 1) * (z_new - z)
            done = bool(((z_new - z).norm(dim=1) <= tol * z_new.norm(dim=1).clamp_min(1e-12)).all()) and it > 50
            z, t = z_new, t_new
            if done:
                break
        u = (z * Dc.view(1, -1)).view(B, J, K)
        c = c0 + z @ AD.t()
        f = 0.5 * (w * (torch.relu(self.lo - c) ** 2 + torch.relu(c - self.hi) ** 2)).sum(1) + mu * self.mass_coef * u.sum((1, 2))
        return {'u': u, 'obj': f + lam * u.norm(dim=2).sum(1), 'L_exact': L / 1.000001, 'iters': it + 1,
                'D_exact': D, 'block_norm_range': [float(n2[n2 > 0].min().sqrt()), float(n2.max().sqrt())]}

    def verify_dll(self, u, tag):
        """EPANET run with the design installed through the API (MASS source + pattern per station)."""
        u = np.asarray(u.detach().cpu().numpy() if torch.is_tensor(u) else u).reshape(self.J, self.K)
        with self.open(tag) as pr:
            for j in np.flatnonzero(u.sum(1) > 0):
                uref = float(u[j].max())
                pidx = pr.add_pattern(f'b{j}', (u[j] / uref).tolist())
                pr.set_source(self.cand[j], api.EN_MASS, uref, pidx)
            pr.solve_hydraulics(self.workdir / f'{self.name}{tag}.hyd')
            _, q, _ = pr.run_quality_reference('nextQ')
        with torch.no_grad():
            c = self.c_full(torch.as_tensor(u, dtype=self.dt, device=self.dev).view(1, self.J, self.K))[0].cpu().numpy()
        m_dll = self.metrics(q.reshape(-1)[self.obs])
        m_eng = self.metrics(c.reshape(-1)[self.obs])
        return {'max_rel_err_all_nodes': float(np.abs(q - c).max() / max(np.abs(q).max(), 1e-300)),
                'coverage_dll': m_dll['coverage'], 'coverage_engine': m_eng['coverage'],
                'viol_dll': m_dll['viol'], 'viol_engine': m_eng['viol']}

    def curves(self, c_full):
        """Hourly demand-weighted mean / min over junctions of a full snapshot array (for figures)."""
        c = np.asarray(c_full)
        W = self.W_np
        rows = np.flatnonzero(W.sum(1) > 0)
        return {'t_h': [float(self.times[s]) / 3600 for s in rows],
                'mean_mgL': [float((W[s] * c[s]).sum() / W[s].sum()) for s in rows],
                'min_mgL': [float(c[s][W[s] > 0].min()) for s in rows],
                'frac_below_lo': [float(W[s][c[s] < self.lo_rep].sum() / W[s].sum()) for s in rows]}


def periodicity(P):
    """Plant-only concentrations of the design window vs the same clock times one day earlier."""
    c = P.c_plant.cpu().numpy()
    by_t = {int(t): s for s, t in enumerate(P.times)}
    is_junc = P.net.node_type == api.EN_JUNCTION
    diffs = []
    for s, t in enumerate(P.times):
        if P.W_np[s].sum() > 0 and int(t) - 86400 in by_t:
            diffs.append(np.abs(c[s, is_junc] - c[by_t[int(t) - 86400], is_junc]))
    if not diffs:
        return None
    d = np.stack(diffs)
    return {'max_abs_mgL': float(d.max()), 'mean_abs_mgL': float(d.mean()), 'snapshots': len(diffs)}


def run_network(name, lib, workdir, args):
    t_start = time.perf_counter()
    P = BoosterProblem(name, lib, workdir, args)
    try:
        return _run(P, args, t_start)
    finally:
        P.free_graphs()
        P.pr.__exit__()


def _run(P, args, t_start):
    name, T = P.name, time.perf_counter
    net, dev, dt = P.net, P.dev, P.dt
    print(f'[{name}] nodes={net.nnodes} junctions={int(np.sum(net.node_type[1:] == api.EN_JUNCTION))} tanks={net.ntanks} '
          f'links={net.nlinks} dur={net.duration // 3600}h qstep={net.qstep}s K={P.K} candidates={P.J} n_obs={len(P.obs)} '
          f'existing_sources={len(net.sources)} | record={P.t_record:.1f}s build={P.t_build:.1f}s setup={P.t_setup:.1f}s '
          f'dll_run={P.t_dll * 1e3:.0f}ms | {P.sch_info}', flush=True)
    R = {'network': name, 'theta': P.theta_info, 'sizes': {'nodes': net.nnodes, 'links': net.nlinks, 'candidates': P.J,
         'slots': P.K, 'decision_vars': P.J * P.K, 'n_obs': len(P.obs), 'existing_sources': len(net.sources),
         'duration_h': net.duration / 3600, 'eval_h': list(P.cfg['eval_h']), **P.sch_info},
         'settings': {'lo_report': P.lo_rep, 'lo_design': P.lo, 'margin': args.margin, 'hi': P.hi, 'mu_rel': args.mu_rel,
                      'mu': P.mu, 'V0': P.V0, 'M_ref_kg_d': P.M_ref, 'mass_coef_kg_d_per_unit': P.mass_coef,
                      'tol': args.tol, 'iters_max': args.iters, 'ctol': 0.0},
         'candidates': [net.node_ids[j] for j in P.cand]}
    tm = {'record_s': P.t_record, 'build_s': P.t_build, 'setup_s': P.t_setup, 'dll_quality_run_s': P.t_dll}
    c_plant_obs = P.c_plant.reshape(-1)[P.obs_t]
    R['plant_only'] = dict(P.metrics(c_plant_obs, torch.zeros(P.J, P.K, dtype=dt, device=dev)),
                           curves=P.curves(P.c_plant.cpu().numpy()))
    R['periodicity_plant_only'] = periodicity(P) if P.cfg['periodic'] else None
    print(f'  plant only: coverage={R["plant_only"]["coverage"]:.4f} mean_short={R["plant_only"]["mean_short_mgL"]:.4f} '
          f'min_c={R["plant_only"]["min_c_mgL"]:.3f} V0={P.V0:.3e} M_ref={P.M_ref:.2f}kg/d mu={P.mu:.3e} '
          f'periodicity={R["periodicity_plant_only"]}', flush=True)
    t0 = T(); R['group_scale'] = P.group_scale(); tm['group_scale_s'] = T() - t0
    t0 = T(); R['lipschitz'] = P.lipschitz(); tm['lipschitz_s'] = T() - t0
    mu1 = torch.full((1,), P.mu, dtype=dt, device=dev)
    lam0 = P.lambda_zero(mu1)
    R['lambda_zero'] = lam0
    fracs = [float(x) for x in args.lam_fracs.split(',')]
    B = len(fracs)
    lam = torch.tensor([f * lam0 for f in fracs], dtype=dt, device=dev)
    muB = torch.full((B,), P.mu, dtype=dt, device=dev)
    zero = torch.zeros(B, dtype=dt, device=dev)
    print(f'  L={P.L:.4e} lambda_0={lam0:.4e} scale={R["group_scale"]} graph_capture_B{B}...', flush=True)
    t0 = T(); sweep = P.solve(lam, muB, label='group lasso sweep'); tm['sweep_s'] = T() - t0
    support = sweep['u'].norm(dim=2) > 0
    t0 = T(); refit = P.solve(zero, muB, mask=support, u0=sweep['u'], label='refit on supports'); tm['refit_s'] = T() - t0
    t0 = T(); const = P.solve(zero, muB, mask=support, const_rate=True, u0=sweep['u'], label='constant rate on supports'); tm['const_s'] = T() - t0
    tm['graph_capture_s'] = P.free_graphs()
    R['sweep'] = [{'lam_frac': f, **s} for f, s in zip(fracs, sweep['stats'])]
    R['refit'] = [{'lam_frac': f, **s, 'u': refit['u'][b].cpu().numpy().round(6).tolist(),
                   'curves': P.curves(P.c_full(refit['u'][b:b + 1]).detach()[0].cpu().numpy())}
                  for b, (f, s) in enumerate(zip(fracs, refit['stats']))]
    R['const_rate'] = [{'lam_frac': f, **s} for f, s in zip(fracs, const['stats'])]
    R['solver'] = {k: {kk: v[kk] for kk in ('iters', 'wall_s', 'L_bumps', 'L_final', 'B', 'history')} for k, v in
                   (('sweep', sweep), ('refit', refit), ('const_rate', const))}
    # ---- greedy forward selection with the same inner solver (selection budget), then one batched refit
    #      of the G nested supports at the full budget (the numbers that are compared with the lasso path) ----
    G = min(args.greedy_max, max(s['n_stations'] for s in refit['stats']))
    S, greedy, t0, u_greedy, u_stages = [], [], T(), None, []
    for stage in range(G):
        rest = [j for j in range(P.J) if j not in S]
        masks = torch.zeros(len(rest), P.J, dtype=dt, device=dev)
        if S:
            masks[:, torch.as_tensor(S, device=dev)] = 1.0
        masks[torch.arange(len(rest), device=dev), torch.as_tensor(rest, device=dev)] = 1.0
        u0 = None if u_greedy is None else u_greedy.unsqueeze(0).expand(len(rest), P.J, P.K).contiguous()
        res = P.solve(torch.zeros(len(rest), dtype=dt, device=dev), torch.full((len(rest),), P.mu, dtype=dt, device=dev),
                      mask=masks, u0=u0, n_iter=args.greedy_iters, label=f'greedy stage {stage + 1}')
        b = int(np.argmin([s['obj'] for s in res['stats']]))
        S.append(rest[b])
        u_greedy = res['u'][b].clone()
        u_stages.append(u_greedy)
        greedy.append({'stage': stage + 1, 'added': net.node_ids[P.cand[rest[b]]], **res['stats'][b],
                       'wall_s': res['wall_s'], 'B': len(rest), 'iters': res['iters'], 'L_bumps': res['L_bumps'],
                       'n_converged': sum(s['converged'] for s in res['stats']),
                       'obj_all': [s['obj'] for s in res['stats']]})
        P.free_graphs()
    if G:
        masks = torch.zeros(G, P.J, dtype=dt, device=dev)
        for g in range(G):
            masks[g, torch.as_tensor(S[:g + 1], device=dev)] = 1.0
        gre = P.solve(torch.zeros(G, dtype=dt, device=dev), torch.full((G,), P.mu, dtype=dt, device=dev), mask=masks,
                      u0=torch.stack(u_stages), label='greedy refit (nested supports)')
        for g in range(G):
            greedy[g]['refit'] = dict(gre['stats'][g], u=gre['u'][g].cpu().numpy().round(6).tolist())
        u_greedy = gre['u'][-1].clone()
        R['solver']['greedy_refit'] = {kk: gre[kk] for kk in ('iters', 'wall_s', 'L_bumps', 'L_final', 'B', 'history')}
        P.free_graphs()
    tm['greedy_s'] = T() - t0
    R['greedy'] = greedy
    # ---- mass weight sweep (relative), all stations allowed ----
    mu_rels = [float(x) for x in args.mu_sweep.split(',')]
    mus = torch.tensor([m * P.V0 / P.M_ref for m in mu_rels], dtype=dt, device=dev)
    t0 = T(); msw = P.solve(torch.zeros(len(mus), dtype=dt, device=dev), mus, label='mu sweep, all stations'); tm['mu_sweep_s'] = T() - t0
    R['mu_sweep'] = [{'mu_rel': m, **s} for m, s in zip(mu_rels, msw['stats'])]
    R['solver']['mu_sweep'] = {kk: msw[kk] for kk in ('iters', 'wall_s', 'L_bumps', 'L_final', 'B', 'history')}
    P.free_graphs()
    # ---- dense response matrix: same program, explicit A ----
    if args.matrix:
        t0 = T(); A, c0 = P.response_matrix(args.chunk); tA = T() - t0
        t0 = T(); dense = P.solve_dense(A, c0, lam, muB); tD = T() - t0
        per = []
        for b in range(B):
            ud, uf = dense['u'][b], sweep['u'][b]
            per.append({'lam_frac': fracs[b], 'obj_free': sweep['stats'][b]['obj'], 'obj_dense': float(dense['obj'][b]),
                        'rel_obj_diff': float(abs(sweep['stats'][b]['obj'] - dense['obj'][b]) / max(abs(dense['obj'][b]), 1e-300)),
                        'rel_u_diff': float((uf - ud).norm() / max(ud.norm(), 1e-300)),
                        'n_stations_dense': int((ud.norm(dim=1) > 0).sum()), 'n_stations_free': sweep['stats'][b]['n_stations']})
        De, Dp = dense['D_exact'], P.D
        act = De > 0
        d_rel = float(((Dp[act] - De[act]).abs() / De[act]).max()) if bool(act.any()) else 0.0
        R['response_matrix'] = {'n_obs': int(A.shape[0]), 'n_cols': int(A.shape[1]), 'build_s': tA, 'dense_solve_s': tD,
                                'dense_iters': dense['iters'], 'L_exact': dense['L_exact'], 'L_power': P.L / 1.2,
                                'block_norm_range_exact': dense['block_norm_range'], 'D_probe_max_rel_err': d_rel,
                                'D_support_match': bool(((Dp > 0) == act).all()),
                                'projected_dll_runs_s': P.J * P.K * P.t_dll, 'per_lam': per}
        print(f'  response matrix {A.shape[0]}x{A.shape[1]} built in {tA:.1f}s; dense solve {tD:.1f}s ({dense["iters"]} it); '
              f'max rel obj diff {max(p["rel_obj_diff"] for p in per):.1e}; max rel u diff {max(p["rel_u_diff"] for p in per):.1e}; '
              f'L exact/power {dense["L_exact"]:.4e}/{P.L / 1.2:.4e}; D probe max rel err {d_rel:.2f}', flush=True)
        del A
        tm['matrix_s'] = tA + tD
    # ---- EPANET check of the designs ----
    t0 = T()
    R['dll_check'] = [{'kind': 'refit', 'lam_frac': f, **P.verify_dll(refit['u'][b], f'_chk{b}')} for b, f in enumerate(fracs)]
    if u_greedy is not None:
        R['dll_check'].append({'kind': 'greedy', 'stations': len(S), **P.verify_dll(u_greedy, '_chkg')})
    tm['dll_check_s'] = T() - t0
    print(f'  DLL check: max rel err {max(d["max_rel_err_all_nodes"] for d in R["dll_check"]):.1e}; '
          f'coverage dll-engine max diff {max(abs(d["coverage_dll"] - d["coverage_engine"]) for d in R["dll_check"]):.1e}', flush=True)
    tm['total_s'] = T() - t_start
    R['timing'] = tm
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='Net3', help='comma-separated: ' + ','.join(CONFIGS))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--lo', type=float, default=0.2, help='minimum residual (mg/L) at which compliance is reported')
    ap.add_argument('--margin', type=float, default=0.02, help='design margin: the program enforces lo + margin')
    ap.add_argument('--hi', type=float, default=4.0, help='maximum residual (mg/L)')
    ap.add_argument('--mu-rel', type=float, default=0.1, help='mass weight: dosing M_ref (0.1 mg/L on the whole supply) '
                                                                'costs mu_rel x the plant-only violation')
    ap.add_argument('--mu-sweep', default='0.01,0.1,1,10', help='relative mass weights, all stations allowed')
    ap.add_argument('--lam-fracs', default='0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0')
    ap.add_argument('--iters', type=int, default=5000)
    ap.add_argument('--greedy-iters', type=int, default=2500, help='budget of the greedy selection stages')
    ap.add_argument('--tol', type=float, default=1e-7)
    ap.add_argument('--greedy-max', type=int, default=8)
    ap.add_argument('--matrix', action='store_true', help='also build the dense response matrix and solve with it')
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--no-tanks', action='store_true', help='junction candidates only')
    ap.add_argument('--theta-json', default=None, help='calib.py result JSON whose best ADJ theta is installed')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    results = {'platform': platform.platform(), 'torch': torch.__version__, 'epanet': lib.version(), 'args': vars(args),
               'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else platform.processor(),
               'networks': []}
    with tempfile.TemporaryDirectory(prefix='exp1_', dir=os.environ.get('TMPDIR')) as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in args.network.split(','):
            results['networks'].append(run_network(name, lib, workdir, args))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1))
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
