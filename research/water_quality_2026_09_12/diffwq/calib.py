"""Gate G4: Kb / Kw calibration against the measured chlorine data sets (Fairfield, Bellingham,
North Penn) with the exact adjoint gradient of the replayed EPANET map, against derivative-free
and finite-difference baselines that call the EPANET DLL itself.

Forward map: hydraulics solved once by the DLL; quality in EN_stepQ mode (one snapshot per quality
step) with TOLERANCE = 0 (mode M1: no segment merging, so the transport schedule does not depend on
the reaction parameters and one recorded schedule replays the DLL exactly for every theta).
The identical loss is evaluated through the DLL (set EN_KBULK / EN_KWALL / EN_TANK_KBULK, run
EN_stepQ) for the baselines, so all optimizers see the same objective (checked to ~1e-12).

theta = [Kb (1/day, pipes and tanks), Kw_g (ft/day) for each diameter group g]; all <= 0 (decay).
Observations enter through linear interpolation between the two snapshots bracketing each
sample time; loss = sum of squared residuals (mg/L)^2.

The B = 1 adjoint gradient runs eagerly (method adj) or, on a CUDA device, through a CUDA-graph
capture of the replay (method adjg, same capture as uq.Engine): the serial chain of a few thousand
small kernels is launch-latency bound, so the graph replay is what makes the exact gradient
comparable in wall time with (p + 1) DLL forward runs.

Usage (from research/water_quality_2026_09_12):
    python -m diffwq.calib --datasets Fairfield,Bellingham,NorthPenn --methods all --seeds 5 --out g4_calib.json
    python -m diffwq.calib --datasets Fairfield --methods fd,adj,adjg --repeat 3 --device cuda --out g4_graph.json
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
from scipy.optimize import differential_evolution, minimize

from . import epanet_api as api
from .calib_data import DATASETS, prepare_inp, load_observations
from .hydfile import read_hydfile
from .network import extract, SECperDAY
from .replicate import Replicator
from .torch_wq import TorchWQ

HERE = Path(__file__).resolve().parent
DIAM_EDGES = (7.0, 9.0, 13.0)                 # inches: <=6 | 8 | 10-12 | >=16
GROUP_NAMES = ('<=6in', '8in', '10-12in', '>=16in')


def obs_indices(obs, times, qstep, N1):
    """Linear interpolation between the two stepQ snapshots (times = k * qstep) bracketing each
    sample: flat indices into q.reshape(-1) with q[n_snap, N1], and the weight of the later one."""
    j = np.minimum((obs.t // qstep).astype(np.int64), len(times) - 2)
    w = (obs.t - times[j]) / qstep
    return j * N1 + obs.node, (j + 1) * N1 + obs.node, w


class CalibProblem:
    """One data set: opened project (kept open for DLL evaluations), recorded schedule, torch engine."""

    def __init__(self, name, lib, workdir, device='cpu', ctol=0.0, kb_bounds=(-5.0, 0.0), kw_bounds=(-5.0, 0.0),
                 free_tank_kb=False, fix_kb=False, graph=False, verbose=True, engine='adjoint'):
        self.name, self.dev = name, torch.device(device)
        self.workdir = Path(workdir)
        self.inp = prepare_inp(name, self.workdir)
        self.pr = api.Project(lib, self.inp, self.workdir / f'{name}.rpt')
        pr = self.pr
        self.obs = load_observations(name, pr)
        self.qstep, self.dur = pr.time_param(api.EN_QUALSTEP), pr.time_param(api.EN_DURATION)
        self.ctol = float(ctol)
        pr.set_option(api.EN_TOLERANCE, self.ctol)
        hyd = self.workdir / f'{name}.hyd'
        pr.solve_hydraulics(hyd)
        self.net = extract(pr)
        self.trace = read_hydfile(hyd)
        t0 = time.perf_counter()
        rep = Replicator(self.net, self.trace, ctol=self.ctol, record=True)
        self.times, _, _ = rep.run('stepQ')
        self.t_record = time.perf_counter() - t0
        self.sch, self.reactflag = rep.schedule, rep.reactflag
        t0 = time.perf_counter()
        self.eng = TorchWQ(self.net, self.sch, self.reactflag, device=device, engine=engine)
        self.t_build = time.perf_counter() - t0
        n = self.net
        # ---- parameter groups: Kb shared by all pipes (and tanks unless free_tank_kb); Kw by diameter class
        L, T = n.nlinks, n.ntanks
        diam_in = np.array([pr.link_value(k, api.EN_DIAMETER) if k else 0.0 for k in range(L + 1)])
        self.is_pipe = np.concatenate(([False], n.link_type[1:] == api.EN_PIPE))
        raw = np.digitize(diam_in, DIAM_EDGES)
        present = sorted({int(g) for g in raw[self.is_pipe]})
        remap = {g: i for i, g in enumerate(present)}
        self.group_names = [GROUP_NAMES[g] for g in present]
        self.group_count = [int(np.sum(raw[self.is_pipe] == g)) for g in present]
        self.link_group = np.array([remap[int(g)] if p else 0 for g, p in zip(raw, self.is_pipe)], dtype=np.int64)
        self.G = len(present)
        self.free_tank_kb = bool(free_tank_kb)
        self.fix_kb = bool(fix_kb)                    # Kb held at the INP (bottle-test) value, Kw groups only
        kb0 = float(np.median(n.kb[self.is_pipe]) * SECperDAY)
        kw0 = [float(np.median(n.kw[self.is_pipe & (self.link_group == g)]) * SECperDAY) for g in range(self.G)]
        tkb0 = [float(np.median(n.tank_kb[1:]) * SECperDAY)] if (free_tank_kb and T) else ([kb0] if free_tank_kb else [])
        self.kb_fixed = kb0
        self.kw_off = 0 if fix_kb else 1
        self.param_names = ([] if fix_kb else ['Kb']) + [f'Kw[{g}]' for g in self.group_names] + (['Kb_tank'] if free_tank_kb else [])
        self.ntheta = len(self.param_names)
        self.bounds = ([] if fix_kb else [tuple(kb_bounds)]) + [tuple(kw_bounds)] * self.G + ([tuple(kb_bounds)] if free_tank_kb else [])
        self.theta_nominal = np.array(([] if fix_kb else [kb0]) + kw0 + tkb0)
        # ---- observation operator on the flattened snapshot tensor [n_snap * (N+1)]
        self._i0, self._i1, self._w = obs_indices(self.obs, self.times, self.qstep, n.nnodes + 1)
        self.obs_i0 = torch.as_tensor(self._i0, device=self.dev)
        self.obs_i1 = torch.as_tensor(self._i1, device=self.dev)
        self.obs_w = torch.as_tensor(self._w, dtype=torch.float64, device=self.dev)
        self.y = torch.as_tensor(self.obs.y, dtype=torch.float64, device=self.dev)
        self.lg_t = torch.as_tensor(self.link_group, device=self.dev)
        # c0 / src cached on the device: forward() would otherwise copy them host->device on every
        # call, which is not allowed inside a CUDA-graph capture
        self.c0_t = torch.as_tensor(np.asarray(n.c0 * n.ucf['QUALITY']), dtype=torch.float64, device=self.dev).view(1, -1)
        self.src_t = torch.as_tensor(np.asarray([n.sources.get(i, (0, 0.0, 0))[1] for i in range(n.nnodes + 1)]),
                                     dtype=torch.float64, device=self.dev).view(1, -1)
        self.graphed, self.graph_error, self.t_graph = None, None, 0.0
        if graph and self.dev.type == 'cuda':
            t0 = time.perf_counter()
            try:
                th = torch.as_tensor(self.theta_nominal, device=self.dev).view(1, -1).requires_grad_(True)
                self.graphed = torch.cuda.make_graphed_callables(self.simulate_torch, (th,), num_warmup_iters=3,
                                                                 allow_unused_input=True)
            except Exception as exc:
                self.graph_error = f'{type(exc).__name__}: {str(exc)[:300]}'
            self.t_graph = time.perf_counter() - t0
        self.n_dll, self.t_dll, self.n_grad, self.t_grad = 0, 0.0, 0, 0.0
        if verbose:
            print(f'[{name}] nodes={n.nnodes} links={L} pipes={int(self.is_pipe.sum())} tanks={T} obs={len(self.obs)} '
                  f'sites={len(self.obs.site_node_id)} dropped={self.obs.n_dropped} qstep={self.qstep}s dur={self.dur // 3600}h '
                  f'groups={dict(zip(self.group_names, self.group_count))} record={self.t_record:.1f}s build={self.t_build:.2f}s '
                  f'versions={self.sch.nver} ops={len(self.sch.ops)}'
                  + (f' graph={self.t_graph:.1f}s' if self.graphed is not None else '')
                  + (f' graph capture failed: {self.graph_error}' if self.graph_error else ''), flush=True)

    def close(self):
        self.pr.__exit__(None, None, None)

    # ---- theta -> per-link / per-tank coefficients (EPANET user units) ----
    def expand_np(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        L, T = self.net.nlinks, self.net.ntanks
        kb_val = self.kb_fixed if self.fix_kb else theta[0]
        kb = np.full(L + 1, kb_val); kw = theta[self.kw_off + self.link_group]
        tkb = np.full(T + 1, theta[-1] if self.free_tank_kb else kb_val)
        return kb, kw, tkb

    def expand_torch(self, theta):
        B = theta.shape[0]
        L, T = self.net.nlinks, self.net.ntanks
        kb_col = torch.full_like(theta[:, 0:1], self.kb_fixed) if self.fix_kb else theta[:, 0:1]
        kb = kb_col.expand(B, L + 1)
        kw = theta[:, self.kw_off + self.lg_t]
        tkb = (theta[:, -1:] if self.free_tank_kb else kb_col).expand(B, T + 1)
        return kb, kw, tkb

    # ---- simulated observations ----
    def simulate_torch(self, theta):
        """theta [B, ntheta] -> simulated observations [B, M] (differentiable)."""
        B = theta.shape[0]
        kb, kw, tkb = self.expand_torch(theta)
        out = self.eng.forward(kb=kb, kw=kw, tank_kb=tkb, c0=self.c0_t.expand(B, -1), src=self.src_t.expand(B, -1)).reshape(B, -1)
        return out[:, self.obs_i0] * (1.0 - self.obs_w) + out[:, self.obs_i1] * self.obs_w

    def simulate_dll(self, theta, ctol=None):
        """Same map through the EPANET DLL (EN_setlinkvalue / EN_setnodevalue + EN_stepQ loop)."""
        t0 = time.perf_counter()
        pr, n = self.pr, self.net
        kb, kw, tkb = self.expand_np(theta)
        for k in range(1, n.nlinks + 1):
            if n.link_type[k] <= api.EN_PIPE:
                pr.set_link_value(k, api.EN_KBULK, kb[k]); pr.set_link_value(k, api.EN_KWALL, kw[k])
        for i in range(1, n.ntanks + 1):
            if not n.tank_is_reservoir[i]:
                pr.set_node_value(int(n.tank_node[i]), api.EN_TANK_KBULK, tkb[i])
        if ctol is not None:
            pr.set_option(api.EN_TOLERANCE, ctol)
        _, q, _ = pr.run_quality_reference('stepQ')
        if ctol is not None:
            pr.set_option(api.EN_TOLERANCE, self.ctol)
        flat = q.reshape(-1)
        self.n_dll += 1; self.t_dll += time.perf_counter() - t0
        return flat[self._i0] * (1.0 - self._w) + flat[self._i1] * self._w

    # ---- losses ----
    def sse_dll(self, theta, ctol=None, hist=None):
        r = self.simulate_dll(theta, ctol) - self.obs.y
        v = float(r @ r)
        if hist is not None:
            hist.append(v)
        return v

    def sse_grad(self, theta, hist=None, use_graph=False):
        """(SSE, dSSE/dtheta) from one adjoint sweep; theta 1-D numpy. use_graph replays the captured
        CUDA graph of the B = 1 map (CalibProblem(graph=True) on a CUDA device) instead of running eagerly."""
        t0 = time.perf_counter()
        th = torch.as_tensor(np.asarray(theta, dtype=np.float64), device=self.dev).view(1, -1).requires_grad_(True)
        sim = self.graphed(th) if (use_graph and self.graphed is not None) else self.simulate_torch(th)
        r = sim[0] - self.y
        loss = r @ r
        loss.backward()
        self.n_grad += 1; self.t_grad += time.perf_counter() - t0
        v = loss.item()
        if hist is not None:
            hist.append(v)
        return v, th.grad[0].cpu().numpy().copy()

    def sse_batch(self, theta):
        """theta [B, ntheta] tensor -> SSE per row [B] (differentiable)."""
        r = self.simulate_torch(theta) - self.y
        return (r * r).sum(dim=1)

    def jacobian(self, theta):
        """Residual Jacobian [M, ntheta] from a single batched adjoint sweep (B = M one-hot outputs)."""
        M = len(self.obs)
        th = torch.as_tensor(np.asarray(theta, dtype=np.float64), device=self.dev).repeat(M, 1).requires_grad_(True)
        sim = self.simulate_torch(th)
        sim.diagonal().sum().backward()
        return th.grad.cpu().numpy().copy()

    def fit_stats(self, theta, sim=None):
        """RMSE / MAE / bias / per-site RMSE and linearised least-squares statistics at theta."""
        if sim is None:
            with torch.no_grad():
                sim = self.simulate_torch(torch.as_tensor(np.asarray(theta, dtype=np.float64), device=self.dev).view(1, -1))[0].cpu().numpy()
        r = sim - self.obs.y
        M, p = len(r), self.ntheta
        out = {'sse': float(r @ r), 'rmse': float(np.sqrt(np.mean(r * r))), 'mae': float(np.abs(r).mean()),
               'bias': float(r.mean()), 'corr': float(np.corrcoef(sim, self.obs.y)[0, 1]),
               'site_rmse': {self.obs.site_node_id[s]: float(np.sqrt(np.mean(r[self.obs.site == s] ** 2)))
                             for s in range(len(self.obs.site_node_id))}}
        J = self.jacobian(theta)
        JtJ = J.T @ J
        sigma2 = out['sse'] / max(M - p, 1)
        ev, V = np.linalg.eigh(JtJ)
        out['jtj_eigvals'] = ev[::-1].tolist()
        out['jtj_eigvecs'] = V[:, ::-1].T.tolist()
        try:
            cov = sigma2 * np.linalg.inv(JtJ)
            sd = np.sqrt(np.diag(cov))
            out['ci95_halfwidth'] = (1.96 * sd).tolist()
            out['correlation'] = (cov / np.outer(sd, sd)).tolist()
        except np.linalg.LinAlgError:
            out['ci95_halfwidth'] = None
        out['sigma'] = float(np.sqrt(sigma2))
        return out


# ----------------------------------------------------------------------------- optimizers
def _record(prob, method, theta, t_wall, extra=None, n_eval=None, seed=None):
    theta = np.asarray(theta, dtype=np.float64)
    st = prob.fit_stats(theta)
    st.pop('jtj_eigvecs', None)
    rec = {'method': method, 'theta': theta.tolist(), 'wall_s': t_wall, 'n_eval': n_eval, 'seed': seed}
    rec.update({k: st[k] for k in ('sse', 'rmse', 'mae', 'bias', 'corr', 'site_rmse', 'ci95_halfwidth', 'correlation', 'jtj_eigvals')})
    rec['sse_dll_check'] = prob.sse_dll(theta)
    if extra:
        rec.update(extra)
    return rec


def _best_so_far(hist, every):
    b = np.minimum.accumulate(np.asarray(hist, dtype=np.float64)) if hist else np.zeros(0)
    return b[::every].tolist()


def run_de(prob, seed, popsize=15, maxiter=300, tol=1e-5):
    """SciPy differential evolution (best1bin) on the DLL objective; no polishing step."""
    hist = []
    n0, t0 = prob.n_dll, time.perf_counter()
    res = differential_evolution(lambda x: prob.sse_dll(x, hist=hist), prob.bounds, seed=seed, popsize=popsize,
                                 maxiter=maxiter, tol=tol, atol=0.0, mutation=(0.5, 1.0), recombination=0.7,
                                 polish=False, init='latinhypercube')
    return _record(prob, 'DE', res.x, time.perf_counter() - t0, n_eval=prob.n_dll - n0, seed=seed,
                   extra={'nit': int(res.nit), 'success': bool(res.success), 'message': str(res.message),
                          'best_every_100': _best_so_far(hist, 100)})


def run_nm(prob, x0, maxfev=4000):
    hist = []
    n0, t0 = prob.n_dll, time.perf_counter()
    res = minimize(lambda x: prob.sse_dll(x, hist=hist), x0, method='Nelder-Mead', bounds=prob.bounds,
                   options={'xatol': 1e-5, 'fatol': 1e-10, 'maxfev': maxfev, 'adaptive': True})
    return _record(prob, 'NM', res.x, time.perf_counter() - t0, n_eval=prob.n_dll - n0,
                   extra={'nit': int(res.nit), 'success': bool(res.success), 'message': str(res.message),
                          'best_every_10': _best_so_far(hist, 10)})


def run_fd_lbfgsb(prob, x0, eps=1e-8, ctol=None, maxiter=200, label='FD-LBFGSB'):
    """L-BFGS-B with SciPy's forward-difference gradient of the DLL objective (absolute step eps).
    ctol != None evaluates the DLL at that TOLERANCE (e.g. the EPANET default 0.01) instead of 0."""
    hist = []
    n0, t0 = prob.n_dll, time.perf_counter()
    res = minimize(lambda x: prob.sse_dll(x, ctol, hist=hist), x0, method='L-BFGS-B', bounds=prob.bounds, jac=None,
                   options={'eps': eps, 'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9, 'maxcor': 20})
    rec = _record(prob, label, res.x, time.perf_counter() - t0, n_eval=prob.n_dll - n0,
                  extra={'nit': int(res.nit), 'success': bool(res.success), 'message': str(res.message),
                         'eps': eps, 'ctol_eval': prob.ctol if ctol is None else ctol, 'best_every_10': _best_so_far(hist, 10)})
    if ctol is not None:                                # objective as seen by this optimizer
        rec['sse_at_eval_ctol'] = prob.sse_dll(res.x, ctol)
    return rec


def run_adj_lbfgsb(prob, x0, maxiter=200, label='ADJ-LBFGSB', use_graph=False):
    """L-BFGS-B with the exact adjoint gradient (torch replay, B = 1): eager, or the captured CUDA graph."""
    hist = []
    n0, t0 = prob.n_grad, time.perf_counter()
    res = minimize(lambda x: prob.sse_grad(x, hist=hist, use_graph=use_graph), x0, method='L-BFGS-B', jac=True,
                   bounds=prob.bounds, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9, 'maxcor': 20})
    return _record(prob, label, res.x, time.perf_counter() - t0, n_eval=prob.n_grad - n0,
                   extra={'nit': int(res.nit), 'success': bool(res.success), 'message': str(res.message),
                          'graph': bool(use_graph and prob.graphed is not None), 'best_every_1': _best_so_far(hist, 1)})


def run_adj_multistart(prob, B, seed, iters=300, lr=0.05, polish=True):
    """B random starts (Latin hypercube in the box) advanced together by Adam in one batched
    adjoint program (bounds through a sigmoid map), then the best member polished by L-BFGS-B."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in prob.bounds]); hi = np.array([b[1] for b in prob.bounds])
    u = (rng.permuted(np.tile(np.arange(B), (prob.ntheta, 1)), axis=1).T + rng.uniform(size=(B, prob.ntheta))) / B
    u = np.clip(u, 1e-3, 1 - 1e-3)
    z = torch.as_tensor(np.log(u / (1 - u)), device=prob.dev).requires_grad_(True)
    lo_t, hi_t = torch.as_tensor(lo, device=prob.dev), torch.as_tensor(hi, device=prob.dev)
    opt = torch.optim.Adam([z], lr=lr)
    t0 = time.perf_counter(); n0 = prob.n_grad
    best = None
    for it in range(iters):
        opt.zero_grad()
        theta = lo_t + (hi_t - lo_t) * torch.sigmoid(z)
        sse = prob.sse_batch(theta)
        sse.sum().backward()
        opt.step()
        cur = sse.detach().cpu().numpy()
        if best is None:
            best = (cur.copy(), theta.detach().cpu().numpy().copy())
        else:
            better = cur < best[0]
            best[0][better] = cur[better]; best[1][better] = theta.detach().cpu().numpy()[better]
    prob.n_grad += iters
    t_adam = time.perf_counter() - t0
    ends = best[1]; end_sse = best[0]
    order = np.argsort(end_sse)
    if polish:
        rec = run_adj_lbfgsb(prob, ends[order[0]], label='ADJ-MS-Adam+LBFGSB')
        rec['n_eval'] = rec['n_eval'] + iters
        rec['wall_s'] += t_adam
    else:
        rec = _record(prob, 'ADJ-MS-Adam', ends[order[0]], t_adam, n_eval=iters, seed=seed)
    rec.update({'seed': seed, 'B': B, 'adam_iters': iters, 'wall_adam_s': t_adam,
                'ms_end_sse': end_sse[order].tolist(), 'ms_end_theta': ends[order].tolist()})
    return rec


# ----------------------------------------------------------------------------- checks & driver
def consistency(prob, seed=0, n=3, h=1e-4):
    """Torch loss vs DLL loss at random theta (same map, Ctol = 0) and adjoint gradient vs DLL
    central differences at one of them."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in prob.bounds]); hi = np.array([b[1] for b in prob.bounds])
    out = {'loss_rel_diff': [], 'grad_rel_err': None}
    for i in range(n):
        th = lo + (hi - lo) * rng.uniform(0.05, 0.95, size=prob.ntheta)
        a, g = prob.sse_grad(th)
        b = prob.sse_dll(th)
        out['loss_rel_diff'].append(abs(a - b) / max(abs(b), 1e-300))
        if i == 0:
            errs = []
            for j in range(prob.ntheta):
                p, m = th.copy(), th.copy(); p[j] += h; m[j] -= h
                fd = (prob.sse_dll(p) - prob.sse_dll(m)) / (2 * h)
                errs.append(abs(fd - g[j]) / max(abs(fd), abs(g[j]), 1e-300))
            out['grad_rel_err'] = errs
    return out


def timing(prob, reps=5):
    """Mean wall time of one DLL quality run (Ctol = 0 and 0.01), one eager adjoint gradient (B = 1) and,
    when captured, one replay of its CUDA graph; fd_grad_equiv_s = (p + 1) DLL runs at Ctol = 0, the cost
    of one forward-difference gradient as L-BFGS-B evaluates it."""
    th = prob.theta_nominal
    t = {}
    for label, ctol in (('dll_ctol0_s', None), ('dll_ctol0.01_s', 0.01)):
        t0 = time.perf_counter()
        for _ in range(reps):
            prob.simulate_dll(th, ctol)
        t[label] = (time.perf_counter() - t0) / reps
    t['fd_grad_equiv_s'] = (prob.ntheta + 1) * t['dll_ctol0_s']
    for label, use_graph in (('adjoint_grad_B1_s', False), ('adjoint_grad_B1_graph_s', True)):
        if use_graph and prob.graphed is None:
            continue
        prob.sse_grad(th, use_graph=use_graph)
        t0 = time.perf_counter()
        for _ in range(reps):
            prob.sse_grad(th, use_graph=use_graph)
        t[label] = (time.perf_counter() - t0) / reps
    return t


def run_dataset(name, lib, workdir, args):
    prob = CalibProblem(name, lib, workdir, device=args.device, fix_kb=args.fix_kb, graph='adjg' in args.methods,
                        engine=args.engine)
    try:
        return _run_dataset(prob, name, args)
    finally:
        prob.close()


def _run_dataset(prob, name, args):
    res = {'dataset': name, 'inp': str(prob.inp), 'param_names': prob.param_names, 'bounds': prob.bounds, 'fix_kb': args.fix_kb,
           'groups': dict(zip(prob.group_names, prob.group_count)), 'n_obs': len(prob.obs),
           'sites': prob.obs.site_node_id, 'n_dropped': prob.obs.n_dropped, 'qstep': prob.qstep, 'duration': prob.dur,
           'schedule': {'steps': prob.sch.nsteps, 'versions': prob.sch.nver, 'ops': len(prob.sch.ops),
                        'record_s': prob.t_record, 'build_s': prob.t_build},
           'theta_nominal': prob.theta_nominal.tolist(), 'device': args.device,
           'graph': {'requested': 'adjg' in args.methods, 'captured': prob.graphed is not None,
                     'capture_s': prob.t_graph, 'error': prob.graph_error}}
    res['consistency'] = consistency(prob)
    res['timing'] = timing(prob)
    print(f"  consistency: loss rel diff {max(res['consistency']['loss_rel_diff']):.1e}, grad vs DLL FD "
          f"{max(res['consistency']['grad_rel_err']):.1e}; timing {res['timing']}", flush=True)
    runs = [_record(prob, 'nominal(INP)', prob.theta_nominal, 0.0)]
    x0 = prob.theta_nominal
    methods = args.methods
    todo = []
    if 'de' in methods:
        todo += [('DE', lambda s=s: run_de(prob, s, popsize=args.de_popsize, maxiter=args.de_maxiter)) for s in range(args.seeds)]
    if 'nm' in methods:
        todo.append(('NM', lambda: run_nm(prob, x0)))
    # the deterministic gradient runs are repeated --repeat times in interleaved rounds (wall-clock statistics)
    for _ in range(max(1, args.repeat)):
        if 'fd' in methods:
            todo.append(('FD-LBFGSB', lambda: run_fd_lbfgsb(prob, x0, eps=1e-8)))
        if 'adj' in methods:
            todo.append(('ADJ-LBFGSB', lambda: run_adj_lbfgsb(prob, x0)))
        if 'adjg' in methods and prob.graphed is not None:
            todo.append(('ADJ-LBFGSB-graph', lambda: run_adj_lbfgsb(prob, x0, label='ADJ-LBFGSB-graph', use_graph=True)))
    if 'adjg' in methods and prob.graphed is None:
        print(f"  adjg skipped: {'graph capture failed: ' + prob.graph_error if prob.graph_error else 'no CUDA graph on device ' + args.device}",
              flush=True)
    if 'fd' in methods:
        todo.append(('FD-LBFGSB@ctol0.01', lambda: run_fd_lbfgsb(prob, x0, eps=1e-8, ctol=0.01, label='FD-LBFGSB@ctol0.01')))
        todo.append(('FD-LBFGSB@ctol0.01,eps1e-3', lambda: run_fd_lbfgsb(prob, x0, eps=1e-3, ctol=0.01, label='FD-LBFGSB@ctol0.01,eps1e-3')))
    if 'ms' in methods:
        todo += [('ADJ-MS', lambda s=s: run_adj_multistart(prob, args.ms_B, s, iters=args.ms_iters)) for s in range(args.ms_seeds)]
    seen = {}
    for label, fn in todo:
        rec = fn()
        rec['rep'] = seen[label] = seen.get(label, -1) + 1
        runs.append(rec)
        print(f"  {rec['method']:<28s} rep={rec['rep']} seed={rec.get('seed')} rmse={rec['rmse']:.4f} sse={rec['sse']:.5f} "
              f"n_eval={rec['n_eval']} wall={rec['wall_s']:.1f}s theta={np.round(rec['theta'], 4).tolist()}", flush=True)
    res['runs'] = runs
    res['counters'] = {'n_dll': prob.n_dll, 't_dll': prob.t_dll, 'n_grad': prob.n_grad, 't_grad': prob.t_grad}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='Fairfield,Bellingham,NorthPenn')
    ap.add_argument('--methods', default='all',
                    help='comma list of de,nm,fd,adj,adjg,ms or all (adjg = adj through the CUDA-graph capture; needs --device cuda)')
    ap.add_argument('--repeat', type=int, default=1, help='repetitions of the FD (Ctol 0) / ADJ / ADJ-graph runs for wall-clock statistics')
    ap.add_argument('--seeds', type=int, default=5, help='DE seeds')
    ap.add_argument('--de-popsize', type=int, default=15)
    ap.add_argument('--de-maxiter', type=int, default=300)
    ap.add_argument('--ms-seeds', type=int, default=2)
    ap.add_argument('--ms-B', type=int, default=32)
    ap.add_argument('--ms-iters', type=int, default=200)
    ap.add_argument('--fix-kb', action='store_true', help='hold Kb at the INP value; calibrate the Kw groups only')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--engine', default='adjoint', choices=['adjoint', 'fused'], help='torch replay engine (fused = one op per step)')
    ap.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = half the cores)')
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.methods = ['de', 'nm', 'fd', 'adj', 'adjg', 'ms'] if args.methods == 'all' else args.methods.split(',')
    lib = api.Library(args.lib)
    torch.set_num_threads(args.threads or int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or max(1, os.cpu_count() // 2))
    cpu_name = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu_name = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), cpu_name)
    results = {'platform': platform.platform(), 'cpu': cpu_name, 'threads': torch.get_num_threads(), 'torch': torch.__version__,
               'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else cpu_name,
               'epanet': lib.version(), 'args': vars(args), 'datasets': []}
    with tempfile.TemporaryDirectory(prefix='g4_') as workdir:
        # EPANET names its scratch hydraulics file with _tempnam() (Windows), which does not create
        # the file: concurrently started processes can collide. Give each process its own TMP.
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in args.datasets.split(','):
            results['datasets'].append(run_dataset(name, lib, workdir, args))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1))
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
