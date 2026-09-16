"""Exp2: contamination source identification (node, start, duration, mass) from sparse sensor series.

A conservative contaminant is injected at an unknown junction j* with an unknown box profile (start slot,
duration, mass rate); n_s online sensors report noisy concentrations at every hydraulic event time. With
Ctol = 0 the sensor series are an exact linear map of the slot strengths u[k, j] (mg/min at node j in
pattern slot k), c = A u, so identification is the sparse inverse problem

    min_{u >= 0}  1/2 || W (A u - d) ||^2  +  lam sum_j ||u_j||_2       (group lasso over nodes)

solved matrix-free on the recorded EPANET schedule. One adjoint row sweep per batch of scenarios (one-hot
cotangents, `chunk` rows of A per graphed backward pass; n_obs passes independent of the number of
candidates) accumulates the normal equations G_j = A_j^T A_j, b_j = A_j^T d of EVERY junction on the growing
observation window (rows are time-major, windows are row prefixes). At each window end:
  (A) parametric: the exact box GLRT over all (node, start slot, duration <= tau_max, rate) hypotheses from
      2-D prefix sums of G_j (the maximum-likelihood source under the box model class; deviance gap to the
      truth and the size of its statistical equivalence set are recorded);
  (B) nonparametric: the top-M GLRT nodes (+ the truth, flagged) are fitted by an l1-penalized NNLS
      (support) + debiased NNLS on the free-form profile and ranked by a BIC-type score; the first
      `exhaustive` scenarios are refined over every junction to validate the top-M restriction.
Sensors: greedy detection-likelihood placement on batched training scenarios. Data: EPANET-exact truth
(DLL check), additive Gaussian noise, confirmed alarm (`confirm` consecutive readings >= thr at one
sensor), results vs time since the first alarm. The injection started before the alarm but may continue
after it, so the admissible slots of a window are all slots starting before the window end.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp2_source --network LTown --device cuda --out exp2_LTown.json
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from . import epanet_api as api
from .hydfile import read_hydfile
from .network import extract
from .replicate import Replicator
from .torch_wq import TorchWQ, default_src
from .verify_m0 import ROOT, all_links, all_tanks_kb, init_qual, qual_type

CONFIGS = {
    # BattLeDIM L-Town: 5-min hydraulic / quality / pattern steps -> 5-min injection slots, 48 h horizon
    'LTown': {'inp': ROOT / 'networks/L-Town/L-TOWN.inp', 'duration_h': 48, 'slot_s': 300, 'n_sensors': 10,
              'unbalanced': None},
    # BWSN Network 2 (12,523 junctions): hourly patterns refined to 15-min slots; the hydraulics become
    # unbalanced once (t = 27 h) with the stock options, so Unbalanced = Continue 10 (EPANET's remedy)
    'BWSN2': {'inp': ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp',
              'duration_h': 48, 'slot_s': 900, 'n_sensors': 20, 'unbalanced': 10},
}
FT2M = 0.3048


def timed(fn, repeat=3):
    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


class SourceProblem:
    """Recorded network (conservative CHEM, no background) with virtual MASS sources at every junction."""

    def __init__(self, name, lib, workdir, args):
        cfg = CONFIGS[name]
        self.name, self.cfg, self.args, self.lib = name, cfg, args, lib
        self.dev, self.dt = torch.device(args.device), torch.float64
        self.sync = torch.cuda.synchronize if self.dev.type == 'cuda' else (lambda: None)
        self.workdir = Path(workdir)
        t0 = time.perf_counter()
        self.pr = pr = self.open()
        hyd = self.workdir / f'{name}.hyd'
        pr.solve_hydraulics(hyd)
        self.net = net = extract(pr)
        self.trace = read_hydfile(hyd)
        self.t_dll = timed(lambda: pr.run_quality_reference('nextQ'))
        self.slot, self.K = int(net.pstep), int(net.duration // net.pstep)
        N1 = net.nnodes + 1
        self.junc = [i for i in range(1, N1) if net.node_type[i] == api.EN_JUNCTION]
        rep = Replicator(net, self.trace, record=True, src_slots=self.K, virtual_sources={j: api.EN_MASS for j in self.junc})
        t1 = time.perf_counter(); self.times, _, _ = rep.run('nextQ'); self.t_record = time.perf_counter() - t1
        t1 = time.perf_counter()
        self.eng = TorchWQ(net, rep.schedule, rep.reactflag, device=self.dev, dtype=self.dt, engine='fused')
        self.eng.set_fixed_reaction(kb=torch.zeros(net.nlinks + 1), kw=torch.zeros(net.nlinks + 1), tank_kb=torch.zeros(net.ntanks + 1))
        self.t_build = time.perf_counter() - t1
        self.sch_info = {'versions': int(rep.schedule.nver), 'steps': int(rep.schedule.nsteps), 'fused_ops': len(self.eng.fsch.ops),
                         'fused_terms': int(self.eng.T_total), 'virtual_terms': int(rep.hits['virtual_source']),
                         'snapshots': len(self.times), 'hydraulic_periods': len(self.trace.periods)}
        self.n_out = len(self.times)
        self.times_t = torch.as_tensor(np.asarray(self.times, dtype=np.float64), device=self.dev)
        self.is_junc = torch.zeros(N1, dtype=self.dt, device=self.dev)
        self.is_junc[torch.as_tensor(self.junc, device=self.dev)] = 1.0
        assert not net.sources, 'Exp2 assumes no background sources'
        # pipe-length graph (m) for distance errors
        k = np.arange(1, net.nlinks + 1)
        w = np.where(net.link_type[k] <= api.EN_PIPE, net.length[k] * FT2M, 1e-3)
        self.graph = coo_matrix((np.r_[w, w], (np.r_[net.n1[k], net.n2[k]], np.r_[net.n2[k], net.n1[k]])), shape=(N1, N1)).tocsr()
        self._graphs, self.sens, self.sens_t = {}, None, None
        self.n_sweep = 0
        self.t_setup = time.perf_counter() - t0

    def open(self, tag=''):
        cfg = self.cfg
        pr = api.Project(self.lib, cfg['inp'], self.workdir / f'{self.name}{tag}.rpt')
        qual_type(api.EN_CHEM)(pr); init_qual(0.0, reservoirs=0.0)(pr)
        all_links(kb=0.0, kw=0.0)(pr); all_tanks_kb(0.0)(pr)
        pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
        pr.set_option(api.EN_TOLERANCE, 0.0)
        if cfg['unbalanced'] is not None:
            pr.set_option(api.EN_UNBALANCED, cfg['unbalanced'])
        if cfg['slot_s'] < pr.time_param(api.EN_PATTERNSTEP):
            pr.resample_patterns(cfg['slot_s'])
        return pr

    # ---- the linear map, sensors ----
    def c_full(self, src, eager=False):
        """[B, n_out, N+1]; CUDA-graph replay per batch size (forward and backward captured) unless eager."""
        B = src.shape[0]
        if self.dev.type == 'cuda' and not self.args.no_graph and not eager:
            if B not in self._graphs:
                with torch.enable_grad():
                    ex = torch.zeros(B, self.K, self.net.nnodes + 1, dtype=self.dt, device=self.dev, requires_grad=True)
                    t0 = time.perf_counter()
                    gph = torch.cuda.make_graphed_callables(lambda x: self.eng.forward(src=x), (ex,), num_warmup_iters=3)
                    self._graphs[B] = (gph, time.perf_counter() - t0)
            return self._graphs[B][0](src)
        return self.eng.forward(src=src)

    def free_graphs(self):
        t = sum(v[1] for v in self._graphs.values())
        self._graphs = {}
        gc.collect()
        if self.dev.type == 'cuda':
            torch.cuda.empty_cache()
        return t

    def c_sens(self, src, eager=False):
        """[B, n_out, n_s] at the sensor nodes."""
        return self.c_full(src, eager)[:, :, self.sens_t]

    # ---- scenarios ----
    def sample_scenarios(self, n, rng):
        """(node, start slot, duration in slots, mass rate mg/min): junction uniform, start in [0, start_max_h),
        duration uniform in [dur_h], rate log-uniform in [rate]."""
        a = self.args
        d_lo, d_hi = [max(1, int(float(x) * 3600 // self.slot)) for x in a.dur_h.split(',')]
        r_lo, r_hi = [float(x) for x in a.rate.split(',')]
        return [(int(j), int(k0), int(d), float(m)) for j, k0, d, m in zip(
            rng.choice(self.junc, n), rng.integers(0, int(a.start_max_h * 3600 // self.slot), n),
            rng.integers(d_lo, d_hi + 1, n), np.exp(rng.uniform(np.log(r_lo), np.log(r_hi), n)))]

    def src_of(self, scen):
        """[B, K, N+1] slot strengths of a list of scenarios."""
        u = torch.zeros(len(scen), self.K, self.net.nnodes + 1, dtype=self.dt, device=self.dev)
        for b, (j, k0, d, m) in enumerate(scen):
            u[b, k0:k0 + d, j] = m
        return u

    def truth_series(self, scen, batch):
        """Noise-free sensor series [n, n_out, n_s] of the scenarios (batched eager forwards)."""
        out = []
        with torch.no_grad():
            for s in range(0, len(scen), batch):
                out.append(self.c_sens(self.src_of(scen[s:s + batch]), eager=True).clone())
        return torch.cat(out, 0)

    def first_alarm(self, series):
        """series [B, n_out, C] -> first alarm time per column [B, C]: `confirm` consecutive readings >= thr
        (time of the confirming reading), inf if never."""
        hit = series >= self.args.thr
        n = self.args.confirm
        conf = hit[:, n - 1:]
        for i in range(1, n):
            conf = conf & hit[:, n - 1 - i:hit.shape[1] - i]
        t = self.times_t[n - 1:].view(1, -1, 1)
        return torch.where(conf, t, torch.full_like(t, float('inf'))).amin(1)

    def place_sensors(self, n_train, rng, batch):
        """Greedy detection-likelihood placement (BWSN objective Z4) on n_train training scenarios: at each
        step the junction that alarms (noise-free series) for the most still-undetected scenarios; ties by
        mean alarm time. Returns the sensor node indices and the coverage curve."""
        scen = self.sample_scenarios(n_train, rng)
        det, tdet = [], []
        with torch.no_grad():
            for s in range(0, n_train, batch):
                c = self.c_full(self.src_of(scen[s:s + batch]), eager=True)              # [B, n_out, N+1]
                first = self.first_alarm(c)                                              # [B, N+1]
                det.append(torch.isfinite(first))
                tdet.append(first)
        det = torch.cat(det, 0) & self.is_junc.bool().view(1, -1)                          # [n, N+1]
        tdet = torch.cat(tdet, 0)
        covered = torch.zeros(n_train, dtype=torch.bool, device=self.dev)
        chosen, curve = [], []
        for _ in range(self.cfg['n_sensors']):
            new = det & ~covered.view(-1, 1)
            gain = new.sum(0).to(self.dt)
            mean_t = torch.where(new, tdet, torch.zeros_like(tdet)).sum(0) / gain.clamp_min(1.0)
            score = gain - 1e-9 * mean_t                                                     # ties: earlier detection
            if chosen:
                score[torch.as_tensor(chosen, device=self.dev)] = -1.0
            j = int(score.argmax())
            chosen.append(j)
            covered |= det[:, j]
            curve.append(float(covered.float().mean()))
        detectable = float(det.any(1).float().mean())
        self.set_sensors(chosen)
        return {'n_train': n_train, 'sensors': [self.net.node_ids[j] for j in chosen], 'coverage_curve': curve,
                'detectable_fraction': detectable}

    def set_sensors(self, nodes):
        self.sens = list(nodes)
        self.sens_t = torch.as_tensor(self.sens, device=self.dev)

    def make_data(self, scen, rng, batch):
        """Sensor data with additive N(0, noise^2) errors and the first alarm time (any sensor) of every
        scenario on the noisy data; scenarios never alarmed are dropped, alarms at or before the injection
        start are false alarms (counted, dropped)."""
        c = self.truth_series(scen, batch)
        eps = torch.as_tensor(rng.normal(0.0, self.args.noise, size=tuple(c.shape)), dtype=self.dt, device=self.dev)
        d = c + eps
        first = self.first_alarm(d).amin(1)                                                  # [n]
        start = torch.as_tensor([k0 * self.slot for _, k0, _, _ in scen], dtype=self.dt, device=self.dev)
        false_alarm = torch.isfinite(first) & (first <= start)
        keep = torch.isfinite(first) & ~false_alarm
        idx = torch.nonzero(keep).flatten().tolist()
        return ([scen[i] for i in idx], c[keep], d[keep], first[keep],
                {'n_sampled': len(scen), 'n_detected': len(idx), 'n_false_alarm': int(false_alarm.sum()),
                 'n_undetected': int((~torch.isfinite(first)).sum()),
                 'detection_delay_h': ((first[keep] - start[keep]) / 3600).tolist(),
                 'first_detection_h': (first[keep] / 3600).tolist()})


    # ---- rows of the sensor response matrix ----
    def rows(self, r0, m):
        """Rows r0 .. r0+m (time-major: row = time index x n_s + sensor) of the sensor response matrix A as
        [m, K, N+1]: one graphed adjoint pass of `chunk` one-hot cotangents (zero rows pad the batch)."""
        K, N1, n_obs, ch = self.K, self.net.nnodes + 1, self.n_out * len(self.sens), self.args.chunk
        ar = torch.arange(m, device=self.dev)
        E = torch.zeros(ch, n_obs, dtype=self.dt, device=self.dev)
        E[ar, r0 + ar] = 1.0
        u0 = torch.zeros(ch, K, N1, dtype=self.dt, device=self.dev, requires_grad=True)
        (self.c_sens(u0).reshape(ch, -1) * E).sum().backward()
        self.n_sweep += 1
        return u0.grad[:m]

    # ---- stage A: exact box GLRT of every node from its normal equations ----
    def box_glrt(self, G, b, taus):
        """Profile likelihood of the box hypotheses (start slot k0, duration tau in `taus`, constant rate
        u <= u_max) of every candidate from its normal equations G [M, k, k], b [M, k]: S = box sum of b,
        N2 = box x box sum of G (2-D prefix sums), u = min(S_+ / N2, u_max), explained energy
        2 u S - u^2 N2 (= S_+^2 / N2 unconstrained: ||d_signal||^2 cos^2 of the box direction, maximal at
        the truth up to noise). Returns the score and the best k0, tau, u [M]."""
        M, k = b.shape
        u_max = self.args.u_max
        best = torch.zeros(M, dtype=self.dt, device=self.dev); u_best = torch.zeros_like(best)
        k_best = torch.zeros(M, dtype=torch.long, device=self.dev); tau_best = torch.zeros_like(k_best)
        Sb = torch.zeros(M, k + 1, dtype=self.dt, device=self.dev); Sb[:, 1:] = b.cumsum(1)
        node_chunk = max(1, 2 ** 28 // (8 * (k + 1) ** 2))
        diag = lambda X: torch.diagonal(X, dim1=1, dim2=2)
        for m0 in range(0, M, node_chunk):
            m1 = min(m0 + node_chunk, M)
            C = torch.zeros(m1 - m0, k + 1, k + 1, dtype=self.dt, device=self.dev)
            C[:, 1:, 1:] = G[m0:m1].cumsum(1).cumsum(2)
            for t in taus:
                if t > k:
                    continue
                n = k + 1 - t
                S = Sb[m0:m1, t:] - Sb[m0:m1, :n]                                                  # [m, n] box sums, k0 = 0..k-t
                N2 = diag(C[:, t:, t:]) - diag(C[:, :n, t:]) - diag(C[:, t:, :n]) + diag(C[:, :n, :n])
                u = torch.where(N2 > 1e-200, torch.relu(S) / N2.clamp_min(1e-200), torch.zeros_like(S)).clamp(max=u_max)
                sc = 2.0 * u * S - u * u * N2
                v, k0 = sc.max(1)
                upd = v > best[m0:m1]
                best[m0:m1] = torch.where(upd, v, best[m0:m1])
                u_best[m0:m1] = torch.where(upd, u.gather(1, k0.view(-1, 1)).view(-1), u_best[m0:m1])
                k_best[m0:m1] = torch.where(upd, k0, k_best[m0:m1])
                tau_best[m0:m1] = torch.where(upd, torch.full_like(k0, t), tau_best[m0:m1])
        return best, k_best, tau_best, u_best

    # ---- normal equations of every junction on growing windows: one adjoint row sweep ----
    def sweep_normal(self, nodes, d, bounds):
        """Normal equations of the nodes U on growing observation windows from one adjoint row sweep. Rows of
        the sensor response matrix A (time-major: row = time index x n_s + sensor) come from one-hot cotangent
        backward passes, `chunk` rows per graphed pass, so the cost is n_rows passes independent of |U|.
        Running accumulators G = A_U[:n]^T A_U[:n] [|U|, K, K] and b = A_U[:n]^T d_i[:n] [n_d, |U|, K] for the
        series d [n_d, n_out, n_s]; yields (n, G, b) at every row count n in `bounds` (ascending; the
        accumulators are views, consume them before resuming)."""
        U = torch.as_tensor(nodes, device=self.dev)
        K, n_obs, ch = self.K, self.n_out * len(self.sens), self.args.chunk
        bounds = sorted(set(int(n) for n in bounds))
        assert 0 < bounds[0] and bounds[-1] <= n_obs
        need = (2 * len(nodes) * K * K + len(nodes) * K * ch * 2) * 8
        assert need <= self.args.gpu_budget_gb * 2 ** 30, \
            f'normal equations of {len(nodes)} nodes need {need / 2 ** 30:.1f} GB > --gpu-budget-gb; reduce --chunk'
        d_flat = d.reshape(d.shape[0], -1)
        G = torch.zeros(len(nodes), K, K, dtype=self.dt, device=self.dev)
        b = torch.zeros(d.shape[0], len(nodes), K, dtype=self.dt, device=self.dev)

        def accumulate(Rt, r0, a, e):                                                             # rows r0+a .. r0+e of A_U
            G.baddbmm_(Rt[:, :, a:e], Rt[:, :, a:e].transpose(1, 2))
            b.add_((Rt[:, :, a:e] @ d_flat[:, r0 + a:r0 + e].T).permute(2, 0, 1))

        r0, bi = 0, 0
        while r0 < bounds[-1]:
            m = min(ch, bounds[-1] - r0)
            Rt = self.rows(r0, m)[:, :, U].permute(2, 1, 0)                                          # [|U|, K, m] rows r0..r0+m of A_U
            a = 0
            while bi < len(bounds) and bounds[bi] <= r0 + m:                                      # bounds inside the chunk: split the rows
                e = bounds[bi] - r0
                if e > a:
                    accumulate(Rt, r0, a, e)
                a = e
                yield bounds[bi], G, b
                bi += 1
            if a < m:
                accumulate(Rt, r0, a, m)
            r0 += m

    def refine(self, G, b, dd, n_rows, n_iter=3000, check=200, rtol=1e-9):
        """G [M, k, k], b [M, k], dd = ||d||^2: normal equations of the candidates' response columns of slots
        [0, k) on the window rows. Per candidate (i) support selection by the l1-penalized NNLS
        min_{u >= 0} 1/2 ||A_j u - d||^2 + sum_k lam_k u_k with lam_k = sigma sqrt(2 ln k) ||a_jk|| (universal
        threshold: pure noise gives u = 0), (ii) debiased NNLS on the selected support, (iii) RSS =
        ||d||^2 - 2 b.u + u^T G u and the BIC-type score RSS / sigma^2 + |S| ln n_rows (RSS when sigma = 0).
        FISTA batched over candidates, step 1 / ||G||_inf (>= lambda_max), stopped when the largest relative
        step over the batch falls below rtol (checked every `check` iterations). Returns dict of rss, score,
        support size [M], iterations and the debiased profiles [M, k] (mg/min per slot)."""
        M, k = b.shape
        sigma = self.args.noise
        G = G.contiguous()
        L = G.abs().sum(2).amax(1).clamp_min(1e-300).view(M, 1) * 1.000001
        lam = sigma * np.sqrt(2.0 * np.log(max(k, 2))) * torch.diagonal(G, dim1=1, dim2=2).clamp_min(0.0).sqrt()
        iters = []

        def fista(b_eff, mask):
            z = torch.zeros(M, k, dtype=self.dt, device=self.dev); y, t = z.clone(), torch.ones(M, dtype=self.dt, device=self.dev)
            for it in range(1, n_iter + 1):
                g = (G @ y.unsqueeze(2)).squeeze(2) - b_eff
                z_new = torch.relu(y - g / L) * mask
                restart = ((y - z_new) * (z_new - z)).sum(1) > 0
                t_new = torch.where(restart, torch.ones_like(t), 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t)))
                y = z_new + torch.where(restart, torch.zeros_like(t), (t - 1.0) / t_new).view(M, 1) * (z_new - z)
                if it % check == 0 and float(((z_new - z).abs().amax(1) / z_new.abs().amax(1).clamp_min(1e-300)).amax()) < rtol:
                    z, t = z_new, t_new
                    break
                z, t = z_new, t_new
            iters.append(it)
            return z

        z1 = fista(b - lam, torch.ones_like(b))
        supp = (z1 > 1e-3 * z1.amax(1, keepdim=True)).to(self.dt)
        z = fista(b, supp)
        rss = (dd - 2.0 * (b * z).sum(1) + ((G @ z.unsqueeze(2)).squeeze(2) * z).sum(1)).clamp_min(0.0)
        nsup = supp.sum(1)
        score = rss / sigma ** 2 + nsup * np.log(n_rows) if sigma > 0 else rss
        return {'rss': rss, 'score': score, 'supp': nsup, 'n_obs': float(n_rows), 'iters': iters}, z

    def profile_stats(self, u, k_hi):
        """Start / end (h) of the slots carrying >= 5% of the peak rate, total mass (kg)."""
        u = u.detach().cpu().numpy()
        act = np.flatnonzero(u >= 0.05 * u.max()) if u.max() > 0 else np.array([], dtype=int)
        mass_kg = float(u.sum() * self.slot / 60.0 * 1e-6)
        if len(act) == 0:
            return {'start_h': None, 'end_h': None, 'mass_kg': mass_kg}
        return {'start_h': float(act[0] * self.slot / 3600), 'end_h': float((act[-1] + 1) * self.slot / 3600), 'mass_kg': mass_kg}

    def verify_dll(self, scen, tag):
        """EPANET run with the scenario's source installed through the API vs the engine's sensor series."""
        j, k0, d, m = scen
        pat = np.zeros(self.K); pat[k0:k0 + d] = 1.0
        with self.open(tag) as pr:
            pidx = pr.add_pattern('inj', pat.tolist())
            pr.set_source(j, api.EN_MASS, m, pidx)
            pr.solve_hydraulics(self.workdir / f'{self.name}{tag}.hyd')
            _, q, _ = pr.run_quality_reference('nextQ')
        with torch.no_grad():
            c = self.c_full(self.src_of([scen]), eager=True)[0].cpu().numpy()
        return {'node': self.net.node_ids[j], 'max_rel_err_all_nodes': float(np.abs(q - c).max() / max(np.abs(q).max(), 1e-300)),
                'max_c_mgL': float(np.abs(q).max()), 'nodes_reached': int((np.abs(q).max(0) > 1e-9).sum())}



def run_network(name, lib, workdir, args):
    t_start = time.perf_counter()
    P = SourceProblem(name, lib, workdir, args)
    try:
        return _run(P, args, t_start)
    finally:
        P.free_graphs()
        P.pr.__exit__()


def _run(P, args, t_start):
    name, T, net, dev, dt = P.name, time.perf_counter, P.net, P.dev, P.dt
    print(f'[{name}] nodes={net.nnodes} junctions={len(P.junc)} links={net.nlinks} dur={net.duration // 3600}h qstep={net.qstep}s '
          f'slot={P.slot}s K={P.K} unknowns={P.K * len(P.junc)} snapshots={P.n_out} | record={P.t_record:.1f}s build={P.t_build:.1f}s '
          f'setup={P.t_setup:.1f}s dll_run={P.t_dll * 1e3:.0f}ms | {P.sch_info}', flush=True)
    rng = np.random.default_rng(args.seed)
    R = {'network': name, 'sizes': {'nodes': net.nnodes, 'junctions': len(P.junc), 'links': net.nlinks, 'slots': P.K,
                                    'slot_s': P.slot, 'unknowns': P.K * len(P.junc), 'snapshots': P.n_out,
                                    'duration_h': net.duration / 3600, **P.sch_info},
         'settings': {k: getattr(args, k) for k in ('noise', 'thr', 'confirm', 'start_max_h', 'dur_h', 'rate', 'windows', 'top_m', 'u_max',
                                                    'equiv_gap', 'exhaustive', 'chunk', 'seed', 'n_train', 'n_test')}}
    tm = {'record_s': P.t_record, 'build_s': P.t_build, 'setup_s': P.t_setup, 'dll_quality_run_s': P.t_dll}
    # ---- sensors ----
    t0 = T()
    if args.sensors:
        P.set_sensors([net.node_ids.index(s) for s in args.sensors.split(',')])
        R['placement'] = {'sensors': [net.node_ids[j] for j in P.sens], 'given': True}
    else:
        R['placement'] = P.place_sensors(args.n_train, rng, args.place_batch)
    tm['placement_s'] = T() - t0
    print(f'  sensors ({len(P.sens)}): {R["placement"]["sensors"]} | {tm["placement_s"]:.1f}s '
          + (f'coverage={R["placement"]["coverage_curve"][-1]:.3f} detectable={R["placement"]["detectable_fraction"]:.3f}'
             if 'coverage_curve' in R['placement'] else ''), flush=True)
    # ---- test scenarios and data ----
    t0 = T()
    scen, c_true, d, t_det, info = P.make_data(P.sample_scenarios(args.n_test, rng), rng, args.place_batch)
    tm['data_s'] = T() - t0
    R['data'] = info
    R['dll_check'] = [P.verify_dll(s, f'_chk{i}') for i, s in enumerate(scen[:3])]
    print(f'  scenarios: {info["n_detected"]}/{info["n_sampled"]} detected ({info["n_undetected"]} undetected, '
          f'{info["n_false_alarm"]} false alarms); median delay {np.median(info["detection_delay_h"]):.2f}h; DLL check max rel err '
          f'{max(c["max_rel_err_all_nodes"] for c in R["dll_check"]):.1e}', flush=True)
    wins = [float('inf') if w in ('inf', 'all') else float(w) for w in args.windows.split(',')]
    n_w, n, K, n_s = len(wins), len(scen), P.K, len(P.sens)
    tau_max = max(1, int(float(args.dur_h.split(',')[1]) * 3600 // P.slot))
    taus = list(range(1, tau_max + 1))
    R['settings']['tau_max_slots'] = tau_max
    junc_pos = {j: p for p, j in enumerate(P.junc)}
    dd_all = (d.reshape(n, -1) ** 2).cumsum(1)                                   # ||d_i[:r]||^2 of the time-major row prefixes
    sig2 = args.noise ** 2
    per_scen, t_sweep_total, t_screen_total, t_refine_total, t_exh_total, t0_all = [], 0.0, 0.0, 0.0, 0.0, T()

    def fit_window(G, b, i, n_rows, k_hi, cands, j_true):
        """Exact refinement of the candidates (rows of G, b = cands) on the first n_rows rows of scenario i."""
        P.sync(); t1 = T()
        fit, prof = P.refine(G[:, :k_hi, :k_hi], b[:, :k_hi], float(dd_all[i, n_rows - 1]), n_rows)
        sc_l, rss_l, sup_l = fit['score'].tolist(), fit['rss'].tolist(), fit['supp'].tolist()
        ib = int(np.argmin(sc_l)); it = cands.index(j_true)
        # equivalence: BIC-type score within `equiv_gap` of the best (deviance units; relative when sigma = 0)
        tol = args.equiv_gap if args.noise > 0 else 1e-6 * max(sc_l[ib], 1e-300)
        P.sync()
        return {'best_idx': ib, 'rank_true': 1 + sum(s < sc_l[it] for s in sc_l), 'n_equiv': sum(s <= sc_l[ib] + tol for s in sc_l),
                'identifiable_within_tol': sc_l[it] <= sc_l[ib] + tol, 'score_best': sc_l[ib], 'score_true': sc_l[it],
                'score_gap_true': sc_l[it] - sc_l[ib], 'score_tol': tol, 'res_best': rss_l[ib] ** 0.5, 'res_true': rss_l[it] ** 0.5,
                'supp_best': sup_l[ib], 'supp_true': sup_l[it], 'n_obs': int(fit['n_obs']), 'fista_iters': fit['iters'], 'profile': prof[ib],
                'wall_refine_s': T() - t1}

    def screen_window(G, b, i, k_hi, j_true):
        """Exact box GLRT of every junction on the window (G, b): the parametric (node, k0, tau, u) estimate,
        its deviance gap to the truth, the top-M candidates (+ the truth, flagged) of the NNLS refinement."""
        P.sync(); t1 = T()
        g_score, g_k, g_tau, g_u = P.box_glrt(G[:, :k_hi, :k_hi], b[:, :k_hi], taus)
        dev = g_score / sig2 if sig2 > 0 else g_score                             # explained deviance of the best box
        order = torch.argsort(dev, descending=True).tolist()
        cands = [P.junc[p] for p in order[:args.top_m] if dev[p] > 0]
        pt, p1 = junc_pos[j_true], order[0]
        in_screen = j_true in cands
        if not in_screen:
            cands.append(j_true)
        tol = args.equiv_gap if sig2 > 0 else 1e-6 * float(dev[p1])
        k0, dur = scen[i][1], scen[i][2]
        P.sync()
        diag = {'glrt_rank_true': 1 + int((dev > dev[pt]).sum()) if dev[pt] > 0 else None, 'glrt_top1': p1 == pt,
                'glrt_dev_true': float(dev[pt]), 'glrt_dev_top1': float(dev[p1]), 'glrt_gap_true': float(dev[p1] - dev[pt]),
                'glrt_n_equiv': int((dev >= dev[p1] - tol).sum()), 'glrt_identifiable_within_tol': bool(dev[pt] >= dev[p1] - tol),
                'glrt_top1_node': P.junc[p1], 'n_positive_glrt': int((dev > 0).sum()),
                'glrt_best_start_h': int(g_k[p1]) * P.slot / 3600, 'glrt_best_dur_h': int(g_tau[p1]) * P.slot / 3600,
                'glrt_best_rate_mg_min': float(g_u[p1]), 'glrt_best_mass_kg': float(g_u[p1]) * int(g_tau[p1]) * P.slot / 60 * 1e-6,
                'glrt_start_err_h': (int(g_k[p1]) - k0) * P.slot / 3600, 'glrt_dur_err_h': (int(g_tau[p1]) - dur) * P.slot / 3600,
                'glrt_mass_rel_err': float(g_u[p1]) * int(g_tau[p1]) / (scen[i][3] * dur) - 1.0,
                'glrt_start_err_h_true': (int(g_k[pt]) - k0) * P.slot / 3600, 'glrt_dur_err_h_true': (int(g_tau[pt]) - dur) * P.slot / 3600,
                'glrt_mass_rel_err_true': float(g_u[pt]) * int(g_tau[pt]) / (scen[i][3] * dur) - 1.0, 'wall_screen_s': T() - t1}
        return cands, in_screen, diag

    for s0 in range(0, n, args.batch):
        idx = list(range(s0, min(s0 + args.batch, n)))
        nb, B = len(idx), len(idx) * n_w
        rep = torch.as_tensor(idx, device=dev).repeat_interleave(n_w)
        t_end = t_det[rep] + torch.tensor(wins, dtype=dt, device=dev).repeat(nb) * 3600.0
        n_rows_of = [int((P.times_t <= t_end[b]).sum()) * n_s for b in range(B)]
        k_hi_of = [min(int(min(float(t_end[b]), net.duration) // P.slot) + 1, K) for b in range(B)]
        # ---- one adjoint row sweep per batch: normal equations of every junction, consumed at each window end ----
        res, t_sc, t_rf, t_ex = {}, 0.0, 0.0, 0.0
        P.sync(); t0 = T()
        for n_rows, G, bv in P.sweep_normal(P.junc, d[idx], n_rows_of):
            for b in [b for b in range(B) if n_rows_of[b] == n_rows]:
                bi, i, k_hi = b // n_w, idx[b // n_w], k_hi_of[b]
                j_true = scen[i][0]
                cands, in_screen, diag = screen_window(G, bv[bi], i, k_hi, j_true)
                sel = torch.as_tensor([junc_pos[j] for j in cands], device=dev)
                f = fit_window(G[sel], bv[bi, sel], i, n_rows, k_hi, cands, j_true)
                t_sc += diag['wall_screen_s']; t_rf += f['wall_refine_s']
                ex = None
                if i < args.exhaustive:                                           # every junction refined: validates top-M
                    fx = fit_window(G, bv[bi], i, n_rows, k_hi, P.junc, j_true)
                    t_ex += fx['wall_refine_s']
                    bx = P.junc[fx['best_idx']]
                    ex = {'best': net.node_ids[bx], 'top1': bx == j_true, 'rank_true': fx['rank_true'], 'n_equiv': fx['n_equiv'],
                          'score_best': fx['score_best'], 'score_true': fx['score_true'], 'n_refined': len(P.junc),
                          'same_best_as_top_m': bx == cands[f['best_idx']], 'wall_refine_s': fx['wall_refine_s']}
                res[b] = (cands, in_screen, diag, f, ex)
        P.sync(); t_sw = T() - t0 - t_sc - t_rf - t_ex
        t_sweep_total += t_sw; t_screen_total += t_sc; t_refine_total += t_rf; t_exh_total += t_ex
        for bi, i in enumerate(idx):
            j_true, k0, dur, rate = scen[i]
            dist = dijkstra(P.graph, indices=j_true)
            out = {'truth': {'node': net.node_ids[j_true], 'start_h': k0 * P.slot / 3600, 'dur_h': dur * P.slot / 3600,
                             'rate_mg_min': rate, 'mass_kg': rate * dur * P.slot / 60 * 1e-6, 't_det_h': float(t_det[i]) / 3600,
                             'peak_sensor_mgL': float(c_true[i].max())}, 'windows': []}
            for wi, w in enumerate(wins):
                b = bi * n_w + wi
                cands, in_screen, diag, f, ex = res[b]
                k_hi = k_hi_of[b]
                jg = diag.pop('glrt_top1_node')
                diag['glrt_top1_dist_m'] = float(dist[jg]) if np.isfinite(dist[jg]) else None
                best = cands[f['best_idx']]
                top1 = in_screen and best == j_true
                ps = P.profile_stats(f['profile'], k_hi)
                wr = {'window_h': w, 'n_obs_steps': n_rows_of[b] // n_s, 'n_obs': f['n_obs'], 'k_hi': k_hi,
                      'n_cands': len(cands) - (0 if in_screen else 1), 'truth_in_screen_topm': in_screen, **diag,
                      'best': net.node_ids[best], 'top1': top1, 'top5': in_screen and f['rank_true'] <= 5,
                      **{k: f[k] for k in ('rank_true', 'n_equiv', 'identifiable_within_tol', 'score_best', 'score_true', 'score_gap_true',
                                           'score_tol', 'res_best', 'res_true', 'supp_best', 'supp_true', 'fista_iters')},
                      'outcome': 'screen_miss' if not in_screen else ('correct' if top1 else ('equivalent' if f['identifiable_within_tol'] else 'misfit')),
                      'dist_m': float(dist[best]) if np.isfinite(dist[best]) else None,
                      'res_noise_expected': args.noise * f['n_obs'] ** 0.5, 'profile_best': ps,
                      'start_err_h': None if ps['start_h'] is None else ps['start_h'] - k0 * P.slot / 3600,
                      'dur_err_h': None if ps['start_h'] is None else (ps['end_h'] - ps['start_h']) - dur * P.slot / 3600,
                      'mass_rel_err': ps['mass_kg'] / out['truth']['mass_kg'] - 1.0,
                      'wall_sweep_s': t_sw / nb, 'wall_refine_s': f['wall_refine_s'],
                      'projected_exhaustive_dll_s': len(P.junc) * k_hi * P.t_dll}
                if wi == n_w - 1:
                    wr['u_best_mg_min'] = f['profile'].cpu().numpy().round(3).tolist()
                if ex is not None:
                    ex['dist_m'] = float(dist[net.node_ids.index(ex['best'])]) if np.isfinite(dist[net.node_ids.index(ex['best'])]) else None
                    wr['exhaustive'] = ex
                out['windows'].append(wr)
            per_scen.append(out)
        del res
        done = len(per_scen)
        acc = [np.mean([s['windows'][wi]['top1'] for s in per_scen]) for wi in range(n_w)]
        acg = [np.mean([s['windows'][wi]['glrt_top1'] for s in per_scen]) for wi in range(n_w)]
        scr = [np.mean([s['windows'][wi]['truth_in_screen_topm'] for s in per_scen]) for wi in range(n_w)]
        exs = [x['exhaustive'] for s in per_scen for x in s['windows'] if 'exhaustive' in x]
        print(f'  {done}/{n} scenarios | sweep {t_sw:.1f}s screen {t_sc:.1f}s refine {t_rf:.1f}s exhaustive {t_ex:.1f}s '
              f'| glrt top1 {np.round(acg, 3).tolist()} screened {np.round(scr, 3).tolist()} nnls top1 {np.round(acc, 3).tolist()} '
              f'| exhaustive agrees {sum(e["same_best_as_top_m"] for e in exs)}/{len(exs)} | elapsed {T() - t0_all:.0f}s', flush=True)
    tm['sweep_s'], tm['screen_s'], tm['refine_s'], tm['exhaustive_s'] = t_sweep_total, t_screen_total, t_refine_total, t_exh_total
    tm['identification_s'] = T() - t0_all
    tm['adjoint_row_passes'] = P.n_sweep
    tm['graph_capture_s'] = P.free_graphs()
    R['scenarios'] = per_scen
    R['summary'] = summarize(per_scen, wins)
    for wi, w in enumerate(wins):
        print(f'  window {w}h: {R["summary"][wi]}', flush=True)
    tm['total_s'] = T() - t_start
    R['timing'] = tm
    return R


def summarize(per_scen, wins):
    def med(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.median(xs)) if xs else None
    out = []
    for wi, w in enumerate(wins):
        ws = [s['windows'][wi] for s in per_scen]
        n = len(ws)
        scr = [x for x in ws if x['truth_in_screen_topm']]
        out.append({'window_h': w, 'n': n,
                    # parametric (box GLRT over every junction)
                    'glrt_top1': float(np.mean([x['glrt_top1'] for x in ws])),
                    'glrt_top5': float(np.mean([x['glrt_rank_true'] is not None and x['glrt_rank_true'] <= 5 for x in ws])),
                    'glrt_correct_or_equivalent': float(np.mean([x['glrt_top1'] or x['glrt_identifiable_within_tol'] for x in ws])),
                    'median_glrt_rank_true': med([x['glrt_rank_true'] for x in ws]), 'median_glrt_n_equiv': med([x['glrt_n_equiv'] for x in ws]),
                    'median_glrt_gap_true': med([x['glrt_gap_true'] for x in ws]),
                    'median_glrt_top1_dist_m': med([x['glrt_top1_dist_m'] for x in ws]),
                    'median_glrt_top1_dist_m_wrong': med([x['glrt_top1_dist_m'] for x in ws if not x['glrt_top1']]),
                    'median_abs_glrt_start_err_h': med([abs(x['glrt_start_err_h']) for x in ws]),
                    'median_abs_glrt_dur_err_h': med([abs(x['glrt_dur_err_h']) for x in ws]),
                    'median_abs_glrt_mass_rel_err': med([abs(x['glrt_mass_rel_err']) for x in ws]),
                    'median_abs_glrt_start_err_h_true': med([abs(x['glrt_start_err_h_true']) for x in ws]),
                    'median_abs_glrt_dur_err_h_true': med([abs(x['glrt_dur_err_h_true']) for x in ws]),
                    'median_abs_glrt_mass_rel_err_true': med([abs(x['glrt_mass_rel_err_true']) for x in ws]),
                    # nonparametric (NNLS + BIC over the top-M GLRT candidates)
                    'top1': float(np.mean([x['top1'] for x in ws])), 'top5': float(np.mean([x['top5'] for x in ws])),
                    'truth_in_screen_topm': float(np.mean([x['truth_in_screen_topm'] for x in ws])),
                    'top1_given_screened': float(np.mean([x['top1'] for x in scr])) if scr else None,
                    'correct_or_equivalent': float(np.mean([x['outcome'] in ('correct', 'equivalent') for x in ws])),
                    'median_score_gap_true': med([x['score_gap_true'] for x in ws]),
                    'median_res_true_over_noise': med([x['res_true'] / x['res_noise_expected'] for x in ws]),
                    'outcomes': {k: sum(x['outcome'] == k for x in ws) for k in ('correct', 'equivalent', 'screen_miss', 'misfit')},
                    'median_dist_m': med([x['dist_m'] for x in ws]), 'median_dist_m_wrong': med([x['dist_m'] for x in ws if not x['top1']]),
                    'median_n_equiv': med([x['n_equiv'] for x in ws]),
                    'median_abs_start_err_h': med([abs(x['start_err_h']) if x['start_err_h'] is not None else None for x in ws]),
                    'median_abs_dur_err_h': med([abs(x['dur_err_h']) if x['dur_err_h'] is not None else None for x in ws]),
                    'median_abs_mass_rel_err': med([abs(x['mass_rel_err']) for x in ws]),
                    'median_fista_iters': med([sum(x['fista_iters']) for x in ws]),
                    'mean_wall_sweep_s': float(np.mean([x['wall_sweep_s'] for x in ws])),
                    'mean_wall_screen_s': float(np.mean([x['wall_screen_s'] for x in ws])),
                    'mean_wall_refine_s': float(np.mean([x['wall_refine_s'] for x in ws])),
                    'mean_projected_exhaustive_dll_s': float(np.mean([x['projected_exhaustive_dll_s'] for x in ws]))})
        ex = [x['exhaustive'] for x in ws if 'exhaustive' in x]
        if ex:
            out[-1]['exhaustive'] = {'n': len(ex), 'top1': float(np.mean([e['top1'] for e in ex])),
                                     'same_best_as_top_m': float(np.mean([e['same_best_as_top_m'] for e in ex])),
                                     'median_rank_true': med([e['rank_true'] for e in ex]), 'median_n_equiv': med([e['n_equiv'] for e in ex]),
                                     'mean_wall_refine_s': float(np.mean([e['wall_refine_s'] for e in ex]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='LTown', help='comma-separated: ' + ','.join(CONFIGS))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sensors', default=None, help='comma-separated node IDs (default: greedy placement)')
    ap.add_argument('--n-train', type=int, default=512, help='training scenarios of the sensor placement')
    ap.add_argument('--n-test', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', type=float, default=0.1, help='sensor noise s.d. (mg/L)')
    ap.add_argument('--thr', type=float, default=0.5, help='alarm threshold (mg/L)')
    ap.add_argument('--confirm', type=int, default=2, help='consecutive readings >= thr that confirm an alarm')
    ap.add_argument('--start-max-h', type=float, default=24.0)
    ap.add_argument('--dur-h', default='1,4', help='injection duration range (h)')
    ap.add_argument('--rate', default='2e4,2e5', help='mass rate range (mg/min), log-uniform')
    ap.add_argument('--windows', default='2,6,inf', help='hours of data after the first detection')
    ap.add_argument('--top-m', type=int, default=30, help='GLRT candidates refined by NNLS')
    ap.add_argument('--u-max', type=float, default=1e7, help='rate bound (mg/min) of the GLRT amplitude')
    ap.add_argument('--equiv-gap', type=float, default=10.0,
                    help='candidates whose deviance / BIC-type score is within this gap of the best are statistically equivalent')
    ap.add_argument('--batch', type=int, default=8, help='scenarios per adjoint row sweep')
    ap.add_argument('--place-batch', type=int, default=32, help='scenarios per eager forward (placement, truth)')
    ap.add_argument('--chunk', type=int, default=64, help='one-hot cotangents (rows of A) per graphed adjoint pass')
    ap.add_argument('--exhaustive', type=int, default=20, help='scenarios also refined over every junction (validation of top-M)')
    ap.add_argument('--gpu-budget-gb', type=float, default=24.0, help='GPU budget of the normal-equation accumulators')
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lib', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    results = {'platform': platform.platform(), 'torch': torch.__version__, 'epanet': lib.version(), 'args': vars(args),
               'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else platform.processor(),
               'networks': []}
    with tempfile.TemporaryDirectory(prefix='exp2_', dir=os.environ.get('TMPDIR')) as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in args.network.split(','):
            results['networks'].append(run_network(name, lib, workdir, args))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1, default=float), encoding='utf-8')
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
