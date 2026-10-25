"""R1-5: complete source identification on BWSN2 beyond 48 h with the time-chunked replay (chunked.py) and two-level
normal equations. New module; exp2_source.py, exp2_v2.py and their results untouched.

Level 1 screens every junction with the exact box GLRT of exp2_source from per-junction box statistics accumulated in
the adjoint row sweep: for every duration tau and start k0, N2 = ||A_j 1_[k0, k0+tau)||^2 is the sum over rows of the
squared box sum of the row, and b = A_j^T d, so K x |taus| values per junction are held instead of the K x K normal
matrix (the same score of the same sums as box_glrt, which takes N2 from differences of 2-D prefix sums of G). Level 2
builds the full normal equations of the union of the batch's top-M candidates in a second sweep and refines them by the
l1-NNLS of exp2_source. Rows come from ChunkedWQ.rows_src (adjoint only; a batch starts at its last row).
Scenarios, sensors and noise reproduce the 48-h v2 study of --ref (exp2_v2.py): the test scenarios are the same draws,
the sensors are the reference set, and the noise stream of every sensor starts with its 48-h values, so windows that end
within 48 h see the data of the 48-h study.
--validate compares, on the recorded schedule, the chunked replay with the unchunked fused engine (one forward-plus-adjoint
source gradient and rows of the sensor response) before the identification.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp2_long --ref final_v2/exp2_v2_c2_200.json --hours 168 --max-events 16 --validate --out f.json
"""
import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

from . import epanet_api as api
from . import exp2_source as E
from . import exp2_v2 as E2
from .chunked import ChunkedWQ
from .torch_wq import TorchWQ
try:
    import resource
except ImportError:                                                   # not on Windows
    resource = None


def gpu_gb(fn='max_memory_allocated'):
    return getattr(torch.cuda, fn)() / 2 ** 30 if torch.cuda.is_available() else 0.0


class SourceProblemLong(E2.SourceProblemV2):
    def __init__(self, name, lib, workdir, args):
        made = E.TorchWQ
        E.TorchWQ = lambda *a, **k: ChunkedWQ(*a, n_chunks=args.n_chunks, chunk_terms=args.chunk_terms, **k)
        try:
            super().__init__(name, lib, workdir, args)
        finally:
            E.TorchWQ = made

    def rows(self, r0, m):
        """Rows r0 .. r0+m (time-major) of the sensor response matrix A as [m, K, N+1]: chunked adjoint, no forward.
        Short batches are padded with their last row (same reverse start) so the row buffer keeps its size."""
        n_s, ch = len(self.sens), self.args.chunk
        r = torch.arange(r0, r0 + m, device=self.dev)
        if m < ch:
            r = torch.cat([r, r[-1:].expand(ch - m)])
        self.n_sweep += 1
        return self.eng.rows_src(r // n_s, self.sens_t[r % n_s])[:m]

    def noise(self, n, n_out):
        """v2 noise (one stream per sensor node) whose first n_ref snapshots equal those of the 48-h study."""
        n_ref = self.args.n_out_ref
        cols = []
        for j in self.sens:
            g = np.random.default_rng([self.args.seed, 3, int(j)])
            a = g.normal(0.0, self.args.noise, size=(n, n_ref))
            if n_out > n_ref:
                a = np.concatenate((a, g.normal(0.0, self.args.noise, size=(n, n_out - n_ref))), 1)
            cols.append(a[:, :n_out])
        return torch.as_tensor(np.stack(cols, 2), dtype=self.dt, device=self.dev)

    # ---- level 1: box statistics of every junction from one adjoint row sweep ----
    def sweep_box(self, nodes, d, bounds, taus):
        """Running N2[t, j, k0] = sum_r (sum_{k in [k0, k0+tau_t)} A[r, k, j])^2 and b[i, j, k] = sum_r A[r, k, j] d_i[r]
        over the rows r < n, yielded at every row count n in `bounds` (views: consume before resuming)."""
        U = torch.as_tensor(nodes, device=self.dev)
        K, ch, dev, dt = self.K, self.args.chunk, self.dev, self.dt
        bounds = sorted(set(int(n) for n in bounds))
        d_flat = d.reshape(d.shape[0], -1)
        N2 = torch.zeros(len(taus), len(nodes), K, dtype=dt, device=dev)
        b = torch.zeros(d.shape[0], len(nodes), K, dtype=dt, device=dev)

        def accumulate(Rt, r0, a, e):                                          # rows r0+a .. r0+e of A_U, Rt [|U|, K, m]
            X = Rt[:, :, a:e]
            P = torch.zeros(len(nodes), K + 1, e - a, dtype=dt, device=dev)
            P[:, 1:] = X.cumsum(1)
            for ti, t in enumerate(taus):
                if t <= K:
                    s = P[:, t:] - P[:, :K + 1 - t]                              # box sums, k0 = 0 .. K - t
                    N2[ti, :, :K + 1 - t] += (s * s).sum(2)
            del P
            b.add_((X @ d_flat[:, r0 + a:r0 + e].T).permute(2, 0, 1))

        r0, bi = 0, 0
        while r0 < bounds[-1]:
            m = min(ch, bounds[-1] - r0)
            Rt = self.rows(r0, m)[:, :, U].permute(2, 1, 0)
            a = 0
            while bi < len(bounds) and bounds[bi] <= r0 + m:
                e = bounds[bi] - r0
                if e > a:
                    accumulate(Rt, r0, a, e)
                a = e
                yield bounds[bi], N2, b
                bi += 1
            if a < m:
                accumulate(Rt, r0, a, m)
            r0 += m

    def box_glrt_stats(self, N2all, b, taus, k):
        """exp2_source.box_glrt for the boxes inside the first k slots, from the box statistics of sweep_box."""
        M = b.shape[0]
        u_max, dev, dt = self.args.u_max, self.dev, self.dt
        best = torch.zeros(M, dtype=dt, device=dev); u_best = torch.zeros_like(best)
        k_best = torch.zeros(M, dtype=torch.long, device=dev); tau_best = torch.zeros_like(k_best)
        Sb = torch.zeros(M, k + 1, dtype=dt, device=dev); Sb[:, 1:] = b[:, :k].cumsum(1)
        for ti, t in enumerate(taus):
            if t > k:
                continue
            n = k + 1 - t
            S = Sb[:, t:] - Sb[:, :n]
            N2 = N2all[ti, :, :n]
            u = torch.where(N2 > 1e-200, torch.relu(S) / N2.clamp_min(1e-200), torch.zeros_like(S)).clamp(max=u_max)
            sc = 2.0 * u * S - u * u * N2
            v, k0 = sc.max(1)
            upd = v > best
            best = torch.where(upd, v, best)
            u_best = torch.where(upd, u.gather(1, k0.view(-1, 1)).view(-1), u_best)
            k_best = torch.where(upd, k0, k_best)
            tau_best = torch.where(upd, torch.full_like(k0, t), tau_best)
        return best, k_best, tau_best, u_best


# ---------------------------------------------------------------------------------------------- validation
def validate(P, args):
    """Chunked replay vs the unchunked fused engine on the same schedule: one forward-plus-adjoint source gradient and
    rows of the sensor response at early, middle and late times. Both engines held at once (the unchunked one only here)."""
    dev, K, N1 = P.dev, P.K, P.net.nnodes + 1
    g = np.random.default_rng(7)
    x = torch.as_tensor(g.uniform(0, 1e3, size=(1, K, N1)) * (g.uniform(size=(1, K, N1)) < 0.01), dtype=P.dt, device=dev)
    w = torch.as_tensor(g.uniform(0.5, 1.5, size=(1, P.n_out, N1)), dtype=P.dt, device=dev)
    n_s, n_obs = len(P.sens), P.n_out * len(P.sens)
    picks = [np.sort(g.choice(np.arange(lo, hi), 4, replace=False)) for lo, hi in
             ((0, n_obs // 10), (n_obs // 2 - n_obs // 20, n_obs // 2 + n_obs // 20), (n_obs - n_obs // 10, n_obs))]

    def measure(eng, rows_fn, other_GB):
        """Warm-up, then the best of three timed forward-plus-adjoint passes; memory of this engine alone (other_GB:
        resident memory of the other engine held at the same time)."""
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = gpu_gb('memory_allocated')
        best = float('inf')
        for _ in range(4):
            xx = x.clone().requires_grad_(True)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            y = eng.forward(src=xx); (y * w).sum().backward(); torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0) if _ else best
            if _ == 0:
                first = time.perf_counter() - t0
        out = {'fwd_adj_B1_s': best, 'fwd_adj_B1_first_call_s': first, 'gpu_resident_GB': base - other_GB,
               'gpu_peak_fwd_adj_B1_GB': gpu_gb() - other_GB, 'y': y.detach().cpu().numpy(), 'grad': xx.grad.cpu().numpy()}
        del xx, y
        out['rows'] = [rows_fn(eng, torch.as_tensor(p, device=dev)) for p in picks]
        return out

    def rows_chunked(eng, r):
        return eng.rows_src(r // n_s, P.sens_t[r % n_s]).cpu().numpy()

    def rows_fused(eng, r):
        u0 = torch.zeros(len(r), K, N1, dtype=P.dt, device=dev, requires_grad=True)
        y = eng.forward(src=u0)
        y[torch.arange(len(r), device=dev), r // n_s, P.sens_t[r % n_s]].sum().backward()
        return u0.grad.cpu().numpy()

    A = measure(P.eng, rows_chunked, 0.0)
    P.eng._rows_buf = None
    gc.collect(); torch.cuda.empty_cache()
    chunked_GB = gpu_gb('memory_allocated')            # resident memory of the chunked engine
    t0 = time.perf_counter()
    ref = TorchWQ(P.net, P.eng.sch, P.eng.reactflag, device=dev, dtype=P.dt, engine='fused')
    z = lambda n: torch.zeros(n, dtype=P.dt)
    ref.set_fixed_reaction(kb=z(P.net.nlinks + 1), kw=z(P.net.nlinks + 1), tank_kb=z(P.net.ntanks + 1))
    build_ref = time.perf_counter() - t0
    Bm = measure(ref, rows_fused, chunked_GB)
    del ref
    gc.collect(); torch.cuda.empty_cache()
    rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-300))
    return {'chunk_info': P.eng.chunk_info(), 'build_unchunked_s': build_ref,
            'forward_max_rel': rel(A['y'], Bm['y']), 'grad_src_max_rel': rel(A['grad'], Bm['grad']),
            'rows_max_rel': [rel(a, b) for a, b in zip(A['rows'], Bm['rows'])],
            'rows_picked': [p.tolist() for p in picks],
            'chunked': {k: v for k, v in A.items() if k not in ('y', 'grad', 'rows')},
            'unchunked': {k: v for k, v in Bm.items() if k not in ('y', 'grad', 'rows')}}


# ---------------------------------------------------------------------------------------------- identification
def run_two_level(P, args, t_start):
    """exp2_source._run with the two-level normal equations (no exhaustive validation of the top-M restriction)."""
    name, T, net, dev, dt = P.name, time.perf_counter, P.net, P.dev, P.dt
    rng = np.random.default_rng(args.seed)
    R = {'network': name, 'sizes': {'nodes': net.nnodes, 'junctions': len(P.junc), 'links': net.nlinks, 'slots': P.K,
                                    'slot_s': P.slot, 'unknowns': P.K * len(P.junc), 'snapshots': P.n_out,
                                    'duration_h': net.duration / 3600, **P.sch_info},
         'chunks': P.eng.chunk_info(),
         'settings': {k: getattr(args, k) for k in ('noise', 'thr', 'confirm', 'start_max_h', 'dur_h', 'rate', 'windows', 'top_m',
                                                    'u_max', 'equiv_gap', 'chunk', 'seed', 'n_train', 'n_test', 'max_events')}}
    tm = {'record_s': P.t_record, 'build_s': P.t_build, 'setup_s': P.t_setup, 'dll_quality_run_s': P.t_dll}
    t0 = T(); R['placement'] = P.place_sensors(args.n_train, rng, args.place_batch); tm['placement_s'] = T() - t0
    t0 = T()
    scen, c_true, d, t_det, info = P.make_data(P.sample_scenarios(args.n_test, rng), rng, args.place_batch)
    tm['data_s'] = T() - t0
    R['data'] = info
    if args.max_events:
        scen, c_true, d, t_det = scen[:args.max_events], c_true[:args.max_events], d[:args.max_events], t_det[:args.max_events]
    R['dll_check'] = [P.verify_dll(s, f'_chk{i}') for i, s in enumerate(scen[:3])]
    print(f'  scenarios: {info["n_detected"]}/{info["n_sampled"]} detected, identifying {len(scen)}; DLL check max rel err '
          f'{max(c["max_rel_err_all_nodes"] for c in R["dll_check"]):.1e}', flush=True)
    wins = [float('inf') if w in ('inf', 'all') else float(w) for w in args.windows.split(',')]
    n_w, n, K, n_s = len(wins), len(scen), P.K, len(P.sens)
    tau_max = max(1, int(float(args.dur_h.split(',')[1]) * 3600 // P.slot))
    taus = list(range(1, tau_max + 1))
    R['settings']['tau_max_slots'] = tau_max
    junc_pos = {j: p for p, j in enumerate(P.junc)}
    dd_all = (d.reshape(n, -1) ** 2).cumsum(1)
    sig2 = args.noise ** 2
    per_scen, t_l1_total, t_l2_total, t_screen_total, t_refine_total, t0_all = [], 0.0, 0.0, 0.0, 0.0, T()
    peaks = []

    def fit_window(G, b, i, n_rows, k_hi, cands, j_true):
        P.sync(); t1 = T()
        fit, prof = P.refine(G[:, :k_hi, :k_hi], b[:, :k_hi], float(dd_all[i, n_rows - 1]), n_rows)
        sc_l, rss_l, sup_l = fit['score'].tolist(), fit['rss'].tolist(), fit['supp'].tolist()
        ib = int(np.argmin(sc_l)); it = cands.index(j_true)
        tol = args.equiv_gap if args.noise > 0 else 1e-6 * max(sc_l[ib], 1e-300)
        P.sync()
        return {'best_idx': ib, 'rank_true': 1 + sum(s < sc_l[it] for s in sc_l), 'n_equiv': sum(s <= sc_l[ib] + tol for s in sc_l),
                'identifiable_within_tol': sc_l[it] <= sc_l[ib] + tol, 'score_best': sc_l[ib], 'score_true': sc_l[it],
                'score_gap_true': sc_l[it] - sc_l[ib], 'score_tol': tol, 'res_best': rss_l[ib] ** 0.5, 'res_true': rss_l[it] ** 0.5,
                'supp_best': sup_l[ib], 'supp_true': sup_l[it], 'n_obs': int(fit['n_obs']), 'fista_iters': fit['iters'], 'profile': prof[ib],
                'wall_refine_s': T() - t1}

    def screen_window(N2, b, i, k_hi, j_true):
        P.sync(); t1 = T()
        g_score, g_k, g_tau, g_u = P.box_glrt_stats(N2, b, taus, k_hi)
        dev_ = g_score / sig2 if sig2 > 0 else g_score
        order = torch.argsort(dev_, descending=True).tolist()
        cands = [P.junc[p] for p in order[:args.top_m] if dev_[p] > 0]
        pt, p1 = junc_pos[j_true], order[0]
        in_screen = j_true in cands
        if not in_screen:
            cands.append(j_true)
        tol = args.equiv_gap if sig2 > 0 else 1e-6 * float(dev_[p1])
        k0, dur = scen[i][1], scen[i][2]
        P.sync()
        diag = {'glrt_rank_true': 1 + int((dev_ > dev_[pt]).sum()) if dev_[pt] > 0 else None, 'glrt_top1': p1 == pt,
                'glrt_dev_true': float(dev_[pt]), 'glrt_dev_top1': float(dev_[p1]), 'glrt_gap_true': float(dev_[p1] - dev_[pt]),
                'glrt_n_equiv': int((dev_ >= dev_[p1] - tol).sum()), 'glrt_identifiable_within_tol': bool(dev_[pt] >= dev_[p1] - tol),
                'glrt_top1_node': P.junc[p1], 'n_positive_glrt': int((dev_ > 0).sum()),
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
        torch.cuda.reset_peak_memory_stats()
        # ---- level 1: box GLRT of every junction ----
        lev, t_sc = {}, 0.0
        P.sync(); t0 = T(); n0 = P.n_sweep
        for n_rows, N2, bv in P.sweep_box(P.junc, d[idx], n_rows_of, taus):
            for b in [b for b in range(B) if n_rows_of[b] == n_rows]:
                bi, i, k_hi = b // n_w, idx[b // n_w], k_hi_of[b]
                lev[b] = screen_window(N2, bv[bi], i, k_hi, scen[i][0])
                t_sc += lev[b][2]['wall_screen_s']
        P.sync(); t_l1 = T() - t0 - t_sc; p1 = P.n_sweep - n0; peak1 = gpu_gb()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        # ---- level 2: normal equations of the batch's candidates, NNLS refinement ----
        union = sorted({j for b in lev for j in lev[b][0]})
        upos = {j: p for p, j in enumerate(union)}
        res, t_rf = {}, 0.0
        P.sync(); t0 = T(); n0 = P.n_sweep
        for n_rows, G, bv in P.sweep_normal(union, d[idx], n_rows_of):
            for b in [b for b in range(B) if n_rows_of[b] == n_rows]:
                bi, i, k_hi = b // n_w, idx[b // n_w], k_hi_of[b]
                cands, in_screen, diag = lev[b]
                sel = torch.as_tensor([upos[j] for j in cands], device=dev)
                f = fit_window(G[sel], bv[bi, sel], i, n_rows, k_hi, cands, scen[i][0])
                t_rf += f['wall_refine_s']
                res[b] = (cands, in_screen, diag, f)
        P.sync(); t_l2 = T() - t0 - t_rf; p2 = P.n_sweep - n0; peak2 = gpu_gb()
        gc.collect(); torch.cuda.empty_cache()
        peaks.append({'batch': s0 // args.batch, 'events': nb, 'level1_passes': p1, 'level2_passes': p2, 'union_candidates': len(union),
                      'level1_sweep_s': t_l1, 'level2_sweep_s': t_l2, 'screen_s': t_sc, 'refine_s': t_rf,
                      'gpu_peak_level1_GB': peak1, 'gpu_peak_level2_GB': peak2})
        t_l1_total += t_l1; t_l2_total += t_l2; t_screen_total += t_sc; t_refine_total += t_rf
        for bi, i in enumerate(idx):
            j_true, k0, dur, rate = scen[i]
            dist = dijkstra(P.graph, indices=j_true)
            out = {'truth': {'node': net.node_ids[j_true], 'start_h': k0 * P.slot / 3600, 'dur_h': dur * P.slot / 3600,
                             'rate_mg_min': rate, 'mass_kg': rate * dur * P.slot / 60 * 1e-6, 't_det_h': float(t_det[i]) / 3600,
                             'peak_sensor_mgL': float(c_true[i].max())}, 'windows': []}
            for wi, w in enumerate(wins):
                b = bi * n_w + wi
                cands, in_screen, diag, f = res[b]
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
                      'wall_sweep_s': (t_l1 + t_l2) / nb, 'wall_sweep_level1_s': t_l1 / nb, 'wall_sweep_level2_s': t_l2 / nb,
                      'wall_refine_s': f['wall_refine_s'], 'projected_exhaustive_dll_s': len(P.junc) * k_hi * P.t_dll}
                out['windows'].append(wr)
            per_scen.append(out)
        acg = [np.mean([s['windows'][wi]['glrt_top1'] for s in per_scen]) for wi in range(n_w)]
        acc = [np.mean([s['windows'][wi]['top1'] for s in per_scen]) for wi in range(n_w)]
        print(f'  {len(per_scen)}/{n} events | level 1 {t_l1:.0f}s ({p1} passes, peak {peak1:.1f} GB) level 2 {t_l2:.0f}s ({p2} passes, '
              f'{len(union)} candidates, peak {peak2:.1f} GB) | glrt top1 {np.round(acg, 3).tolist()} nnls top1 {np.round(acc, 3).tolist()} '
              f'| elapsed {T() - t0_all:.0f}s', flush=True)
    tm.update(level1_sweep_s=t_l1_total, level2_sweep_s=t_l2_total, sweep_s=t_l1_total + t_l2_total, screen_s=t_screen_total,
              refine_s=t_refine_total, identification_s=T() - t0_all, adjoint_row_passes=P.n_sweep, total_s=T() - t_start)
    R['batches'] = peaks
    R['scenarios'] = per_scen
    R['summary'] = E.summarize(per_scen, wins)
    R['timing'] = tm
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='48-h v2 result JSON: settings, sensors, per-scenario results')
    ap.add_argument('--hours', type=int, required=True)
    ap.add_argument('--max-events', type=int, default=16, help='detected test events identified (0: all)')
    ap.add_argument('--chunk', type=int, default=32, help='rows of the sensor response per adjoint pass')
    ap.add_argument('--n-chunks', type=int, default=None)
    ap.add_argument('--chunk-terms', type=int, default=int(1.5e7), help='terms per replay chunk (if --n-chunks is not given)')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    J = json.load(open(a.ref))
    args = argparse.Namespace(**J['args'])
    args.device, args.no_graph, args.exhaustive = 'cuda', True, 0
    args.chunk, args.n_chunks, args.chunk_terms, args.max_events = a.chunk, a.n_chunks, a.chunk_terms, a.max_events
    args.n_out_ref = int(J['sizes']['snapshots'])
    E.CONFIGS[J['network']]['duration_h'] = a.hours
    lib = api.Library(a.lib)
    res = {'ref': a.ref, 'hours': a.hours, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'epanet': lib.version(),
           'args': {k: v for k, v in vars(args).items()}}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='exp2long_', dir=os.environ.get('TMPDIR')) as wd:
        P = SourceProblemLong(J['network'], lib, wd, args)
        res['host_peak_rss_GB_after_setup'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20 if resource else None
        print(f"[{P.name} {a.hours} h] record {P.t_record:.0f}s build {P.t_build:.0f}s | {P.sch_info} | {P.eng.chunk_info()} | "
              f"GPU resident {gpu_gb('memory_allocated'):.1f} GB", flush=True)
        P.v2_sensors = [P.net.node_ids.index(s) for s in J['placement']['sensors']]
        P.set_sensors(P.v2_sensors)
        if J['v2']['config']['place'] == 'v2' and getattr(args, 'scen_rule', 'full') == 'full':
            # exp2_v2 drew its n_train_v2 placement scenarios (with redraws of stagnant sources) before the test scenarios:
            # advance the redraw stream in the same way, so the 200 test scenarios are those of the reference
            P.sample_scenarios(int(J['v2']['n_train_v2']), np.random.default_rng([args.seed, 1]))
            P.n_resampled = 0
        if a.validate:
            res['validation'] = validate(P, args)
            v = res['validation']
            print(f"  validation: forward {v['forward_max_rel']:.1e} grad {v['grad_src_max_rel']:.1e} rows {['%.1e' % x for x in v['rows_max_rel']]} | "
                  f"B=1 chunked {v['chunked']['fwd_adj_B1_s']:.2f}s {v['chunked']['gpu_peak_fwd_adj_B1_GB']:.1f} GB, unchunked "
                  f"{v['unchunked']['fwd_adj_B1_s']:.2f}s {v['unchunked']['gpu_peak_fwd_adj_B1_GB']:.1f} GB", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
        res['result'] = run_two_level(P, args, t0)
        P.pr.__exit__()
    res['total_s'] = time.perf_counter() - t0
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    s = res['result']['summary']
    print('wrote', a.out, '| glrt top1', [round(w['glrt_top1'], 3) for w in s], '| nnls top1', [round(w['top1'], 3) for w in s])


if __name__ == '__main__':
    main()
