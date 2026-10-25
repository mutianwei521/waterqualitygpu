"""R2-11, part 3 (optional certificate): mixed-integer optimum of the booster siting problem on Net3 for a station budget m,
with a linear (L1) version of the exp1_booster objective, solved by HiGHS (scipy.optimize.milp).

Variables: rates u >= 0 (candidates x slots), station indicators z in {0, 1}, shortfalls s >= 0 below the design threshold
lo and excesses e >= 0 above hi (only for node-hours that could exceed hi at the rate bound). Constraints: c0 + A u + s >= lo,
c0 + A u - e <= hi, u_jk <= U z_j, sum z <= m, with the explicit response matrix A from the replay. Objective:
sum w (s + e) + mu1 * mass, mu1 = mu_rel * V0_L1 / M_ref (the L1 analogue of the exp1 mass weight, V0_L1 = plant-only value).
HiGHS gets the program scaled (x = u / U in [0, 1], costs divided by V0_L1): in mg/min and mg/L the mass cost of one rate
unit is ~1e-9, below the solver's optimality tolerances, and the unscaled program returned a wrong 'optimal' bound. Every
solution is re-evaluated with the dense A in numpy. The rate bound U is a multiple of the largest rate of the nominal designs;
rates at the bound are counted. The supports of the greedy selection and of the GA (if given) are evaluated with the same
program with z fixed (an LP): their gap to the MILP, and, as a check of the certificate, that none is below its dual bound.
The supports are also refitted with the original squared program.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp1_milp --designs RESULTS/exp1_Net3.json --m 3 --ga 'ga_Net3_m3_s*.json' --out milp_Net3_m3.json
"""
import argparse
import glob
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

from . import epanet_api as api
from .exp1_booster import BoosterProblem


class L1Program:
    """Scaled MILP / LP data of the L1 booster program and its exact (dense, unscaled) objective."""

    def __init__(self, A, c0, w, lo, hi, U, mu_rel, M_ref, mass_coef, m, K):
        n, JK = A.shape
        J = JK // K
        self.A, self.c0, self.w, self.lo, self.hi, self.mass_coef = A, c0, w, lo, hi, mass_coef
        self.n, self.JK, self.J, self.K = n, JK, J, K
        self.Ur = np.repeat(U, K)
        self.V0 = float((w * (np.clip(lo - c0, 0, None) + np.clip(c0 - hi, 0, None))).sum())
        self.mu1 = mu_rel * self.V0 / M_ref
        As = A * self.Ur                                                     # response per unit scaled rate x
        As = sparse.csr_matrix(np.where(np.abs(As) > 1e-12 * np.abs(As).max(), As, 0.0))
        self.ex = ex = np.flatnonzero(c0 + np.clip(A, 0.0, None) @ self.Ur > hi)   # node-hours that could exceed hi at all
        self.ne = ne = len(ex)
        I_n, I_e = sparse.identity(n, format='csr'), sparse.identity(ne, format='csr')
        Z = lambda r, c: sparse.csr_matrix((r, c))
        rows = [sparse.hstack([-As, Z(n, J), -I_n, Z(n, ne)])]                          # -A u - s <= c0 - lo
        lb, ub = [np.full(n, -np.inf)], [c0 - lo]
        if ne:
            rows.append(sparse.hstack([As[ex], Z(ne, J), Z(ne, n), -I_e]))                # A u - e <= hi - c0
            lb.append(np.full(ne, -np.inf)); ub.append(hi - c0[ex])
        rows.append(sparse.hstack([sparse.identity(JK, format='csr'), -sparse.kron(sparse.identity(J), np.ones((K, 1))),
                                   Z(JK, n), Z(JK, ne)]))                                  # x_jk - z_j <= 0
        lb.append(np.full(JK, -np.inf)); ub.append(np.zeros(JK))
        rows.append(sparse.hstack([Z(1, JK), sparse.csr_matrix(np.ones((1, J))), Z(1, n), Z(1, ne)]))   # sum z <= m
        lb.append([-np.inf]); ub.append([m])
        self.M = sparse.vstack(rows).tocsr()
        self.lb, self.ub = np.concatenate(lb), np.concatenate(ub)
        self.cost = np.concatenate([self.mu1 * mass_coef * self.Ur, np.zeros(J), w, w[ex]]) / self.V0

    def exact(self, u):
        """L1 objective of the rates u [J*K] (mg/min) with the dense A."""
        c = self.c0 + self.A @ u
        return float((self.w * (np.clip(self.lo - c, 0, None) + np.clip(c - self.hi, 0, None))).sum()
                     + self.mu1 * self.mass_coef * u.sum())

    def solve(self, integer, z_fix=None, time_limit=1800, gap=1e-4):
        JK, J, nv = self.JK, self.J, self.M.shape[1]
        lo_b, hi_b = np.zeros(nv), np.concatenate([np.ones(JK + J), np.full(nv - JK - J, np.inf)])
        if z_fix is not None:
            lo_b[JK:JK + J] = z_fix; hi_b[JK:JK + J] = z_fix
        integ = np.zeros(nv)
        if integer:
            integ[JK:JK + J] = 1
        t0 = time.perf_counter()
        res = milp(self.cost, constraints=LinearConstraint(self.M, self.lb, self.ub), integrality=integ,
                   bounds=Bounds(lo_b, hi_b), options={'time_limit': time_limit, 'mip_rel_gap': gap, 'disp': False})
        out = {'status': int(res.status), 'message': str(res.message), 'wall_s': time.perf_counter() - t0,
               'obj': float(res.fun) * self.V0 if res.x is not None else None}
        if integer:
            db, mg, nn = (getattr(res, k, None) for k in ('mip_dual_bound', 'mip_gap', 'mip_node_count'))
            out.update(dual_bound=float(db) * self.V0 if db is not None else None, mip_gap=float(mg) if mg is not None else None,
                       nodes=int(nn) if nn is not None else None)
        if res.x is not None:
            x, zr = np.clip(res.x[:JK], 0.0, 1.0), res.x[JK:JK + J]
            z = np.round(zr).astype(int)
            out['u'], out['z'] = x * self.Ur, z
            out['obj_exact'] = self.exact(out['u'])
            out['rates_at_bound'] = int((x > 1 - 1e-6).sum())
            out['max_link_violation'] = float(np.max(x.reshape(J, self.K) - zr[:, None]))
            out['z_fractionality'] = float(np.abs(zr - z).max())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='Net3')
    ap.add_argument('--designs', required=True)
    ap.add_argument('--m', type=int, default=3)
    ap.add_argument('--ga', default='', help='glob of exp1_ga result JSONs with the same m')
    ap.add_argument('--u-mult', type=float, default=10.0, help='rate bound U = u_mult x the largest rate of the nominal designs')
    ap.add_argument('--time-limit', type=float, default=1800.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lib', default=None)
    a = ap.parse_args()
    for k, v in dict(lo=0.2, margin=0.02, hi=4.0, mu_rel=0.1, iters=5000, tol=1e-7, chunk=256, no_graph=False, no_tanks=False,
                     theta_json=None).items():
        setattr(a, k, v)
    nom = json.load(open(a.designs, encoding='utf-8'))['networks'][0]
    lib = api.Library(a.lib)
    wd = Path(tempfile.mkdtemp(prefix='diffwq_milp_', dir=os.environ.get('TMPDIR')))
    P = BoosterProblem(a.network, lib, wd, a)
    try:
        J, K = P.J, P.K
        t0 = time.perf_counter(); A_t, c0_t = P.response_matrix(a.chunk); t_A = time.perf_counter() - t0
        A, c0, w = A_t.cpu().numpy(), c0_t.cpu().numpy(), P.w.cpu().numpy()
        umax = max(float(np.max(np.abs(np.array(d['u'])))) for d in [g['refit'] for g in nom['greedy']] + nom['refit'])
        U = np.full(J, a.u_mult * umax)
        prog = L1Program(A, c0, w, P.lo, P.hi, U, a.mu_rel, P.M_ref, P.mass_coef, a.m, K)
        print(f'[{a.network}] MILP: {J} binaries, {prog.M.shape[1] - J} continuous, {prog.M.shape[0]} rows, {prog.M.nnz} nonzeros; '
              f'A built in {t_A:.1f}s; U = {U[0]:.4g} mg/min ({a.u_mult:g} x {umax:.4g}); V0_L1 = {prog.V0:.4e}; mu1 = {prog.mu1:.4e}',
              flush=True)
        lp = prog.solve(False, time_limit=a.time_limit)
        mip = prog.solve(True, time_limit=a.time_limit)
        print(f"  LP relaxation {lp['obj']:.6e}; MILP {mip['obj']:.6e} (exact {mip.get('obj_exact', float('nan')):.6e}) dual bound "
              f"{mip['dual_bound']} gap {mip['mip_gap']} nodes {mip['nodes']} ({mip['message']}, {mip['wall_s']:.0f}s); "
              f"rates at bound {mip.get('rates_at_bound')}", flush=True)
        ids = P.net.node_ids
        sup = lambda z: [ids[P.cand[j]] for j in np.flatnonzero(z)]
        strip = lambda d: {k: v for k, v in d.items() if k not in ('u', 'z')}
        R = {'network': a.network, 'm': a.m, 'n_obs': prog.n, 'candidates': J, 'slots': K, 'excess_rows': prog.ne,
             'u_mult': a.u_mult, 'umax_nominal_mg_min': umax, 'U_mg_min': float(U[0]), 'mu1': prog.mu1, 'V0_L1': prog.V0,
             'A_build_s': t_A, 'lp_relaxation': strip(lp), 'milp': strip(mip), 'supports': {}}
        if mip.get('u') is not None:
            R['milp']['stations'] = sup(mip['z'])
            R['milp']['u'] = np.round(mip['u'].reshape(J, K), 6).tolist()
        cand = {'MILP': mip.get('z')}
        g = next(x for x in nom['greedy'] if x['stage'] == a.m)
        cand[f'greedy {a.m}'] = (np.abs(np.array(g['refit']['u'])).sum(1) > 0).astype(int)
        pos = {ids[P.cand[j]]: j for j in range(J)}
        for f in sorted(glob.glob(a.ga)) if a.ga else []:
            r = json.load(open(f, encoding='utf-8'))
            zz = np.zeros(J, int); zz[[pos[s] for s in r['best_set']]] = 1
            cand[f"GA seed {r['seed']}"] = zz
        dt, dev = P.dt, P.dev
        mu = torch.full((1,), P.mu, dtype=dt, device=dev)
        for k, z in cand.items():
            if z is None:
                continue
            fx = prog.solve(False, z_fix=z.astype(float), time_limit=a.time_limit)
            mask = torch.as_tensor(z, dtype=dt, device=dev).view(1, J)
            sq = P.solve(torch.zeros(1, dtype=dt, device=dev), mu, mask=mask, label=f'squared refit on {k} support')
            st = sq['stats'][0]
            R['supports'][k] = {'stations': sup(z), 'L1_obj': fx['obj'], 'L1_obj_exact': fx.get('obj_exact'),
                                'rates_at_bound': fx.get('rates_at_bound'),
                                'L1_gap_to_milp': (fx['obj'] - mip['obj']) / mip['obj'] if mip.get('obj') else None,
                                'L1_gap_to_dual_bound': ((fx['obj'] - mip['dual_bound']) / mip['dual_bound']
                                                         if mip.get('dual_bound') else None),
                                'squared_obj': st['obj'], 'coverage': st['coverage'], 'mass_kg_d': st['mass_kg_d']}
            print(f"  {k:12s} {sup(z)}: L1 {fx['obj']:.6e} (exact {fx.get('obj_exact', float('nan')):.6e}, gap to MILP "
                  f"{R['supports'][k]['L1_gap_to_milp']}) | squared {st['obj']:.6e} coverage {st['coverage']:.4f} "
                  f"mass {st['mass_kg_d']:.3f}", flush=True)
            P.free_graphs()
        db = mip.get('dual_bound')
        R['certificate_consistent'] = (None if db is None else
                                       bool(all(v['L1_obj'] >= db * (1 - 1e-6) for v in R['supports'].values())))
        print(f"  certificate consistent (every support >= dual bound): {R['certificate_consistent']}", flush=True)
        Path(a.out).write_text(json.dumps(R, default=float), encoding='utf-8')
    finally:
        P.free_graphs()
        P.pr.__exit__()


if __name__ == '__main__':
    main()
