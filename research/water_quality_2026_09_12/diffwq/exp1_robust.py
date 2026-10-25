"""R2-11, parts 1-2: sensitivity of the reported booster designs to hydraulic conditions and chlorine demand.

For each network (Net3, North Penn) and scenario the booster problem is rebuilt with the perturbed hydraulics or kinetics
(the schedule is recorded again; at Ctol = 0 it does not depend on the kinetics, but the problem object is rebuilt for
simplicity), and the nominal designs of exp1_booster.py (greedy stations, group-lasso support refitted, constant rates on
that support) are
  (a) evaluated as installed in the library through the API (the truth for a fixed design), and
  (b) re-optimised under the scenario: schedules refitted on the same support, and the group-lasso support at the same
      relative penalty re-solved and refitted.
The mass weight is kept at its nominal value, so that every scenario solves the same trade-off.
Scenarios: demand multiplier 0.8 / 0.9 / 1.1 / 1.2; one operational change (Net3: pump 10 switched two hours later;
North Penn: consumer demand patterns two hours earlier, wells and transfers as operated); bulk and wall
coefficients x0.5 / x2.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp1_robust --network Net3 --designs RESULTS/exp1_Net3.json --out robust_Net3.json
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
from .exp1_booster import BoosterProblem, CONFIGS
from .verify_m0 import inp_replace

EN_DEMANDMULT, EN_DEMANDPATTERN, EN_PATTERN = 4, 23, 2
DESIGN = {'Net3': {'greedy': 3, 'lasso': 0.1}, 'NorthPenn': {'greedy': 8, 'lasso': 0.3}}
CONFIGS['Net3_pump2'] = dict(CONFIGS['Net3'], inp_edit=inp_replace(('Link 10 OPEN AT TIME 1', 'Link 10 OPEN AT CLOCKTIME 3:00 AM'),
                                                                    ('Link 10 CLOSED AT TIME 15', 'Link 10 CLOSED AT CLOCKTIME 5:00 PM')))


def scenarios(net):
    op = ('pump 10 two hours later', 'config', 'Net3_pump2') if net == 'Net3' else ('consumer demand two hours earlier', 'shift', 2)
    return ([('nominal', None, None)] + [(f'demand x{f}', 'demand', f) for f in (0.8, 0.9, 1.1, 1.2)] + [op]
            + [(f'kb x{f}', 'kb', f) for f in (0.5, 2.0)] + [(f'kw x{f}', 'kw', f) for f in (0.5, 2.0)])


class RobustProblem(BoosterProblem):
    """BoosterProblem whose projects (recording and library checks alike) carry one scenario modification."""

    def __init__(self, name, lib, workdir, args, mod):
        self.mod, self.mod_log = mod, {}
        super().__init__(name, lib, workdir, args)

    def open(self, tag=''):
        pr = super().open(tag)
        kind, val = self.mod
        if kind == 'demand':
            pr.set_option(EN_DEMANDMULT, val)
        elif kind in ('kb', 'kw'):
            prop = api.EN_KBULK if kind == 'kb' else api.EN_KWALL
            for k in range(1, pr.count(api.EN_LINKCOUNT) + 1):
                if pr.link_type(k) <= api.EN_PIPE:
                    pr.set_link_value(k, prop, val * pr.link_value(k, prop))
            if kind == 'kb':
                for i in range(1, pr.count(api.EN_NODECOUNT) + 1):
                    if pr.node_type(i) == api.EN_TANK:
                        pr.set_node_value(i, api.EN_TANK_KBULK, val * pr.node_value(i, api.EN_TANK_KBULK))
        elif kind == 'shift':
            # consumer demand patterns val hours earlier, supply kept as operated (North Penn: wells and transfers are
            # negative demands with measured schedules; source-quality patterns untouched). Demands without a pattern
            # follow the default pattern (hydraul.c). A pattern of at most one day is rotated; the tail of a longer one
            # (North Penn: measured 36 h series, not periodic) is taken from the same clock time one day earlier.
            step = pr.time_param(api.EN_PATTERNSTEP)
            nn = pr.count(api.EN_NODECOUNT)
            src = {int(v) for i in range(1, nn + 1) if (v := pr.node_value(i, api.EN_SOURCEPAT)) is not None}
            defpat, sign = int(pr.option(EN_DEMANDPATTERN)), {}
            for i in range(1, nn + 1):
                if pr.node_type(i) == api.EN_JUNCTION and (b := pr.node_value(i, api.EN_BASEDEMAND)) != 0:
                    sign.setdefault(int(pr.node_value(i, EN_PATTERN)) or defpat, set()).add(b > 0)
            consumer = sorted(p for p, s in sign.items() if p and s == {True} and p not in src)
            n, day = int(round(val * 3600 / step)), int(round(86400 / step))
            lens = {}
            for p in consumer:
                v = [float(x) for x in pr.pattern(p)]
                L = lens[p] = len(v)
                if L <= day:
                    pr.set_pattern(p, v[n % L:] + v[:n % L])
                else:
                    pr.set_pattern(p, [v[i if i < L else i - day * ((i - L) // day + 1)] for i in range(n, n + L)])
            self.mod_log = {'pattern_step_s': step, 'shift_periods': n, 'periods_per_day': day, 'default_pattern': defpat,
                            'patterns_shifted': consumer, 'pattern_lengths': lens,
                            'supply_patterns_kept': sorted(p for p, s in sign.items() if p and s == {False}),
                            'mixed_sign_patterns_kept': sorted(p for p, s in sign.items() if p and len(s) == 2),
                            'source_patterns_kept': sorted(src - {0})}
        return pr


def library_metrics(P, u, tag):
    """Design u [J, K] installed in the library (MASS sources with patterns); metrics of the library run."""
    u = np.asarray(u, float).reshape(P.J, P.K)
    with P.open(tag) as pr:
        for j in np.flatnonzero(u.sum(1) > 0):
            uref = float(u[j].max())
            pidx = pr.add_pattern(f'b{j}', (u[j] / uref).tolist())
            pr.set_source(P.cand[j], api.EN_MASS, uref, pidx)
        pr.solve_hydraulics(P.workdir / f'{P.name}{tag}.hyd')
        _, q, _ = pr.run_quality_reference('nextQ')
    with torch.no_grad():
        c = P.c_full(torch.as_tensor(u, dtype=P.dt, device=P.dev).view(1, P.J, P.K))[0].cpu().numpy()
    m = P.metrics(q.reshape(-1)[P.obs], torch.as_tensor(u, dtype=P.dt, device=P.dev))
    m['max_rel_err_replay_vs_library'] = float(np.abs(q - c).max() / max(np.abs(q).max(), 1e-300))
    m.pop('stations', None); m.pop('station_mass_kg_d', None)
    return m


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def run(args):
    nom = json.load(open(args.designs, encoding='utf-8'))['networks'][0]
    spec = DESIGN[args.network]
    g = next(x for x in nom['greedy'] if x['stage'] == spec['greedy'])
    l = next(x for x in nom['refit'] if abs(x['lam_frac'] - spec['lasso']) < 1e-12)
    designs = {f"greedy {spec['greedy']} stations": np.array(g['refit']['u']), f"lasso {spec['lasso']} refit": np.array(l['u'])}
    mu_nom = nom['settings']['mu']
    lib = api.Library(args.lib)
    wd = Path(tempfile.mkdtemp(prefix='diffwq_rob_', dir=os.environ.get('TMPDIR')))
    out = {'network': args.network, 'mu_nominal': mu_nom, 'designs': {}, 'scenarios': []}
    kc = f"constant rate on lasso {spec['lasso']} support"
    if args.resume:                                   # continue an interrupted run: its designs and finished scenarios are kept
        prev = json.load(open(args.resume, encoding='utf-8'))
        out['designs'], out['scenarios'] = prev['designs'], prev['scenarios']
        designs[kc] = np.array(prev['designs'][kc]['u'])
    done = {s['scenario'] for s in out['scenarios']}
    keep = [x.strip() for x in args.scenarios.split(',')] if args.scenarios else None
    for label, kind, val in [sc for sc in scenarios(args.network) if (keep is None or sc[0] in keep) and sc[0] not in done]:
        if label != 'nominal' and kc not in designs:
            raise SystemExit('the constant-rate design comes from the nominal scenario: include it or pass --resume')
        t0 = time.perf_counter()
        name = val if kind == 'config' else args.network
        P = RobustProblem(name, lib, wd, args, (kind, val))
        try:
            P.mu = mu_nom
            dt, dev = P.dt, P.dev
            mu1 = torch.full((1,), mu_nom, dtype=dt, device=dev)
            if label == 'nominal':              # constant rates on the lasso support (not stored by exp1_booster)
                ul = torch.as_tensor(designs[f"lasso {spec['lasso']} refit"], dtype=dt, device=dev).view(1, P.J, P.K)
                cr = P.solve(torch.zeros(1, dtype=dt, device=dev), mu1, mask=(ul.norm(dim=2) > 0), const_rate=True, u0=ul,
                             label='constant rate on lasso support')
                designs[f"constant rate on lasso {spec['lasso']} support"] = cr['u'][0].cpu().numpy()
                P.free_graphs()
                out['designs'] = {k: {'stations': [P.net.node_ids[P.cand[j]] for j in np.flatnonzero(np.abs(v).sum(1) > 0)],
                                      'u': np.round(v, 6).tolist()} for k, v in designs.items()}
            rec = {'scenario': label, 'kind': kind, 'value': val, 'mod_log': P.mod_log, 'record_s': P.t_record,
                   'plant_only': P.metrics(P.c_plant.reshape(-1)[P.obs_t]), 'fixed': {}, 'refit': {}}
            rec['plant_only'].pop('stations', None)
            for k, u in designs.items():                              # (a) fixed designs in the library
                rec['fixed'][k] = library_metrics(P, u, f'_fx{len(rec["fixed"])}')
            for k, u in designs.items():                              # (b1) schedules refitted on the same support
                ut = torch.as_tensor(u, dtype=dt, device=dev).view(1, P.J, P.K)
                res = P.solve(torch.zeros(1, dtype=dt, device=dev), mu1, mask=(ut.norm(dim=2) > 0), u0=ut,
                              const_rate=k.startswith('constant'), label=f'{label}: refit {k}')
                st = res['stats'][0]
                rec['refit'][k] = {kk: st[kk] for kk in ('coverage', 'mean_short_mgL', 'min_c_mgL', 'max_c_mgL', 'mass_kg_d', 'obj', 'n_stations')}
                P.free_graphs()
            lam0 = P.lambda_zero(mu1)                                 # (b2) group-lasso support re-solved, then refitted
            sw = P.solve(torch.tensor([spec['lasso'] * lam0], dtype=dt, device=dev), mu1, label=f'{label}: lasso {spec["lasso"]}')
            sup = sw['u'].norm(dim=2) > 0
            rf = P.solve(torch.zeros(1, dtype=dt, device=dev), mu1, mask=sup, u0=sw['u'], label=f'{label}: lasso refit')
            st = rf['stats'][0]
            new = [P.net.node_ids[P.cand[j]] for j in torch.nonzero(sup[0]).flatten().tolist()]
            rec['lasso_resolved'] = {**{kk: st[kk] for kk in ('coverage', 'mean_short_mgL', 'min_c_mgL', 'mass_kg_d', 'obj', 'n_stations')},
                                     'stations': new, 'jaccard_with_nominal': jaccard(new, out['designs'][f"lasso {spec['lasso']} refit"]['stations'])}
            P.free_graphs()
            rec['wall_s'] = time.perf_counter() - t0
            out['scenarios'].append(rec)
            fx = {k: round(v['coverage'], 4) for k, v in rec['fixed'].items()}
            print(f"[{args.network}] {label}: plant {rec['plant_only']['coverage']:.4f} | fixed {fx} | lasso re-solved "
                  f"{rec['lasso_resolved']['n_stations']} stations cov {rec['lasso_resolved']['coverage']:.4f} "
                  f"J={rec['lasso_resolved']['jaccard_with_nominal']:.2f} | {rec['wall_s']:.0f}s", flush=True)
            Path(args.out).write_text(json.dumps(out, default=float), encoding='utf-8')
        finally:
            P.free_graphs()
            P.pr.__exit__()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', required=True, choices=list(DESIGN))
    ap.add_argument('--designs', required=True, help='exp1_booster result JSON of the nominal study')
    ap.add_argument('--theta-json', default=None)
    ap.add_argument('--scenarios', default='', help='comma-separated subset of the scenario labels (smoke tests)')
    ap.add_argument('--resume', default='', help='output JSON of an interrupted run (its designs and finished scenarios are kept)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lib', default=None)
    a = ap.parse_args()
    # settings of the nominal study (exp1_booster defaults)
    for k, v in dict(lo=0.2, margin=0.02, hi=4.0, mu_rel=0.1, iters=5000, tol=1e-7, chunk=256, no_graph=False, no_tanks=False).items():
        setattr(a, k, v)
    run(a)


if __name__ == '__main__':
    main()
