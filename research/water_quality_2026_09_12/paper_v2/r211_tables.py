"""R2-11: sensitivity of the booster designs to hydraulics and chlorine demand (Table S-robust) and comparison with other
siting methods on the nominal problem (Table S-methods). Inputs: final_5090_v2/r1_R11R_*/robust_*.json (exp1_robust.py),
r1_R11G_*/ga_*.json (exp1_ga.py), r1_R11M_*/milp_Net3_m3.json (exp1_milp.py) and the nominal study final_5090/exp1.
Writes paper_v2/tables/T_S_robust.md, T_S_methods.md and r211_numbers.json."""
import argparse
import glob
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V2 = ROOT / 'final_5090_v2'
ap = argparse.ArgumentParser()
ap.add_argument('--robust', default=str(V2 / 'r1_R11R_*' / 'robust_*.json'))
ap.add_argument('--ga', default=str(V2 / 'r1_R11G_*' / 'ga_*.json'))
ap.add_argument('--milp', default=str(V2 / 'r1_R11M_*' / 'milp_Net3_m3.json'))
ap.add_argument('--show', default='NorthPenn:8:0,16,17',
                help="GA runs listed as rows of Table S-methods, e.g. 'NorthPenn:8:0,1,2' (otherwise all runs if at most three, "
                     "else seeds 0-2); every run is in Table S-GA and in the summary row. Default: the authors' choice for North "
                     "Penn, its three runs with the highest objective")
a = ap.parse_args()
SHOW = {(x.split(':')[0], int(x.split(':')[1])): [int(v) for v in x.split(':')[2].split(',')] for x in a.show.split(';') if x}
NET = {'Net3': 'Net3', 'NorthPenn': 'North Penn'}
NUM = {}
pc = lambda x: f'{100 * x:.1f}'
pc2 = lambda x: f'{100 * x:.2f}'
TOL = 0.05                                     # % objective difference counted as equal to greedy


def nominal(net):
    return json.load(open(ROOT / 'final_5090' / 'exp1' / f'exp1_{net}.json', encoding='utf-8'))['networks'][0]


# ------------------------------------------------------------------------------------------------------ Table S-robust
md = ['# Sensitivity of the booster designs to hydraulic conditions and chlorine demand (R2-11)\n',
      'Coverage: demand-weighted share (%) of node-hours of the evaluation window at or above 0.2 mg/L. *Fixed*: the nominal '
      'design (stations and hourly schedule) installed in EPANET as mass-rate sources and run by the library under the '
      'scenario. *Refit*: schedule re-optimised under the scenario on the same stations. *Re-solved*: group lasso at the same '
      'relative penalty under the scenario, refitted; J is the Jaccard index of its stations with the nominal support. The mass '
      'weight is kept at its nominal value; since the refit minimises violation plus mass, under milder scenarios it may trade a '
      'little coverage for less chlorine (second table). Every scenario is recorded again (at Ctol = 0 the kinetic ones would '
      'not need it).\n']
for f in sorted(glob.glob(a.robust)):
    d = json.load(open(f, encoding='utf-8'))
    net = d['network']
    keys = list(d['designs'])
    nst = {k: len(v['stations']) for k, v in d['designs'].items()}
    short = {k: (f'greedy, {nst[k]} stations' if k.startswith('greedy') else
                 f'group lasso {k.split()[1]}λ0, {nst[k]} stations' if k.startswith('lasso') else f'constant rate, {nst[k]} stations')
             for k in keys}
    md += [f"\n## {NET[net]}\n", 'Designs: ' + '; '.join(short[k] for k in keys) + ' (nominal study, Table S5).\n']
    head = ['scenario', 'plant only'] + [f'{short[k]}: fixed / refit' for k in keys] + ['lasso re-solved: stations / coverage / J']
    md += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    err, lossmax = 0.0, {k: 0.0 for k in keys}
    for s in d['scenarios']:
        row = [s['scenario'], pc(s['plant_only']['coverage'])]
        for k in keys:
            fx, rf = s['fixed'][k]['coverage'], s['refit'][k]['coverage']
            row.append(f'{pc(fx)} / {pc(rf)}')
            err = max(err, s['fixed'][k]['max_rel_err_replay_vs_library'])
            if s['scenario'] != 'nominal':
                lossmax[k] = max(lossmax[k], rf - fx)
            NUM[f"{net}|{s['scenario']}|{k}|fixed_coverage"] = fx
            NUM[f"{net}|{s['scenario']}|{k}|refit_coverage"] = rf
            NUM[f"{net}|{s['scenario']}|{k}|fixed_max_c"] = s['fixed'][k]['max_c_mgL']
            NUM[f"{net}|{s['scenario']}|{k}|refit_mass"] = s['refit'][k]['mass_kg_d']
        L = s['lasso_resolved']
        row.append(f"{L['n_stations']} / {pc(L['coverage'])} / {L['jaccard_with_nominal']:.2f}")
        NUM[f"{net}|{s['scenario']}|plant_coverage"] = s['plant_only']['coverage']
        NUM[f"{net}|{s['scenario']}|lasso_resolved"] = {kk: L[kk] for kk in ('n_stations', 'coverage', 'mass_kg_d', 'jaccard_with_nominal')}
        md.append('| ' + ' | '.join(row) + ' |')
    NUM[f'{net}|max_rel_err_replay_vs_library'] = err
    NUM[f'{net}|max_coverage_gain_of_refit'] = lossmax
    md += ['', f'Replayed versus library concentrations of every fixed design in every scenario: maximum relative difference {err:.1e}.', '']
    head = ['scenario'] + [f'{short[k]}: fixed max mg/L' for k in keys] + [f'{short[k]}: refit kg/d' for k in keys] + ['re-solved kg/d']
    md += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for s in d['scenarios']:
        md.append('| ' + ' | '.join([s['scenario']] + [f"{s['fixed'][k]['max_c_mgL']:.2f}" for k in keys]
                                    + [f"{s['refit'][k]['mass_kg_d']:.2f}" for k in keys]
                                    + [f"{s['lasso_resolved']['mass_kg_d']:.2f}"]) + ' |')
    nm = {k: s for s in d['scenarios'] for k in [s['scenario']]}
    if 'consumer demand two hours earlier' in nm:
        md += ['', f"Consumer demand shift: patterns {nm['consumer demand two hours earlier']['mod_log']['patterns_shifted']} moved "
               f"(supply patterns {nm['consumer demand two hours earlier']['mod_log']['supply_patterns_kept']} kept as operated)."]
(HERE / 'tables' / 'T_S_robust.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print('\n'.join(md))

# ----------------------------------------------------------------------------------------------------- Table S-methods
ga = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(a.ga)) if not f.endswith('ga_smoke.json')]
mf = sorted(glob.glob(a.milp))
milp = json.load(open(mf[-1], encoding='utf-8')) if mf else None
md = ['# Booster siting methods on the nominal problem (R2-11)\n',
      'Same candidates, objective (squared shortfall and excess plus the nominal mass weight) and inner schedule fit for every '
      'method; coverage and mass from the library run of the design. GPU times on one RTX 5090; the MILP runs on 8 CPU cores. '
      'The MILP solves the L1 version of the objective; its stations are then refitted with the squared objective, and the '
      'L1 column gives each support\'s optimal L1 value relative to the MILP optimum. The objective has no station cost, so '
      'objectives are comparable only at equal station count.\n']
head = ['network', 'method', 'stations', 'objective (×10⁻⁵)', 'coverage %', 'mass kg/d', 'wall s', 'L1 gap to MILP', 'notes']
md += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
notes = []
for net, mg, lam in (('Net3', [3, 8], 0.1), ('NorthPenn', [8], 0.3)):
    nom = nominal(net)
    walls = [g['wall_s'] for g in nom['greedy']]
    rows = []
    for m in mg:
        g = next(x for x in nom['greedy'] if x['stage'] == m)
        st = g['stations']
        gap = (milp['supports'].get(f'greedy {m}', {}).get('L1_gap_to_milp') if (milp and net == 'Net3' and m == milp['m']) else None)
        rows.append([NET[net], f'greedy forward selection, m = {m}', f"{m}: {', '.join(st)}", f"{g['refit']['obj'] * 1e5:.3f}",
                     pc2(g['refit']['coverage']), f"{g['refit']['mass_kg_d']:.2f}", f'{sum(walls[:m]):.0f}',
                     '' if gap is None else f'{100 * gap:+.2f}%',
                     f"{m} stages, {sum(len(nom['candidates']) - k for k in range(m)):,} sets"])
        runs = sorted([r for r in ga if r['network'] == net and r['m'] == m], key=lambda r: r['seed'])
        show = SHOW.get((net, m), [r['seed'] for r in runs] if len(runs) <= 3 else [0, 1, 2])
        rank = {r['seed']: i + 1 for i, r in enumerate(sorted(runs, key=lambda r: r['final']['obj']))}     # 1 = lowest objective
        if len(runs) > len(show):
            worst = sorted(rank, key=rank.get)[-len(show):]
            kind = ('the runs with the highest objective' if sorted(show) == sorted(worst) else
                    'the first seeds' if sorted(show) == [r['seed'] for r in runs[:len(show)]] else 'selected runs')
            notes.append(f"{NET[net]}, m = {m}: the seed rows are {kind} of {len(runs)} (seeds {', '.join(map(str, show))}; "
                         f"rank 1 = lowest objective); the row 'all {len(runs)} seeds' summarises every run, each of which is "
                         f"listed in Table S-GA.")
        for r in sorted([r for r in runs if r['seed'] in show], key=lambda r: rank[r['seed']]):
            F, lib = r['final'], r['library']
            gap = (milp['supports'].get(f"GA seed {r['seed']}", {}).get('L1_gap_to_milp')
                   if (milp and net == 'Net3' and m == milp['m']) else None)
            lab = f"genetic algorithm, m = {m}, seed {r['seed']}" + (f" (rank {rank[r['seed']]} of {len(runs)})"
                                                                    if len(runs) > len(show) else '')
            rows.append([NET[net], lab, f"{m}: {', '.join(r['best_set'])}",
                         f"{F['obj'] * 1e5:.3f}", pc2(lib['coverage']), f"{lib['mass_kg_d']:.2f}",
                         f"{r['search_wall_s']:.0f}", '' if gap is None else f'{100 * gap:+.2f}%',
                         f"pop {r['pop']}, {r['gens']} gen, {r['sets_evaluated']} sets"])
        if len(runs) > len(show):                                   # every run enters the summary row
            dlt = np.array([100 * (r['final']['obj'] / g['refit']['obj'] - 1) for r in runs])
            ob = np.array([r['final']['obj'] * 1e5 for r in runs])
            cv = np.array([100 * r['library']['coverage'] for r in runs])
            ms = np.array([r['library']['mass_kg_d'] for r in runs])
            wl = np.array([r['search_wall_s'] for r in runs])
            rows.append([NET[net], f'genetic algorithm, m = {m}, all {len(runs)} seeds (Table S-GA)', '—',
                         f'median {np.median(ob):.3f} ({ob.min():.3f}-{ob.max():.3f})',
                         f'median {np.median(cv):.2f} ({cv.min():.2f}-{cv.max():.2f})',
                         f'median {np.median(ms):.2f} ({ms.min():.2f}-{ms.max():.2f})', f'median {np.median(wl):.0f}', '',
                         f'{int((dlt > TOL).sum())} of {len(runs)} worse than greedy, {int((dlt < -TOL).sum())} better; objective vs greedy median {np.median(dlt):+.1f}% '
                         f'({dlt.min():+.1f}% to {dlt.max():+.1f}%)'])
        for r in runs:
            NUM[f"{net}|GA m{m} s{r['seed']}"] = {'stations': r['best_set'], 'obj': r['final']['obj'], 'coverage': r['library']['coverage'],
                                                  'mass': r['library']['mass_kg_d'], 'wall': r['search_wall_s'],
                                                  'sets': r['sets_evaluated'], 'obj_greedy': g['refit']['obj']}
        if milp and net == 'Net3' and m == milp['m']:
            s = milp['supports']['MILP']
            rows.append([NET[net], f'MILP (HiGHS, L1 objective), m = {m}', f"{m}: {', '.join(s['stations'])}",
                         f"{s['squared_obj'] * 1e5:.3f}", pc2(s['coverage']), f"{s['mass_kg_d']:.2f}", f"{milp['milp']['wall_s']:.0f}",
                         '0 (optimum)', f"rel. gap {milp['milp']['mip_gap']:.1e}, {milp['milp']['nodes']} nodes, CPU"])
            NUM['Net3|MILP'] = {'stations': s['stations'], 'squared_obj': s['squared_obj'], 'coverage': s['coverage'],
                                'mass': s['mass_kg_d'], 'L1_opt': milp['milp']['obj'], 'dual_bound': milp['milp']['dual_bound'],
                                'gap': milp['milp']['mip_gap'], 'wall': milp['milp']['wall_s'],
                                'L1_gaps': {k: v['L1_gap_to_milp'] for k, v in milp['supports'].items()},
                                'certificate_consistent': milp['certificate_consistent']}
    r = next(x for x in nom['refit'] if abs(x['lam_frac'] - lam) < 1e-12)
    c = next(x for x in nom['const_rate'] if abs(x['lam_frac'] - lam) < 1e-12)
    t = nom['timing']
    rows.append([NET[net], f'group lasso, lambda = {lam} lambda0, refitted', str(r['n_stations']), f"{r['obj'] * 1e5:.3f}",
                 pc2(r['coverage']), f"{r['mass_kg_d']:.2f}", f"{t['sweep_s'] + t['refit_s']:.0f}", '', '14 penalties in one batch'])
    rows.append([NET[net], 'constant rate on the same stations', str(c['n_stations']), f"{c['obj'] * 1e5:.3f}",
                 pc2(c['coverage']), f"{c['mass_kg_d']:.2f}", f"{t['const_s']:.0f}", '', '14 supports in one batch'])
    md += ['| ' + ' | '.join(map(str, x)) + ' |' for x in rows]
md += [''] + notes
(HERE / 'tables' / 'T_S_methods.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print('\n'.join(md))

# ----------------------------------------------------------------------------------------------------- Table S-GA (all runs)
md = ['# Every genetic-algorithm run (R2-11)\n',
      'Same settings in every run (population 64, 60 generations, at most 1,000 warm-started inner iterations, tournaments of '
      'three, four elites, uniform crossover repaired to m stations, mutation probability 0.3); only the random seed differs. '
      'Objective relative to greedy selection with the same number of stations. *Collapse*: first generation in which at least '
      'half of the population is the best set.\n']
for (net, m) in sorted({(r['network'], r['m']) for r in ga}):
    g = next(x for x in nominal(net)['greedy'] if x['stage'] == m)
    runs = sorted([r for r in ga if r['network'] == net and r['m'] == m], key=lambda r: r['seed'])
    md += [f"\n## {NET[net]}, m = {m} (greedy: objective {g['refit']['obj'] * 1e5:.3f} ×10⁻⁵, coverage {pc2(g['refit']['coverage'])}%, "
           f"{g['refit']['mass_kg_d']:.2f} kg/d)\n"]
    head = ['seed', 'stations', 'objective (×10⁻⁵)', 'vs greedy', 'coverage %', 'mass kg/d', 'stations shared with greedy',
            'sets evaluated', 'collapse (generation)', 'wall s']
    md += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in runs:
        col = next((h['gen'] for h in r['history'] if h['median_obj'] <= h['best_obj'] * (1 + 1e-12)), None)
        md.append('| ' + ' | '.join(map(str, [
            r['seed'], ', '.join(r['best_set']), f"{r['final']['obj'] * 1e5:.3f}", f"{100 * (r['final']['obj'] / g['refit']['obj'] - 1):+.1f}%",
            pc2(r['library']['coverage']), f"{r['library']['mass_kg_d']:.2f}", len(set(r['best_set']) & set(g['stations'])),
            r['sets_evaluated'], '—' if col is None else col, f"{r['search_wall_s']:.0f}"])) + ' |')
    if len(runs) > 1:
        dlt = np.array([100 * (r['final']['obj'] / g['refit']['obj'] - 1) for r in runs])
        md += ['', f'{len(runs)} runs: {int((dlt > TOL).sum())} worse, {int((dlt < -TOL).sum())} better and {int((abs(dlt) <= TOL).sum())} equal '
               f'(within {TOL}%) relative to greedy selection; objective '
               f'relative to greedy median {np.median(dlt):+.1f}%, range {dlt.min():+.1f}% to {dlt.max():+.1f}%.']
        NUM[f'{net}|GA m{m} summary'] = {'runs': len(runs), 'worse': int((dlt > TOL).sum()), 'better': int((dlt < -TOL).sum()),
                                         'median_pct': float(np.median(dlt)), 'min_pct': float(dlt.min()), 'max_pct': float(dlt.max())}
(HERE / 'tables' / 'T_S_ga.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print('\n'.join(md))
(HERE / 'tables' / 'r211_numbers.json').write_text(json.dumps(NUM, indent=1, default=float), encoding='utf-8')
