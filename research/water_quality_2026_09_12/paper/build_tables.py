"""Build every table / number quoted in the manuscript from the 5090 result JSONs.
Usage (from research/water_quality_2026_09_12):  python paper/build_tables.py
Writes paper/tables/T*.md and paper/tables/numbers.json. Final-engine runs under final_5090/* take precedence;
the earlier 5090 runs (v2_5090, g3s_5090, exp*_5090, g4_graph_5090, h2_5090, hpc/results_5090) are the fallback.
"""
import glob
import json
import os
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'tables'
OUT.mkdir(exist_ok=True)
FINAL = ROOT / 'final_5090'
NUM = {}


def pick(final_sub, fallback_sub):
    d = FINAL / final_sub
    if d.is_dir() and any(d.glob('*.json')):
        return d
    return ROOT / fallback_sub


DIRS = {
    'g1': ROOT / 'hpc/results_5090/diffwq_g123_1586077',
    'g2fd': ROOT / 'hpc/results_5090/diffwq_g123_1586077',
    'g3L': ROOT / 'hpc/results_5090/diffwq_g3L_1586093',
    'v2': ROOT / 'v2_5090',
    'g3s': pick('g3s', 'g3s_5090'),
    'fds': FINAL / 'fds',
    'calib': ROOT / 'v2_5090',          # g4_fused_*_cuda.json: fused engine (job 1587178); g4_graph_5090 is the v1 run
    'h2': ROOT / 'h2_5090',
    'exp1': pick('exp1', 'exp1_5090'),
    'exp2': pick('exp2', 'exp2_5090'),
    'exp3': pick('exp3', 'exp3_5090'),
}


def J(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def fmt(x, sig=3):
    if x is None:
        return '-'
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f'{x:,}'
    if isinstance(x, float):
        if x == 0:
            return '0'
        if abs(x) < 1e-3 or abs(x) >= 1e5:
            return f'{x:.{sig - 1}e}'
        return f'{x:.{sig}g}'
    return str(x)


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for r in rows:
        out.append('| ' + ' | '.join(fmt(v) if not isinstance(v, str) else v for v in r) + ' |')
    return '\n'.join(out) + '\n'


def write(name, title, blocks):
    text = f'# {title}\n\n' + '\n'.join(blocks)
    (OUT / name).write_text(text, encoding='utf-8')
    print('wrote', OUT / name)


def maxnum(obj):
    """max over every numeric leaf named *rel_err* / *rel* (used for fd_dll dicts of unknown depth)."""
    best = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 'rel' in k:
                best = v if best is None else max(best, v)
            elif isinstance(v, (dict, list)):
                m = maxnum(v)
                if m is not None:
                    best = m if best is None else max(best, m)
    elif isinstance(obj, list):
        for v in obj:
            m = maxnum(v)
            if m is not None:
                best = m if best is None else max(best, m)
    return best


# ----------------------------------------------------------------------------------------------------------- T1
def t1_exactness():
    blocks = []
    rows = []
    for label, fn in (('core, nextQ', 'g1_core_nextQ.json'), ('core, stepQ', 'g1_core_stepQ.json'),
                      ('variants (tank models, overflow, sources, kinetics, Ctol=0)', 'g1_variants_nextQ.json'),
                      ('corpus (all QUALITY != NONE networks)', 'g1_corpus_nextQ.json')):
        res = J(DIRS['g1'] / fn)['results']
        rows.append([label, len(res), max(r['max_rel_diff'] for r in res), max(r['max_abs_diff'] for r in res),
                     max(abs(r['mass_balance_diff']) for r in res), sum(r['segments_created'] for r in res),
                     sum(r['seconds_python'] for r in res)])
    NUM['g1_cases'] = sum(r[1] for r in rows)
    NUM['g1_segments'] = sum(r[5] for r in rows)
    NUM['g1_max_rel'] = max(r[2] for r in rows)
    blocks.append('## G1: replicator vs libepanet2.so (Linux/gcc build, EPANET 2.3, 20306)\n')
    blocks.append(table(['set', 'cases', 'max rel diff', 'max abs diff', 'max |mass-balance diff|', 'segments', 'python s'], rows))

    g2 = J(DIRS['g3s'] / 'g2_fused_spmm.json')
    fd0 = {r['network']: r for r in J(DIRS['g2fd'] / 'g2_nextQ_ctol0_cuda.json')['results']}
    rows = []
    for r in g2['results']:
        f0 = fd0.get(r['network'], {})
        rows.append([r['network'], r['steps'], r['versions'], r.get('fused_ops'), r.get('fused_terms'),
                     r['forward_rel_vs_dll'], r['adjoint_vs_autograd']['max_rel_err'],
                     maxnum(r.get('fd_torch')), maxnum(f0.get('fd_dll')), r['seconds']['forward_backward']])
    NUM['g2_cases'] = len(rows)
    NUM['g2_max_fwd_rel_vs_dll'] = max(r[5] for r in rows)
    NUM['g2_max_fused_vs_adjoint'] = max(r[6] for r in rows)
    blocks.append(f"\n## G2: fused engine, SpMM path forced ({g2['engine']}, {g2['device']}; DLL FD column from the Ctol=0 run)\n")
    blocks.append(table(['case', 'steps', 'versions', 'fused ops', 'fused terms', 'fwd rel vs DLL', 'fused vs v1 adjoint grad',
                         'max rel err vs torch FD', 'max rel err vs DLL FD (Ctol=0)', 'fwd+bwd s'], rows))
    write('T1_engine_exactness.md', 'Engine exactness (G1 forward replication, G2 gradients)', blocks)



# ------------------------------------------------------------------------------------------------ T1b FD sweep
def t1b_fd_sweep():
    files = sorted(glob.glob(str(DIRS['fds'] / 'g2_fd_sweep_*.json')))
    if not files:
        print('FD sweep: no cluster files yet in', DIRS['fds'])
        return
    blocks, rows = [], []
    for fn in files:
        d = J(fn)
        for label, run in d['runs'].items():
            hs = [f"{row['rel_err']:.1e}" for row in run['sweep']]
            rows.append([f"{d['case']} / {d['param']} (n={run['n_params']})", label, run['ctol_user'], run['autograd']] + hs)
    hcols = [f"h={row['h']:.0e}" for row in J(files[0])['runs']['M0_inp_ctol']['sweep']]
    blocks.append('Relative error of the EPANET central difference (direction: +1 on every parameter) against the replay gradient.\n')
    blocks.append(table(['case / parameter', 'mode', 'Ctol', 'replay gradient'] + hcols, rows))
    write('T1b_fd_step_sweep.md', 'Finite-difference step sweep against EPANET (cluster libepanet2.so)', blocks)


# ----------------------------------------------------------------------------------------------------------- T2
def t2_cost():
    blocks = []
    v2 = J(DIRS['v2'] / 'g3_bench_v2_cuda.json')
    rows = []
    for r in v2['results']:
        e = r['engines']
        def at(eng, B, key='graph_forward_backward_s'):
            for b in e[eng]['batches']:
                if b['B'] == B and key in b:
                    return b[key]
            return None
        rows.append([r['network'], r['n_theta'], r['ops'], e['fused'].get('ops'), e['fused'].get('terms'),
                     r['dll_quality_run_s'], r['fd_gradient_cost_s'],
                     at('adjoint', 1), at('fused_nf', 1), at('fused', 1),
                     (at('adjoint', 1) or 0) / at('fused', 1) if at('fused', 1) else None,
                     at('fused', 1024, 'graph_forward_backward_s') and at('fused', 1024) / 1024])
        NUM[f"v2_speedup_{r['network']}"] = rows[-1][10]
    blocks.append('## v1 -> v2: per-operator adjoint vs fused replay (CUDA graph, fwd+bwd, B=1; per-sample at B=1024)\n')
    blocks.append(table(['case', 'n_theta', 'v1 ops', 'fused ops', 'fused terms', 'DLL run s', 'FD grad s',
                         'adjoint s', 'fused_nf s', 'fused s', 'fused speedup', 'fused B=1024 per-sample s'], rows))

    rows = []
    for fn in sorted(glob.glob(str(DIRS['g3s'] / 'g3_shared_*.json'))):
        d = J(fn)
        for r in d['results']:
            e = r['engines']
            engs = [k for k in ('fused_gather', 'fused_spmm', 'fused') if k in e]
            Bs = [b['B'] for b in e[engs[0]]['batches']]
            for B in Bs:
                row = [r['network'], r['n_theta'], B]
                worst = 0.0
                for eng in engs:
                    b = next((x for x in e[eng]['batches'] if x['B'] == B), {})
                    t = b.get('graph_forward_backward_s', b.get('forward_backward_s'))
                    row += [t, (b.get('jac_rows') or {}).get('rows_per_s')]
                    v = b.get('vs_first_engine') or {}
                    worst = max(worst, v.get('out_rel_err', 0) or 0, v.get('grad_rel_err', 0) or 0, v.get('jac_rel_err', 0) or 0)
                tg = row[3]
                ts = row[5] if len(engs) > 1 else None
                ta = row[7] if len(engs) > 2 else None
                row += [tg / ts if (tg and ts) else None, tg / ta if (tg and ta) else None, worst]
                rows.append(row)
                NUM[f"v3_{r['network']}_B{B}_spmm_speedup"] = row[-3]
                NUM[f"v3_{r['network']}_B{B}_auto_speedup"] = row[-2]
    blocks.append('\n## v3: shared-parameter batches (B initial-quality / source columns); gather vs cuSPARSE SpMM vs automatic per-op choice\n')
    blocks.append(table(['case', 'n_theta', 'B', 'gather s', 'gather jac rows/s', 'spmm s', 'spmm jac rows/s', 'auto s', 'auto jac rows/s',
                         'gather/spmm', 'gather/auto', 'max rel diff between engines'], rows))

    rows = []
    for fn in sorted(glob.glob(str(DIRS['g3L'] / 'g3_*.json'))):
        for r in J(fn)['results']:
            b = [x for x in r['engines']['adjoint']['batches'] if x.get('graph_forward_backward_s')] if 'engines' in r else \
                [x for x in r['batches'] if x.get('graph_forward_backward_s')]
            if b:
                x = max(b, key=lambda y: y['B'])
                rows.append([r['network'], r['n_theta'], x['B'], x['graph_forward_backward_s'] / x['B'], x.get('graph_peak_mem_MB'),
                             r['fd_gradient_cost_s'] / (x['graph_forward_backward_s'] / x['B'])])
    blocks.append('\n## v1 largest feasible batch on 32 GB (CUDA graph)\n')
    blocks.append(table(['case', 'n_theta', 'B_max', 'per-sample fwd+bwd s', 'peak MB', 'FD grad cost / per-sample'], rows))
    write('T2_engine_cost.md', 'Engine cost (v1 adjoint, v2 fused, v3 SpMM)', blocks)



# ----------------------------------------------------------------------------------------------------------- T3
def t3_calibration():
    blocks, rows, trows = [], [], []
    for fn in sorted(glob.glob(str(DIRS['calib'] / 'g4_fused_*_cuda.json'))):
        for d in J(fn)['datasets']:
            tag = d['dataset'] + (' (Kb fixed)' if d['fix_kb'] else '')
            by = {}
            for r in d['runs']:
                by.setdefault(r['method'], []).append(r)
            for m, rs in by.items():
                rows.append([tag, m, len(rs), st.median(r['rmse'] for r in rs), min(r['rmse'] for r in rs), max(r['rmse'] for r in rs),
                             st.median(r['wall_s'] for r in rs), st.median(r['n_eval'] for r in rs) if rs[0]['n_eval'] is not None else None])
                if not d['fix_kb'] and m in ('FD-LBFGSB', 'ADJ-LBFGSB-graph'):
                    NUM[f"calib_{d['dataset']}_{m}_wall_s"] = rows[-1][6]
                    NUM[f"calib_{d['dataset']}_{m}_rmse"] = rows[-1][3]
            t = d['timing']
            trows.append([tag, d['n_obs'], len(d['sites']), t['dll_ctol0_s'], t['dll_ctol0.01_s'], t['fd_grad_equiv_s'],
                          t['adjoint_grad_B1_s'], t['adjoint_grad_B1_graph_s'], max(d['consistency']['loss_rel_diff']),
                          max(d['consistency']['grad_rel_err'])])
    blocks.append('## Field calibration (Kb + 4 diameter-class Kw), each method repeated; wall-clock per full fit\n')
    blocks.append(table(['dataset', 'method', 'repeats', 'median RMSE mg/L', 'min', 'max', 'median wall s', 'median evaluations'], rows))
    blocks.append('\n## One-gradient cost and engine/DLL consistency at the nominal point\n')
    blocks.append(table(['dataset', 'n_obs', 'sites', 'DLL run (Ctol 0) s', 'DLL run (Ctol 0.01) s', 'FD gradient s',
                         'adjoint gradient s', 'adjoint gradient, graph s', 'loss rel diff engine vs DLL', 'grad rel err vs DLL FD'], trows))
    h2 = DIRS['h2'] / 'h2_uq.json'
    if h2.exists():
        rows = []
        for d in J(h2)['datasets']:
            l = d['loso']
            sc = l['site_coverage95']
            n_tot = sum(v['n'] for v in sc.values())
            cov = sum(v['n'] * v['pred_sampled'] for v in sc.values()) / n_tot
            cov_lin = sum(v['n'] * v['pred_lin'] for v in sc.values()) / n_tot
            cov_param = sum(v['n'] * v['param_only_lin'] for v in sc.values()) / n_tot
            rows.append([d['dataset'], d['n_obs'], d['fit']['rmse'], sum(d['fit']['active']), l['cv_rmse'], l['param_share_of_pred_var'],
                         cov_lin, cov, cov_param, d['sampling']['S'], d['sampling']['wall_obs_s']])
            NUM[f"h2_{d['dataset']}_loso_cov95_lin"] = cov_lin
            NUM[f"h2_{d['dataset']}_loso_cov95_sampled"] = cov
        blocks.append('\n## H2: leave-one-site-out predictive check (Laplace sampling on the GPU)\n')
        blocks.append(table(['dataset', 'n_obs', 'fit RMSE', 'active bounds', 'LOSO RMSE', 'parameter share of predictive variance',
                             'LOSO 95% coverage (linearised)', 'LOSO 95% coverage (sampled)', 'LOSO coverage (parameter-only band)',
                             'samples', 'sampling wall s'], rows))
    write('T3_field_calibration.md', 'Field calibration and uncertainty (G4 / H2)', blocks)


# ----------------------------------------------------------------------------------------------------------- T4
def t4_exp1():
    blocks = []
    for fn in sorted(glob.glob(str(DIRS['exp1'] / 'exp1_*.json'))):
        d = J(fn)
        for n in d['networks']:
            s, p = n['sizes'], n['plant_only']
            blocks.append(f"## {n['network']}: {s['nodes']} nodes / {s['links']} links, {s['duration_h']:g} h, {s['slots']} slots, "
                          f"{s['candidates']} candidates, {s['decision_vars']} decision variables, {s['n_obs']} observations, "
                          f"{s['fused_terms']:,} fused terms; lo = {n['settings']['lo_report']} mg/L, hi = {n['settings']['hi']} mg/L\n")
            blocks.append(f"plant only: coverage {p['coverage']:.4f}, mean shortfall {p['mean_short_mgL']:.4f} mg/L, mass {p['mass_kg_d']:.3f} kg/d\n")
            NUM[f"exp1_{n['network']}_plant_only_coverage"] = p['coverage']
            rows = [[g['stage'], g['added'], g['n_stations'], g['coverage'], g['mean_short_mgL'], g['mass_kg_d'], g['wall_s'], g['B']]
                    for g in n['greedy']]
            blocks.append('\n### Greedy forward selection (batched refit over all remaining candidates per stage)\n')
            blocks.append(table(['stage', 'added', 'stations', 'coverage', 'mean shortfall mg/L', 'mass kg/d', 'wall s', 'batch'], rows))
            for g in n['greedy']:
                NUM[f"exp1_{n['network']}_greedy{g['stage']}_coverage"] = g['coverage']
            rows = []
            for a, b, c in zip(n['sweep'], n['refit'], n['const_rate']):
                rows.append([a['lam_frac'], a['n_stations'], a['coverage'], a['mass_kg_d'], b['n_stations'], b['coverage'], b['mass_kg_d'],
                             c['n_stations'], c['coverage'], c['mass_kg_d']])
            blocks.append('\n### Group-lasso path (lambda fraction), refit on the support, constant-rate baseline on the same support\n')
            blocks.append(table(['lam frac', 'path stations', 'coverage', 'mass kg/d', 'refit stations', 'coverage', 'mass kg/d',
                                 'const-rate stations', 'coverage', 'mass kg/d'], rows))
            rm = n.get('response_matrix')
            if rm:
                rows = [[x['lam_frac'], x['obj_free'], x['obj_dense'], x['rel_obj_diff'], x['rel_u_diff'], x['n_stations_free'], x['n_stations_dense']]
                        for x in rm['per_lam']]
                blocks.append(f"\n### Explicit response matrix cross-check: {rm['n_obs']} x {rm['n_cols']}, build {rm['build_s']:.2f} s, "
                              f"dense solve {rm['dense_solve_s']:.1f} s ({rm['dense_iters']} iters), L_exact {rm['L_exact']:.3g} vs "
                              f"L_power {rm['L_power']:.3g}, D-probe max rel err {rm['D_probe_max_rel_err']:.2e}, support match {rm['D_support_match']}\n")
                blocks.append(table(['lam frac', 'obj matrix-free', 'obj dense', 'rel obj diff', 'rel u diff', 'stations free', 'stations dense'], rows))
            dc = n.get('dll_check') or []
            if dc:
                worst = max((x['max_rel_err_all_nodes'] for x in dc), default=None)
                blocks.append(f"\nDLL re-insertion of every reported design: {len(dc)} designs, max rel err {worst:.2e}\n")
                NUM[f"exp1_{n['network']}_dll_max_rel_err"] = worst
            t = n['timing']
            blocks.append('\ntiming s: ' + ', '.join(f"{k} {v:.1f}" for k, v in t.items() if isinstance(v, (int, float))) + '\n')
    write('T4_exp1_booster.md', 'Exp1: booster siting and dosing schedule as one convex program', blocks)



# ----------------------------------------------------------------------------------------------------------- T5
def t5_exp2():
    blocks = []
    for fn in sorted(glob.glob(str(DIRS['exp2'] / 'exp2_*.json'))):
        for n in J(fn)['networks']:
            s, dat, pl, t = n['sizes'], n['data'], n['placement'], n['timing']
            blocks.append(f"## {n['network']}: {s['nodes']} nodes / {s['links']} links, {s['duration_h']:g} h, {s['slots']} slots of {s['slot_s']} s, "
                          f"{s['unknowns']:,} unknowns ({s['junctions']} candidate nodes), {s['fused_terms']:,} fused terms\n")
            delays = dat.get('detection_delay_h') or []
            blocks.append(f"sensors {len(pl['sensors'])} (greedy detection coverage on {pl['n_train']} training scenarios, detectable fraction "
                          f"{pl['detectable_fraction']:.3f}); test scenarios {dat['n_sampled']}, detected {dat['n_detected']}, false alarms "
                          f"{dat['n_false_alarm']}, median detection delay {st.median(delays) if delays else float('nan'):.2f} h\n")
            NUM[f"exp2_{n['network']}_detected"] = dat['n_detected']
            rows = []
            for w in n['summary']:
                ex = w.get('exhaustive') or {}
                rows.append([w['window_h'], w['n'], w['glrt_top1'], w['glrt_top5'], w['glrt_correct_or_equivalent'], w['median_glrt_n_equiv'],
                             w['median_abs_glrt_start_err_h'], w['median_abs_glrt_mass_rel_err'], w['top1'], w['correct_or_equivalent'],
                             w['mean_wall_refine_s'], w['mean_projected_exhaustive_dll_s'], ex.get('n'), ex.get('same_best_as_top_m')])
                NUM[f"exp2_{n['network']}_w{w['window_h']}_glrt_top1"] = w['glrt_top1']
                NUM[f"exp2_{n['network']}_w{w['window_h']}_glrt_equiv"] = w['glrt_correct_or_equivalent']
            blocks.append('\n### Identification by observation window after first detection (GLRT = parametric box-constrained likelihood ratio; '
                          'NNLS = non-parametric l1 profile)\n')
            blocks.append(table(['window h', 'scenarios', 'GLRT top-1', 'GLRT top-5', 'GLRT correct or equivalent', 'median equivalent set',
                                 'median |start err| h', 'median mass rel err', 'NNLS top-1', 'NNLS correct or equiv', 'refine wall s per scenario',
                                 'projected exhaustive DLL s', 'exhaustive n', 'same best as top-M'], rows))
            dc = n.get('dll_check') or []
            if dc:
                blocks.append(f"\nDLL check of the truth runs: {len(dc)} scenarios, max rel err {max(x['max_rel_err_all_nodes'] for x in dc):.2e}\n")
            blocks.append('\ntiming s: ' + ', '.join(f"{k} {v:.1f}" for k, v in t.items() if isinstance(v, (int, float))) + '\n')
    write('T5_exp2_source.md', 'Exp2: contamination source identification from sparse sensors', blocks)


# ----------------------------------------------------------------------------------------------------------- T6
def t6_exp3():
    blocks, rows = [], []
    f = DIRS['exp3'] / 'exp3_field.json'
    if f.exists():
        for d in J(f)['field']:
            a, g = d['audit'], d['prescription']['greedy']['logdet']
            rows.append([d['dataset'], d['n_obs'], len(d['sites']), d['n_pipes'], d['rmse_g4'], a['n_sensitive'], a['n_identifiable'], a['p_eff'],
                         a['median_sd_over_tau'], d['prescription']['n_candidates'], len(g['added_sites']), g['n_identifiable_after'],
                         g['p_eff_after'], d['jac_check']['max_abs_diff'], d['timing']['total_s']])
            NUM[f"exp3_field_{d['dataset']}_identifiable"] = (a['n_identifiable'], g['n_identifiable_after'])
        blocks.append('## Field systems at theta*: pipe-by-pipe Kw Laplace audit and D-optimal (log-det) sensor prescription (+5 sites)\n')
        blocks.append(table(['dataset', 'n_obs', 'sites', 'pipes', 'RMSE at theta*', 'pipes sensitive', 'identifiable (sd < 0.5 prior)', 'p_eff',
                             'median sd/prior', 'candidate sites', 'sites added', 'identifiable after', 'p_eff after', 'Jacobian check', 'total s'], rows))
    for fn in sorted(glob.glob(str(DIRS['exp3'] / 'exp3_*.json'))):
        if fn.endswith('exp3_field.json'):
            continue
        for n in J(fn)['networks']:
            s, c, des = n['sizes'], n['cost'], n['design']
            blocks.append(f"\n## {n['network']}: {s['nodes']} nodes / {s['pipes']} pipes, {s['duration_h']:g} h, {s['n_t']} sampling instants, "
                          f"Kw in {s['kw_unit']}; {n['n_sensors']} random sensors + {n['n_add']} prescribed; groups {n['groups']}\n")
            blocks.append(f"DLL check {n['dll_check']['max_rel_err']:.1e}; Jacobian rows vs batched autograd {n['jac_check']['max_abs_diff']:.1e}; "
                          f"one adjoint gradient of {c['n_params']} pipe Kw {c['adjoint_grad_B1_s']:.3f} s vs projected DLL FD {c['fd_grad_dll_projected_s']:.3g} s "
                          f"({c['speedup_vs_fd_dll']:.3g}x) and batched-engine FD {c['fd_grad_engine_batched_projected_s']:.3g} s ({c['speedup_vs_fd_engine_batched']:.3g}x); "
                          f"candidate Jacobian {des['candidate_rows']:,} rows in {des['wall_candidate_jacobian_s']:.1f} s\n")
            NUM[f"exp3_{n['network']}_speedup_vs_fd_dll"] = c['speedup_vs_fd_dll']
            rows = []
            for k, v in n['sets'].items():
                a, ke = v['audit'], v.get('kw_error') or {}
                pred = des['predicted'].get(k, {})
                m, gp, ho, sf = ke.get('map', {}), ke.get('grouped_prior_mean', {}), v.get('holdout_rmse_mgL') or {}, v.get('sensor_fit_rmse_mgL') or {}
                rows.append([k, v['n_sensors'], v['n_obs'], pred.get('n_identifiable'), a['n_identifiable'], a['p_eff'], a['n_sensitive'],
                             m.get('rmse_identifiable'), gp.get('rmse_identifiable'), m.get('median_rel_abs_err_identifiable'),
                             gp.get('median_rel_abs_err_identifiable'), m.get('rmse'), gp.get('rmse'), m.get('coverage_2sd'),
                             ho.get('map_nonsensor'), ho.get('grouped_nonsensor'), sf.get('map'), sf.get('noise_sd'),
                             v['map']['wall_s'], a['wall_jacobian_s'] + a['wall_laplace_s']])
                NUM[f"exp3_{n['network']}_{k}_identifiable"] = a['n_identifiable']
                NUM[f"exp3_{n['network']}_{k}_map_rel_err_identifiable"] = m.get('median_rel_abs_err_identifiable')
            blocks.append(table(['sensor set', 'sensors', 'n_obs', 'predicted identifiable', 'identifiable (sd < 0.5 prior)', 'p_eff', 'sensitive pipes',
                                 'Kw RMSE identifiable: MAP', 'grouped prior', 'median rel err identifiable: MAP', 'grouped prior',
                                 'Kw RMSE all: MAP', 'grouped prior', 'MAP 2-sd coverage', 'hold-out node RMSE mg/L: MAP', 'grouped',
                                 'sensor fit RMSE (MAP)', 'noise sd', 'MAP wall s', 'audit wall s'], rows))
            blocks.append('\ntiming s: ' + ', '.join(f"{k} {v:.1f}" for k, v in n['timing'].items() if isinstance(v, (int, float))) + '\n')
    write('T6_exp3_pipe_kw.md', 'Exp3: pipe-by-pipe wall-decay identifiability and sensor prescription', blocks)


if __name__ == '__main__':
    for k, v in DIRS.items():
        print(f'{k:6s} <- {v.relative_to(ROOT) if v.is_relative_to(ROOT) else v}  {"(missing)" if not v.exists() else ""}')
    t1_exactness()
    t1b_fd_sweep()
    t2_cost()
    t3_calibration()
    t4_exp1()
    t5_exp2()
    t6_exp3()
    (OUT / 'numbers.json').write_text(json.dumps(NUM, indent=1, default=str), encoding='utf-8')
    print('wrote', OUT / 'numbers.json', len(NUM), 'numbers')
