"""R1-1 / R2-2: effect of disabling parcel merging (Ctol = 0) in EPANET 2.3. New module; v1 files untouched.
Part 1 (library only): Ctol 0 vs the default 0.01 at the same hydraulics and quality step, over the corpus, the field
networks, L-Town and BWSN2, three durations and eight quality conditions, with a fine-step Ctol 0 reference.
Part 2 (library only): the paper's estimates and designs re-simulated at Ctol 0.01 (calibration, booster designs,
source-identification scenarios, Exp3 truth fields).
Part 3 (replicator + GPU): schedule size, live segments, recording, replay time and memory at Ctol 0 and 0.01.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.ctol_study --part 1 --lib LIB --out final_ctol --workers 64
"""
import argparse, json, os, tempfile, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from . import epanet_api as api
from .hydfile import read_hydfile
from .verify_m0 import CORE, ROOT, all_links, all_tanks_kb, init_qual, qual_type, scan_corpus

COND = {'native': None, 'cl_k0': (0.0, 0.0), 'cl_b05': (-0.5, 0.0), 'cl_b2': (-2.0, 0.0), 'cl_w05': (0.0, -0.5),
        'cl_b05_w05': (-0.5, -0.5), 'cl_b2_w05': (-2.0, -0.5), 'age': 'age', 'trace': 'trace'}
BIG = 5000                                                          # nodes above which the reference step is 30 s


def pair(a, b):
    d = np.abs(a - b)
    return {'max': float(d.max()), 'p99': float(np.quantile(d, 0.99)), 'rmse': float(np.sqrt(np.mean(d * d))),
            'frac_gt_0.01': float(np.mean(d > 0.01)), 'frac_gt_0.05': float(np.mean(d > 0.05)), 'scale': float(np.abs(b).max())}


def open_net(lib, item, wd, tag):
    pr = api.Project(lib, item['inp'], str(Path(wd) / f'{tag}_{os.getpid()}.rpt'))
    if item.get('unbalanced'):
        pr.set_option(api.EN_UNBALANCED, item['unbalanced'])
    return pr


def apply_cond(pr, cond):
    c = COND[cond]
    if c is None:
        return pr.qual_info()[0] != api.EN_NONE
    if c == 'age':
        qual_type(api.EN_AGE)(pr); init_qual(0.0)(pr); return True
    if c == 'trace':
        res = [i for i in range(1, pr.count(api.EN_NODECOUNT) + 1) if pr.node_type(i) == api.EN_RESERVOIR] or \
              [i for i in range(1, pr.count(api.EN_NODECOUNT) + 1) if pr.node_type(i) == api.EN_TANK]
        if not res:
            return False
        qual_type(api.EN_TRACE, pr.node_id(res[0]))(pr); return True
    qual_type(api.EN_CHEM)(pr); init_qual(0.5, reservoirs=1.0)(pr); all_links(kb=c[0], kw=c[1])(pr); all_tanks_kb(c[0])(pr)
    return True


_W = {}


def _w_init(lib_path, wd):
    if os.name == 'nt':                       # EPANET's Windows scratch-file names can collide across processes
        tmp = Path(wd) / f'tmp_{os.getpid()}'; tmp.mkdir(parents=True, exist_ok=True)
        os.environ['TMP'] = os.environ['TEMP'] = str(tmp)
    _W['lib'], _W['wd'] = api.Library(lib_path), wd


def p1_task(t):
    item, cond, dur = t
    out = {'net': item['label'], 'cond': cond, 'duration_h': dur}
    try:
        pr = open_net(_W['lib'], item, _W['wd'], 'p1')
        if not apply_cond(pr, cond):
            pr.__exit__(); out['skipped'] = 'no quality / no trace node'; return out
        if dur:
            pr.set_time_param(api.EN_DURATION, int(dur * 3600))
        q0 = pr.time_param(api.EN_QUALSTEP)
        pr.solve_hydraulics(Path(_W['wd']) / f'p1_{os.getpid()}.hyd')
        runs = {}
        ref_step = 30 if pr.count(api.EN_NODECOUNT) > BIG else 10
        for qs in sorted({q0, 60, ref_step}):
            for ct in ((0.0, 0.01) if qs != ref_step or qs in (q0, 60) else (0.0,)):
                pr.set_time_param(api.EN_QUALSTEP, int(qs)); pr.set_option(api.EN_TOLERANCE, ct)
                t0 = time.perf_counter(); _, q, mb = pr.run_quality_reference('nextQ')
                runs[(qs, ct)] = (q, mb, time.perf_counter() - t0)
        pr.__exit__()
        out.update(qstep_input=int(q0), ref_step=ref_step, n_nodes=int(runs[(q0, 0.0)][0].shape[1] - 1))
        for qs in sorted({q0, 60}):
            a, b = runs[(qs, 0.01)], runs[(qs, 0.0)]
            out[f'q{qs}'] = {'ctol001_vs_ctol0': pair(a[0], b[0]), 'mb_ctol0': b[1], 'mb_ctol001': a[1],
                             'run_s_ctol0': b[2], 'run_s_ctol001': a[2],
                             'ctol0_vs_ref': pair(b[0], runs[(ref_step, 0.0)][0]), 'ctol001_vs_ref': pair(a[0], runs[(ref_step, 0.0)][0])}
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'[:200]
    return out


def part1(a, pool, wd):
    from .calib_data import prepare_inp
    lib = api.Library(a.lib)
    items = [{'label': n, 'inp': str(p)} for n, p in scan_corpus(lib, wd)]
    labels = {i['label'] for i in items}
    extra = [('Bellingham', prepare_inp('Bellingham', wd), None),
             ('L-TOWN', ROOT / 'networks/L-Town/L-TOWN.inp', None),
             ('BWSN_Network_2', ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp', 10)]
    for lab, p, ub in extra:
        it = next((i for i in items if i['label'] == lab), None)
        if it is None:
            items.append({'label': lab, 'inp': str(p), 'unbalanced': ub})
        elif ub:
            it['unbalanced'] = ub
    tasks = [(it, c, d) for it in items for c in COND for d in (None, 48, 168)]
    tasks.sort(key=lambda t: ('BWSN' not in t[0]['label'], 'TOWN' not in t[0]['label']))     # long jobs first
    print(f'part 1: {len(items)} networks, {len(tasks)} tasks', flush=True)
    res = pool.map(p1_task, tasks, chunksize=1)
    return {'networks': [i['label'] for i in items], 'records': res}


# ------------------------------------------------------------------ part 1f: fast version (incremental, resumable)
def p1f_task(t):
    """kind 'main': Ctol 0 vs 0.01 at the input step and at 60 s; kind 'ref': Ctol 0 reference (10 s; 30 s above BIG
    nodes; 60 s for 168 h) against the input-step runs at both tolerances."""
    item, cond, dur, kind = t
    out = {'net': item['label'], 'cond': cond, 'duration_h': dur, 'kind': kind}
    t_all = time.perf_counter()
    try:
        pr = open_net(_W['lib'], item, _W['wd'], 'p1f')
        if not apply_cond(pr, cond):
            pr.__exit__(); out['skipped'] = 'no quality / no trace node'; return out
        if dur:
            pr.set_time_param(api.EN_DURATION, int(dur * 3600))
        q0 = int(pr.time_param(api.EN_QUALSTEP)); nn = pr.count(api.EN_NODECOUNT)
        pr.solve_hydraulics(Path(_W['wd']) / f'p1f_{os.getpid()}.hyd')

        def run(qs, ct):
            pr.set_time_param(api.EN_QUALSTEP, int(qs)); pr.set_option(api.EN_TOLERANCE, ct)
            t0 = time.perf_counter(); _, q, mb = pr.run_quality_reference('nextQ')
            return q, mb, time.perf_counter() - t0
        out.update(qstep_input=q0, n_nodes=nn)
        if kind == 'main':
            for qs in sorted({q0, 60}):
                a, b = run(qs, 0.01), run(qs, 0.0)
                out[f'q{qs}'] = {'ctol001_vs_ctol0': pair(a[0], b[0]), 'mb_ctol0': b[1], 'mb_ctol001': a[1],
                                 'run_s_ctol0': b[2], 'run_s_ctol001': a[2]}
        else:
            ref = 60 if dur == 168 else (30 if nn > BIG else 10)
            r = run(ref, 0.0); a, b = run(q0, 0.01), run(q0, 0.0)
            out.update(ref_step=ref, ref_run_s=r[2], ctol0_vs_ref=pair(b[0], r[0]), ctol001_vs_ref=pair(a[0], r[0]))
        pr.__exit__()
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'[:200]
        try:
            pr.__exit__()
        except Exception:
            pass
    out['task_s'] = time.perf_counter() - t_all
    return out


def part1f(a, pool, wd, out_dir):
    from .calib_data import prepare_inp
    lib = api.Library(a.lib)
    items = [{'label': n, 'inp': str(p)} for n, p in scan_corpus(lib, wd)]
    for lab, p, ub in (('Bellingham', prepare_inp('Bellingham', wd), None), ('L-TOWN', ROOT / 'networks/L-Town/L-TOWN.inp', None),
                       ('BWSN_Network_2', ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp', 10)):
        it = next((i for i in items if i['label'] == lab), None)
        if it is None:
            items.append({'label': lab, 'inp': str(p), 'unbalanced': ub})
        elif ub:
            it['unbalanced'] = ub
    heavy = lambda it: 0 if 'BWSN' in it['label'] else (1 if 'TOWN' in it['label'] else 2)
    tasks = [(it, c, d, k) for k in ('main', 'ref') for it in items for c in COND for d in (None, 48, 168)]
    tasks.sort(key=lambda t: (t[3] != 'main', heavy(t[0]), -(t[2] or 24)))
    log = out_dir / f'ctol_part1f{getattr(a, "log_tag", "")}.jsonl'
    key = lambda r: f"{r['net']}|{r['cond']}|{r['duration_h']}|{r['kind']}"
    done = {key(json.loads(l)) for l in open(log, encoding='utf-8')} if log.exists() else set()
    todo = [t for t in tasks if f"{t[0]['label']}|{t[1]}|{t[2]}|{t[3]}" not in done]
    if getattr(a, 'only_keys', None):
        want = {l.strip() for l in open(a.only_keys, encoding='utf-8') if l.strip()}
        todo = [t for t in todo if f"{t[0]['label']}|{t[1]}|{t[2]}|{t[3]}" in want]
    if getattr(a, 'reverse', False):
        todo = todo[::-1]
    print(f'part 1f: {len(items)} networks, {len(tasks)} tasks, {len(done)} already done, {len(todo)} to run', flush=True)
    t0 = time.perf_counter()
    with open(log, 'a', encoding='utf-8') as fh:
        for i, r in enumerate(pool.imap_unordered(p1f_task, todo, chunksize=1), 1):
            fh.write(json.dumps(r, default=float) + '\n'); fh.flush()
            if i % 20 == 0 or i == len(todo):
                nm = sum(1 for t in todo[:0])
                print(f'  {i}/{len(todo)} done ({len(done) + i}/{len(tasks)} overall), {time.perf_counter() - t0:.0f}s, last {key(r)} '
                      f'{r.get("task_s", 0):.0f}s{" ERROR " + r["error"] if "error" in r else ""}', flush=True)
    return {'records': [json.loads(l) for l in open(log, encoding='utf-8')]}


# ------------------------------------------------------------------ part 2
def coverage_W(trace, times, node_type, eval_h):
    by_t = {int(p.t): p for p in trace.periods}
    W = np.zeros((len(times), len(node_type)))
    isj = node_type == api.EN_JUNCTION
    for s, t in enumerate(times):
        p = by_t.get(int(t))
        if p is not None and eval_h[0] * 3600 <= t < eval_h[1] * 3600:
            W[s, isj] = np.maximum(np.asarray(p.demand)[isj], 0.0) * p.hydstep
    return W / W.sum()


def p2_booster(t):
    """One booster design re-simulated at Ctol 0 and 0.01; coverage = demand-weighted share of node-hours in [0.2, 4] mg/L."""
    net_name, cand_ids, u, label = t
    from .exp1_booster import CONFIGS, apply_g4_theta
    from .calib_data import prepare_inp
    cfg, lib, wd = CONFIGS[net_name], _W['lib'], _W['wd']
    inp = prepare_inp(cfg['dataset'], wd) if 'dataset' in cfg else cfg['inp']
    if cfg.get('inp_edit'):
        inp = cfg['inp_edit'](inp, wd, f'{net_name}_{os.getpid()}')
    pr = api.Project(lib, inp, str(Path(wd) / f'b_{os.getpid()}.rpt'))
    for f in cfg['setup']:
        f(pr)
    if net_name == 'NorthPenn':
        apply_g4_theta(pr, ROOT / 'research/water_quality_2026_09_12/g4_graph_5090/g4_graph_NorthPenn_cuda.json')
    if cfg['duration_h']:
        pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
    u = np.asarray(u)
    for j in np.flatnonzero(u.sum(1) > 0):
        ur = float(u[j].max())
        pr.set_source(pr.node_index(cand_ids[j]), api.EN_MASS, ur, pr.add_pattern(f'b{j}', (u[j] / ur).tolist()))
    hyd = Path(wd) / f'b_{os.getpid()}.hyd'
    pr.solve_hydraulics(hyd)
    trace = read_hydfile(hyd)
    nt = np.array([0] + [pr.node_type(i) for i in range(1, pr.count(api.EN_NODECOUNT) + 1)])
    out = {'net': net_name, 'design': label}
    for ct in (0.0, 0.01):
        pr.set_option(api.EN_TOLERANCE, ct)
        times, q, _ = pr.run_quality_reference('nextQ')
        W = coverage_W(trace, times, nt, cfg['eval_h'])
        m = W > 0
        out[f'ctol{ct}'] = {'coverage': float((W * ((q >= 0.2) & (q <= 4.0))).sum()), 'min_c_eval': float(q[m].min()), 'q': q}
    d = np.abs(out['ctol0.01'].pop('q') - out['ctol0.0'].pop('q'))
    out['max_abs_diff_mgL'] = float(d.max())
    pr.__exit__()
    return out


def p2_source(t):
    """One contamination scenario at Ctol 0 and 0.01: noise-free first alarm (2 readings >= 0.5 mg/L) at the sensors."""
    net, node_id, start_h, dur_h, rate, sensors = t
    from .exp2_source import CONFIGS
    cfg, lib, wd = CONFIGS[net], _W['lib'], _W['wd']
    pr = api.Project(lib, cfg['inp'], str(Path(wd) / f's_{os.getpid()}.rpt'))
    qual_type(api.EN_CHEM)(pr); init_qual(0.0, reservoirs=0.0)(pr); all_links(kb=0.0, kw=0.0)(pr); all_tanks_kb(0.0)(pr)
    pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
    if cfg['unbalanced'] is not None:
        pr.set_option(api.EN_UNBALANCED, cfg['unbalanced'])
    if cfg['slot_s'] < pr.time_param(api.EN_PATTERNSTEP):
        pr.resample_patterns(cfg['slot_s'])
    slot = pr.time_param(api.EN_PATTERNSTEP); K = int(pr.time_param(api.EN_DURATION) // slot)
    pat = np.zeros(K); k0 = int(round(start_h * 3600 / slot)); pat[k0:k0 + int(round(dur_h * 3600 / slot))] = 1.0
    pr.set_source(pr.node_index(node_id), api.EN_MASS, float(rate), pr.add_pattern('inj', pat.tolist()))
    pr.solve_hydraulics(Path(wd) / f's_{os.getpid()}.hyd')
    sidx = [pr.node_index(s) for s in sensors]
    out = {'node': node_id, 'start_h': start_h}
    series = {}
    for ct in (0.0, 0.01):
        pr.set_option(api.EN_TOLERANCE, ct)
        times, q, _ = pr.run_quality_reference('nextQ')
        c = q[:, sidx]
        hit = c >= 0.5
        conf = hit[1:] & hit[:-1]
        tt = np.where(conf.any(1))[0]
        out[f'alarm_h_ctol{ct}'] = float(times[1 + tt[0]] / 3600) if tt.size else None
        series[ct] = (c, q)
    out['max_abs_diff_sensors_mgL'] = float(np.abs(series[0.01][0] - series[0.0][0]).max())
    out['max_abs_diff_all_mgL'] = float(np.abs(series[0.01][1] - series[0.0][1]).max())
    out['max_c_mgL'] = float(np.abs(series[0.0][1]).max())
    pr.__exit__()
    return out


def p2_exp3(t):
    net, kw_truth, sensors = t
    from .exp3_pipe_kw import CONFIGS
    cfg, lib, wd = CONFIGS[net], _W['lib'], _W['wd']
    pr = api.Project(lib, cfg['inp'], str(Path(wd) / f'e3_{os.getpid()}.rpt'))
    qual_type(api.EN_CHEM)(pr); init_qual(0.5, reservoirs=1.0)(pr); all_links(kb=-0.5, kw=0.0)(pr); all_tanks_kb(-0.5)(pr)
    pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
    if cfg['unbalanced'] is not None:
        pr.set_option(api.EN_UNBALANCED, cfg['unbalanced'])
    pipes = [k for k in range(1, pr.count(api.EN_LINKCOUNT) + 1) if pr.link_type(k) <= api.EN_PIPE]
    for k, v in zip(pipes, kw_truth):
        pr.set_link_value(k, api.EN_KWALL, float(v))
    pr.solve_hydraulics(Path(wd) / f'e3_{os.getpid()}.hyd')
    sidx = [pr.node_index(s) for s in sensors]
    res = {}
    for ct in (0.0, 0.01):
        pr.set_option(api.EN_TOLERANCE, ct)
        times, q, _ = pr.run_quality_reference('stepQ')
        keep = np.flatnonzero(np.asarray(times) % 900 == 0)
        res[ct] = q[keep]
    pr.__exit__()
    d = np.abs(res[0.01] - res[0.0])
    return {'net': net, 'sensors': pair(res[0.01][:, sidx], res[0.0][:, sidx]), 'all_nodes': pair(res[0.01], res[0.0]),
            'noise_sd_mgL': 0.02, 'frac_sensor_readings_diff_gt_noise': float(np.mean(d[:, sidx] > 0.02))}


def part2(a, pool, wd):
    from .calib import CalibProblem
    R, RES = {}, ROOT / 'research/water_quality_2026_09_12'
    lib = api.Library(a.lib)
    H = json.load(open(RES / 'h2_5090/h2_uq.json'))
    cal = []
    for d in H['datasets']:
        P = CalibProblem(d['dataset'], lib, wd, device='cpu', verbose=False)
        th = np.array(d['fit']['theta'])
        r0, r1 = (P.simulate_dll(th, ct) - P.obs.y for ct in (0.0, 0.01))
        rec = {'dataset': d['dataset'], 'rmse_ctol0': float(np.sqrt(np.mean(r0 ** 2))), 'rmse_ctol001': float(np.sqrt(np.mean(r1 ** 2))),
               'max_abs_pred_diff_mgL': float(np.abs(r1 - r0).max())}
        held0, held1 = np.zeros(len(P.obs.y)), np.zeros(len(P.obs.y))
        for s, f in enumerate(d['loso']['folds']):
            m = P.obs.site == s
            held0[m] = (P.simulate_dll(np.array(f['theta']), 0.0) - P.obs.y)[m]
            held1[m] = (P.simulate_dll(np.array(f['theta']), 0.01) - P.obs.y)[m]
        rec.update(loso_rmse_ctol0=float(np.sqrt(np.mean(held0 ** 2))), loso_rmse_ctol001=float(np.sqrt(np.mean(held1 ** 2))))
        cal.append(rec); print('[2 calib]', rec, flush=True)
    R['calibration'] = cal
    tasks = []
    for net in ('Net3', 'NorthPenn'):
        n = json.load(open(RES / f'final_5090/exp1/exp1_{net}.json'))['networks'][0]
        for r in n['refit']:
            tasks.append((net, n['candidates'], r['u'], f"lasso_{r['lam_frac']}"))
        g = n['greedy'][-1]
        if 'refit' in g:
            tasks.append((net, n['candidates'], g['refit']['u'], f"greedy_{g['stage']}"))
    R['booster'] = pool.map(p2_booster, tasks, chunksize=1)
    print('[2 booster] max |dcoverage| %.2e' % max(abs(r['ctol0.01']['coverage'] - r['ctol0.0']['coverage']) for r in R['booster']), flush=True)
    src_tasks = []
    for net, f in (('BWSN2', 'final_v2/exp2_v2_c2_200.json'), ('LTown', 'final_v2/exp2_v2_lt10.json')):
        if not (RES / f).exists():
            print('missing', f, flush=True); continue
        J = json.load(open(RES / f))
        src_tasks += [(net, s[0], s[1], s[2], s[3], J['placement']['sensors']) for s in J['v2']['scenarios_all']]
    sr = pool.map(p2_source, src_tasks, chunksize=1)
    R['source'] = [dict(r, net=t[0]) for r, t in zip(sr, src_tasks)]
    e3 = []
    for net in ('LTown', 'BWSN2'):
        n = json.load(open(RES / f'final_5090/exp3/exp3_{net}.json'))['networks'][0]
        e3.append((net, n['truth']['kw'], n['sets']['random']['sensors']))
    R['exp3'] = pool.map(p2_exp3, e3, chunksize=1)
    return R


# ------------------------------------------------------------------ part 3
def part3(a, wd, nets):
    import torch
    from .network import extract
    from .replicate import Replicator
    from .torch_wq import TorchWQ
    from .calib_data import prepare_inp
    lib = api.Library(a.lib)
    paths = {'Net1': ROOT / CORE['Net1'], 'Net2': ROOT / CORE['Net2'], 'Net3': ROOT / CORE['Net3'], 'Jilin': ROOT / CORE['Jilin'],
             'Fairfield': prepare_inp('Fairfield', wd), 'Bellingham': prepare_inp('Bellingham', wd), 'NorthPenn': prepare_inp('NorthPenn', wd),
             'LTown': ROOT / 'networks/L-Town/L-TOWN.inp',
             'BWSN2': ROOT / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp'}
    out = []
    for name in nets:
        for dur in ((48,) if name == 'BWSN2' else (48, 168)):
            for ct in (0.0, 0.01):
                item = {'inp': str(paths[name]), 'unbalanced': 10 if name == 'BWSN2' else None}
                pr = open_net(lib, item, wd, 'p3')
                apply_cond(pr, 'cl_b05_w05'); pr.set_time_param(api.EN_DURATION, dur * 3600); pr.set_option(api.EN_TOLERANCE, ct)
                hyd = Path(wd) / f'p3_{name}.hyd'
                pr.solve_hydraulics(hyd)
                t0 = time.perf_counter(); pr.run_quality_reference('nextQ'); t_lib = time.perf_counter() - t0
                net = extract(pr); pr.__exit__()
                rep = Replicator(net, read_hydfile(hyd), ctol=net.ctol, record=True)
                t0 = time.perf_counter(); rep.run('nextQ'); t_rec = time.perf_counter() - t0
                sch = rep.schedule
                ev = np.concatenate([np.stack([sch.ver_birth, np.ones(sch.nver)], 1), np.stack([sch.ver_death, -np.ones(sch.nver)], 1)])
                ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
                live = int(np.cumsum(ev[:, 1]).max())
                rec = {'net': name, 'duration_h': dur, 'ctol': ct, 'nodes': int(net.nnodes), 'links': int(net.nlinks),
                       'steps': int(sch.nsteps), 'versions': int(sch.nver), 'ops': len(sch.ops),
                       'terms': int(sum(len(v[0]) for op in sch.ops for v in op.terms.values())),
                       'segments_created': int(rep.hits.get('outflow_newseg', 0)), 'merges': int(rep.hits.get('outflow_merge', 0)),
                       'peak_live_versions': live, 'library_run_s': t_lib, 'record_s': t_rec}
                t0 = time.perf_counter(); eng = TorchWQ(net, sch, rep.reactflag, device='cuda', engine='fused'); rec['build_s'] = time.perf_counter() - t0
                rec['fused_terms'] = int(eng.T_total)
                th = {k: torch.tensor(v, dtype=torch.float64, device='cuda').view(1, -1).requires_grad_(True) for k, v in
                      {'kb': net.kb * 86400, 'kw': net.kw * 86400, 'tank_kb': net.tank_kb * 86400}.items()}
                ts = []
                for r in range(4):
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
                    eng.forward(**th)[0].sum().backward(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                rec['replay_fwd_bwd_s'] = float(np.median(ts[1:])); rec['gpu_peak_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
                out.append(rec); print('[3]', rec, flush=True)
                del eng, rep, sch; torch.cuda.empty_cache()
                yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['1', '1f', '2', '3'])
    ap.add_argument('--lib', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--nets', default='')
    ap.add_argument('--reverse', action='store_true', help='part 1f: run the task list from the end (second machine)')
    ap.add_argument('--log-tag', default='', help='part 1f: suffix of the jsonl log')
    ap.add_argument('--only-keys', default=None, help='part 1f: run only the task keys (net|cond|duration_h|kind) listed in this file')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wd = tempfile.mkdtemp(prefix=f'ct{a.part}_', dir=str(out))
    t0 = time.perf_counter()
    tag = a.part + ('_' + a.nets.replace(',', '_') if a.nets else '') + a.log_tag
    save = lambda R: (out / f'ctol_part{tag}.json').write_text(json.dumps(R, default=float), encoding='utf-8')
    if a.part == '3':
        for R in part3(a, wd, a.nets.split(',')):
            save({'records': R, 'wall_s': time.perf_counter() - t0})
    else:
        pool = get_context('spawn' if os.name == 'nt' else 'fork').Pool(a.workers, initializer=_w_init, initargs=(a.lib, wd))
        R = part1(a, pool, wd) if a.part == '1' else (part1f(a, pool, wd, out) if a.part == '1f' else part2(a, pool, wd))
        R['wall_s'] = time.perf_counter() - t0
        save(R); pool.close(); pool.join()
    print('done part', tag, f'{time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
