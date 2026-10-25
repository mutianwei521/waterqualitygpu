"""R1-1 / R2-2: SI tables of the merging-tolerance study (Ctol = 0 versus the EPANET default 0.01).

Reads final_ctol/ctol_part1f_merged.jsonl (3090, EPANET library runs on 55 networks), final_ctol/ctol_part2.json
(application results recomputed at Ctol = 0.01) and final_5090_v2/r1_B_*/ctol_part3_*.json (5090: schedule size and
cost). Writes paper_v2/tables/T_S_ctol.md and paper_v2/tables/ctol_numbers.json.
"""
import glob
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'tables'
CONDS = [('native', 'native INP quality'), ('cl_k0', 'Cl, kb = 0, kw = 0'), ('cl_b05', 'Cl, kb = -0.5'), ('cl_b2', 'Cl, kb = -2'),
         ('cl_w05', 'Cl, kw = -0.5'), ('cl_b05_w05', 'Cl, kb = -0.5, kw = -0.5'), ('cl_b2_w05', 'Cl, kb = -2, kw = -0.5'),
         ('age', 'water age'), ('trace', 'source trace')]
DURS = [(None, 'INP'), (48, '48 h'), (168, '168 h')]
NUM = {}


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]) + '\n'


def mm(v, f='{:.1e}'):
    """median (max)"""
    return f'{f.format(np.median(v))} ({f.format(np.max(v))})' if len(v) else '-'


def rel(x):
    return {k: x[k] / x['scale'] for k in ('max', 'p99', 'rmse')} if x['scale'] > 0 else {k: 0.0 for k in ('max', 'p99', 'rmse')}


SAME_FILE = {'FOS': 'foss_poly_1', 'Net3-NH2CL': 'net3-bio'}    # byte-identical input files under two names (identical results): counted once
R = [r for r in (json.loads(l) for l in open(ROOT / 'final_ctol' / 'ctol_part1f_merged.jsonl', encoding='utf-8')) if r['net'] not in SAME_FILE]
main = {(r['net'], r['cond'], r['duration_h']): r for r in R if r['kind'] == 'main' and 'q60' in r}
ref = {(r['net'], r['cond'], r['duration_h']): r for r in R if r['kind'] == 'ref' and 'ctol0_vs_ref' in r}
nets = sorted({k[0] for k in main})
blocks = [f'# Merging tolerance: Ctol = 0 versus the default 0.01 ({len(nets)} distinct networks, EPANET 2.3 library)\n',
          'Input files that occur twice in the corpus, or under two names with identical content, are counted once.\n',
          'Δ is the absolute difference over all nodes and reporting times, divided by the largest concentration (age, trace '
          'percentage) of the Ctol = 0 run of the same case ("peak"); entries are median (maximum) over networks. Chlorine '
          'cases start from 0.5 mg/L with reservoirs at 1.0 mg/L, so the peak is 1 mg/L and relative values equal mg/L.\n']

rows, A = [], {}
for c, cl in CONDS:
    for d, dl in DURS:
        rs = [r for k, r in main.items() if k[1] == c and k[2] == d]
        if not rs:
            continue
        x = [r[f"q{r['qstep_input']}"] for r in rs]
        rr = [rel(v['ctol001_vs_ctol0']) for v in x]
        r60 = [rel(r['q60']['ctol001_vs_ctol0']) for r in rs]
        gt = [v['ctol001_vs_ctol0']['frac_gt_0.01'] for v in x]
        dmb = [abs(v['mb_ctol0'] - v['mb_ctol001']) for v in x]
        A[(c, d)] = rr
        rows.append([cl, dl, len(rs), mm([q['max'] for q in rr]), mm([q['p99'] for q in rr]), mm([q['rmse'] for q in rr]),
                     mm([q['rmse'] for q in r60]), f'{100 * max(gt):.1f}%', f'{max(dmb):.0e}'])
blocks.append('## S-ctol-1. Effect of disabling merging on the simulated quality (same hydraulics and quality step)\n')
blocks.append(table(['condition', 'duration', 'networks', 'max Δ / peak', '99th pct Δ / peak', 'RMSE / peak',
                     'RMSE / peak at 60 s step', 'node-times with Δ > 0.01, max', 'max change of mass balance'], rows))
allr = [q for v in A.values() for q in v]
cl_keys = [k for k in A if k[0].startswith('cl_')]
NUM.update(networks=len(nets), cases=len(main),
           rmse_rel_median_cl=float(np.median([q['rmse'] for k in cl_keys for q in A[k]])),
           rmse_rel_max_all=float(max(q['rmse'] for q in allr)), max_rel_max_all=float(max(q['max'] for q in allr)),
           p99_rel_max_all=float(max(q['p99'] for q in allr)),
           max_rel_median_cl=float(np.median([q['max'] for k in cl_keys for q in A[k]])))

rows, ratio_all, closer_cl = [], [], []
for c, cl in CONDS:
    for d, dl in DURS:
        ks = [k for k in ref if k[1] == c and k[2] == d and k in main]
        if not ks:
            continue
        e0 = [rel(ref[k]['ctol0_vs_ref'])['rmse'] for k in ks]
        e1 = [rel(ref[k]['ctol001_vs_ref'])['rmse'] for k in ks]
        closer = [ref[k]['ctol0_vs_ref']['rmse'] <= ref[k]['ctol001_vs_ref']['rmse'] for k in ks]
        tol = [main[k][f"q{main[k]['qstep_input']}"]['ctol001_vs_ctol0']['rmse'] / ref[k]['ctol0_vs_ref']['rmse']
               for k in ks if ref[k]['ctol0_vs_ref']['rmse'] > 1e-6 * ref[k]['ctol0_vs_ref']['scale']]   # step error resolved
        ratio_all += tol
        if c.startswith('cl_'):
            closer_cl += closer
        rows.append([cl, dl, len(ks), mm(e0), mm(e1), f'{100 * np.mean(closer):.0f}%', mm(tol, '{:.2f}')])
blocks.append('\n## S-ctol-2. Accuracy against a fine-step reference (Ctol = 0, 10 s; 60 s for 168 h; 30 s above 5,000 nodes)\n')
blocks.append(table(['condition', 'duration', 'networks', 'RMSE / peak, Ctol = 0', 'RMSE / peak, Ctol = 0.01',
                     'Ctol = 0 at least as close', 'tolerance effect / step error (RMSE ratio, where the step error exceeds 1e-6 of peak)'], rows))
NUM.update(closer_share_cl=float(np.mean(closer_cl)), tol_over_step_median=float(np.median(ratio_all)),
           tol_over_step_p90=float(np.quantile(ratio_all, 0.9)))

P3 = [r for f in sorted(glob.glob(str(ROOT / 'final_5090_v2' / 'r1_B_*' / 'ctol_part3_*.json')))
      for r in json.load(open(f, encoding='utf-8'))['records']]
order = ['Net1', 'Net2', 'Net3', 'Jilin', 'Fairfield', 'Bellingham', 'NorthPenn', 'LTown', 'BWSN2']
by = {(r['net'], r['duration_h'], r['ctol']): r for r in P3}
rows = []
for n in order:
    for d in (48, 168):
        a, b = by.get((n, d, 0.0)), by.get((n, d, 0.01))
        if not a or not b:
            continue

        def f(k, fmt='{:.3g}'):
            return f"{fmt.format(a[k])} / {fmt.format(b[k])}"
        rows.append([n, f'{d} h', a['nodes'], f('library_run_s'), f('record_s'), f('peak_live_versions', '{:,}'),
                     f('fused_terms', '{:,}'), f('replay_fwd_bwd_s'), f('gpu_peak_MB', '{:.0f}')])
        if n == 'BWSN2':
            NUM.update(bwsn2_lib_ratio=a['library_run_s'] / b['library_run_s'], bwsn2_live_ratio=a['peak_live_versions'] / b['peak_live_versions'],
                       bwsn2_terms_ratio=b['fused_terms'] / a['fused_terms'], bwsn2_replay_ratio=b['replay_fwd_bwd_s'] / a['replay_fwd_bwd_s'],
                       bwsn2_mem_ratio=b['gpu_peak_MB'] / a['gpu_peak_MB'], bwsn2_record_ratio=a['record_s'] / b['record_s'])
blocks.append('\n## S-ctol-3. Schedule size and cost, Ctol = 0 / Ctol = 0.01 (RTX 5090 node, chlorine kb = kw = -0.5, B = 1)\n')
blocks.append(table(['network', 'duration', 'nodes', 'library run s', 'record s', 'peak live segments', 'fused terms',
                     'replay forward + adjoint s', 'GPU peak MB'], rows))

P2 = json.load(open(ROOT / 'final_ctol' / 'ctol_part2.json', encoding='utf-8'))
rows = [[c['dataset'], f"{c['rmse_ctol0']:.4f} / {c['rmse_ctol001']:.4f}", f"{c['loso_rmse_ctol0']:.4f} / {c['loso_rmse_ctol001']:.4f}",
         f"{c['max_abs_pred_diff_mgL']:.4f}"] for c in P2['calibration']]
blocks.append('\n## S-ctol-4. Application results recomputed with the library at Ctol = 0.01\n')
blocks.append('Field calibration at the reported optimum (RMSE in mg/L, Ctol = 0 / 0.01):\n\n')
blocks.append(table(['dataset', 'RMSE', 'leave-one-site-out RMSE', 'max |Δ prediction| mg/L'], rows))
bo = P2['booster']
dcov = [abs(b['ctol0.0']['coverage'] - b['ctol0.01']['coverage']) for b in bo]
dmin = [abs(b['ctol0.0']['min_c_eval'] - b['ctol0.01']['min_c_eval']) for b in bo]
src = P2['source']
alarm_diff = [s for s in src if s['alarm_h_ctol0.0'] != s['alarm_h_ctol0.01']]
det_change = [s for s in src if (s['alarm_h_ctol0.0'] is None) != (s['alarm_h_ctol0.01'] is None)]
blocks.append(f"\nBooster designs ({len(bo)} designs on {len(set(b['net'] for b in bo))} networks): coverage changes by at most "
              f"{100 * max(dcov):.2f} percentage points and the minimum evaluated concentration by at most {max(dmin):.4f} mg/L.\n")
blocks.append(f"Source identification: {len(alarm_diff)} of {len(src)} test injections change their first-alarm time and {len(det_change)} "
              f"change detection; the largest sensor difference is {max(s['max_abs_diff_sensors_mgL'] for s in src):.3g} mg/L.\n")
for e in P2['exp3']:
    blocks.append(f"Pipe identifiability ({e['net']}): sensor readings differ by at most {e['sensors']['max']:.4f} mg/L (RMSE "
                  f"{e['sensors']['rmse']:.4f}); {100 * e['frac_sensor_readings_diff_gt_noise']:.2f}% of readings differ by more than the "
                  f"noise sd {e['noise_sd_mgL']} mg/L.\n")
NUM.update(calib_rmse_change_max=max(abs(c['rmse_ctol0'] - c['rmse_ctol001']) for c in P2['calibration']),
           booster_cov_pp_max=100 * max(dcov), source_alarm_changes=len(alarm_diff), source_detect_changes=len(det_change), source_n=len(src))
(OUT / 'T_S_ctol.md').write_text('\n'.join(blocks), encoding='utf-8')
(OUT / 'ctol_numbers.json').write_text(json.dumps(NUM, indent=1), encoding='utf-8')
print(json.dumps(NUM, indent=1))
