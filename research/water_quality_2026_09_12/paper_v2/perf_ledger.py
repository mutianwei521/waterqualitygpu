"""Performance and memory ledger (R1-4, R1-5, R2-4, R2-6): read-only summary of existing result JSON files.
No computation is re-run. Every time is labelled with the hardware it was measured on; every finite-difference cost is
labelled measured / projected-from-one-run / projected-from-batched-replay. Output: paper_v2/tables/perf_ledger.md,
paper_v2/figures/out/figS_perf_scaling.{pdf,png}.
Run from research/water_quality_2026_09_12:  python paper_v2/perf_ledger.py"""
import glob, json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[1]
OUT_T = RES / 'paper_v2' / 'tables'; OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F = RES / 'paper_v2' / 'figures' / 'out'; OUT_F.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(RES / p, encoding='utf-8'))
H5090, H3090, HV100 = 'RTX 5090', 'RTX 3090', 'V100'
f3 = lambda x: '' if x is None else (f'{x:.3g}' if isinstance(x, float) else str(x))
pts = []                                         # (hardware, label, fused_terms, B, fwd_bwd_s, peak_MB)


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(f3(c) for c in r) + ' |' for r in rows])


# ---------------- A: scale and preprocessing ----------------
A = []
g2 = J('v2_5090/g2_fused.json')
for r in g2['results']:
    if 'seconds' in r:
        A.append(['G2 ' + r['network'], H5090, None, r['steps'], r['versions'], r['terms'], r['fused_terms'], r['seconds']['record'], r['seconds']['build']])
for f, lab in (('final_5090/g3s/g3_shared_LTown_stepQ.json', 'L-Town stepQ 48 h'), ('final_5090/g3s/g3_shared_BWSN2_stepQ.json', 'BWSN2 stepQ 48 h')):
    c = J(f)['results'][0]; e = c['engines']['fused']
    A.append([lab, H5090, c['nodes'], c['steps'], c['versions'], None, e['terms'], c['record_s'], e['build_s']])
for f, lab in (('final_5090/exp1/exp1_Net3.json', 'Exp1 Net3 168 h'), ('final_5090/exp1/exp1_NorthPenn.json', 'Exp1 North Penn 36 h'),
               ('final_5090/exp2/exp2_LTown.json', 'Exp2 L-Town 48 h'), ('final_5090/exp2/exp2_BWSN2.json', 'Exp2 BWSN2 48 h'),
               ('final_5090/exp3/exp3_LTown.json', 'Exp3 L-Town 48 h'), ('final_5090/exp3/exp3_BWSN2.json', 'Exp3 BWSN2 48 h')):
    n = J(f)['networks'][0]; s, t = n['sizes'], n['timing']
    A.append([lab, H5090, s.get('nodes'), s.get('steps'), s.get('versions'), None, s.get('fused_terms'), t.get('record_s'), t.get('build_s')])
for d in J('h2_5090/h2_uq.json')['datasets']:
    s = d['schedule']; A.append([f"Calibration {d['dataset']}", H5090, None, s['steps'], s['versions'], None, None, s['record_s'], s['build_s']])
p3 = sorted(glob.glob(str(RES / 'final_5090_v2' / 'r1_B_*' / 'ctol_part3_*.json')))       # re-measured on the 5090 (job B, 2026-10-22)
P3 = [r for f in p3 for r in json.load(open(f))['records']]
for r in P3:
    A.append([f"{r['net']} {r['duration_h']} h Ctol {r['ctol']}", H5090, r['nodes'], r['steps'], r['versions'], r['terms'], r['fused_terms'],
              r['record_s'], r['build_s']])
    pts.append((H5090, f"{r['net']} {r['duration_h']}h c{r['ctol']}", r['fused_terms'], 1, r['replay_fwd_bwd_s'], r['gpu_peak_MB']))

# ---------------- B: execution and gradient ----------------
B = []
for c in J('v2_5090/g3_bench_v2_cuda.json')['results']:
    for eng in ('adjoint', 'fused'):
        e = c['engines'][eng]; b1 = next(b for b in e['batches'] if b['B'] == 1)
        ok = [b for b in e['batches'] if 'forward_backward_s' in b]
        bmax = max(ok, key=lambda b: b['B'])
        B.append([c['network'], eng, H5090, c['n_theta'], c['dll_quality_run_s'], b1['forward_backward_s'], b1.get('graph_forward_backward_s'),
                  b1.get('peak_mem_MB'), bmax['B'], bmax.get('graph_forward_backward_s', bmax['forward_backward_s']) / bmax['B'], bmax.get('graph_peak_mem_MB', bmax.get('peak_mem_MB'))])
        if eng == 'fused':
            for b in ok:
                pts.append((H5090, c['network'], e.get('terms') or c.get('terms'), b['B'], b['forward_backward_s'], b.get('peak_mem_MB')))
for f, lab in (('final_5090/g3s/g3_shared_LTown_stepQ.json', 'L-Town'), ('final_5090/g3s/g3_shared_BWSN2_stepQ.json', 'BWSN2')):
    c = J(f)['results'][0]
    for eng in ('fused_gather', 'fused_spmm', 'fused'):
        e = c['engines'][eng]
        for b in e['batches']:
            if 'forward_backward_s' in b and eng == 'fused_gather':     # SpMM engines hold sparse structures built for shared batches
                pts.append((H5090, f'{lab} {eng}', e['terms'], b['B'], b['forward_backward_s'], b.get('peak_mem_MB')))
        b1 = next(b for b in e['batches'] if b['B'] == 1)
        oom = [b['B'] for b in e['batches'] if 'error' in b]
        B.append([f'{lab} shared', eng, H5090, c['n_theta'], c['dll_quality_run_s'], b1['forward_backward_s'], b1.get('graph_forward_backward_s'),
                  b1.get('peak_mem_MB'), max(b['B'] for b in e['batches'] if 'forward_backward_s' in b), None, f'OOM at B={oom}' if oom else None])

# calibration: graph capture, per-gradient, full fits (all measured, 5090)
CAL = []
for f in sorted(glob.glob(str(RES / 'g4_graph_5090' / 'g4_graph_*_cuda.json'))):
    d = json.load(open(f))['datasets'][0]
    walls = {}
    for r in d['runs']:
        walls.setdefault(r['method'], []).append(r['wall_s'])
    tm = d.get('timing', {})
    CAL.append([Path(f).stem.replace('g4_graph_', '').replace('_cuda', ''), H5090, d['graph'].get('capture_s'),
                tm.get('grad_eager_s', tm.get('adjoint_grad_B1_s')), tm.get('grad_graph_s'), tm.get('dll_ctol0_s'),
                np.median(walls.get('FD-LBFGSB', [np.nan])), np.median(walls.get('ADJ-LBFGSB', [np.nan])), np.median(walls.get('ADJ-LBFGSB-graph', [np.nan]))])

# applications: full-task wall clock and its parts (5090; v2 exp2 on 3090 is listed separately)
APP = []
for f, lab in (('final_5090/exp1/exp1_Net3.json', 'Exp1 Net3'), ('final_5090/exp1/exp1_NorthPenn.json', 'Exp1 North Penn'),
               ('final_5090/exp2/exp2_LTown.json', 'Exp2 L-Town (v1, 10 sensors)'), ('final_5090/exp2/exp2_BWSN2.json', 'Exp2 BWSN2 (v1, 20 sensors)'),
               ('final_5090/exp3/exp3_LTown.json', 'Exp3 L-Town'), ('final_5090/exp3/exp3_BWSN2.json', 'Exp3 BWSN2')):
    t = J(f)['networks'][0]['timing']
    APP.append([lab, H5090, t.get('record_s'), t.get('build_s'), t.get('graph_capture_s'), t.get('total_s'),
                ', '.join(f'{k} {v:.0f}' for k, v in t.items() if k.endswith('_s') and k not in ('record_s', 'build_s', 'graph_capture_s', 'total_s', 'setup_s') and v > 1)])
for f in sorted(glob.glob(str(RES / 'final_5090_v2' / 'r1_A*' / 'exp2_v2_*.json'))):     # v2 Exp2 as reported (5090, jobs A1, A2)
    R = json.load(open(f)); t = R['timing']
    whole = t['setup_s'] + R['v2']['placement_v2_s'] + t.get('data_s', 0) + t['identification_s']
    APP.append([f"Exp2 v2 {R['v2']['config']['tag']} ({len(R['placement']['sensors'])} sensors)", H5090, t.get('record_s'), t.get('build_s'),
                t.get('graph_capture_s'), whole, f"sweep {t.get('sweep_s', 0):.0f}, screen {t.get('screen_s', 0):.0f}, refine {t.get('refine_s', 0):.0f}, "
                f"NNLS check {t.get('exhaustive_s', 0):.0f}"])
for f in sorted(glob.glob(str(RES / 'final_v2' / 'exp2_v2_*.json'))):       # sensor-count comparison (3090; accuracy only)
    R = json.load(open(f)); t = R['timing']
    APP.append([f"Exp2 v2 {R['v2']['config']['tag']} ({len(R['placement']['sensors'])} sensors)", H3090, t.get('record_s'), t.get('build_s'),
                t.get('graph_capture_s'), t.get('total_s'), f"sweep {t.get('sweep_s', 0):.0f}, refine {t.get('refine_s', 0):.0f}"])

# ---------------- C: provenance of every finite-difference cost ----------------
C = [['Calibration FD L-BFGS-B fits (Ctol 0 and 0.01)', 'measured', 'library called N times inside the optimiser', 'g4_graph_5090, h2_5090'],
     ['FD step sweep 1e-1 to 1e-7 (Table S2)', 'measured', 'two library runs per step and direction', 'final_5090/fds'],
     ['Engine benchmark FD gradient 22-887 ms (Table S3)', 'projected from one run', '(n_theta + 1) x measured library run', 'bench.py line 100'],
     ['Ratio to FD at B_max 2,000-6,100 (Table S3, TOC)', 'projected from one run', 'projected FD gradient / measured per-set replay cost', 'bench.py'],
     ['Exp1 library cost of the response columns', 'projected from one run', 'candidates x slots x library run', 'exp1_booster.py line 558'],
     ['Exp2 exhaustive library cost 9e3-2e4 s / 6e6-1e7 s', 'projected from one run', 'junctions x admissible slots x library run', 'exp2_source.py line 538'],
     ['Exp3 FD gradient with the library 3e5 s, 4.3e6-fold', 'projected from one run', '(P + 1) x library run', 'exp3_pipe_kw.py line 399'],
     ['Exp3 FD gradient with the batched replay 89 s', 'projected from batched replay', 'passes x measured batched forward', 'exp3_pipe_kw.py line 400'],
     ['Parallel FD (R2-4)', 'measured', '1, 2, 4 and 8 worker processes on the 8 cores of one 5090 allocation; large cases timed on a column subset and scaled', 'final_5090_v2/r1_T4_*/fd_parallel.json']]

# ---------------- D: scaling fits ----------------
fit = {}
for hw in (H5090, HV100):
    q = np.array([(t, b, s, m) for h, _, t, b, s, m in pts if h == hw and t and s and m and b == 1], dtype=float)
    if len(q) >= 3:
        lm = np.polyfit(np.log10(q[:, 0]), np.log10(q[:, 3]), 1); ls = np.polyfit(np.log10(q[:, 0]), np.log10(q[:, 2]), 1)
        fit[hw] = {'n': len(q), 'mem_exponent': lm[0], 'MB_per_Mterm_at_1e7': 10 ** np.polyval(lm, 7) / 10, 'time_exponent': ls[0],
                   'time_s_at_1e7_terms': 10 ** np.polyval(ls, 7)}
bw = [r for r in P3 if r['net'] == 'BWSN2' and r['ctol'] == 0.0]
TS = [[r['net'], r['ctol'], r['steps'] if r['duration_h'] == 48 else None, r['replay_fwd_bwd_s'] if r['duration_h'] == 48 else None, None, None] for r in P3 if r['duration_h'] == 48]
for row in TS:
    r2 = next((r for r in P3 if r['net'] == row[0] and r['ctol'] == row[1] and r['duration_h'] == 168), None)
    if r2:
        row[4], row[5] = r2['steps'] / row[2], r2['replay_fwd_bwd_s'] / row[3]
LONG = [json.load(open(f)) for f in sorted(glob.glob(str(RES / 'final_5090_v2' / 'r1_C_*' / 'bwsn2_*h.json')), key=lambda f: int(Path(f).stem.split('_')[1][:-1]))]
mb = fit[H5090]['MB_per_Mterm_at_1e7']
T48, V48, K48, N1 = 59856795, 8835017, 192, 12528
EX = [['BWSN2 replay, B = 1 (fused terms x MB per term)', f'{T48/1e6:.0f} M terms: {T48/1e6*mb/1024:.1f} GB', f'{3.5*T48/1e6:.0f} M terms: {3.5*T48/1e6*mb/1024:.1f} GB'],
      ['Exp2 adjoint state + cotangent, 32 rows (2 x M x 32 x 8 B)', f'{2*(V48+(3+K48)*N1)*32*8/2**30:.1f} GB', f'{2*(3.5*V48+(3+4*K48)*N1)*32*8/2**30:.1f} GB'],
      ['Exp2 normal equations of every junction (N x K^2 x 8 B)', f'{(N1-5)*K48**2*8/2**30:.1f} GB', f'{(N1-5)*(4*K48)**2*8/2**30:.1f} GB']]

# ---------------- write ----------------
md = ['# Performance and memory ledger (read-only summary of existing results)',
      'Times are only comparable within one hardware column. 5090 = paper; the tolerance study part 3 (B), the BWSN2 long horizons (C) '
      'and the parallel finite differences (T4) were re-measured on the 5090 node on 2026-10-22 (final_5090_v2/).', '',
      '## A. Scale and preprocessing', table(['case', 'hardware', 'nodes', 'steps', 'versions', 'recorded terms', 'fused terms', 'record s', 'build s'], A), '',
      '## B. Engine execution, B = 1 and largest batch', table(['case', 'engine', 'hardware', 'n_theta', 'library run s', 'fwd+adj B=1 eager s',
      'fwd+adj B=1 graph s', 'peak MB B=1', 'largest B', 'per-set s at largest B', 'peak MB at largest B / note'], B), '',
      '## B2. Calibration: graph capture, one gradient, complete fits (all measured)', table(['data set', 'hardware', 'graph capture s', 'gradient eager s',
      'gradient graph s', 'library run s', 'FD fit median s', 'adjoint eager fit s', 'adjoint graph fit s'], CAL), '',
      '## B3. Applications: complete-task wall clock', table(['task', 'hardware', 'record s', 'build s', 'graph capture s', 'total s', 'main parts (s)'], APP), '',
      '## C. Provenance of finite-difference costs', table(['quantity', 'label', 'how obtained', 'source'], C), '',
      '## D2. One-gradient time follows the number of quality steps (RTX 5090, B = 1, eager, median of 3 warm runs)', table(['network', 'Ctol', 'steps 48 h', 'fwd+adj 48 h s', 'steps ratio 168/48 h', 'time ratio 168/48 h'], TS), '',
      '## D3. BWSN2 Exp2 schedule beyond 48 h (measured on one RTX 5090 32 GB; normal equations sized from the recorded schedule, not allocated; fwd+adj is a single cold run)',
      table(['hours', 'slots K', 'versions', 'fused terms', 'host peak RSS GB', 'record s', 'build s', 'GPU peak fwd+adj B=1 GB', 'fwd+adj B=1 s',
             'GPU peak adjoint row pass (32 rows) GB', 'normal equations GB (sized)'],
            [[r['hours'], r['K_slots'], r['versions'], r['fused_terms'], r['host_peak_rss_GB'], r['record_s'], r['build_s'], r['gpu_peak_fwd_adj_B1_GB'],
              r['fwd_adj_B1_s'], r.get('gpu_peak_row_pass_GB') or ('OOM' if r.get('row_pass_error') else None),
              r['sized_not_allocated']['normal_equations_all_junctions_GB']] for r in LONG]), '',
      '## D. Scaling fits (B = 1, log-log; memory: gather path only)', table(['hardware', 'points', 'memory exponent vs fused terms', 'MB per million terms at 1e7 terms', 'time exponent', 's at 1e7 terms'],
      [[h, v['n'], v['mem_exponent'], v['MB_per_Mterm_at_1e7'], v['time_exponent'], v['time_s_at_1e7_terms']] for h, v in fit.items()]), '',
      '## Limits that must be stated', '- Time-chunked replay is not implemented and has no benchmark; it must not be described as a validated capability.',
      '- Largest validated problem for the full source identification: BWSN2, 48 h, about 6e7 fused terms, 32 GB (Exp2 needed 32 rows per adjoint pass after an OOM at 64).',
      '- Measured beyond 48 h (D3): one B = 1 gradient of the 168 h BWSN2 schedule (2.1e8 terms) fits in 12.6 GB, but the identification does not: '
      'the adjoint row pass runs out of memory at 32 rows and the normal equations alone would need 42 GB. Longer identification needs node-split '
      'normal equations (several GPUs or host memory) or time-chunked replay, neither of which is implemented.',
      '- Any statement about longer horizons or finer steps is an extrapolation of the fits in D, labelled as such.']
if bw:
    md.append(f"- 5090 BWSN2 48 h, Ctol 0: {bw[0]['fused_terms']:,} fused terms, B = 1 forward+adjoint peak {bw[0]['gpu_peak_MB']:.0f} MB.")
(OUT_T / 'perf_ledger.md').write_text('\n'.join(md), encoding='utf-8')
json.dump({'fits': fit, 'points': pts}, open(OUT_T / 'perf_ledger_points.json', 'w'), default=float)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for hw, col in ((H5090, '#6a51a3'), (HV100, '#2c7fb8')):
    q = [(t, b, s, m) for h, _, t, b, s, m in pts if h == hw and t and s and b == 1]
    if q:
        q = np.array(q, dtype=float)
        ax[0].loglog(q[:, 0], q[:, 3], 'o', ms=4, color=col, label=hw); ax[1].loglog(q[:, 0], q[:, 2], 'o', ms=4, color=col, label=hw)
for gb in (24, 32):
    ax[0].axhline(gb * 1024, ls='--', lw=0.8, color='0.5'); ax[0].text(1.2e4, gb * 1024 * 1.08, f'{gb} GB', fontsize=7, color='0.4')
ax[0].set_xlabel('fused terms'); ax[0].set_ylabel('peak GPU memory, B = 1 (MB)'); ax[1].set_xlabel('fused terms'); ax[1].set_ylabel('forward + adjoint, B = 1 (s)')
for a in ax:
    a.spines[['top', 'right']].set_visible(False); a.legend(frameon=False, fontsize=7)
ax[0].text(-0.18, 1.02, 'a', transform=ax[0].transAxes, fontweight='bold'); ax[1].text(-0.18, 1.02, 'b', transform=ax[1].transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT_F / 'figS_perf_scaling.pdf'); fig.savefig(OUT_F / 'figS_perf_scaling.png', dpi=300)
print('wrote', OUT_T / 'perf_ledger.md', '| fits', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in fit.items()})
