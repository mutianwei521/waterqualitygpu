"""R1-3 / R2-3: SI tables of the independent gradient validation (adjoint of the replay versus central differences of
the compiled EPANET library), from the gradcheck_v2 run on the RTX 5090 node.

Usage: python gradcheck_tables.py [RUN_DIR]   (default: final_5090_v2/r1_G_1723889, the run with long-double loss sums)
Writes paper_v2/tables/T_S_gradcheck.md and gradcheck_numbers.json.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'final_5090_v2' / 'r1_G_1723889'
GROUPS = [('A', 'Engine: 25 configurations x 3 parameter points (G2 set)'),
          ('BL', 'Field calibration maps (grouped, per pipe), L-Town per-pipe wall map, slot-source maps of Net3 and L-Town'),
          ('BW', 'BWSN2: per-pipe wall map (14,820 pipes) and slot-source map')]
CLASS = {'smooth': 'smooth branch', 'affine': 'affine (slot sources)', 'kink_kw0': 'wall coefficient perturbed from kw = 0',
         'switch_all_zero_reaction': 'perturbation from the all-zero reaction point', 'setpoint': 'set-point source near its switch',
         'bound': 'parameter on a bound'}


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]) + '\n'


rows, fails, xrows, NUM, meta = [], [], [], {}, {}
for g, desc in GROUPS:
    d = json.load(open(RUN / f'gradcheck_{g}.json', encoding='utf-8'))
    meta[g] = d['meta']
    by = {}
    for m in d['maps']:
        if m['label'].startswith('X_'):
            b = m['boundary']
            xrows.append([b['kb'], f"{b['min_reaction_factor']:.3f}" if 'min_reaction_factor' in b else '-', b['replicator_clamp_hits'],
                          f"{b['max_rel_replay_vs_library_base']:.1e}" if 'max_rel_replay_vs_library_base' in b else 'replay refused (f <= 0)',
                          ', '.join(f"{s['rel_err']:.1e}" for s in b.get('steps', [])) or '-'])
            continue
        for dr in m['directions']:
            by.setdefault(dr['class'], []).append((m['label'], dr))
            if dr['class'] in ('smooth', 'affine') and not dr['pass']:
                e = {s['h']: s['rel_err'] for s in dr['steps']}
                fails.append([g, m['label'], dr['kind'], dr.get('tag') or '-', f"{abs(dr['adjoint']):.2e}", f"{dr['min_err']:.1e}",
                              '-' if dr['slope_1e-2_1e-3'] is None else f"{dr['slope_1e-2_1e-3']:.2f}",
                              ', '.join(f'{h:g}: {v:.1e}' for h, v in sorted(e.items(), reverse=True))])
    for c, lst in by.items():
        mins = np.array([dr['min_err'] for _, dr in lst])
        npass = sum(bool(dr['pass']) for _, dr in lst) if c in ('smooth', 'affine') else None
        npre = sum(bool(dr['pass_preregistered']) for _, dr in lst) if c in ('smooth', 'affine') else None
        has4 = all(any(st['h'] == 1e-4 for st in dr['steps']) for _, dr in lst)
        pre = '-' if npre is None else (f'{npre}' if has4 or c == 'affine' else 'n/a (no 1e-4 step)')
        rows.append([g, CLASS.get(c, c), len(lst), '-' if npass is None else f'{npass}', pre,
                     f'{np.median(mins):.1e}', f'{np.quantile(mins, 0.95):.1e}', f'{mins.max():.1e}'])
        NUM[f'{g}_{c}'] = {'n': len(lst), 'pass': npass, 'pass_pre': npre, 'median_min_err': float(np.median(mins)), 'max_min_err': float(mins.max())}
md = ['# Independent gradient validation: adjoint of the replay versus central differences of the library (R1-3 / R2-3)\n',
      f"Run {RUN.name}: {meta['A']['gpu']}, torch {meta['A']['torch']}, EPANET {meta['A']['epanet']}, seed {meta['A']['seed']}; "
      'library losses summed in long double (eps {:.1e}). Relative error |FD - g.d| / max(|g.d|, |FD|); steps h = 1e-2 to 1e-6 '
      '(BWSN2 per-pipe map: 1e-2, 1e-3, 1e-5). Pass, smooth branch: minimum error <= 1e-6 and central-difference slope between '
      'h = 1e-2 and 1e-3 in [1.7, 2.3] (pre-registered: slope between 1e-2 and 1e-4); affine maps: maximum error <= 1e-9.\n'.format(meta['A'].get('loss_sum_eps', float('nan'))),
      '## S-grad-1. Summary by group and class (errors are the minimum over step sizes of each direction)\n',
      table(['group', 'class', 'directions', 'pass', 'pass (pre-registered slope)', 'median', '95th percentile', 'maximum'], rows),
      '\n## S-grad-2. Smooth and affine directions that did not pass\n',
      table(['group', 'map', 'direction', 'parameter', '|g.d|', 'minimum error', 'slope 1e-2 to 1e-3', 'error by step'], fails) if fails else 'None.\n',
      '\n## S-grad-3. Reaction-factor boundary (Net1, bulk coefficient on all pipes, 300 s step)\n',
      table(['kb (1/d)', 'minimum factor f', 'library clamp hits', 'replay vs library', 'FD error at h = 1e-3, 1e-5'], xrows)]
(HERE / 'tables' / 'T_S_gradcheck.md').write_text('\n'.join(md), encoding='utf-8')
(HERE / 'tables' / 'gradcheck_numbers.json').write_text(json.dumps(NUM, indent=1), encoding='utf-8')
print(json.dumps(NUM, indent=1)); print('failures:', len(fails))
