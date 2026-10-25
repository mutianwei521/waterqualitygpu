"""R1-8: sensitivity of the pipe-identifiability counts to the threshold and to the prior standard deviation.

Threshold scan: a pipe counts as identifiable when posterior sd / prior sd < t, for t in 0.3, 0.5 (paper), 0.7, 0.9,
using the stored per-pipe ratios of the field audits (existing sites and after the five prescribed sites) and of the
synthetic studies (random sensor set). Prior scan: the field audits re-run with prior sd tau = 0.1, 0.3 and 1.0 ft/d
(final_5090_v2/r1_B2_*/exp3_field_tau*.json); reported are the count at t = 0.5, the effective number of determined
pipe parameters p_eff (threshold-free) and the rank correlation of the per-pipe ratios with the tau = 0.3 audit.
Writes paper_v2/tables/T_S_ident.md.
"""
import glob
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TS = (0.3, 0.5, 0.7, 0.9)


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]) + '\n'


def counts(ratio):
    r = np.asarray(ratio, float)
    return ' / '.join(str(int((r < t).sum())) for t in TS)


field = json.load(open(ROOT / 'final_5090' / 'exp3' / 'exp3_field.json', encoding='utf-8'))['field']
rows = []
for x in field:
    after = x['prescription']['greedy']['logdet']['pipe_sd_over_tau_after']
    rows.append([x['dataset'], f"field audit, tau = {x['field_tau_ft_d']} ft/d", len(x['pipe_sd_over_tau']), counts(x['pipe_sd_over_tau']), counts(after)])
for net in ('LTown', 'BWSN2'):
    n = json.load(open(ROOT / 'final_5090' / 'exp3' / f'exp3_{net}.json', encoding='utf-8'))['networks'][0]
    ps = n['posterior_sd']
    ratio = np.asarray(ps['random'], float) / np.asarray(ps['prior_sd'], float)
    rows.append([net, 'synthetic, random sensors, tau = 0.5 x class value', ratio.size, counts(ratio), '-'])
md = ['# Pipe-level identifiability: dependence on the threshold and on the prior (R1-8)\n',
      '## S-ident-1. Identifiable pipes for thresholds posterior sd / prior sd < 0.3 / 0.5 / 0.7 / 0.9\n',
      table(['network', 'setting', 'pipes', 'existing sensors', 'after five prescribed sites'], rows)]

runs = sorted(glob.glob(str(ROOT / 'final_5090_v2' / 'r1_B2_*' / 'exp3_field_tau*.json')))
if runs:
    by = {}
    for f in runs:
        for x in json.load(open(f, encoding='utf-8'))['field']:
            by.setdefault(x['dataset'], {})[x['field_tau_ft_d']] = x
    prow = []
    for ds, taus in by.items():
        ref = np.asarray(taus.get(0.3, next(iter(taus.values())))['pipe_sd_over_tau'], float)
        for tau in sorted(taus):
            x = taus[tau]; a = x['audit']; r = np.asarray(x['pipe_sd_over_tau'], float)
            after = np.asarray(x['prescription']['greedy']['logdet']['pipe_sd_over_tau_after'], float)
            rho = spearmanr(r, ref).correlation if tau != 0.3 else 1.0
            prow.append([ds, tau, a['n_identifiable'], f"{a['p_eff']:.1f}", f"{a['median_sd_over_tau']:.3f}", int((after < 0.5).sum()),
                         f"{x['prescription']['greedy']['logdet'].get('p_eff_after', float('nan')):.1f}", f'{rho:.3f}'])
    md += ['\n## S-ident-2. Field audits with prior sd tau = 0.1, 0.3 and 1.0 ft/d (threshold 0.5)\n',
           table(['data set', 'tau ft/d', 'identifiable pipes', 'p_eff', 'median sd / tau', 'identifiable after five sites', 'p_eff after five sites',
                  'rank correlation of sd / tau with tau = 0.3'], prow)]
(HERE / 'tables' / 'T_S_ident.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))
