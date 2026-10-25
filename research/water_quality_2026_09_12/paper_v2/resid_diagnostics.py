"""R1-6: diagnostics of the residual assumptions behind the calibration bands (independent, equal-variance errors).

Uses the leave-one-site-out residuals stored by uq.py (h2_5090/h2_uq.json, loso.per_obs; observations are in file order,
i.e. chronological within a site). Reports the share of residual variance explained by site means (one-way ANOVA), the
lag-1 autocorrelation within sites before and after removing the site means, the spread of site standard deviations
with a Brown-Forsythe test, the rank correlation of |residual| with the prediction, and the coverage of the 95%
predictive band per site. Writes paper_v2/tables/T_S_resid.md.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]) + '\n'


def lag1(r, s, demean):
    a, b = [], []
    for i in np.unique(s):
        e = r[s == i] - (r[s == i].mean() if demean else 0.0)
        a += list(e[:-1]); b += list(e[1:])
    return float(np.corrcoef(a, b)[0, 1])


rows, srows = [], []
for ds in json.load(open(ROOT / 'h2_5090' / 'h2_uq.json', encoding='utf-8'))['datasets']:
    L = ds['loso']; po = L['per_obs']
    s, r, y = np.array(po['site']), np.array(po['resid'], float), np.array(po['y'], float)
    ok = ~np.isnan(r); s, r, y = s[ok], r[ok], y[ok]
    sites = np.unique(s)
    groups = [r[s == i] for i in sites]
    ni = np.array([len(g) for g in groups]); mi = np.array([g.mean() for g in groups])
    share = float(np.sum(ni * (mi - r.mean()) ** 2) / np.sum((r - r.mean()) ** 2))
    p_site = stats.f_oneway(*groups).pvalue
    sd = np.array([g.std(ddof=1) for g in groups if len(g) > 2])
    p_var = stats.levene(*[g for g in groups if len(g) > 2], center='median').pvalue
    rho = stats.spearmanr(np.abs(r), y - r)
    cov = L['site_coverage95']
    inside = sum(v['pred_lin'] * v['n'] for v in cov.values()); total = sum(v['n'] for v in cov.values())
    worst = min(cov.items(), key=lambda kv: kv[1]['pred_lin'])
    wo = (inside - worst[1]['pred_lin'] * worst[1]['n']) / (total - worst[1]['n'])
    rows.append([ds['dataset'], len(r), len(sites), f'{share:.2f} (p = {p_site:.1g})', f'{lag1(r, s, False):.2f}', f'{lag1(r, s, True):.2f}',
                 f'{sd.min():.3f}-{sd.max():.3f} (p = {p_var:.2f})', f'{rho.correlation:.2f} (p = {rho.pvalue:.1g})',
                 f"{inside / total:.3f}", f"{sum(v['pred_lin'] < 0.95 for v in cov.values())} of {len(cov)}",
                 f"site {worst[0]}: {worst[1]['pred_lin']:.2f} ({worst[1]['n']} obs); others {wo:.3f}"])
    for k, v in cov.items():
        srows.append([ds['dataset'], k, v['n'], f"{v['pred_lin']:.2f}", f"{v['pred_sampled']:.2f}", f"{v['param_only_lin']:.2f}",
                      f"{np.mean(r[s == list(cov).index(k)]):+.3f}"])
md = ['# Residual diagnostics of the leave-one-site-out test (R1-6)\n',
      'Leave-one-site-out residuals of the grouped calibration (Kb + four diameter-class Kw), linearised 95% predictive band.\n',
      table(['data set', 'observations', 'sites', 'variance share of site means (ANOVA p)', 'lag-1 autocorrelation within sites',
             'same after removing site means', 'site sd range (Brown-Forsythe p)', 'Spearman |r| vs prediction', '95% coverage',
             'sites below 0.95', 'worst site; coverage without it'], rows),
      '\n## Coverage of the 95% predictive band by held-out site\n',
      table(['data set', 'site', 'observations', 'linearised', 'sampled', 'parameter only', 'mean residual mg/L'], srows)]
(HERE / 'tables' / 'T_S_resid.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md[:3]))
