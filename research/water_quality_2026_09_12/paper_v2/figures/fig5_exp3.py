"""Figure 5: pipe-by-pipe wall-decay identifiability and sensor prescription.

Figure contract: an asymmetric map-plus-quantification figure shows which wall
coefficients the existing samples inform, all saved truth/MAP comparisons with
their original two-standard-deviation intervals, and the effect of added sites.
No saved observations or intervals are cropped, averaged or interpolated.
Python/matplotlib; 176 mm width; editable PDF/SVG and 600-dpi PNG/TIFF.
"""
import sys
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import paperfig as pf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

pf.setup()
CYAN, TEAL, GOLD = '#56B4E9', '#009E73', '#E69F00'
PURPLE, MAGENTA, VERM = '#7B61A8', '#CC79A7', '#D55E00'
INK, GREY, LIGHT, NAVY = '#30363D', '#707A84', '#D6DCE1', '#2F5D7C'
IDENT = {'ident': PURPLE, 'part': MAGENTA, 'prior': '#CBD3DA'}
plt.rcParams.update({'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': GREY,
                     'xtick.color': INK, 'ytick.color': INK, 'grid.color': '#E7EAED',
                     'grid.linewidth': .35, 'axes.grid.axis': 'y',
                     # Keep mathematical subscripts and log exponents >=5 pt after the
                     # 176-to-165-mm manuscript reduction.
                     'axes.labelsize': 7.8, 'xtick.labelsize': 7.8, 'ytick.labelsize': 7.8})
SYN = [('LTown', 'L-Town', MAGENTA, 'm/d'), ('BWSN2', 'BWSN2', TEAL, 'ft/d')]
FIELD = [('Fairfield', 'Fairfield', PURPLE), ('Bellingham', 'Bellingham', CYAN),
         ('NorthPenn', 'North Penn', TEAL)]
SETS = [('random', 'Existing sensors', NAVY), ('random+random', '+ Random sites', CYAN),
        ('random+dopt', '+ D-optimal sites', GOLD), ('random+count', '+ Count-greedy sites', TEAL)]
TH = 0.5           # identifiable: posterior sd < TH x prior sd
PART = 0.9         # map colour class boundary: partially informed pipes
H_MM = 184
SMALL = pf.FS['small']
DASH = (0, (3, 1.5))     # 'after prescription' line style, shared with Fig. 2d
SITE = dict(existing=dict(marker='s', mfc='white', mec=NAVY, mew=0.8), random=dict(marker='o', mfc='white', mec=CYAN, mew=0.8),
            dopt=dict(marker='^', mfc=GOLD, mec='white', mew=0.45), count=dict(marker='D', mfc=TEAL, mec='white', mew=0.45))

syn = {k: pf.load(pf.DATA['exp3'] / f'exp3_{k}.json')['networks'][0] for k, *_ in SYN}
field = {f['dataset']: f for f in pf.load(pf.DATA['exp3'] / 'exp3_field.json')['field']}


def ratio(n):
    return np.array(n['posterior_sd']['random']) / np.array(n['posterior_sd']['prior_sd'])


fig = pf.figure(height_mm=H_MM)
fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc='white', ec='none', zorder=-10))
# The original equal-row grid left a 52-mm inter-row band.  Preserve the upper
# audit/map row, give the quantitative c--e row more height, and reduce the band
# to about 40 mm.  This moves c--e upward by 12 mm while retaining room for the
# two map keys and the panel-specific legends between the rows.
gs = fig.add_gridspec(2, 3, left=0.078, right=0.98, top=0.945, bottom=0.075, wspace=0.46, hspace=0.67,
                      height_ratios=[1.0, 1.22])

# (a) posterior/prior sd of every pipe's Kw, ranked - the audit on the existing sensors
a = fig.add_subplot(gs[0, 0])
ytxt = {'LTown': 0.30, 'BWSN2': 0.10}       # direct labels in the empty lower right (both curves sit at 1.0 there)
for key, label, col, unit in SYN:
    n = syn[key]; r = np.sort(ratio(n))
    a.plot(np.arange(1, len(r) + 1), r, color=col, lw=1.2)
    pf.note(a, 0.98, ytxt[key], f"{label}, {len(r):,} pipes\n{int((r < TH).sum())} identifiable, "
            f"$p_\\mathrm{{eff}}$ = {n['sets']['random']['audit']['p_eff']:.0f}", color=col, ha='right', va='center',
            fontsize=7.8)
pf.reference_line(a, y=TH, color=VERM, ls=(0, (1.5, 1.5)), lw=.7)
pf.note(a, 0.98, 0.505, 'identifiable below', color=GREY, ha='right', va='bottom')
a.set_xscale('log'); pf.log_ticks(a, 'x'); a.set_xlim(1, 2e4); a.set_ylim(0, 1.04)
a.set_xlabel('Pipes ranked by posterior sd'); a.set_ylabel('Posterior sd / prior sd of $K_w$')
pf.label_cell(fig, gs[0, 0], 'a')
pf.header(fig, gs[0, 0], 'Network-wide audit', fontsize=7.7, color=INK)

# (b) L-Town map: which pipes the 10 sensors inform, and where the prescriptions add sensors
b = fig.add_subplot(gs[0, 1:])
n = syn['LTown']; rmap = dict(zip(n['truth']['pipe_ids'], ratio(n)))
coords, links = pf.read_inp_geometry(pf.INP['LTown'])
segs = {'prior': [], 'part': [], 'ident': []}
for lid, u, v in links:
    if u in coords and v in coords:
        x = rmap.get(lid, 1.0)
        segs['ident' if x < TH else 'part' if x < PART else 'prior'].append([coords[u], coords[v]])
for cls, lw, z in (('prior', 0.5, 1), ('part', 1.2, 2), ('ident', 2.0, 3)):
    b.add_collection(LineCollection(segs[cls], colors=IDENT[cls], linewidths=lw, zorder=z, capstyle='round'))
xy = np.array(list(coords.values())); (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
b.set_xlim(x0 - 0.01 * (x1 - x0), x1 + 0.01 * (x1 - x0)); b.set_ylim(y0 - 0.02 * (y1 - y0), y1 + 0.02 * (y1 - y0)); b.set_aspect('equal'); b.axis('off')
pts = lambda ids: np.array([coords[i] for i in ids if i in coords])  # noqa: E731
for ids, kind, ms in ((n['sets']['random']['sensors'], 'existing', 3.8), (n['design']['random_added'], 'random', 3.8),
                      (n['design']['greedy']['random+dopt']['added'], 'dopt', 4.8), (n['design']['greedy']['random+count']['added'], 'count', 4.0)):
    p = pts(ids); b.plot(p[:, 0], p[:, 1], ls='none', ms=ms, zorder=5 if kind in ('existing', 'random') else 6, **SITE[kind])
cb1, cb2 = gs[0, 1].get_position(fig), gs[0, 2].get_position(fig)
xc = (cb1.x0 + cb2.x1) / 2
fig.legend(handles=[Line2D([], [], color=IDENT['prior'], lw=1.5, label=f'Prior-dominated: {len(segs["prior"])}'),
                    Line2D([], [], color=IDENT['part'], lw=1.5, label=f'Partly informed: {len(segs["part"])}'),
                    Line2D([], [], color=IDENT['ident'], lw=2.2, label=f'Identifiable: {len(segs["ident"])}')],
           loc='upper center', bbox_to_anchor=(xc, cb1.y0 - 1.5 * pf.MM / (H_MM * pf.MM)), ncol=3,
           fontsize=pf.FS['legend'], frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.5)
fig.legend(handles=[Line2D([], [], ls='none', ms=3.8, label=f"existing ({n['n_sensors']})", **SITE['existing']),
                    Line2D([], [], ls='none', ms=3.8, label=f"+{n['n_add']} random", **SITE['random']),
                    Line2D([], [], ls='none', ms=4.8, label=f"+{n['n_add']} D-optimal", **SITE['dopt']),
                    Line2D([], [], ls='none', ms=4.0, label=f"+{n['n_add']} count-greedy", **SITE['count'])],
           loc='upper center', bbox_to_anchor=(xc, cb1.y0 - 5.5 * pf.MM / (H_MM * pf.MM)), ncol=4,
           fontsize=pf.FS['legend'], frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.5)
pf.label_cell(fig, gs[0, 1], 'b')
pf.header(fig, gs[0, 1:], f"L-Town \u00b7 {n['n_sensors']} random sensors \u00b7 48 h, 15-min sampling \u00b7 {len(rmap)} pipes", color=MAGENTA)

# (c) truth vs MAP per pipe: identifiable pipes move to the truth, the rest stay at the grouped prior
gc = gs[1, 0].subgridspec(2, 1, hspace=1.08)
clipped = {}
for j, (key, label, col, unit) in enumerate(SYN):
    ax = fig.add_subplot(gc[j]); n = syn[key]; r = ratio(n)
    kt = -np.array(n['truth']['kw']); km = -np.array(n['map_kw']['random']); sd = np.array(n['posterior_sd']['random'])
    ident = r < TH
    ax.scatter(kt[~ident], km[~ident], s=2.0, color='#AEB7BF', alpha=0.4, lw=0, rasterized=True, zorder=1)
    ax.errorbar(kt[ident], km[ident], yerr=2 * sd[ident], fmt='o', color=col, ms=2.8, mew=0, elinewidth=0.6, capsize=1.2, zorder=3)
    # Positive truth is displayed on a log axis so all tail values remain visible.
    # The reference is the analytic identity y=x; no measured series is interpolated.
    assert np.all(kt > 0)
    lo, hi = kt.min() / 1.2, kt.max() * 1.2
    identity = np.geomspace(lo, hi, 200)
    ax.plot(identity, identity, '-', color=GREY, lw=.6, zorder=2)
    ax.set_xscale('log'); pf.log_ticks(ax, 'x')
    ax.set_xlim(lo, hi)
    ymin = min(0., float((km - 2 * sd)[ident].min()))
    ymax = max(float(km.max()), float((km + 2 * sd)[ident].max()))
    ax.set_ylim(ymin - .07 * (ymax - ymin), ymax + .12 * (ymax - ymin))
    ax.text(.03, .93, 'Reference: MAP = true', transform=ax.transAxes, fontsize=7,
            color=GREY, ha='left', va='top')
    clipped[label] = int(((kt < lo) | (kt > hi)).sum())
    assert clipped[label] == 0
    ke = n['sets']['random']['kw_error']
    pf.note(ax, 0, 1.10, f"{label}: {int(ident.sum())} / {len(r):,} identifiable\nRMSE {ke['map']['rmse_identifiable']:.3f}; prior "
            f"{ke['grouped_prior_mean']['rmse_identifiable']:.3f} {unit}", color=col, va='bottom', fontsize=7)
    ax.set_ylabel(f'MAP $|K_w|$ ({unit})'); ax.set_xlabel(f'True $|K_w|$ ({unit}, log scale)', labelpad=1)
pf.label_cell(fig, gs[1, 0], 'c')

# (d) sensor prescription: identifiable pipes and p_eff, predicted by the audit vs realised after re-estimation
gd = gs[1, 1].subgridspec(2, 1, hspace=0.23)
d1, d2 = fig.add_subplot(gd[0]), fig.add_subplot(gd[1])
w, xs = 0.8, {}
for g, (key, label, col, unit) in enumerate(SYN):
    n = syn[key]
    for i, (s, sl, sc) in enumerate(SETS):
        x = g * 5 + i; xs.setdefault(key, []).append(x)
        aud, pred = n['sets'][s]['audit'], n['design']['predicted'][s]
        tint = plt.matplotlib.colors.to_rgba(sc, .23)
        d1.bar(x, aud['n_identifiable'], w, color=tint, edgecolor=sc, lw=.6, zorder=2); d1.plot([x - w / 2, x + w / 2], [pred['n_identifiable']] * 2, '-', color=INK, lw=0.9, zorder=3)
        d2.bar(x, aud['p_eff'], w, color=tint, edgecolor=sc, lw=.6, zorder=2); d2.plot([x - w / 2, x + w / 2], [pred['p_eff']] * 2, '-', color=INK, lw=0.9, zorder=3)
for ax, yl in ((d1, 'Identifiable pipes'), (d2, '$p_\\mathrm{eff}$ (effective\nparameters)')):
    ax.set_ylabel(yl); ax.set_xticks([1.5, 6.5]); ax.set_xlim(-0.8, 8.8)
d1.set_xticklabels([]); d2.set_xticklabels([f"{label}\n{syn[key]['n_sensors']} \u2192 {syn[key]['n_sensors'] + syn[key]['n_add']} sensors" for key, label, _, _ in SYN])
pf.legend(d1, handles=[Patch(facecolor=plt.matplotlib.colors.to_rgba(sc,.23), edgecolor=sc, label=sl) for _, sl, sc in SETS] + [Line2D([], [], color=INK, lw=0.9, label='Audit prediction')],
          loc='lower left', bbox_to_anchor=(-.03, 1.10), borderaxespad=0, labelspacing=0.25, handlelength=1.2, handleheight=0.8)
d1.set_ylim(0, 35)
pf.label_cell(fig, gs[1, 1], 'd')

# (e) field systems at theta*: audit of every pipe with the existing sites and after five prescribed sites
e = fig.add_subplot(gs[1, 2])
hand = []
for key, label, col in FIELD:
    f = field[key]; r0 = np.sort(f['pipe_sd_over_tau']); pres = f['prescription']['greedy']['logdet']; r1 = np.sort(pres['pipe_sd_over_tau_after'])
    e.plot(np.arange(1, len(r0) + 1), r0, '-', color=col, lw=1.0)
    e.plot(np.arange(1, len(r1) + 1), r1, ls=DASH, color=col, lw=1.0)
    hand.append(Line2D([], [], color=col, lw=1.2, label=f"{label}: {int((r0 < TH).sum())}\u2192{pres['n_identifiable_after']}; "
                                                        f"{f['audit']['p_eff']:.1f}\u2192{pres['p_eff_after']:.1f}"))
pf.reference_line(e, y=TH, color=VERM, ls=(0, (1.5, 1.5)), lw=.7)
e.set_xscale('log'); pf.log_ticks(e, 'x'); e.set_xlim(1, 500); e.set_ylim(0, 1.04); e.set_yticks([0, 0.5, 1.0])
e.set_xlabel('Pipes ranked by posterior sd'); e.set_ylabel('Posterior sd / prior sd of $K_w$')
pf.legend(e, handles=hand + [Line2D([], [], color=GREY, lw=1.0, label='existing sites'), Line2D([], [], color=GREY, lw=1.0, ls=DASH, label='+5 D-optimal sites')],
          loc='lower left', bbox_to_anchor=(-.08, 1.05), fontsize=7.8, borderaxespad=0, labelspacing=0.35, handlelength=1.6, ncol=1, title='Identifiable pipes; $p_\\mathrm{eff}$',
          title_fontsize=7.8, alignment='left')
pf.label_cell(fig, gs[1, 2], 'e')

pf.save(fig, 'fig5_exp3')
source_rows = {'synthetic_pipes': [], 'prescription': [], 'field_pipes': [], 'sites': []}
for key, label, col, unit in SYN:
    n = syn[key]
    rr = ratio(n)
    for i, pipe_id in enumerate(n['truth']['pipe_ids']):
        source_rows['synthetic_pipes'].append(dict(network=key, pipe=pipe_id, unit=unit,
            posterior_over_prior_sd=float(rr[i]), true_kw=n['truth']['kw'][i],
            map_kw=n['map_kw']['random'][i], posterior_sd=n['posterior_sd']['random'][i],
            identifiable=bool(rr[i] < TH)))
    for set_key, set_label, colour in SETS:
        for metric in ['n_identifiable', 'p_eff']:
            source_rows['prescription'].append(dict(network=key, sensor_set=set_key, metric=metric,
                realised=n['sets'][set_key]['audit'][metric], predicted=n['design']['predicted'][set_key][metric]))
        for node in n['sets'][set_key]['sensors']:
            source_rows['sites'].append(dict(network=key, sensor_set=set_key, node=node))
for key, label, col in FIELD:
    f = field[key]
    before = f['pipe_sd_over_tau']; after = f['prescription']['greedy']['logdet']['pipe_sd_over_tau_after']
    assert len(before) == len(after)
    for i, (b0, b1) in enumerate(zip(before, after)):
        source_rows['field_pipes'].append(dict(network=key, original_pipe_index=i,
                                             existing_ratio=b0, after_dopt_ratio=b1))
counts = {key: len(rows) for key, rows in source_rows.items()}
assert counts['synthetic_pipes'] == 15727 and counts['prescription'] == 16 and counts['field_pipes'] == 693
for key, rows in source_rows.items():
    with (pf.OUT / f'fig5_source_{key}.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
inputs = [pf.DATA['exp3'] / f'exp3_{key}.json' for key, *_ in SYN] + [pf.DATA['exp3'] / 'exp3_field.json', pf.INP['LTown']]
manifest = {'dimensions_mm': [176, H_MM], 'data_rows': counts,
    'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    'synthetic_pipe_counts': {key: len(syn[key]['truth']['pipe_ids']) for key, *_ in SYN},
    'identifiable_interval_counts': {key: int((ratio(syn[key]) < TH).sum()) for key, *_ in SYN},
    'map_class_counts': {key: len(values) for key, values in segs.items()},
    'c_points_outside_axes': clipped,
    'c_limits': 'All truth and MAP points shown; all identifiable-pipe MAP +/- 2 posterior-sd intervals shown, including negative lower endpoints. Positive truth uses a logarithmic x axis; y is linear. The grey curve is the analytic identity, not a fit.',
    'encoding': {'synthetic_networks': {key: col for key, _, col, _ in SYN},
                 'field_networks': {key: col for key, _, col in FIELD},
                 'sensor_sets': {key: col for key, _, col in SETS}, 'map_classes': IDENT,
                 'identifiability_threshold': VERM},
    'data_policy': 'All saved arrays and pointwise 2-sd intervals retained; only sorting for ranked curves, the original threshold classes and original audit summaries are used.'}
(pf.OUT / 'fig5_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
for key, label, col, unit in SYN:
    n = syn[key]; r = ratio(n)
    print(f"== {label}: {len(r)} pipes; sd/prior < {TH}: {(r < TH).sum()}, < {PART}: {(r < PART).sum()}, sensitive {n['sets']['random']['audit']['n_sensitive']}; "
          f"sets -> " + '; '.join(f"{s}: ident {n['sets'][s]['audit']['n_identifiable']} (pred {n['design']['predicted'][s]['n_identifiable']}), p_eff {n['sets'][s]['audit']['p_eff']:.1f} (pred {n['design']['predicted'][s]['p_eff']:.1f})" for s, _, _ in SETS))
print(f"L-Town map classes: prior {len(segs['prior'])}, partial {len(segs['part'])}, identifiable {len(segs['ident'])} (links {len(links)}, pipes {len(rmap)}); (c) prior-dominated points beyond axis limit: {clipped}")
for key, label, _ in FIELD:
    f = field[key]; r0 = np.array(f['pipe_sd_over_tau'])
    print(f"{label}: min sd/tau {r0.min():.3f}, identifiable {(r0 < TH).sum()} -> logdet {f['prescription']['greedy']['logdet']['n_identifiable_after']}, count {f['prescription']['greedy']['count']['n_identifiable_after']}; added {f['prescription']['greedy']['logdet']['added_sites']}")
