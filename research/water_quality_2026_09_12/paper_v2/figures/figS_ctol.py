"""Supplementary figure (R1-1 / R2-2): effect of disabling segment merging (Ctol = 0 versus the default 0.01).

a, RMSE of the Ctol = 0.01 minus Ctol = 0 difference over all nodes and reporting times, relative to the case peak, one
point per network, by condition and duration. b, the same difference relative to the time-step error of the Ctol = 0 run
against a fine-step reference. c, Ctol = 0 / Ctol = 0.01 ratios of library run time, peak live segments, fused replay
terms and replay time against network size (RTX 5090 node, 48 h). Identical runs (zero difference) are drawn at 1e-8. Data: paper_v2/tables/T_S_ctol.md (ctol_tables.py).
"""
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paperfig as pf

CONDS = [('native', 'native'), ('cl_k0', 'Cl k0'), ('cl_b05', 'Cl b0.5'), ('cl_b2', 'Cl b2'), ('cl_w05', 'Cl w0.5'),
         ('cl_b05_w05', 'Cl b0.5 w0.5'), ('cl_b2_w05', 'Cl b2 w0.5'), ('age', 'age'), ('trace', 'trace')]
DURS = [(None, 'INP', pf.C['blue']), (48, '48 h', pf.C['purple']), (168, '168 h', pf.C['green'])]
FLOOR = 1e-8


def load():
    same_file = {'FOS', 'Net3-NH2CL'}                 # byte-identical copies of foss_poly_1 and net3-bio, counted once
    R = [r for r in (json.loads(l) for l in open(pf.RES / 'final_ctol' / 'ctol_part1f_merged.jsonl', encoding='utf-8')) if r['net'] not in same_file]
    main = {(r['net'], r['cond'], r['duration_h']): r for r in R if r['kind'] == 'main' and 'q60' in r}
    ref = {(r['net'], r['cond'], r['duration_h']): r for r in R if r['kind'] == 'ref' and 'ctol0_vs_ref' in r}
    p3 = [r for f in sorted(glob.glob(str(pf.RES / 'final_5090_v2' / 'r1_B_*' / 'ctol_part3_*.json')))
          for r in json.load(open(f, encoding='utf-8'))['records']]
    return main, ref, p3


def diff(r):
    x = r[f"q{r['qstep_input']}"]['ctol001_vs_ctol0']
    return x['rmse'] / x['scale'] if x['scale'] > 0 else 0.0


def strip(ax, values, ylabel, ref_y, ref_label):
    rng = np.random.default_rng(0)
    for i, (c, _) in enumerate(CONDS):
        for j, (d, _, col) in enumerate(DURS):
            v = np.maximum(np.asarray(values.get((c, d), []), float), FLOOR)
            if len(v):
                x = i + (j - 1) * 0.26 + rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(x, v, s=4, color=col, alpha=0.55, linewidths=0, zorder=2)
                ax.hlines(np.median(v), i + (j - 1) * 0.26 - 0.11, i + (j - 1) * 0.26 + 0.11, color=pf.darken(col), lw=1.0, zorder=3)
    ax.set_yscale('log')
    pf.reference_line(ax, y=ref_y)
    pf.note(ax, len(CONDS) - 0.45, ref_y / 1.4, ref_label, ha='right', va='top', transform=ax.transData)
    ax.set_xticks(range(len(CONDS)), [n for _, n in CONDS], rotation=40, ha='right')
    ax.set_xlim(-0.6, len(CONDS) - 0.4)
    ax.set_ylabel(ylabel)
    pf.despine(ax); pf.grid_only(ax, 'y')


def main():
    pf.setup()
    main_, ref, p3 = load()
    a_vals, b_vals = {}, {}
    for k, r in main_.items():
        a_vals.setdefault(k[1:], []).append(diff(r))
        f = ref.get(k)
        if f and f['ctol0_vs_ref']['rmse'] > 1e-6 * f['ctol0_vs_ref']['scale']:
            b_vals.setdefault(k[1:], []).append(r[f"q{r['qstep_input']}"]['ctol001_vs_ctol0']['rmse'] / f['ctol0_vs_ref']['rmse'])

    fig = pf.figure(height_mm=78)
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.80, bottom=0.25, wspace=0.42, width_ratios=[1.25, 1.25, 1])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    strip(ax_a, a_vals, 'RMSE of difference / peak', 0.01, '1% of peak')
    strip(ax_b, b_vals, 'Tolerance effect / step error', 1.0, 'equal')
    pf.legend(ax_a, handles=[Line2D([], [], ls='none', marker='o', ms=3, color=col, label=lab) for _, lab, col in DURS],
              loc='lower left', bbox_to_anchor=(0, 1.03), ncol=3, borderaxespad=0)

    by = {(r['net'], r['duration_h'], r['ctol']): r for r in p3}
    nets = sorted({r['net'] for r in p3 if r['duration_h'] == 48}, key=lambda n: by[(n, 48, 0.0)]['nodes'])
    nodes = np.array([by[(n, 48, 0.0)]['nodes'] for n in nets])
    for key, lab, col, mk in (('library_run_s', 'library run time', pf.C['yellow'], 'o'), ('peak_live_versions', 'peak live segments', pf.C['purple'], 's'),
                              ('fused_terms', 'fused replay terms', pf.C['blue'], 'D'), ('replay_fwd_bwd_s', 'replay time', pf.C['green'], '^')):
        ratio = np.array([by[(n, 48, 0.0)][key] / by[(n, 48, 0.01)][key] for n in nets])
        ax_c.plot(nodes, ratio, color=col, lw=0.8, marker=mk, ms=3.2, label=lab, zorder=3)
    ax_c.set_xscale('log'); ax_c.set_yscale('log')
    pf.reference_line(ax_c, y=1.0)
    ax_c.set_xlabel('Nodes'); ax_c.set_ylabel('Ctol = 0 / Ctol = 0.01 (48 h)')
    pf.despine(ax_c); pf.grid_only(ax_c, 'y')
    pf.legend(ax_c, loc='lower left', fontsize=pf.FS['legend'], frameon=False, handlelength=1.6)
    for ax, letter in ((ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c')):
        pf.panel_label(ax, letter, dy=1.16)
    pf.save(fig, 'figS_ctol')
    plt.close(fig)


if __name__ == '__main__':
    main()
