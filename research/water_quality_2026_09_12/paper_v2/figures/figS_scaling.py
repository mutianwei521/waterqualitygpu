"""Supplementary figure (R2-6): how runtime and memory scale with the recorded dependencies, the quality steps and the
network size, on the RTX 5090 node (B = 1).

a, peak GPU memory of one forward-plus-adjoint replay against the fused terms of the schedule (Net1-3, Jilin, three field
networks, L-Town at 48 h and 168 h and BWSN2 at 48 h, both tolerances; BWSN2 96 h and 168 h from the long-horizon run).
b, time of one forward-plus-adjoint replay against the number of quality steps (Ctol = 0, median of three warm runs).
c, recording time of the Python replicator against the number of recorded versions (Ctol = 0).
Data: final_5090_v2/r1_B_*/ctol_part3_*.json, final_5090_v2/r1_C_*/bwsn2_*h.json (paper_v2/tables/perf_ledger.md).
"""
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paperfig as pf


def load():
    p3 = [r for f in sorted(glob.glob(str(pf.RES / 'final_5090_v2' / 'r1_B_*' / 'ctol_part3_*.json')))
          for r in json.load(open(f, encoding='utf-8'))['records']]
    long = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(str(pf.RES / 'final_5090_v2' / 'r1_C_*' / 'bwsn2_*h.json')))]
    return p3, long


def guide(ax, x, y, slope, label, **kw):
    """Line of the given log-log slope through the geometric mean of the points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.exp(np.mean(np.log(y) - slope * np.log(x)))
    xs = np.array([x.min() / 1.6, x.max() * 1.6])
    ax.plot(xs, c * xs ** slope, color=pf.C['grey'], lw=0.6, ls=(0, (3, 2)), zorder=1, **kw)
    return c


def main():
    pf.setup()
    p3, long = load()
    big = lambda r: r['net'] == 'BWSN2'
    fig = pf.figure(height_mm=66)
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.86, bottom=0.17, wspace=0.42)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    for ct, col, mk in ((0.0, pf.C['purple'], 'o'), (0.01, pf.C['yellow'], 's')):
        q = [r for r in p3 if r['ctol'] == ct]
        ax_a.scatter([r['fused_terms'] for r in q], [r['gpu_peak_MB'] / 1024 for r in q], s=10, marker=mk, color=col, linewidths=0, zorder=3)
    ax_a.scatter([r['fused_terms'] for r in long], [r['gpu_peak_fwd_adj_B1_GB'] for r in long], s=16, marker='D', color=pf.C['green'], linewidths=0, zorder=3)
    for r in long:
        pf.note(ax_a, r['fused_terms'] / 1.25, r['gpu_peak_fwd_adj_B1_GB'], f"BWSN2 {r['hours']} h", ha='right', va='center', transform=ax_a.transData)
    x = [r['fused_terms'] for r in p3] + [r['fused_terms'] for r in long]
    y = [r['gpu_peak_MB'] / 1024 for r in p3] + [r['gpu_peak_fwd_adj_B1_GB'] for r in long]
    c = guide(ax_a, x, y, 1.0, 'linear')
    pf.reference_line(ax_a, y=32)
    pf.note(ax_a, 3e4, 32 * 1.25, '32 GB', va='bottom', transform=ax_a.transData)
    pf.note(ax_a, 0.03, 0.80, f'{c * 1e6 * 1024:.0f} MB per\nmillion terms')
    ax_a.set_xscale('log'); ax_a.set_yscale('log')
    ax_a.set_xlabel('Fused terms'); ax_a.set_ylabel('Peak GPU memory, B = 1 (GB)')
    pf.legend(ax_a, handles=[Line2D([], [], ls='none', marker='o', ms=3, color=pf.C['purple'], label='Ctol = 0'),
                             Line2D([], [], ls='none', marker='s', ms=3, color=pf.C['yellow'], label='Ctol = 0.01'),
                             Line2D([], [], ls='none', marker='D', ms=3, color=pf.C['green'], label='BWSN2 96/168 h')],
              loc='lower right', frameon=False)

    z = [r for r in p3 if r['ctol'] == 0.0]
    for r in z:
        ax_b.scatter(r['steps'], r['replay_fwd_bwd_s'], s=12, color=pf.C['green'] if big(r) else (pf.C['blue'] if r['duration_h'] == 168 else pf.C['purple']),
                     linewidths=0, zorder=3)
    c = guide(ax_b, [r['steps'] for r in z], [r['replay_fwd_bwd_s'] for r in z], 1.0, 'linear')
    pf.note(ax_b, 0.03, 0.97, f'{c * 1e6:.0f} µs per step\n(11 to 12,527 nodes)')
    ax_b.set_xscale('log'); ax_b.set_yscale('log')
    ax_b.set_xlabel('Quality steps'); ax_b.set_ylabel('Forward + adjoint, B = 1 (s)')
    pf.legend(ax_b, handles=[Line2D([], [], ls='none', marker='o', ms=3, color=pf.C['purple'], label='48 h'),
                             Line2D([], [], ls='none', marker='o', ms=3, color=pf.C['blue'], label='168 h'),
                             Line2D([], [], ls='none', marker='o', ms=3, color=pf.C['green'], label='BWSN2 48 h')],
              loc='lower right', frameon=False)

    ax_c.scatter([r['versions'] for r in z], [r['record_s'] for r in z], s=12, color=pf.C['purple'], linewidths=0, zorder=3)
    c = guide(ax_c, [r['versions'] for r in z], [r['record_s'] for r in z], 1.0, 'linear')
    pf.note(ax_c, 0.03, 0.97, f'{c * 1e6:.0f} µs per version\n(Python replicator)')
    ax_c.set_xscale('log'); ax_c.set_yscale('log')
    ax_c.set_xlabel('Recorded versions'); ax_c.set_ylabel('Recording time (s)')

    for ax, letter in ((ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c')):
        pf.despine(ax); pf.grid_only(ax, 'y'); pf.panel_label(ax, letter, dy=1.06)
    pf.save(fig, 'figS_scaling')
    plt.close(fig)


if __name__ == '__main__':
    main()
