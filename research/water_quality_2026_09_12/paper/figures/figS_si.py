"""Supporting Figures S1–S7 for the ES&T Supporting Information, drawn from paper/tables/*.md (the same result
tables that feed Tables S1–S7) in the shared paperfig style.

    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg python figS_si.py
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import paperfig as pf

TAB = pf.PAPER / 'tables'
C, MM = pf.C, pf.MM
pf.setup()


# ----------------------------------------------------------------------------------------------- helpers
def md_tables(path):
    """All pipe tables of a markdown file as lists of row-cell lists (header first)."""
    tabs, cur = [], []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if set(''.join(cells)) <= set('-: '):
                continue
            cur.append(cells)
        elif cur:
            tabs.append(cur)
            cur = []
    if cur:
        tabs.append(cur)
    return tabs


def num(s):
    s = s.replace(',', '').strip()
    if s in ('-', '', 'inf'):
        return np.nan if s != 'inf' else np.inf
    return float(s)


def box(ax, x, y, w, h, text, fc, ec=None, fs=None, bold_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.012', fc=fc,
                                ec=ec or pf.darken(fc, 0.7), lw=0.6, transform=ax.transAxes, zorder=2))
    if bold_first and '\n' in text:
        first, rest = text.split('\n', 1)
        ax.text(x + w / 2, y + h - 0.035, first, transform=ax.transAxes, ha='center', va='top', fontsize=fs or pf.FS['body'],
                fontweight='bold', zorder=3)
        ax.text(x + w / 2, y + h - 0.035 - 0.11, rest, transform=ax.transAxes, ha='center', va='top',
                fontsize=fs or pf.FS['small'], linespacing=1.2, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha='center', va='center', fontsize=fs or pf.FS['small'],
                linespacing=1.2, zorder=3)


def arrow(ax, p, q, ls='-', color=None, text=None, tpos=0.5, toff=(0, 0.03), fs=None):
    ax.annotate('', xy=q, xytext=p, xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=0.7, color=color or C['ink'], ls=ls, shrinkA=1, shrinkB=1,
                                mutation_scale=7), zorder=4)
    if text:
        x = p[0] + (q[0] - p[0]) * tpos + toff[0]
        y = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(x, y, text, transform=ax.transAxes, ha='center', va='bottom', fontsize=fs or pf.FS['small'],
                color=color or C['ink'], linespacing=1.15, zorder=5)


# ----------------------------------------------------------------------------------------------- Figure S1: flow
def fig_s1():
    # Keep the standalone icon-led algorithm diagram as the canonical S1.
    from figS1_flow_redesign import export
    export()


# ----------------------------------------------------------------------------------------------- Figure S2: step sweep
_S2_S4_REGENERATED = False


def _ensure_s2_s4_redesign():
    """Regenerate the coupled S2--S4 redesign once per process."""
    global _S2_S4_REGENERATED
    if not _S2_S4_REGENERATED:
        from figS2_S4_redesign import main as export_s2_s4
        export_s2_s4()
        _S2_S4_REGENERATED = True


def fig_s2():
    _ensure_s2_s4_redesign()
    return
    rows = md_tables(TAB / 'T1b_fd_step_sweep.md')[0]
    hdr, rows = rows[0], rows[1:]
    h = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    PCOL = {'kb': C['purple'], 'kw': C['green'], 'c0': C['blue']}
    PLAB = {'kb': '$K_b$', 'kw': '$K_w$', 'c0': '$c_0$'}
    NET = {'Jilin_kw': ('Jilin', 'o'), 'Net1': ('Net1', 's'), 'Net2_MIX2': ('Net2, two-compartment tank', '^'),
           'Net2': ('Net2', 'v'), 'Net2_react': ('Net2, first-order decay', 'D'),
           'Net3_CHEM_kw_mixed_tanks': ('Net3, wall decay', 'P'), 'Net3_CHEM_mixed_tanks': ('Net3', 'X')}
    fig = pf.figure(height_mm=92)
    gs = pf.panel_grid(fig, 1, 2, bottom_mm=34, top_mm=6, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    handles = []
    for r in rows:
        case, par = [s.strip() for s in r[0].split('/')]
        par = par.split()[0]
        mode = 0 if 'M0' in r[1] else 1
        y = np.array([num(v) for v in r[4:11]])
        net, mk = NET[case]
        ax = axes[mode]
        ax.plot(h, y, color=PCOL[par], marker=mk, ms=3.2, lw=0.9, mec=pf.darken(PCOL[par]), mfc=PCOL[par])
        if mode == 1:
            handles.append((Line2D([], [], color=PCOL[par], marker=mk, ms=3.2, lw=0.9, mec=pf.darken(PCOL[par])),
                            f'{net}, {PLAB[par]}'))
    for i, ax in enumerate(axes):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.set_ylim(1e-16, 3)
        ax.set_xlabel('finite-difference step $h$ (engine user units)')
        ax.plot(h, 1e-1 * (h / 1e-1) ** 2 * 1e-2, color=C['grey'], ls=(0, (2, 2)), lw=0.6)
        ax.plot(h, 1e-1 * (h / 1e-1) * 1e-1, color=C['grey'], ls=(0, (2, 2)), lw=0.6)
        ax.text(2e-7, 1e-1 * (2e-7 / 1e-1) ** 2 * 1e-2 * 3, '$h^2$', color=C['grey'], fontsize=pf.FS['small'], ha='left')
        ax.text(2e-7, 1e-1 * (2e-7 / 1e-1) * 1e-1 * 3, '$h$', color=C['grey'], fontsize=pf.FS['small'], ha='left')
        pf.label_cell(fig, gs[0, i], 'ab'[i])
        pf.header(fig, gs[0, i], ['input tolerance, $C_{tol}$ = 0.01 mg/L', 'zero tolerance'][i])
    axes[0].set_ylabel('relative error of the library central difference')
    fig.legend([h_ for h_, _ in handles], [l for _, l in handles], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0),
               fontsize=pf.FS['legend'], frameon=False, columnspacing=1.2, handlelength=1.8)
    pf.save(fig, 'figS2_sweep')


# ----------------------------------------------------------------------------------------------- Figure S3: engine cost
def fig_s3():
    _ensure_s2_s4_redesign()
    return
    t = md_tables(TAB / 'T2_engine_cost.md')
    a, b = t[0], t[1]
    NAME = {'Net1': 'Net1', 'Net2_react': 'Net2, decay', 'Jilin_kw': 'Jilin', 'Net3_CHEM_kw_mixed_tanks': 'Net3, wall decay',
            'BWSN2_kw': 'BWSN2', 'LTown_kw': 'L-Town'}
    fig = pf.figure(height_mm=72)
    gs = pf.panel_grid(fig, 1, 2, bottom_mm=14, top_mm=6, wspace=0.32, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    rows = a[1:]
    labels = [NAME[r[0]] for r in rows]
    series = [('library run', [num(r[5]) for r in rows], C['grey']),
              ('library FD gradient', [num(r[6]) for r in rows], pf.METHOD['fd']),
              ('per-operator adjoint', [num(r[7]) for r in rows], pf.METHOD['per_op']),
              ('fused forward + adjoint, $B$ = 1', [num(r[9]) for r in rows], pf.METHOD['adjoint']),
              ('fused, per set at $B$ = 1,024', [num(r[11]) for r in rows], C['purple_light'])]
    w = 0.16
    x = np.arange(len(rows))
    for k, (lab, vals, col) in enumerate(series):
        pf.outlined_bar(ax, x + (k - 2) * w, vals, width=w, color=col, label=lab)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('wall-clock time (s)')
    ax.set_ylim(3e-6, 40)
    pf.legend(ax, loc='upper left', ncol=2, bbox_to_anchor=(0.0, 1.0), fontsize=6.5)
    pf.label_cell(fig, gs[0, 0], 'a')
    ax = fig.add_subplot(gs[0, 1])
    cases = []
    for r in b[1:]:
        if r[0] not in cases:
            cases.append(r[0])
    cols = [C['green'], C['purple'], C['grey'], C['blue'], C['yellow'], pf.darken(C['purple'])]
    for ci, case in enumerate(cases):
        rr = [r for r in b[1:] if r[0] == case and r[3] != '-']
        B = np.array([num(r[2]) for r in rr])
        for j, (idx, ls) in enumerate(((4, (0, (1.2, 1.2))), (6, (0, (3, 1.5))), (8, '-'))):
            ax.plot(B, [num(r[idx]) for r in rr], color=cols[ci], ls=ls, lw=0.9, marker='o' if ls == '-' else None, ms=2.6,
                    mec=pf.darken(cols[ci]))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('batch size $B$')
    ax.set_ylabel('Jacobian rows per second')
    ax.set_ylim(5, 5e5)
    h1 = [Line2D([], [], color=cols[i], lw=0.9) for i in range(len(cases))]
    l1 = [NAME[c] for c in cases]
    h2 = [Line2D([], [], color=C['ink'], ls=(0, (1.2, 1.2)), lw=0.9), Line2D([], [], color=C['ink'], ls=(0, (3, 1.5)), lw=0.9),
          Line2D([], [], color=C['ink'], ls='-', lw=0.9, marker='o', ms=2.6)]
    l2 = ['gather path', 'sparse (SpMM) path', 'automatic choice']
    pf.legend(ax, h1 + h2, l1 + l2, loc='upper left', ncol=2, fontsize=6.5)
    pf.label_cell(fig, gs[0, 1], 'b')
    pf.save(fig, 'figS3_cost')


# ----------------------------------------------------------------------------------------------- Figure S4: calibration
def fig_s4():
    _ensure_s2_s4_redesign()
    return
    t = md_tables(TAB / 'T3_field_calibration.md')
    cal, loso = t[0][1:], t[2][1:]
    METH = [('nominal(INP)', 'nominal', C['grey']), ('FD-LBFGSB', 'FD, $C_{tol}$ = 0', pf.METHOD['fd']),
            ('FD-LBFGSB@ctol0.01', 'FD, $C_{tol}$ = 0.01', pf.METHOD['fd_default']),
            ('FD-LBFGSB@ctol0.01,eps1e-3', 'FD, $C_{tol}$ = 0.01, step $10^{-3}$', pf.darken(pf.METHOD['fd'], 0.75)),
            ('ADJ-LBFGSB', 'adjoint, eager', C['purple_light']), ('ADJ-LBFGSB-graph', 'adjoint, execution graph', pf.METHOD['adjoint'])]
    SYS = [('Bellingham', 'Bellingham'), ('Fairfield', 'Fairfield'), ('NorthPenn', 'North Penn')]
    fig = pf.figure(height_mm=100)
    gs = pf.panel_grid(fig, 2, 2, bottom_mm=10, top_mm=6, hspace=0.55, wspace=0.3)

    def val(sysk, meth, col, fixed):
        key = sysk + (' (Kb fixed)' if fixed else '')
        for r in cal:
            if r[0] == key and r[1] == meth:
                return num(r[col])
        return np.nan

    x = np.arange(3)
    w = 0.13
    for pi, (fixed, title) in enumerate(((False, 'bulk coefficient free'), (True, 'bulk coefficient fixed'))):
        ax = fig.add_subplot(gs[0, pi])
        for k, (meth, lab, col) in enumerate(METH):
            vals = [val(s, meth, 3, fixed) for s, _ in SYS]
            pf.outlined_bar(ax, x + (k - 2.5) * w, vals, width=w, color=col, label=lab, hatch='///' if fixed else None)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in SYS])
        ax.set_ylim(0, 0.42)
        ax.set_ylabel('site RMSE (mg/L)')
        pf.header(fig, gs[0, pi], title)
        pf.label_cell(fig, gs[0, pi], 'ab'[pi])
        if pi == 0:
            pf.legend(ax, loc='upper left', ncol=2, fontsize=6.2, bbox_to_anchor=(0.0, 1.02))
    ax = fig.add_subplot(gs[1, 0])
    for k, (meth, lab, col) in enumerate(METH[1:]):
        vals = [val(s, meth, 6, False) for s, _ in SYS]
        pf.outlined_bar(ax, x + (k - 2) * 0.15, vals, width=0.15, color=col)
    ax.set_yscale('log')
    ax.set_ylim(0.05, 60)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in SYS])
    ax.set_ylabel('wall-clock time per fit (s)')
    pf.header(fig, gs[1, 0], 'bulk coefficient free')
    pf.label_cell(fig, gs[1, 0], 'c')
    ax = fig.add_subplot(gs[1, 1])
    order = ['Fairfield', 'Bellingham', 'NorthPenn']
    lab3 = ['Fairfield', 'Bellingham', 'North Penn']
    cols3 = [('predictive, linearised', 6, pf.METHOD['adjoint']), ('predictive, sampled', 7, C['purple_light']),
             ('parameter-only', 8, C['light'])]
    for k, (lab, idx, col) in enumerate(cols3):
        vals = [num(next(r for r in loso if r[0] == s)[idx]) for s in order]
        pf.outlined_bar(ax, x + (k - 1) * 0.25, vals, width=0.25, color=col, label=lab)
    pf.reference_line(ax, y=0.95, color=C['grey'])
    ax.set_xticks(x)
    ax.set_xticklabels(lab3)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel('LOSO coverage, 95% band')
    pf.legend(ax, loc='upper left', ncol=3, fontsize=6.2, bbox_to_anchor=(0.0, 1.02))
    pf.label_cell(fig, gs[1, 1], 'd')
    pf.save(fig, 'figS4_calib')


# ----------------------------------------------------------------------------------------------- Figure S5: booster path
def fig_s5():
    from figS5_S7_redesign import export_s5
    return export_s5()
    t = md_tables(TAB / 'T4_exp1_booster.md')
    paths = [tb for tb in t if tb[0][0] == 'lam frac' and tb[0][1] == 'path stations']
    checks = [tb for tb in t if tb[0][0] == 'lam frac' and tb[0][1] == 'obj matrix-free']
    NETS = [('Net3', pf.NET['Net3']), ('North Penn', pf.NET['NorthPenn'])]
    fig = pf.figure(height_mm=95)
    gs = pf.panel_grid(fig, 2, 2, bottom_mm=10, top_mm=5, hspace=0.5, wspace=0.3)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    lam = [r[0] for r in paths[0][1:]]
    xi = np.arange(len(lam))
    STY = [('path solution', (0, (1.2, 1.2)), 1), ('refit on the support', '-', 4), ('constant rate, same support', (0, (3, 1.5)), 7)]
    for (name, col), P, K in zip(NETS, paths, checks):
        rows = P[1:]
        for lab, ls, off in STY:
            axs[0].plot(xi, [num(r[off + 1]) for r in rows], color=col, ls=ls, lw=0.9)
            axs[1].plot(xi, [num(r[off + 2]) for r in rows], color=col, ls=ls, lw=0.9)
            axs[2].plot(xi, [num(r[off]) for r in rows], color=col, ls=ls, lw=0.9)
        rel = np.array([max(num(r[3]), 1e-17) for r in K[1:]])
        axs[3].plot(xi, rel, color=col, marker='o', ms=2.8, lw=0.9, mec=pf.darken(col))
    for k, ax in enumerate(axs):
        ax.set_xticks(xi)
        ax.set_xticklabels(lam, rotation=90, fontsize=6.2)
        ax.set_xlabel('penalty fraction $\\lambda/\\lambda_0$')
        pf.label_cell(fig, gs[k // 2, k % 2], 'abcd'[k])
    axs[0].set_ylabel('demand-weighted coverage')
    axs[0].set_ylim(0.85, 1.005)
    axs[1].set_ylabel('dosed mass (kg/d)')
    axs[1].set_yscale('log')
    axs[2].set_ylabel('selected stations')
    axs[3].set_ylabel('rel. objective difference,\nmatrix-free vs dense')
    axs[3].set_yscale('log')
    axs[3].set_ylim(1e-17, 1)
    h = [Line2D([], [], color=c, lw=0.9) for _, c in NETS] + [Line2D([], [], color=C['ink'], ls=ls, lw=0.9) for _, ls, _ in STY]
    l = [n for n, _ in NETS] + [s for s, _, _ in STY]
    pf.legend(axs[2], h, l, loc='upper left', ncol=1, fontsize=6.3)
    pf.save(fig, 'figS5_booster')


# ----------------------------------------------------------------------------------------------- Figure S6: sources
def fig_s6():
    from figS5_S7_redesign import export_s6
    return export_s6()
    text = (TAB / 'T5_exp2_source.md').read_text(encoding='utf-8')
    t = md_tables(TAB / 'T5_exp2_source.md')
    ident = [tb for tb in t if tb[0][0] == 'window h']
    timings = re.findall(r'timing s: ([^\n]*)', text)
    NETS = [('BWSN2', pf.NET['BWSN2']), ('L-Town', pf.NET['LTown'])]
    fig = pf.figure(height_mm=95)
    gs = pf.panel_grid(fig, 2, 2, bottom_mm=10, top_mm=6, hspace=0.55, wspace=0.3)
    WIN = ['2 h', '6 h', 'all data']
    x = np.arange(3)
    MET = [('GLRT top-1', 2, pf.METHOD['adjoint']), ('GLRT top-5', 3, C['purple_light']),
           ('GLRT correct or equivalent', 4, C['light']), ('NNLS top-1', 8, pf.METHOD['fd'])]
    for pi, ((name, col), T) in enumerate(zip(NETS, ident)):
        ax = fig.add_subplot(gs[0, pi])
        rows = T[1:]
        for k, (lab, idx, c) in enumerate(MET):
            pf.outlined_bar(ax, x + (k - 1.5) * 0.2, [num(r[idx]) for r in rows], width=0.2, color=c, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(WIN)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('fraction of detected scenarios')
        ax.set_xlabel('observation window after the first alarm')
        pf.header(fig, gs[0, pi], f'{name} ({rows[0][1]} detected scenarios)')
        pf.label_cell(fig, gs[0, pi], 'ab'[pi])
        if pi == 0:
            pf.legend(ax, loc='upper left', ncol=2, fontsize=6.2, bbox_to_anchor=(0.0, 1.02))
    ax = fig.add_subplot(gs[1, 0])
    for (name, col), T in zip(NETS, ident):
        rows = T[1:]
        ax.plot(x, [num(r[10]) for r in rows], color=col, marker='o', ms=3, lw=0.9, mec=pf.darken(col), label=f'{name}, adjoint sweep and GLRT')
        ax.plot(x, [num(r[11]) for r in rows], color=col, marker='s', ms=3, lw=0.9, ls=(0, (3, 1.5)), mec=pf.darken(col),
                label=f'{name}, one engine run per candidate and slot (projected)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(WIN)
    ax.set_ylabel('time per scenario (s)')
    ax.set_xlabel('observation window after the first alarm')
    ax.set_ylim(0.1, 1e13)
    pf.legend(ax, loc='upper left', fontsize=6.0, bbox_to_anchor=(0.0, 1.02))
    pf.label_cell(fig, gs[1, 0], 'c')
    ax = fig.add_subplot(gs[1, 1])
    KEYS = [('record_s', 'recording'), ('build_s', 'schedule build'), ('sweep_s', 'adjoint sweeps'), ('screen_s', 'screening'),
            ('refine_s', 'refinement'), ('exhaustive_s', 'exhaustive NNLS check')]
    shades = [C['purple'], C['purple_light'], C['blue'], C['green'], C['yellow'], C['light']]
    for i, (tim, (name, col)) in enumerate(zip(timings, NETS)):
        d = dict((k, float(v)) for k, v in re.findall(r'(\w+) ([\d.]+)', tim))
        bottom = 0
        for (k, lab), sh in zip(KEYS, shades):
            v = d.get(k, 0)
            ax.bar(i, v, bottom=bottom, color=sh, edgecolor=pf.darken(sh, 0.7), lw=0.4, width=0.55, label=lab if i == 0 else None)
            bottom += v
    ax.set_xticks([0, 1])
    ax.set_xticklabels([n for n, _ in NETS])
    ax.set_ylabel('wall-clock time of the study (s)')
    ax.set_ylim(0, 2400)
    pf.legend(ax, loc='upper right', ncol=2, fontsize=6.2)
    pf.label_cell(fig, gs[1, 1], 'd')
    pf.save(fig, 'figS6_source')


# ----------------------------------------------------------------------------------------------- Figure S7: identifiability
def fig_s7():
    from figS5_S7_redesign import export_s7
    return export_s7()
    t = md_tables(TAB / 'T6_exp3_pipe_kw.md')
    field = t[0][1:]
    sets = [tb for tb in t if tb[0][0] == 'sensor set']
    NETS = [('BWSN2', pf.NET['BWSN2']), ('L-Town', pf.NET['LTown'])]
    SETL = {'random': 'existing', 'random+dopt': '+ D-optimal', 'random+count': '+ count-greedy', 'random+random': '+ random'}
    fig = pf.figure(height_mm=95)
    gs = pf.panel_grid(fig, 2, 2, bottom_mm=10, top_mm=6, hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    for (name, col), S in zip(NETS, sets):
        rows = S[1:]
        px = [num(r[3]) for r in rows]
        py = [num(r[4]) for r in rows]
        ax.plot(px, py, ls='none', marker='o', ms=4, color=col, mec=pf.darken(col), label=name)
        groups = {}
        for r, a, b in zip(rows, px, py):
            groups.setdefault((a, b), []).append(SETL[r[0]])
        left = name == 'L-Town'
        for (a, b), labs in groups.items():
            if left and a > 6:
                off, ha = (-5, 3), 'right'
            elif a > 26:
                off, ha = (5, -3), 'left'
            elif a > 22:
                off, ha = (-5, -3), 'right'
            else:
                off, ha = (4, -9), 'left'
            ax.annotate(' / '.join(labs), (a, b), off, textcoords='offset points', fontsize=6.0, color=pf.darken(col), ha=ha)
    ax.plot([0, 34], [0, 34], color=C['grey'], ls=(0, (3, 2)), lw=0.6)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 35)
    ax.set_xlabel('identifiable pipes predicted before re-estimation')
    ax.set_ylabel('identifiable pipes realised')
    pf.legend(ax, loc='upper left', fontsize=6.3)
    pf.label_cell(fig, gs[0, 0], 'a')
    x = np.arange(4)
    ax = fig.add_subplot(gs[0, 1])
    for k, ((name, col), S) in enumerate(zip(NETS, sets)):
        rows = S[1:]
        pf.outlined_bar(ax, x + (k - 0.5) * 0.36, [num(r[5]) for r in rows], width=0.36, color=col, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([SETL[r[0]] for r in sets[0][1:]])
    ax.set_ylabel('effective number of\ndetermined wall parameters')
    pf.legend(ax, loc='upper left', fontsize=6.3)
    pf.label_cell(fig, gs[0, 1], 'b')
    ax = fig.add_subplot(gs[1, 0])
    for k, ((name, col), S) in enumerate(zip(NETS, sets)):
        rows = S[1:]
        ratio = [num(r[7]) / num(r[8]) for r in rows]
        pf.outlined_bar(ax, x + (k - 0.5) * 0.36, ratio, width=0.36, color=col, label=name)
    pf.reference_line(ax, y=1.0, color=C['grey'])
    ax.set_xticks(x)
    ax.set_xticklabels([SETL[r[0]] for r in sets[0][1:]])
    ax.set_ylabel('$K_w$ RMSE, identifiable pipes\n(MAP / grouped prior)')
    ax.set_ylim(0, 1.1)
    pf.label_cell(fig, gs[1, 0], 'c')
    ax = fig.add_subplot(gs[1, 1])
    names = ['Fairfield', 'Bellingham', 'North Penn']
    xf = np.arange(3)
    before = [num(r[6]) for r in field]
    after = [num(r[11]) for r in field]
    pf.outlined_bar(ax, xf - 0.18, before, width=0.36, color=C['light'], label='identifiable pipes, existing sites')
    pf.outlined_bar(ax, xf + 0.18, after, width=0.36, color=pf.METHOD['adjoint'], label='identifiable pipes, + 5 D-optimal sites')
    ax.set_xticks(xf)
    ax.set_xticklabels(names)
    ax.set_ylabel('identifiable pipes')
    ax.set_ylim(0, 32)
    ax2 = ax.twinx()
    ax2.plot(xf - 0.18, [num(r[7]) for r in field], ls='none', marker='D', ms=3.5, color='white', mec=pf.darken(C['yellow']),
             label='effective parameters, existing sites')
    ax2.plot(xf + 0.18, [num(r[12]) for r in field], ls='none', marker='D', ms=3.5, color=pf.darken(C['yellow'], 0.7),
             label='effective parameters, + 5 sites')
    ax2.set_ylabel('effective number of parameters')
    ax2.set_ylim(0, 32)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    pf.legend(ax, h1 + h2, l1 + l2, loc='upper left', fontsize=6.0, bbox_to_anchor=(0.0, 1.02))
    pf.label_cell(fig, gs[1, 1], 'd')
    pf.save(fig, 'figS7_ident')


if __name__ == '__main__':
    for f in (fig_s1, fig_s2, fig_s3, fig_s4, fig_s5, fig_s6, fig_s7):
        f()
        plt.close('all')
