"""Figure 4: contamination source identification from sparse sensors by one adjoint sweep (experiment 2).

Figure contract: an asymmetric quantitative grid with two network maps shows what
the detected scenarios support, their detection delay, and the measured versus
projected computational cost. All saved scenarios and the original summary
statistics, map-selection rules and 5th--95th percentile whiskers are retained.
Python/matplotlib; 176 mm wide; live-text PDF/SVG and 600-dpi PNG/TIFF exports.

a  identification accuracy versus observation window after the first alarm (GLRT top-1 / top-5 / correct-or-equivalent,
   NNLS top-1), L-Town and BWSN2
b  ECDF of the detection delay (injection start -> confirmed alarm) over the detected scenarios
c  wall-clock per scenario: adjoint pipeline (sweep + GLRT screen + NNLS refine) versus projected exhaustive EPANET
d  L-Town map, representative scenario: GLRT top-1 unique and correct at every window, NNLS top-1 wrong
e  BWSN2 map, window-dependence scenario: GLRT top-1 wrong at the shortest window, unique and correct afterwards
f  GLRT equivalence-set size per window and median pipe-distance error of the wrong GLRT top-1 cases

The JSON keeps the GLRT top-1 only as a pipe distance to the truth (exp2_source.py drops 'glrt_top1_node'), so the maps
draw the GLRT marker on the true source when the top-1 is correct and a dashed circle of radius glrt_top1_dist_m
(pipe distance >= straight-line distance) when it is wrong; equivalence sets are available as counts only.
Run: python paper/figures/fig4_exp2.py
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

NETS = [('LTown', 'L-Town'), ('BWSN2', 'BWSN2')]
CYAN, TEAL, GOLD = '#56B4E9', '#009E73', '#E69F00'
PURPLE, MAGENTA, VERM = '#7B61A8', '#CC79A7', '#D55E00'
INK, GREY, LIGHT = '#30363D', '#707A84', '#D6DCE1'
NET_COLOR = {'LTown': MAGENTA, 'BWSN2': TEAL}
US_FLOW_UNITS = {'CFS', 'GPM', 'MGD', 'IMGD', 'AFD'}
FT = 0.3048
H_MM = 166
FIG_W, FIG_H = pf.DOUBLE, H_MM * pf.MM
SMALL = pf.FS['small']
DASH = (0, (3, 2))


def load_net(key):
    net = pf.load(pf.DATA['exp2'] / f'exp2_{key}.json')['networks'][0]
    assert net['network'] == key
    return net


def win(s, w):
    return next(x for x in s['windows'] if x['window_h'] == w)


def wlabel(w):
    return 'full' if w == float('inf') else f'{w:g} h'


def coord_unit_m(inp_path):
    """Metres per coordinate unit: EPANET US flow units imply feet, SI flow units imply metres."""
    section = None
    with open(inp_path, encoding='latin-1') as f:
        for raw in f:
            line = raw.split(';', 1)[0].strip()
            if line.startswith('['):
                section = line.upper()
                continue
            parts = line.split()
            if section == '[OPTIONS]' and len(parts) >= 2 and parts[0].upper() == 'UNITS':
                return FT if parts[1].upper() in US_FLOW_UNITS else 1.0
    raise ValueError(f'no Units option in {inp_path}')


def fmt_time(sec):
    if sec < 60:
        return f'{sec:.1f} s'
    if sec < 3600:
        return f'{sec / 60:.0f} min'
    if sec < 86400:
        return f'{sec / 3600:.1f} h'
    return f'{sec / 86400:.0f} d'


def fmt_fold(r):
    e = int(np.floor(np.log10(r)))
    return rf'${r / 10 ** e:.1f}\times10^{{{e}}}$-fold'


def fmt_dist(d):
    if d is None:
        return 'n/a'
    return f'{d:.0f} m' if d < 1000 else f'{d / 1000:.1f} km'


# ----------------------------------------------------------------------------------------------------------------- a
def panel_accuracy(ax, nets, wins):
    """All 24 saved fractions; marker offsets separate methods without averaging."""
    series = [('glrt_top1', 'GLRT top-1', PURPLE, 'o'), ('glrt_top5', 'GLRT top-5', MAGENTA, 's'),
              ('glrt_correct_or_equivalent', 'GLRT correct/equiv.', TEAL, 'D'), ('top1', 'NNLS top-1', CYAN, '^')]
    ys, labels = [], []
    for gi, (key, name) in enumerate(NETS):
        summ = nets[key]['summary']
        assert [s['window_h'] for s in summ] == wins
        positions = 5.8 - gi * 3.7 - np.arange(len(wins))
        ax.axhspan(positions[-1] - .48, positions[0] + .48, color=NET_COLOR[key], alpha=.065, lw=0, zorder=0)
        ax.text(.02, positions[0] + .58, f'{name}  ·  n = {summ[0]["n"]}', ha='left', va='bottom',
                fontsize=7, color=NET_COLOR[key], fontweight='bold')
        for si, (field, label, colour, marker) in enumerate(series):
            ax.scatter([s[field] for s in summ], positions + (si - 1.5) * .16,
                       marker=marker, s=17, facecolor='white' if si == 1 else colour,
                       edgecolor=colour, linewidth=.7, label=label if gi == 0 else None, zorder=3)
        ys.extend(positions); labels.extend(wlabel(w) for w in wins)
    ax.set_yticks(ys, labels)
    ax.set_ylim(-.6, 7.1); ax.set_xlim(0, 1.045)
    ax.set_xticks([0, .5, 1], ['0', '0.5', '1.0'])
    ax.set_xlabel('Fraction of detected\nscenarios')
    ax.set_ylabel('Window after first alarm')
    ax.grid(axis='x', color='#E7EAED', linewidth=.35)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower left', bbox_to_anchor=(-.1, 1.08), ncol=1, fontsize=7,
              handlelength=1.0, labelspacing=.2, borderaxespad=0)


# ----------------------------------------------------------------------------------------------------------------- b
def panel_delay(ax, nets):
    """ECDF of the detection delay per network over its detected scenarios; the dot marks the median."""
    xmax = 1.05 * max(max(nets[k]['data']['detection_delay_h']) for k, _ in NETS)
    pf.reference_line(ax, y=0.5, color=pf.C['light'])
    ytxt = {'LTown': 1.28, 'BWSN2': 1.11}
    for key, name in NETS:
        d, col = nets[key]['data'], NET_COLOR[key]
        delays = np.sort(np.asarray(d['detection_delay_h'], float))
        assert len(delays) == d['n_detected']
        x = np.r_[0.0, delays, xmax]
        y = np.r_[0.0, np.arange(1, len(delays) + 1) / len(delays), 1.0]
        ax.fill_between(x, y, step='post', color=col, alpha=.075, linewidth=0)
        ax.step(x, y, where='post', color=col, lw=1.15)
        ax.plot(np.median(delays), 0.5, 'o', ms=3, color=col, zorder=3)
        pf.note(ax, 0.02, ytxt[key], f'{name}: {d["n_detected"]}/{d["n_sampled"]} detected\nmedian delay {np.median(delays):.1f} h',
                color=col, fontsize=pf.FS['legend'], ha='left', va='center')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Detection delay (h)')
    ax.set_ylabel('Fraction of detected scenarios')


# ----------------------------------------------------------------------------------------------------------------- c
def cost_breakdown(net):
    """Per-scenario wall-clock at the full-record window (the most expensive one)."""
    s = net['summary'][-1]
    parts = {'sweep': s['mean_wall_sweep_s'], 'screen': s['mean_wall_screen_s'], 'refine': s['mean_wall_refine_s']}
    return parts, sum(parts.values()), s['mean_projected_exhaustive_dll_s']


def panel_cost(ax, nets):
    """Two paired comparisons on a log axis, keeping measured/projected endpoints explicit."""
    for gi, (key, name) in enumerate(NETS):
        _, total, exh = cost_breakdown(nets[key])
        y = 1 - gi
        ax.hlines(y, total, exh, color=LIGHT, lw=2, zorder=1)
        ax.plot(total, y, 'D', ms=4.2, mfc=PURPLE, mec=PURPLE, mew=.6, zorder=3)
        ax.plot(exh, y, 's', ms=4.2, mfc='white', mec=GOLD, mew=.9, zorder=3)
        ax.text(total, y - .15, fmt_time(total), ha='center', va='top', fontsize=7)
        ax.text(exh, y - .15, fmt_time(exh), ha='center', va='top', fontsize=7)
        ax.text(np.sqrt(total * exh), y + .19, fmt_fold(exh / total), ha='center', va='bottom', fontsize=7.5)
        ax.text(.65, y + .45, name, ha='left', va='bottom', fontsize=7, fontweight='bold', color=NET_COLOR[key])
    ax.set_xscale('log'); ax.set_xlim(.5, 7e7); ax.set_ylim(-.5, 1.9)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xticks([1, 1e3, 1e6]); pf.log_ticks(ax, 'x', minor=False)
    ax.set_xlabel('Time per scenario (s)')
    ax.grid(axis='x', color='#E7EAED', lw=.35)
    pf.legend(ax, handles=[Line2D([], [], ls='none', marker='D', color=PURPLE, ms=4, label='Adjoint pipeline'),
                           Line2D([], [], ls='none', marker='s', mfc='white', mec=GOLD, ms=4, label='Library, projected')],
              loc='lower left', bbox_to_anchor=(-.05, 1.08), fontsize=7, borderaxespad=0)


# --------------------------------------------------------------------------------------------------------------- d, e
def choose_ltown(scen, wins):
    """GLRT unique and correct at every window, NNLS top-1 wrong at the full window; median NNLS pipe distance."""
    wl = wins[-1]
    c = [s for s in scen if all(win(s, w)['glrt_top1'] and win(s, w)['glrt_n_equiv'] == 1 for w in wins)
         and not win(s, wl)['top1'] and win(s, wl)['dist_m'] is not None]
    c.sort(key=lambda s: win(s, wl)['dist_m'])
    return (c[len(c) // 2], len(c)) if c else (None, 0)


def choose_bwsn2(scen, wins):
    """GLRT wrong at the shortest window, unique and correct at every later window; largest short-window set."""
    c = [s for s in scen if not win(s, wins[0])['glrt_top1']
         and all(win(s, w)['glrt_top1'] and win(s, w)['glrt_n_equiv'] == 1 for w in wins[1:])]
    return (max(c, key=lambda s: win(s, wins[0])['glrt_n_equiv']), len(c)) if c else (None, 0)


CHOOSE = {'LTown': choose_ltown, 'BWSN2': choose_bwsn2}


def nice_scale_m(width_m):
    """Largest round length not exceeding a quarter of the map width."""
    fits = [L for L in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000) if L <= 0.25 * width_m]
    return fits[-1] if fits else 100


def emptiest_corner(xy, avoid, x0, x1, y0, y1, fx=0.65, fy=0.35):
    """Corner rectangle (fx x fy of the bounding box) with the fewest nodes; corners holding an `avoid` point are excluded."""
    w, h = x1 - x0, y1 - y0
    boxes = {'lower right': (x1 - fx * w, x1, y0, y0 + fy * h), 'upper right': (x1 - fx * w, x1, y1 - fy * h, y1),
             'lower left': (x0, x0 + fx * w, y0, y0 + fy * h), 'upper left': (x0, x0 + fx * w, y1 - fy * h, y1)}

    def inside(p, box):
        xa, xb, ya, yb = box
        return (p[:, 0] >= xa) & (p[:, 0] <= xb) & (p[:, 1] >= ya) & (p[:, 1] <= yb)

    counts = {k: int(inside(xy, b).sum()) + (10 ** 6 if inside(np.asarray(avoid), b).any() else 0) for k, b in boxes.items()}
    return min(counts, key=counts.get), counts


def glrt_lines(scen, wins):
    """One line per run of consecutive windows with the same GLRT outcome, e.g. 'GLRT 6 h, full: correct, unique'."""
    groups = []
    for w in wins:
        x = win(scen, w)
        status = ('correct' if x['glrt_top1'] else f'{fmt_dist(x["glrt_top1_dist_m"])} off') + \
                 (', unique' if x['glrt_n_equiv'] == 1 else f', {x["glrt_n_equiv"]} equiv.')
        if groups and groups[-1][1] == status:
            groups[-1][0].append(wlabel(w))
        else:
            groups.append(([wlabel(w)], status))
    return [f'GLRT {", ".join(ls)}: {st}' for ls, st in groups]


def panel_map(fig, cell, ax, name, net, scen, wins, coords, links, unit_m):
    pf.draw_network(ax, coords, links, color='#C7D0D7', lw=.42)
    xy = np.array(list(coords.values()))
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    w, h = x1 - x0, y1 - y0
    ax.set_xlim(x0 - 0.02 * w, x1 + 0.02 * w)
    pad = 0.50 if w / h > 1.6 else 0.34          # room under the map for the scenario text
    ax.set_ylim(y0 - pad * h, y1 + 0.02 * h)
    ax.set_anchor('N')
    sens = net['placement']['sensors']
    missing = [s for s in sens if s not in coords]
    sx = np.array([coords[s] for s in sens if s in coords])
    ax.plot(sx[:, 0], sx[:, 1], 's', ms=3.2, mfc='white', mec=INK, mew=0.7, ls='none', zorder=4)
    truth, last = scen['truth'], win(scen, wins[-1])
    tx, ty = coords[truth['node']]
    for k_, w_ in enumerate(wins):
        x = win(scen, w_)
        if not x['glrt_top1'] and x['glrt_top1_dist_m'] is not None:
            ax.add_patch(Circle((tx, ty), x['glrt_top1_dist_m'] / unit_m, fill=False, ec=PURPLE, lw=0.8, ls=DASH, zorder=5))
        elif x['glrt_top1']:      # nested rings, one per window where GLRT is already correct
            ax.plot(tx, ty, 'o', ms=8 + 2.6 * k_, mfc='none', mec=PURPLE, mew=0.9, zorder=5)
    bx, by = coords[last['best']]
    ax.plot(bx, by, 'D', ms=4.6, mfc=CYAN, mec='white', mew=0.4, zorder=5)
    ax.plot(tx, ty, '*', ms=8.5, mfc=VERM, mec='white', mew=.3, zorder=6)
    nnls = 'correct' if last['top1'] else f'{fmt_dist(last["dist_m"])} off ({last["outcome"]})'
    lines = [f'Source: {truth["node"]}'] + glrt_lines(scen, wins) + [f'NNLS {wlabel(wins[-1])}: {nnls}']
    corner, counts = 'below map', {}
    ax.text(x1, y0 - 0.13 * h, '\n'.join(lines), ha='right', va='top', fontsize=SMALL, linespacing=1.2)
    L = nice_scale_m(w * unit_m)
    pf.scale_bar(ax, L / unit_m, f'{L / 1000:g} km' if L >= 1000 else f'{L} m', x0, y0 - 0.07 * h)
    delay = truth['t_det_h'] - truth['start_h']
    pf.header(fig, cell, f'{name} \u00b7 alarm {delay:.1f} h', color=NET_COLOR[net['network']], fontsize=7.5)
    return {'missing_sensor_coords': missing, 'annotation_corner': corner, 'corner_node_counts': counts, 'scale_bar_m': L,
            'delay_h': delay, 'lines': lines}


def map_legend(fig, cell_d, cell_e):
    handles = [Line2D([], [], ls='none', marker='s', ms=3.2, mfc='white', mec=INK, mew=0.7, label='sensor'),
               Line2D([], [], ls='none', marker='*', ms=7, mfc=VERM, mec='none', label='True source'),
               Line2D([], [], ls='none', marker='o', ms=7, mfc='none', mec=PURPLE, mew=0.9, label='GLRT correct'),
               Line2D([], [], color=PURPLE, lw=0.7, ls=DASH, label='GLRT wrong: distance bound'),
               Line2D([], [], ls='none', marker='D', ms=4, mfc=CYAN, mec='white', mew=0.4, label='NNLS top-1')]
    bd, be = cell_d.get_position(fig), cell_e.get_position(fig)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=((bd.x0 + be.x1) / 2, bd.y0 - 1.5 * pf.MM / FIG_H),
               ncol=3, fontsize=pf.FS['legend'], frameon=False, handletextpad=0.5, columnspacing=1.2, handlelength=1.6)


# ----------------------------------------------------------------------------------------------------------------- f
def n_wrong(net, w):
    return sum(not win(s, w)['glrt_top1'] for s in net['scenarios'])


def panel_equivalence(ax_top, ax_bot, nets, wins):
    xs = np.arange(len(wins))
    off = {'LTown': -0.19, 'BWSN2': 0.19}
    ymax, emax = 1, 0.0
    for key, name in NETS:
        col, net = NET_COLOR[key], nets[key]
        data = [[win(s, w)['glrt_n_equiv'] for s in net['scenarios']] for w in wins]
        ymax = max(ymax, max(max(d) for d in data))
        ax_top.boxplot(data, positions=xs + off[key], widths=0.3, whis=(5, 95), patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor=plt.matplotlib.colors.to_rgba(col, .18), edgecolor=col, lw=0.75), medianprops=dict(color=col, lw=1.0),
                       whiskerprops=dict(color=col, lw=0.6), capprops=dict(color=col, lw=0.6),
                       flierprops=dict(marker='o', ms=1.6, mfc=col, mec='none', alpha=0.7))
        err = [s['median_glrt_top1_dist_m_wrong'] or 0.0 for s in net['summary']]
        emax = max(emax, max(err))
        ax_bot.vlines(xs + off[key], 0, err, color=col, lw=1.2)
        ax_bot.plot(xs + off[key], err, 'o', ms=4, mfc=col, mec='white', mew=.5, ls='none')
        for x, e, w in zip(xs + off[key], err, wins):
            ax_bot.text(x, e, f'n={n_wrong(net, w)}', ha='center', va='bottom', fontsize=SMALL)
    ax_top.set_yscale('log')
    ax_top.set_ylim(0.75, ymax * 2.5)
    ax_top.set_yticks([1, 10, 100])
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax_top.yaxis.set_minor_formatter(NullFormatter())
    ax_top.tick_params(axis='y', which='minor', left=False)
    ax_top.set_xticks(xs)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel('GLRT equivalent\nsources (count)')
    pf.legend(ax_top, handles=[Patch(facecolor=NET_COLOR[k], alpha=0.7, label=n) for k, n in NETS], loc='lower left',
              bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.0, handleheight=0.8, borderaxespad=0)
    ax_bot.set_ylim(0, emax * 1.3)
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels([wlabel(w) for w in wins])
    ax_bot.set_ylabel('Median error of\nwrong top-1 (m)')
    ax_bot.set_xlabel('Window after first alarm')
    for a in (ax_top, ax_bot):
        a.set_xlim(-0.6, len(wins) - 0.4)


# ------------------------------------------------------------------------------------------------------------ report
def report(nets, wins, chosen, info, unit):
    for key, name in NETS:
        net, d, pl = nets[key], nets[key]['data'], nets[key]['placement']
        delays = np.asarray(d['detection_delay_h'], float)
        print(f'\n== {name}: {len(pl["sensors"])} sensors {pl["sensors"]}')
        print(f'   detected {d["n_detected"]}/{d["n_sampled"]} (false alarms {d["n_false_alarm"]}); delay median {np.median(delays):.2f} h, '
              f'IQR {np.percentile(delays, 25):.2f}-{np.percentile(delays, 75):.2f} h, max {delays.max():.2f} h; {len(net["scenarios"])} scenarios in file')
        for s in net['summary']:
            w = s['window_h']
            neq = np.array([win(x, w)['glrt_n_equiv'] for x in net['scenarios']])
            tot = s['mean_wall_sweep_s'] + s['mean_wall_screen_s'] + s['mean_wall_refine_s']
            ex = s['exhaustive']
            print(f'   [{wlabel(w):>4}] n={s["n"]}  GLRT top-1 {s["glrt_top1"]:.3f}  top-5 {s["glrt_top5"]:.3f}  correct/equiv {s["glrt_correct_or_equivalent"]:.3f}'
                  f' | NNLS top-1 {s["top1"]:.3f}  top-5 {s["top5"]:.3f}  correct/equiv {s["correct_or_equivalent"]:.3f}')
            print(f'          GLRT n_equiv: median {np.median(neq):g}, p95 {np.percentile(neq, 95):g}, max {neq.max()}, unique {np.mean(neq == 1):.3f}'
                  f' | wrong top-1: n={n_wrong(net, w)} (summary implies {round((1 - s["glrt_top1"]) * s["n"])}), median dist {s["median_glrt_top1_dist_m_wrong"]} m')
            print(f'          cost/scenario: sweep {s["mean_wall_sweep_s"]:.3f} + screen {s["mean_wall_screen_s"]:.3f} + refine {s["mean_wall_refine_s"]:.3f}'
                  f' = {tot:.2f} s; projected exhaustive EPANET {s["mean_projected_exhaustive_dll_s"]:.0f} s ({fmt_time(s["mean_projected_exhaustive_dll_s"])}),'
                  f' {s["mean_projected_exhaustive_dll_s"] / tot:.3g}-fold; adjoint all-junction refine (n={ex["n"]}) {ex["mean_wall_refine_s"]:.2f} s, top-1 {ex["top1"]:.2f}')
        s, n_cand = chosen[key]
        t = s['truth']
        print(f'   map scenario ({CHOOSE[key].__doc__.strip()}; {n_cand} candidates): node {t["node"]}, start {t["start_h"]:.2f} h, '
              f'dur {t["dur_h"]:.2f} h, mass {t["mass_kg"]:.2f} kg, t_det {t["t_det_h"]:.2f} h (delay {info[key]["delay_h"]:.2f} h)')
        for w in wins:
            x = win(s, w)
            print(f'      [{wlabel(w):>4}] GLRT top-1 {x["glrt_top1"]}, rank_true {x["glrt_rank_true"]}, n_equiv {x["glrt_n_equiv"]}, dist {x["glrt_top1_dist_m"]} m'
                  f' | NNLS best {x["best"]}, outcome {x["outcome"]}, rank_true {x["rank_true"]}, n_equiv {x["n_equiv"]}, dist {x["dist_m"]} m')
        print(f'   map: coordinate unit {unit[key]} m, scale bar {info[key]["scale_bar_m"]} m, annotation {info[key]["annotation_corner"]} '
              f'(corner node counts {info[key]["corner_node_counts"]}), sensors without coordinates {info[key]["missing_sensor_coords"]}')
        print('   map text: ' + ' | '.join(info[key]['lines']))


def main():
    pf.setup()
    plt.rcParams.update({'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': GREY,
                         'xtick.color': INK, 'ytick.color': INK, 'grid.color': '#E7EAED',
                         'grid.linewidth': .35, 'axes.grid.axis': 'y'})
    nets = {key: load_net(key) for key, _ in NETS}
    wins = [s['window_h'] for s in nets['LTown']['summary']]
    geo = {key: pf.read_inp_geometry(pf.INP[key]) for key, _ in NETS}
    unit = {key: coord_unit_m(pf.INP[key]) for key, _ in NETS}
    chosen = {key: CHOOSE[key](nets[key]['scenarios'], wins) for key, _ in NETS}
    for key, (s, _) in chosen.items():
        assert s is not None, f'no scenario satisfies the {key} selection rule'

    fig = pf.figure(height_mm=H_MM)
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc='white', ec='none', zorder=-10))
    gs_top = fig.add_gridspec(1, 3, left=0.085, right=0.98, top=0.81, bottom=0.535, wspace=0.45, width_ratios=[1.08, 1, 1])
    gs_bot = fig.add_gridspec(1, 3, left=0.065, right=0.98, top=0.43, bottom=0.10, wspace=0.34, width_ratios=[1.10, 1.10, .95])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs_top[0, i]) for i in range(3))
    ax_d, ax_e = fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])
    sub = gs_bot[0, 2].subgridspec(2, 1, hspace=0.52)
    ax_f1, ax_f2 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])

    panel_accuracy(ax_a, nets, wins)
    panel_delay(ax_b, nets)
    panel_cost(ax_c, nets)
    info = {key: panel_map(fig, cell, ax, name, nets[key], chosen[key][0], wins, *geo[key], unit[key])
            for (key, name), cell, ax in zip(NETS, (gs_bot[0, 0], gs_bot[0, 1]), (ax_d, ax_e))}
    panel_equivalence(ax_f1, ax_f2, nets, wins)
    for cell, letter, title in zip((gs_top[0, 0], gs_top[0, 1], gs_top[0, 2]), 'abc',
                                   ['Identification accuracy', 'Detection delay', 'Measured vs projected']):
        pos = cell.get_position(fig)
        fig.text(pos.x0 - 7 * pf.MM / FIG_W, .973, letter, fontsize=9, fontweight='bold', va='top')
        fig.text(pos.x0, .972, title, fontsize=7.7, fontweight='bold', va='top')
    for cell, s in zip((gs_bot[0, 0], gs_bot[0, 1], gs_bot[0, 2]), 'def'):
        pf.label_cell(fig, cell, s)
    map_legend(fig, gs_bot[0, 0], gs_bot[0, 1])
    pf.save(fig, 'fig4_exp2')
    export_source(nets, wins, chosen, info)
    report(nets, wins, chosen, info, unit)


def export_source(nets, wins, chosen, info):
    """Retain unrounded source values and the complete distributions behind panels."""
    rows = {'accuracy': [], 'delay': [], 'cost': [], 'equivalence': [], 'wrong_distance': []}
    for key, _ in NETS:
        net = nets[key]
        for s in net['summary']:
            for field in ['glrt_top1', 'glrt_top5', 'glrt_correct_or_equivalent', 'top1']:
                rows['accuracy'].append(dict(network=key, window=wlabel(s['window_h']), method=field,
                                             fraction=s[field], detected_n=s['n']))
            rows['wrong_distance'].append(dict(network=key, window=wlabel(s['window_h']),
                median_distance_m=s['median_glrt_top1_dist_m_wrong'], wrong_n=n_wrong(net, s['window_h'])))
        for i, d in enumerate(net['data']['detection_delay_h']):
            rows['delay'].append(dict(network=key, scenario_index=i, delay_h=d))
        parts, total, projected = cost_breakdown(net)
        for part, value in [*parts.items(), ('total_measured', total), ('library_projected', projected)]:
            rows['cost'].append(dict(network=key, component=part, seconds=value, window='full'))
        for i, scenario in enumerate(net['scenarios']):
            for w in wins:
                rows['equivalence'].append(dict(network=key, scenario_index=i, window=wlabel(w),
                                                size=win(scenario, w)['glrt_n_equiv']))
    expected = {'accuracy': 24, 'delay': 251, 'cost': 10, 'equivalence': 753, 'wrong_distance': 6}
    assert {k: len(v) for k, v in rows.items()} == expected
    for key, data in rows.items():
        with (pf.OUT / f'fig4_source_{key}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    inputs = [pf.DATA['exp2'] / f'exp2_{key}.json' for key, _ in NETS] + [pf.INP[key] for key, _ in NETS]
    manifest = {'dimensions_mm': [176, H_MM], 'data_rows': expected,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        'selected_source_nodes': {key: chosen[key][0]['truth']['node'] for key, _ in NETS},
        'encoding': {'network': NET_COLOR, 'GLRT_top1': PURPLE, 'GLRT_top5': MAGENTA,
                     'GLRT_correct_or_equivalent': TEAL, 'NNLS': CYAN,
                     'true_source': VERM, 'projected_library': GOLD},
        'intervals': 'Equivalence boxes show quartiles, median and 5th--95th percentile whiskers; all outside observations are drawn.',
        'measurement_limits': ['Accuracy and delay denominators are detected scenarios, not all 200 sampled scenarios.',
             'Pipeline cost is the saved full-window component sum; library cost is projected.',
             'A wrong GLRT map circle is the reported pipe-distance radius, not a confidence region or a recovered node coordinate.'],
        'map_details': info}
    (pf.OUT / 'fig4_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
