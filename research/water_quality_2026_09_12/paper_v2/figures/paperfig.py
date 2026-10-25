"""Shared style, data locations and helpers for the manuscript figures.
Every fig*.py imports this module; run the scripts from any directory (paths are resolved relative to this file).
Data precedence mirrors paper/build_tables.py: final_5090/<stage> when present, otherwise the earlier 5090 run.

Style contract (Nature-family / ES&T compatible):
  * width 176 mm (fits Nature's 183 mm double column and ES&T's 7.00 in = 504 pt hard maximum), height <= 220 mm;
    the figure is placed at ~177 mm, so native size is final size -- buy room for larger type with HEIGHT, never by
    shrinking text
  * Arial 7.5 pt body, 7 pt ticks, 7 pt legends, 9 pt bold lower-case panel letters aligned per grid column
  * glyph floor MIN_PT = 4.9 pt in the exported PDF. mathtext renders sub/superscripts at MATH_SHRINK = 0.7x the
    surrounding size, so any base below 7 pt yields illegal 3.8-4.2 pt exponents: nothing in FS may drop below 7 pt
  * left/bottom spines only, 0.5 pt, outward 2 pt ticks, light-grey major grid behind the data on cartesian panels
    (none on network maps or axis-off panels), no axes titles inside panels except short headers
  * one low-saturation palette of four hues plus greys, each hue supplied as a mid-tone stroke (`purple`), a muted
    fill (`purple_light`), a pale band (`purple_pale`) and a darker same-hue outline (`purple_line`); bars and
    patches take the muted fill with a 0.5 pt `_line` edge (see outlined_bar). Roles: purple = exact adjoint /
    our method / first network
    (L-Town, Net3, Fairfield), yellow = EPANET finite differences / baseline, blue = alternative or secondary method
    (Bellingham as third system), green = third method / second network (BWSN2, North Penn), light tints of the same
    hues for fills and bands, greys = context (network background, prior-dominated)
  * legends frameless, placed in empty space or as a shared strip; direct labels preferred to legends
  * math set in Arial through mathtext so labels and formulas share one typeface
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent            # paper/figures
PAPER = HERE.parent                                # paper
RES = PAPER.parent                                 # research/water_quality_2026_09_12
REPO = RES.parent.parent                           # D:/WDN/quantum (networks/ lives here)
FINAL = RES / 'final_5090'
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 176         # final double-column width in mm; 176 mm = 498.9 pt leaves headroom under the 504 pt ceiling
SINGLE = 86 * MM       # single column (Nature 89 mm, ES&T 3.33 in)
DOUBLE = WIDTH_MM * MM
MAXH = 220 * MM
MAX_PT_W = 504.0       # ES&T maximum figure width (7.00 in). save() measures the PDF media box and asserts on this
DPI = 600              # raster export resolution (Nature: >= 300 dpi for images, 600 dpi for line art)

# Glyph floor. No text in the exported PDF may render below MIN_PT at native size. matplotlib draws mathtext
# sub/superscripts at MATH_SHRINK (~0.7x) of the surrounding font size -- a 5.5 pt base gives a 3.85 pt exponent and
# a 6 pt base a 4.2 pt exponent, both illegal. 7 pt is therefore the hard floor for every entry in FS
# (7 x 0.7 = 4.9 pt), and anything carrying a subscript or exponent should sit at 7 pt or above.
MIN_PT = 4.9
MATH_SHRINK = 0.7
MIN_BASE_PT = round(MIN_PT / MATH_SHRINK, 2)   # 7.0 pt: smallest base whose mathtext exponents still clear MIN_PT

# Four low-saturation hues in four steps each: `<hue>` mid-tone stroke / marker, `<hue>_light` muted fill,
# `<hue>_pale` band or halo, `<hue>_line` darker same-hue outline for fills. Plus a neutral grey ramp.
# Every key that existed before is still present; only the hue values were pulled down in saturation.
C = {'purple': '#7A6BA6', 'blue': '#4E7CA3', 'yellow': '#C79A33', 'green': '#5C9A55',
     'purple_light': '#B6ADD2', 'blue_light': '#A4BCD3', 'yellow_light': '#E2C57C', 'green_light': '#A9CAA2',
     'purple_pale': '#E7E3F1', 'blue_pale': '#E2EAF2', 'yellow_pale': '#F6EDD5', 'green_pale': '#E4F0E1',
     'purple_line': '#4C4179', 'blue_line': '#2F5678', 'yellow_line': '#8A6A18', 'green_line': '#366631',
     'grey_line': '#5E5E5E',
     'ink': '#1A1A1A', 'black': '#000000', 'grey': '#8C8C8C', 'mid': '#B0B0B0', 'light': '#D4D4D4', 'pale': '#EBEBEB',
     'grid': '#E6E6E6', 'white': '#FFFFFF'}

# face colour -> darker same-hue outline, so outlined_bar / edge_for can derive an edge from any palette fill
LINE_OF = {}
for _h, _edge in (('purple', 'purple_line'), ('blue', 'blue_line'), ('yellow', 'yellow_line'), ('green', 'green_line')):
    for _step in ('', '_light', '_pale'):
        LINE_OF[C[_h + _step].lower()] = C[_edge]
for _n in ('grey', 'mid', 'light', 'pale'):
    LINE_OF[C[_n].lower()] = C['grey_line']

# semantic roles shared by all figures
METHOD = {'adjoint': C['purple'], 'fd': C['yellow'], 'fd_default': C['yellow_light'], 'spmm': C['blue'], 'per_op': C['grey'],
          'gather': C['mid']}
NET = {'LTown': C['purple'], 'BWSN2': C['green'], 'Net3': C['purple'], 'NorthPenn': C['green'],
       'Fairfield': C['purple'], 'Bellingham': C['blue'], 'Net1': C['grey'], 'Net2': C['green'], 'Jilin': C['yellow']}
IDENT = {'ident': C['purple'], 'part': C['purple_light'], 'prior': C['light']}      # identifiability classes on network maps
ENGINE_COLORS = {'EPANET FD': METHOD['fd'], 'adjoint (v1)': METHOD['per_op'], 'fused (v2)': METHOD['adjoint'],
                 'SpMM (v3)': METHOD['spmm']}
# Nothing below MIN_BASE_PT (7 pt) may appear here: 'small' at 5.5 pt was what produced 3.85 pt mathtext exponents.
# Keep in sync with setup()'s rcParams (font.size, axes.labelsize, xtick/ytick.labelsize, legend.fontsize, axes.titlesize).
FS = {'body': 7.5, 'tick': 7, 'legend': 7, 'small': 7, 'panel': 9, 'header': 7.5}
assert min(FS.values()) >= MIN_BASE_PT, f'FS entry below the {MIN_BASE_PT} pt mathtext floor: {FS}'


def pick(final_sub, fallback_sub):
    d = FINAL / final_sub
    if d.is_dir() and any(d.glob('*.json')):
        return d
    return RES / fallback_sub


DATA = {
    'g1': RES / 'hpc/results_5090/diffwq_g123_1586077',
    'g2fd': RES / 'hpc/results_5090/diffwq_g123_1586077',
    'g3L': RES / 'hpc/results_5090/diffwq_g3L_1586093',
    'v2': RES / 'v2_5090',
    'g3s': pick('g3s', 'g3s_5090'),
    'fds': FINAL / 'fds',
    'calib': RES / 'v2_5090',                      # g4_fused_*_cuda.json (fused engine); g4_graph_5090 is the v1 run
    'h2': RES / 'h2_5090',
    'exp1': pick('exp1', 'exp1_5090'),
    'exp2': RES / 'final_v2' / 'exp2',  # v2: re-optimised placement (R1-7)
    'exp3': pick('exp3', 'exp3_5090'),
}
INP = {
    'Net3': REPO / 'EPANET-dev/example-networks/Net3.inp',
    'LTown': REPO / 'networks/L-Town/L-TOWN.inp',
    'BWSN2': REPO / 'networks/asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp',
    'NorthPenn': REPO / 'networks/asce-tf-wdst/North Penn Water Authority System/NPCL-1.INP',
    'Fairfield': REPO / 'networks/asce-tf-wdst/Fairfield/FFCL-1.INP',
    'Bellingham': REPO / 'networks/asce-tf-wdst/Bellingham/BELL_CL1.INP',
}


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'mathtext.fallback': 'stixsans',
        'font.size': FS['body'], 'axes.titlesize': FS['header'], 'axes.labelsize': FS['body'],
        'xtick.labelsize': FS['tick'], 'ytick.labelsize': FS['tick'], 'legend.fontsize': FS['legend'],
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.5, 'legend.labelspacing': 0.3,
        'legend.borderaxespad': 0.3, 'legend.columnspacing': 1.0,
        'axes.linewidth': 0.5, 'axes.edgecolor': C['ink'], 'axes.labelcolor': C['ink'],
        'xtick.color': C['ink'], 'ytick.color': C['ink'], 'text.color': C['ink'],
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 2.0, 'ytick.major.size': 2.0, 'xtick.minor.size': 1.2, 'ytick.minor.size': 1.2,
        'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.labelpad': 2.5, 'axes.titlepad': 3.0,
        'lines.linewidth': 1.0, 'lines.markersize': 3.0, 'lines.markeredgewidth': 0.6,
        'patch.linewidth': 0.5, 'hatch.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'both', 'axes.grid.which': 'major', 'axes.axisbelow': True,
        'grid.color': C['grid'], 'grid.linewidth': 0.4, 'grid.linestyle': '-', 'grid.alpha': 1.0,
        'axes.prop_cycle': matplotlib.cycler(color=[C['purple'], C['blue'], C['yellow'], C['green'], C['grey'], C['purple_light']]),
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 150, 'savefig.dpi': DPI, 'axes.unicode_minus': True,
    })


def figure(width=DOUBLE, height_mm=110):
    return plt.figure(figsize=(width, height_mm * MM))


def panel_label(ax, letter, dx=-0.18, dy=1.04):
    """Panel letter relative to an axes (legacy). Prefer label_cell for per-column alignment."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FS['panel'], fontweight='bold', va='bottom', ha='left')


def label_cell(fig, cell, letter, dx_mm=7.0, dy_mm=0.5):
    """Bold lower-case panel letter anchored to a gridspec cell, so letters in one column align independently of tick-label widths."""
    b = cell.get_position(fig)
    w_in, h_in = fig.get_size_inches()
    fig.text(b.x0 - dx_mm * MM / w_in, b.y1 + dy_mm * MM / h_in, letter, fontsize=FS['panel'], fontweight='bold',
             va='bottom', ha='left')


def header(fig, cell, text, dy_mm=0.5, ha='center', fontsize=None, color=None):
    """Short panel header (e.g. network name) centred over a gridspec cell."""
    b = cell.get_position(fig)
    _, h_in = fig.get_size_inches()
    x = (b.x0 + b.x1) / 2 if ha == 'center' else b.x0 if ha == 'left' else b.x1
    fig.text(x, b.y1 + dy_mm * MM / h_in, text, fontsize=fontsize or FS['header'], ha=ha, va='bottom',
             color=color or C['ink'])


def log_ticks(ax, axis='y', numticks=None, minor=False):
    """10^n major tick labels, minor ticks without labels (hidden unless minor=True)."""
    a = ax.yaxis if axis == 'y' else ax.xaxis
    if numticks:
        a.set_major_locator(LogLocator(base=10.0, numticks=numticks))
    a.set_major_formatter(LogFormatterMathtext(base=10.0))
    a.set_minor_formatter(NullFormatter())
    ax.tick_params(axis=axis, which='minor', length=1.2 if minor else 0)


def despine(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)


def darken(color, f=0.62):
    """Same-hue darker shade of `color` (any matplotlib colour spec), for outlines around muted fills."""
    from matplotlib.colors import to_hex, to_rgb
    return to_hex(tuple(max(0.0, min(1.0, ch * f)) for ch in to_rgb(color)))


def edge_for(color, f=0.62):
    """Darker same-hue outline for a palette fill: the curated `<hue>_line` value when `color` is a palette
    colour, otherwise darken(color, f)."""
    from matplotlib.colors import to_hex
    try:
        key = to_hex(color).lower()
    except ValueError:
        return darken(color, f)
    return LINE_OF.get(key, darken(color, f))


def grid_only(ax, axis='y', color=None, lw=0.4, ls='-'):
    """Light major grid on one axis only (default y), for bar panels where a vertical grid is noise."""
    ax.grid(False)
    ax.grid(True, axis=axis, which='major', color=color or C['grid'], lw=lw, ls=ls)
    ax.set_axisbelow(True)
    return ax


def outlined_bar(ax, x, height, width=0.72, color=None, edgecolor=None, lw=0.5, horizontal=False, **kw):
    """Bars in the target style: muted face with a 0.5 pt darker same-hue outline.

    `color` is the face (typically C['<hue>_light'] or a METHOD/NET role colour); the edge defaults to
    edge_for(color). With horizontal=True, `x` is the y position and `height` the bar length (barh)."""
    face = color if color is not None else C['purple_light']
    opts = dict(color=face, edgecolor=edgecolor if edgecolor is not None else edge_for(face), linewidth=lw)
    opts.update(kw)
    if horizontal:
        return ax.barh(x, height, height=width, **opts)
    return ax.bar(x, height, width=width, **opts)


def panel_grid(fig, nrows, ncols, left_mm=12.0, right_mm=1.8, top_mm=6.0, bottom_mm=10.0,
               wspace=0.42, hspace=0.5, **kw):
    """GridSpec with the module's default spacing and margins given in mm rather than figure fractions,
    so the same margins hold whatever height a figure grows to. Extra kwargs (width_ratios, height_ratios,
    ...) pass straight through to fig.add_gridspec; a hero panel is just a width_ratios/height_ratios grid."""
    w_in, h_in = fig.get_size_inches()
    return fig.add_gridspec(nrows, ncols,
                            left=left_mm * MM / w_in, right=1 - right_mm * MM / w_in,
                            top=1 - top_mm * MM / h_in, bottom=bottom_mm * MM / h_in,
                            wspace=wspace, hspace=hspace, **kw)


def legend(ax_or_fig, handles=None, labels=None, **kw):
    """Frameless legend with the shared small style; handles/labels optional."""
    opts = dict(frameon=False, fontsize=FS['legend'], handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
                borderaxespad=0.3, columnspacing=1.0)
    opts.update(kw)
    if handles is not None and labels is not None:
        return ax_or_fig.legend(handles, labels, **opts)
    if handles is not None:
        return ax_or_fig.legend(handles=handles, **opts)
    return ax_or_fig.legend(**opts)


def note(ax, x, y, s, color=None, fontsize=None, **kw):
    """Direct label / annotation in axes fraction coordinates."""
    opts = dict(transform=ax.transAxes, fontsize=fontsize or FS['small'], color=color or C['ink'], ha='left', va='top',
                linespacing=1.25)
    opts.update(kw)
    return ax.text(x, y, s, **opts)


def reference_line(ax, y=None, x=None, color=None, ls=(0, (3, 2)), lw=0.5, **kw):
    col = color or C['grey']
    if y is not None:
        return ax.axhline(y, color=col, ls=ls, lw=lw, zorder=0.5, **kw)
    return ax.axvline(x, color=col, ls=ls, lw=lw, zorder=0.5, **kw)


def _png_to_tif(png, tif):
    """600-dpi RGB TIFF from the PNG export; deflate-compressed through tifffile (Pillow's libtiff crashes on this host),
    uncompressed Pillow as the fallback."""
    import numpy as np
    from PIL import Image
    rgb = Image.open(png).convert('RGB')
    try:
        import tifffile
        tifffile.imwrite(tif, np.asarray(rgb), compression='zlib', photometric='rgb', resolution=(DPI, DPI), resolutionunit='INCH')
    except ImportError:
        rgb.save(tif, dpi=(DPI, DPI))


def _tight_size_in(fig):
    """(width, height) in inches of the tight bounding box actually exported by bbox_inches='tight'."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.width, bb.height


def fit_width(fig, max_pt=None, tries=6):
    """Shrink the canvas until the tight bbox clears the width ceiling, and report by how much.

    bbox_inches='tight' exports the union of every artist, so text placed outside the canvas (panel letters left
    of column 0, legends or notes past the right edge) makes the PDF wider than the figsize. Trimming the canvas by
    the measured overflow pulls that text back in without touching any font size. Returns the overflow in pt at the
    first measurement (0.0 when the figure already fitted)."""
    limit_in = (max_pt or MAX_PT_W) / 72
    first = 0.0
    for i in range(tries):
        w_tight, _ = _tight_size_in(fig)
        over = w_tight - limit_in
        if i == 0:
            first = max(0.0, over * 72)
        if over <= 0.004:            # ~0.3 pt of slack
            break
        w_in, h_in = fig.get_size_inches()
        fig.set_size_inches(max(0.5, w_in - over - 0.004), h_in)
    return first


def _pdf_media_box(path):
    """(width_pt, height_pt) of page 1, or None when no PDF reader is installed."""
    try:
        import fitz
    except ImportError:
        return None
    with fitz.open(path) as d:
        r = d[0].rect
        return float(r.width), float(r.height)


def save(fig, name, formats=('pdf', 'svg', 'png', 'tif')):
    """Vector masters (.pdf, .svg with live text), a 600-dpi .png preview and a 600-dpi .tif for submission.

    pad_inches is 0 so the figure size alone governs the media box: the old 0.5 mm pad added ~2.8 pt per side and
    pushed a 177 mm figure to 504.9 pt, over ES&T's 504 pt ceiling. The measured PDF width is printed for every
    figure and asserted against MAX_PT_W. fit_width() runs first, so text extending past the canvas cannot carry
    the media box over the ceiling; when it has to trim, it says so and the figure should be re-laid out to reclaim
    the space deliberately."""
    over_pt = fit_width(fig)
    if over_pt > 0.3:
        print(f'  fit_width: tight bbox was {over_pt:.1f} pt over {MAX_PT_W:.0f} pt (text outside the canvas); '
              f'canvas trimmed to {fig.get_size_inches()[0] / MM:.1f} mm -- consider widening the gridspec margins instead')
    for ext in formats:
        if ext == 'tif':
            continue
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=0, dpi=DPI)
    if 'tif' in formats and 'png' in formats:
        _png_to_tif(OUT / f'{name}.png', OUT / f'{name}.tif')
    w, h = fig.get_size_inches()
    msg = f'wrote {OUT / name}.{{{",".join(formats)}}}  (figsize {w / MM:.0f} x {h / MM:.0f} mm)'
    box = _pdf_media_box(OUT / f'{name}.pdf') if 'pdf' in formats else None
    if box is not None:
        pw, ph = box
        print(f'{msg}  pdf {pw:.1f} x {ph:.1f} pt = {pw / 72 * 25.4:.1f} x {ph / 72 * 25.4:.1f} mm')
        assert pw <= MAX_PT_W, (f'{name}.pdf is {pw:.1f} pt wide, over the {MAX_PT_W:.0f} pt (7.00 in) ES&T maximum: '
                                f'reduce the figure width (DOUBLE = {DOUBLE / MM:.0f} mm) or trim outward-extending text')
        if ph > MAXH / MM * 72 / 25.4:
            print(f'  NOTE {name}.pdf is {ph / 72 * 25.4:.1f} mm tall, over the {MAXH / MM:.0f} mm page maximum')
    else:
        print(msg)


MAP = {   # EPANET-1 map files of the ASCE field systems (their INP files carry no [COORDINATES] section)
    'NorthPenn': REPO / 'networks/asce-tf-wdst/North Penn Water Authority System/LANSDAL1.MAP',
    'Fairfield': REPO / 'networks/asce-tf-wdst/Fairfield/FRFLD-3.MAP',
    'Bellingham': REPO / 'networks/asce-tf-wdst/Bellingham/DAK-YEW5.MAP',
}


def read_inp_geometry(path, map_path=None):
    """Node coordinates and link endpoints from an EPANET INP file (sections [COORDINATES], [PIPES], [PUMPS], [VALVES]);
    when the INP has no coordinates, they are read from an EPANET-1 .MAP file (`id x y [elev]` lines before any [LABEL*] section)."""
    coords, links, section = {}, [], None
    with open(path, encoding='latin-1') as f:
        for raw in f:
            line = raw.split(';', 1)[0].strip()
            if not line:
                continue
            m = re.match(r'\[(\w+)\]', line)
            if m:
                section = m.group(1).upper()
                continue
            parts = line.split()
            if section == 'COORDINATES' and len(parts) >= 3:
                try:
                    coords[parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
            elif section in ('PIPES', 'PUMPS', 'VALVES') and len(parts) >= 3:
                links.append((parts[0], parts[1], parts[2]))
    if not coords and map_path is not None and Path(map_path).exists():
        with open(map_path, encoding='latin-1') as f:
            for raw in f:
                line = raw.split(';', 1)[0].strip()
                if not line:
                    continue
                if line.startswith('['):
                    if line.upper().startswith('[LABEL'):
                        break
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coords[parts[0]] = (float(parts[1]), float(parts[2]))
                    except ValueError:
                        pass
    return coords, links


def draw_network(ax, coords, links, color=C['light'], lw=0.35, zorder=1, pad=0.02):
    """Network background: thin light-grey links, equal aspect, no axes; `pad` is the margin as a fraction of the extent."""
    import numpy as np
    from matplotlib.collections import LineCollection
    segs = [[coords[a], coords[b]] for _, a, b in links if a in coords and b in coords]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=zorder, capstyle='round'))
    xy = np.array(list(coords.values()))
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    w, h = x1 - x0, y1 - y0
    ax.set_xlim(x0 - pad * w, x1 + pad * w)
    ax.set_ylim(y0 - pad * h, y1 + pad * h)
    ax.set_aspect('equal')
    ax.axis('off')


def scale_bar(ax, length, unit_label, x0, y0, lw=1.2, fontsize=None):
    """Horizontal scale bar of `length` data units starting at (x0, y0) with a text label to its right."""
    ax.plot([x0, x0 + length], [y0, y0], color=C['ink'], lw=lw, solid_capstyle='butt', zorder=7)
    ax.text(x0 + length * 1.06, y0, unit_label, ha='left', va='center', fontsize=fontsize or FS['small'], zorder=7)
