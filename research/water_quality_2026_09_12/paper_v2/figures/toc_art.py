"""ES&T Table-of-Contents graphic: exactly 8.25 x 4.45 cm (ACS standard 3.25 x 1.75 in).

Pictorial version: icons carry the story and words are kept to a minimum. Left to right, the
real L-Town distribution network (its own [COORDINATES], 909 links, transmission mains drawn
heavier) with a tank, a served district and a pipe of Lagrangian parcels -> the recorded
transport schedule as a stack of immutable parcel versions -> a GPU chip that replays it and
runs the adjoint -> the exact gradient vector -> four application cards, each with its own
pictogram (chlorine calibration, booster design, source location, pipe-level identifiability).
A return path along the bottom carries the measured agreement when the result is fed back into
EPANET. Every label starts with a capital letter.

Type floor: every label is >= 7 pt, so mathtext exponents render at 8 x 0.7 = 5.6 pt, clear of
pf.MIN_PT (4.9 pt). No axes, hence no tick labels. The only number shown is
g2_max_fwd_rel_vs_dll from paper/tables/numbers.json (max forward relative difference against
the EPANET DLL over the 25 G2 cases).
Run: python paper/figures/toc_art.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
import paperfig as pf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

pf.setup()
W_CM, H_CM = 8.25, 4.45
W, H = W_CM * 10, H_CM * 10           # mm; the full-figure axes uses mm as data units
BODY, EMPH, NUM = 7.0, 7.5, 8.0       # pt; nothing below pf.MIN_BASE_PT = 7.0
INK, GREY = pf.C['ink'], pf.C['grey']
BLUE, BLUEL, BLUEP = pf.C['blue'], pf.C['blue_light'], pf.C['blue_pale']
BLUEE = pf.C['blue_line']
APP_C = [pf.C['purple'], pf.C['blue'], pf.C['yellow'], pf.C['green']]
APP_E = [pf.C['purple_line'], pf.C['blue_line'], pf.C['yellow_line'], pf.C['green_line']]
assert min(BODY, EMPH, NUM) >= pf.MIN_BASE_PT, 'TOC art font below the mathtext floor'

REL = pf.load(pf.PAPER / 'tables' / 'numbers.json')['g2_max_fwd_rel_vs_dll']

fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
# opaque full-canvas rect: bbox_inches='tight' in pf.save() then exports the exact 8.25 x 4.45 cm box
fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc='white', ec='none', zorder=-5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')


# ---------------------------------------------------------------- small drawing primitives
def arrow(x0, y0, x1, y1, color=INK, lw=0.7, scale=6.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=scale,
                                 connectionstyle=f'arc3,rad={rad}', lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=5))


def rbox(x0, y0, w, h, face, edge, r=0.8, lw=0.6, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=face, ec=edge, lw=lw, zorder=z))


def chip(cx, cy, s=5.4, pins=5, plen=1.15):
    """Processor pictogram: gold pins above and below, dark body, bright die. The left and right
    faces carry no pins, so the flow arrows can meet the body cleanly."""
    for k in range(pins):
        t = -s + 2 * s * (k + 0.5) / pins
        for (xa, ya, xb, yb) in ((cx + t, cy + s, cx + t, cy + s + plen),
                                 (cx + t, cy - s, cx + t, cy - s - plen)):
            ax.plot([xa, xb], [ya, yb], color=pf.C['yellow'], lw=0.9, solid_capstyle='butt', zorder=2)
    rbox(cx - s, cy - s, 2 * s, 2 * s, pf.C['ink'], pf.C['black'], r=0.7, lw=0.5, z=3)
    rbox(cx - s + 1.1, cy - s + 1.1, 2 * s - 2.2, 2 * s - 2.2, pf.C['purple'], pf.C['purple_light'],
         r=0.5, lw=0.5, z=4)
    ax.text(cx, cy, 'GPU', fontsize=EMPH, fontweight='bold', color='white',
            ha='center', va='center', zorder=5)


def city_network(x0, y0, w, h, name='LTown'):
    """The real network of the paper's benchmark, fitted into (x0, y0, w, h) with its aspect ratio
    kept: every link thin, the longest tenth drawn as transmission mains."""
    coords, links = pf.read_inp_geometry(pf.INP[name])
    seg = np.array([[coords[a], coords[b]] for _, a, b in links if a in coords and b in coords],
                   dtype=float)
    lo, hi = seg.reshape(-1, 2).min(0), seg.reshape(-1, 2).max(0)
    s = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
    off = np.array([x0 + (w - s * (hi[0] - lo[0])) / 2, y0 + (h - s * (hi[1] - lo[1])) / 2])
    seg = (seg - lo) * s + off
    ln = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    main = ln >= np.quantile(ln, 0.96)
    ax.add_collection(LineCollection(seg[~main], colors=BLUEL, linewidths=0.25, zorder=2,
                                     capstyle='round'))
    ax.add_collection(LineCollection(seg[main], colors=BLUE, linewidths=0.45, zorder=3,
                                     capstyle='round'))
    return seg, off, s, (hi - lo) * s


NODES = np.array([(0.04, 0.52), (0.20, 0.90), (0.28, 0.28), (0.48, 0.70),
                  (0.56, 0.14), (0.74, 0.52), (0.92, 0.22), (0.86, 0.90)])
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (4, 6), (5, 7)]


def network(x0, y0, w, h, color=BLUEL, edge_c=None, lw=0.55, ms=1.5, hi=(), hi_c=None,
            hi_edges=(), z=3):
    """Schematic network in the box (x0, y0, w, h); hi/hi_edges are drawn in hi_c."""
    p = np.column_stack([x0 + NODES[:, 0] * w, y0 + NODES[:, 1] * h])
    for a, b in EDGES:
        c, lwe = (hi_c, lw + 0.4) if (a, b) in hi_edges else (edge_c or color, lw)
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=c, lw=lwe,
                solid_capstyle='round', zorder=z)
    for k, (xk, yk) in enumerate(p):
        c = hi_c if k in hi else color
        ax.plot([xk], [yk], marker='o', ms=ms + (0.5 if k in hi else 0), mfc=c, mec='none', zorder=z + 1)
    return p


def droplet(cx, cy, s):
    """Water drop: triangular tip over a disc."""
    ax.add_patch(Polygon([(cx, cy + 1.9 * s), (cx - 0.78 * s, cy + 0.25 * s), (cx + 0.78 * s, cy + 0.25 * s)],
                         closed=True, fc=BLUE, ec='none', zorder=4))
    ax.add_patch(Circle((cx, cy), 0.82 * s, fc=BLUE, ec='none', zorder=4))


def sensor(cx, cy, s, color):
    """Sensor pin: filled disc with two signal arcs above it."""
    ax.add_patch(Circle((cx, cy), 0.36 * s, fc=color, ec='none', zorder=6))
    for r in (0.9, 1.5):
        ax.add_patch(Arc((cx, cy), r * s, r * s, theta1=25, theta2=155, color=color, lw=0.5, zorder=6))


def check(cx, cy, s, color):
    ax.plot([cx - s, cx - 0.25 * s, cx + s], [cy, cy - 0.75 * s, cy + 0.9 * s],
            color=color, lw=1.0, solid_capstyle='round', solid_joinstyle='miter', zorder=5)


# ---------------------------------------------------------------- 1. the network and its parcels
GY = 34.4                                                                       # ground line
seg, noff, nsc, nsize = city_network(1.4, 22.6, 19.6, 11.0)
ax.plot([1.4, 21.4], [GY, GY], color=pf.C['light'], lw=0.5, solid_capstyle='butt', zorder=1)
ax.plot([4.5, 4.5], [GY + 0.2, GY + 3.3], color=pf.C['mid'], lw=0.7, zorder=2)  # tank stem
for dx in (-1.35, 1.35):                                                        # splayed legs
    ax.plot([4.5 + dx * 0.5, 4.5 + dx], [GY + 3.3, GY + 0.2], color=pf.C['mid'], lw=0.6, zorder=2)
rbox(2.6, GY + 3.3, 3.8, 2.7, BLUEP, BLUEE, r=0.8, lw=0.6, z=3)                 # tank shell
ax.add_patch(Polygon([(2.35, GY + 6.0), (6.65, GY + 6.0), (4.5, GY + 7.3)], closed=True,
                     fc=BLUEL, ec=BLUEE, lw=0.5, zorder=3))                     # tank roof
ax.plot([2.9, 6.1], [GY + 4.6, GY + 4.6], color=BLUE, lw=0.5, zorder=4)         # water level
ax.plot([4.5, 4.5], [GY + 0.2, 33.4], color=BLUEL, lw=0.6, zorder=3)            # service main
for bx, bw, bh in ((11.4, 1.7, 2.4), (13.5, 2.1, 3.6), (16.0, 1.5, 2.0),        # served district
                   (18.0, 2.3, 3.1)):
    ax.add_patch(Rectangle((bx, GY), bw, bh, fc=pf.C['pale'], ec=pf.C['grey_line'], lw=0.45,
                           zorder=3))
    ax.plot([bx + 0.35, bx + bw - 0.35], [GY + bh - 0.5, GY + bh - 0.5], color=pf.C['mid'],
            lw=0.35, zorder=4)
ax.add_patch(Polygon([(8.3, GY), (10.7, GY), (10.7, GY + 1.5), (9.5, GY + 2.6), (8.3, GY + 1.5)],
                     closed=True, fc=pf.C['pale'], ec=pf.C['grey_line'], lw=0.45, zorder=3))
for k in (3, 27, 61):                                                           # sampled junctions
    ax.plot([seg[k, 1, 0]], [seg[k, 1, 1]], marker='o', ms=1.5, mfc=pf.C['purple'], mec='white',
            mew=0.25, zorder=5)

rbox(1.6, 15.2, 19.6, 6.0, BLUEP, BLUEE, r=3.0, lw=0.7, z=2)                    # pipe wall
ax.plot([3.0, 19.8], [20.2, 20.2], color='white', lw=0.5, alpha=0.9, zorder=3)  # inner highlight
for px, pc in zip((3.5, 8.4, 13.3), (pf.C['purple_light'], BLUEL, pf.C['green_light'])):
    rbox(px, 16.4, 4.1, 3.6, pc, pf.LINE_OF[pc.lower()], r=0.7, lw=0.5, z=4)
arrow(18.0, 18.2, 20.2, 18.2, color=BLUEE, lw=0.7, scale=5.0)
ax.text(11.4, 12.2, 'EPANET', fontsize=EMPH, fontweight='bold', color=INK, ha='center', va='center')

# ---------------------------------------------------------------- 2. the recorded schedule
SLAB_X, SLAB_W, SLAB_D, SKEW = 23.6, 8.6, 1.9, 1.5
for k in range(4):
    y = 18.4 + k * 5.3
    ax.add_patch(Polygon([(SLAB_X, y), (SLAB_X + SLAB_W, y), (SLAB_X + SLAB_W + SKEW, y + SLAB_D),
                          (SLAB_X + SKEW, y + SLAB_D)], closed=True,
                         fc=BLUEP if k % 2 else 'white', ec=pf.C['mid'], lw=0.5, zorder=3))
    cols = (pf.C['purple_light'], pf.C['green_light'], BLUEL)[: 2 + (k % 2)]
    for j, c in enumerate(cols):
        ax.add_patch(Rectangle((SLAB_X + 1.0 + j * 2.5 + SKEW * 0.45, y + 0.45), 1.7, SLAB_D - 0.9,
                               fc=c, ec='none', zorder=4))
ax.text(28.2, 40.6, 'Recorded\nschedule', fontsize=BODY, color=GREY, ha='center', va='center',
        linespacing=1.15)

# ---------------------------------------------------------------- 3. GPU replay and the adjoint
chip(40.6, 27.6)
ax.text(40.6, 38.9, 'Replay\n+ adjoint', fontsize=BODY, color=INK, ha='center', va='center',
        linespacing=1.15)

# ---------------------------------------------------------------- 4. the exact gradient vector
VX0, VX1, VY0, VY1 = 48.8, 53.8, 19.6, 34.6
for xb, s in ((VX0, 1.0), (VX1, -1.0)):
    ax.plot([xb + s * 0.9, xb, xb, xb + s * 0.9], [VY1, VY1, VY0, VY0], color=INK, lw=0.7,
            solid_capstyle='butt', solid_joinstyle='miter', zorder=4)
for k, c in enumerate(APP_C):
    ax.add_patch(Rectangle((50.4, 31.4 - k * 2.9), 1.9, 1.9, fc=c, ec='none', zorder=4))
for k in range(3):
    ax.plot([51.35], [20.3 + k * 0.75], marker='o', ms=0.9, mfc=GREY, mec='none', zorder=4)
ax.text(51.3, 37.0, r'$\partial J/\partial\theta$', fontsize=NUM, color=pf.C['purple_line'],
        ha='center', va='center')

for x0, x1 in ((21.8, 23.2), (33.9, 35.0), (46.2, 48.5)):
    arrow(x0, 27.6, x1, 27.6)

# ---------------------------------------------------------------- 5. four application cards
CX0, CX1 = 59.2, 82.0
CH, GAP, CTOP = 8.55, 0.9, 43.9
LABELS = ['Calibration', 'Booster design', 'Source location', 'Identifiability']
for k, (lab, col, ec) in enumerate(zip(LABELS, APP_C, APP_E)):
    yt = CTOP - k * (CH + GAP)
    yb = yt - CH
    rbox(CX0, yb, CX1 - CX0, CH, 'white', col, r=0.9, lw=0.7, z=2)
    ax.text(CX0 + 1.1, yt - 1.6, lab, fontsize=BODY, fontweight='bold', color=ec,
            ha='left', va='center', zorder=4)
    gy, gh = yb + 0.55, CH - 4.6            # pictogram band, clear of the label above it
    arrow(54.4, 27.6, CX0 - 0.5, yt - CH / 2, color=col, lw=0.7, rad=0.20 * (1.5 - k) / 1.5)
    if k == 0:                                                    # chlorine decay fitted to samples
        droplet(CX0 + 2.0, gy + gh * 0.40, 0.66)
        gx0, gw = CX0 + 4.6, 16.6
        t = np.linspace(0, 1, 60)
        ax.plot(gx0 + t * gw, gy + gh * (0.10 + 0.84 * np.exp(-2.4 * t)), color=ec, lw=0.8, zorder=4)
        ts = np.array([0.06, 0.18, 0.31, 0.44, 0.57, 0.70, 0.84, 0.96])
        dy = np.array([0.10, -0.09, 0.08, -0.07, 0.06, -0.05, 0.05, -0.04])
        ax.plot(gx0 + ts * gw, gy + gh * (0.10 + 0.84 * np.exp(-2.4 * ts) + dy), marker='o',
                ms=1.4, mfc=col, mec='none', ls='none', zorder=5)
    elif k == 1:                                                  # booster node plus its dose schedule
        p = network(CX0 + 1.2, gy, 10.4, gh, color=pf.C['light'], edge_c=pf.C['light'],
                    lw=0.5, ms=1.3, hi=(3,), hi_c=col)
        for r in (0.65, 1.05):
            ax.add_patch(Circle((p[3, 0], p[3, 1]), r, fc='none', ec=col, lw=0.4, alpha=0.75, zorder=4))
        for j, hgt in enumerate((0.30, 0.52, 0.74, 1.0)):
            ax.add_patch(Rectangle((CX0 + 13.6 + j * 2.1, gy), 1.5, gh * hgt, fc=col, ec='none',
                                   alpha=0.85, zorder=4))
    elif k == 2:                                                  # unknown source found from two sensors
        p = network(CX0 + 4.2, gy, 15.2, gh, color=pf.C['light'], edge_c=pf.C['light'],
                    lw=0.5, ms=1.3, hi=(2,), hi_c=col, hi_edges=((2, 3), (2, 4)))
        for r, a in ((0.55, 0.5), (0.95, 0.3), (1.35, 0.16)):
            ax.add_patch(Circle((p[2, 0], p[2, 1]), r, fc=col, ec='none', alpha=a, zorder=3))
        sensor(p[5, 0], p[5, 1], 1.3, ec)
        sensor(p[7, 0], p[7, 1], 1.3, ec)
    else:                                                         # which pipes the samples can resolve
        p = network(CX0 + 1.4, gy, 20.0, gh, color=pf.C['light'], edge_c=pf.C['light'], lw=0.5,
                    ms=1.3, hi=(0, 2, 3), hi_c=col, hi_edges=((0, 2), (2, 3), (3, 5)))
        sensor(p[3, 0], p[3, 1], 1.3, ec)

# ---------------------------------------------------------------- 6. fed back into the engine
ax.plot([40.6, 40.6, 11.4], [20.4, 6.4, 6.4], color=GREY, lw=0.7, solid_capstyle='round',
        solid_joinstyle='round', zorder=3)
arrow(11.4, 6.4, 11.4, 10.8, color=GREY, lw=0.7)
mant, exp = f'{REL:.1e}'.split('e')
check(23.6, 3.2, 1.05, pf.C['green_line'])
ax.text(25.8, 3.1, f'Fed back into EPANET: ${mant}' + r'\times10^{%d}$' % int(exp),
        fontsize=NUM, color=INK, ha='left', va='center')

pf.save(fig, 'toc_art')
print(f'  g2_max_fwd_rel_vs_dll = {REL!r} -> shown as {mant}e{int(exp)}')
