"""Figure 1: colourful, data-faithful transport recording and adjoint workflow.

Panel a is an illustrative schematic with colours tied to scientific roles.
Panels b--f use only saved benchmark values. Panel d contains every one of the
11 x 7 x 2 finite-difference errors; panels e and f preserve every available
batch size without interpolation or cross-network averaging.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import paperfig as pf

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from matplotlib.ticker import LogFormatterMathtext, NullLocator


pf.setup()
mpl.rcParams.update({
    "axes.grid": False,
    "axes.linewidth": 0.55,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

W, H = 176, 194
INK = "#30363D"
GREY = "#707A84"
MID = "#AEB7BF"
LIGHT = "#D6DCE1"
CYAN = "#56B4E9"
TEAL = "#009E73"
GOLD = "#E69F00"
VERM = "#D55E00"
PURPLE = "#7B61A8"
MAGENTA = "#CC79A7"
NAVY = "#2F5D7C"
PALE_CYAN = "#EFF8FC"
PALE_GOLD = "#FFF7E5"
PALE_PURPLE = "#F5F0F8"
PALE_TEAL = "#EAF7F2"
PALE_VERM = "#FBEFEA"
WHITE = "#FFFFFF"
DASH = (0, (3.0, 1.8))

NET = {
    "Net1": NAVY,
    "Net2_react": TEAL,
    "Jilin_kw": GOLD,
    "Net3_CHEM_kw_mixed_tanks": MAGENTA,
    "LTown_kw": MAGENTA,
    "BWSN2_kw": TEAL,
}
NAMES = {
    "Net1": "Net1",
    "Net2_react": "Net2",
    "Jilin_kw": "Jilin",
    "Net3_CHEM_kw_mixed_tanks": "Net3",
    "LTown_kw": "L-Town",
    "BWSN2_kw": "BWSN2",
}
sources = {}


def load(path):
    path = Path(path)
    sources[path.relative_to(pf.RES).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


g1 = [
    r
    for filename in [
        "g1_core_nextQ.json",
        "g1_core_stepQ.json",
        "g1_variants_nextQ.json",
        "g1_corpus_nextQ.json",
    ]
    for r in load(pf.DATA["g1"] / filename)["results"]
]
g2 = sorted(
    load(pf.DATA["g3s"] / "g2_fused_spmm.json")["results"],
    key=lambda x: x["fused_terms"],
)
fd = {
    r["network"]: r
    for r in load(pf.DATA["g2fd"] / "g2_nextQ_ctol0_cuda.json")["results"]
}
sweeps = [load(p) for p in sorted(pf.DATA["fds"].glob("g2_fd_sweep_*.json"))]
v2 = load(pf.DATA["v2"] / "g3_bench_v2_cuda.json")["results"]
shared = {
    r["network"]: r
    for p in sorted(pf.DATA["g3s"].glob("g3_shared_*.json"))
    for r in load(p)["results"]
}

terms = np.array([r["fused_terms"] for r in g2])
forward = np.array([r["forward_rel_vs_dll"] for r in g2])
adjoint = np.array([r["adjoint_vs_autograd"]["max_rel_err"] for r in g2])
quality_flag = np.array([r["qualflag"] for r in g2])
worst, directions, off = [], [], []
for row in g2:
    fd_row = fd.get(row["network"], {})
    groups = fd_row.get("fd_dll") or {}
    key = max(groups, key=lambda k: groups[k]["rel_err"]) if groups else None
    worst.append(groups[key]["rel_err"] if key is not None else np.nan)
    directions.append(key)
    off.append(not fd_row.get("reactflag", True))
worst = np.array(worst)
off = np.array(off)
valid = np.isfinite(worst)

fig = plt.figure(figsize=(W / 25.4, H / 25.4))


def axis(x, y, width, height):
    return fig.add_axes([x / W, y / H, width / W, height / H])


def ft(x, y, string, size=7.3, weight="normal", color=INK, ha="left", va="center", **kwargs):
    return fig.text(
        x / W,
        y / H,
        string,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        **kwargs,
    )


def panel(x, y, letter, title):
    ft(x, y, letter, 9, "bold")
    ft(x + 5, y, title, 8, "bold")


def style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GREY)
    ax.tick_params(labelsize=7.1, colors=INK, length=2, width=0.5)
    ax.grid(
        axis=grid_axis,
        which="major",
        color="#E7EAED",
        linewidth=0.35,
        zorder=0,
    )
    ax.set_axisbelow(True)


def log_axes(ax, x=True, y=True):
    if x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(NullLocator())
    if y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(NullLocator())


def symbol_key(x, y, color, label, marker="o", size=6.7, filled=True):
    fig.add_artist(
        Line2D(
            [(x + 1.0) / W],
            [y / H],
            transform=fig.transFigure,
            color=color,
            marker=marker,
            linestyle="none",
            markersize=3.5,
            markerfacecolor=color if filled else WHITE,
            markeredgecolor=color,
            markeredgewidth=0.6,
        )
    )
    ft(x + 3.2, y, label, size, color=INK)


def line_key(x, y, color, label, linestyle="-", size=6.7):
    fig.add_artist(
        Line2D(
            [x / W, (x + 4.0) / W],
            [y / H, y / H],
            transform=fig.transFigure,
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
        )
    )
    ft(x + 5.2, y, label, size, color=INK)


# a | icon-led scientific workflow
panel(4, 190, "a", "Record once, replay and differentiate")
a = axis(4, 139, 168, 47)
a.set(xlim=(0, 168), ylim=(0, 47))
a.axis("off")


def atext(x, y, string, size=7.1, color=INK, ha="center", weight="normal", **kwargs):
    return a.text(
        x,
        y,
        string,
        fontsize=size,
        color=color,
        ha=ha,
        va="center",
        fontweight=weight,
        **kwargs,
    )


def arrow(start, end, color=INK, linewidth=1.0, linestyle="-", scale=7, **kwargs):
    a.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            **kwargs,
        )
    )


def box(x, y, width, height, face=WHITE, edge=LIGHT, linewidth=0.6, radius=1.0):
    a.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
        )
    )


for x, width, face, edge in [
    (0, 48, PALE_CYAN, "#D3EAF2"),
    (52, 54, PALE_GOLD, "#F0DFC1"),
    (110, 58, PALE_PURPLE, "#E1D6E7"),
]:
    box(x, 5.6, width, 37.6, face, edge, 0.55, 1.6)


def reservoir(x, y):
    a.add_patch(
        Polygon(
            [(x, y + 7), (x, y), (x + 8, y), (x + 8, y + 7)],
            closed=False,
            fill=False,
            edgecolor=NAVY,
            linewidth=1.15,
        )
    )
    a.add_patch(
        Polygon(
            [
                (x + 0.7, y + 0.7),
                (x + 7.3, y + 0.7),
                (x + 7.3, y + 4.6),
                (x + 6.0, y + 5.0),
                (x + 4.6, y + 4.4),
                (x + 3.1, y + 5.0),
                (x + 1.5, y + 4.4),
                (x + 0.7, y + 4.8),
            ],
            facecolor=CYAN,
            edgecolor="none",
        )
    )


def tank(x, y):
    a.add_patch(Rectangle((x, y), 5.5, 8, facecolor=PALE_TEAL, edgecolor=TEAL, linewidth=0.8))
    a.add_patch(Rectangle((x + 0.3, y + 0.4), 4.9, 3.7, facecolor=CYAN, edgecolor="none"))
    a.add_patch(Ellipse((x + 2.75, y + 8), 5.5, 1.8, facecolor=WHITE, edgecolor=TEAL, linewidth=0.8))
    a.add_patch(Ellipse((x + 2.75, y), 5.5, 1.8, facecolor=CYAN, edgecolor=TEAL, linewidth=0.8))


def chip(x, y, width=12, height=11):
    for xx in np.linspace(x + 2, x + width - 2, 5):
        a.plot([xx, xx], [y - 1.1, y + height + 1.1], color=GOLD, linewidth=1.25, zorder=1)
    for yy in np.linspace(y + 1.8, y + height - 1.8, 4):
        a.plot([x - 1.1, x + width + 1.1], [yy, yy], color=GOLD, linewidth=1.25, zorder=1)
    box(x, y, width, height, PURPLE, "#574275", 0.9, 0.8)
    box(x + 1.8, y + 1.8, width - 3.6, height - 3.6, "#DDD2E9", "#DDD2E9", 0.4, 0.3)
    atext(x + width / 2, y + height / 2, "GPU", 7.2, "#4C3868", weight="bold")


atext(24, 40.2, "Fixed hydraulics", 8, NAVY, weight="bold")
atext(24, 36.3, "OWA-EPANET Toolkit 2.3", 6.9, GREY)
nodes = {0: (13, 25), 1: (24, 30), 2: (36, 29), 3: (21, 18), 4: (35, 16), 5: (11, 12)}
for u, v in [(0, 1), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4), (3, 5), (5, 4)]:
    p, q = nodes[u], nodes[v]
    a.plot([p[0], q[0]], [p[1], q[1]], color="#66737C", linewidth=1.45, zorder=1)
a.plot([10.4, 13], [25, 25], color="#66737C", linewidth=1.45)
a.plot([36, 41], [29, 29], color="#66737C", linewidth=1.45)
reservoir(2.4, 21)
tank(40, 27)
for x, y in nodes.values():
    a.add_patch(Circle((x, y), 0.9, facecolor=WHITE, edgecolor="#56656F", linewidth=0.8, zorder=3))
box(16, 8.5, 18.5, 3.7, WHITE, "#7D8B94", 0.55, 0.35)
for j, colour in enumerate([CYAN, TEAL, GOLD, PURPLE]):
    a.add_patch(
        Rectangle(
            (16.5 + j * 4.4, 8.9),
            4.1,
            2.9,
            facecolor=colour,
            edgecolor=WHITE,
            linewidth=0.35,
            zorder=4,
        )
    )
arrow((47.3, 25), (54.2, 25), CYAN, 1.2)
atext(50.8, 28.1, "Q, V", 7.0, NAVY)

atext(79, 40.2, "Record once", 8, GOLD, weight="bold")
atext(79, 36.3, "immutable parcel operations", 6.9, GREY)
operation_colours = [CYAN, GOLD, MAGENTA]
for j, colour in enumerate(operation_colours):
    box(56 + j * 7.6, 26.2, 6.1, 4.2, WHITE, colour, 0.75, 0.45)
    a.add_patch(Circle((57.2 + j * 7.6, 28.3), 0.65, facecolor=colour, edgecolor="none"))
    atext(59.5 + j * 7.6, 28.3, f"c{j + 1}", 6.8, colour)
    arrow((59.0 + j * 7.6, 25.7), (69.2, 20.8), colour, 0.65, scale=5)
box(65, 16.7, 9.2, 4.8, PALE_TEAL, TEAL, 0.75, 0.5)
atext(69.6, 19.1, "c4", 7.0, TEAL, weight="bold")
arrow((74.3, 19.1), (84.7, 19.1), TEAL, 0.9)
for offset, edge in [(2.4, MAGENTA), (1.2, GOLD), (0, TEAL)]:
    box(86 + offset, 13.7 + offset, 15, 16, WHITE, edge, 0.55, 0.55)
cell_colours = [CYAN, TEAL, GOLD, MAGENTA, PURPLE]
for idx, (row, col) in enumerate([(0, 0), (0, 3), (1, 1), (2, 0), (2, 2), (3, 1), (3, 3)]):
    a.add_patch(
        Rectangle(
            (88.1 + col * 2.8, 15.7 + row * 2.8),
            2.0,
            2.0,
            facecolor=cell_colours[idx % len(cell_colours)],
            edgecolor="none",
        )
    )
atext(79, 9.3, "operations → schedule", 6.8, GREY)

atext(139, 40.2, "Replay + differentiate", 8, PURPLE, weight="bold")
atext(139, 36.3, "batched first-order factors", 6.9, GREY)
a.plot([102.5, 108, 108, 132], [24, 24, 31.4, 31.4], color=GOLD, linewidth=0.9)
arrow((132, 31.4), (136, 28.7), GOLD, 0.9, scale=6)
box(112.2, 12.2, 15.8, 17.6, WHITE, PURPLE, 0.65, 0.8)
for j, (label, colour, pale_face) in enumerate([
    (r"$K_b$", TEAL, PALE_TEAL),
    (r"$K_w$", GOLD, PALE_GOLD),
    ("s", MAGENTA, "#F9EEF5"),
    (r"$c_0$", CYAN, PALE_CYAN),
]):
    box(113.4, 13.4 + j * 3.65, 13.4, 3.15, pale_face, colour, 0.5, 0.4)
    atext(120.1, 14.98 + j * 3.65 - (0.12 if label.startswith("$") else 0.0), label, 6.5 if label.startswith("$") else 7.2, colour, weight="bold")
atext(120.1, 33.4, r"$\theta_1\ldots\theta_B$", 7.1, PURPLE)
arrow((128.2, 21.3), (132.4, 22.5), PURPLE, 0.95)
chip(134, 17.4, 12, 11)
atext(140, 14.0, "FP64", 6.8, GREY)
arrow((147.2, 22.9), (151.2, 22.9), TEAL, 0.95)
a.plot([153, 153, 166], [31, 18, 18], color="#929CA4", linewidth=0.5)
xx = np.linspace(153.5, 165.5, 45)
for jj, colour in enumerate([CYAN, GOLD, TEAL]):
    a.plot(
        xx,
        20.0 + jj * 2.4 + 1.8 * np.exp(-(xx - 153.5) / 4) + 0.35 * np.sin((xx - 153.5) / 2 + jj),
        color=colour,
        linewidth=0.9,
    )
atext(159.5, 32.8, r"$c(t;\theta_b)$", 7.2, PURPLE)
atext(159.5, 14.0, "loss L", 6.9, INK)
# one dashed path with a single head (three separate arrows left kinks at the corners)
a.plot([166.2, 166.2, 118.8, 118.8], [16.8, 7.2, 7.2, 10.6], color=VERM, linewidth=1.0, linestyle=DASH,
       solid_joinstyle="miter", dash_joinstyle="miter")
arrow((118.8, 10.4), (118.8, 12.4), VERM, 1.0)
atext(143.5, 9.6, r"reverse  $\partial L/\partial\theta$", 7.1, VERM)


# b | forward fidelity
panel(4, 134, "b", "Forward fidelity")
b = axis(14, 84, 40, 37)
style(b)
log_axes(b)
b.set_xlim(1e4, 1e6)
b.set_xticks([1e4, 1e5, 1e6])
b.set_ylim(5e-16, 5e-14)
b.set_yticks([1e-15, 1e-14])
b.set_xlabel("Recorded terms", fontsize=7.3)
b.set_ylabel("Relative concentration error", fontsize=7.3)
q_style = {
    1: (TEAL, "o", "CHEM"),
    2: (GOLD, "s", "AGE"),
    3: (PURPLE, "D", "TRACE"),
}
handles = []
for flag, (colour, marker, label) in q_style.items():
    mask = quality_flag == flag
    b.scatter(
        terms[mask],
        forward[mask],
        s=16,
        marker=marker,
        facecolor=mpl.colors.to_rgba(colour, 0.30),
        edgecolor=colour,
        linewidth=0.65,
        zorder=3,
    )
    handles.append(
        Line2D([], [], marker=marker, linestyle="none", color=colour, markerfacecolor=WHITE, markersize=3.5, label=label)
    )
b.legend(
    handles=handles,
    loc="lower left",
    bbox_to_anchor=(0.50, 0.025),
    borderaxespad=0,
    fontsize=6.4,
    handlelength=0.5,
    handletextpad=0.25,
    labelspacing=0.15,
)
b.text(0.05, 0.94, r"max 1.7 × 10$^{-14}$", transform=b.transAxes, fontsize=7.3, color=TEAL, va="top")
ft(14, 128.5, "GPU · 25 configurations", 6.9, color=PURPLE)
ft(14, 124.7, r"CPU · 91 cases · $\Delta=0$", 6.9, color=TEAL)


# c | adjoint and finite-difference checks
panel(59, 134, "c", "Gradient checks")
c = axis(69, 84, 40, 37)
style(c)
log_axes(c)
c.set_xlim(1e4, 1e6)
c.set_xticks([1e4, 1e5, 1e6])
c.set_ylim(5e-17, 1e-2)
c.set_yticks([1e-16, 1e-12, 1e-8, 1e-4])
c.set_xlabel("Recorded terms", fontsize=7.3)
c.set_ylabel("Relative gradient error", fontsize=7.3)
c.scatter(
    terms,
    adjoint,
    s=16,
    marker="o",
    facecolor=mpl.colors.to_rgba(PURPLE, 0.25),
    edgecolor=PURPLE,
    linewidth=0.65,
    zorder=4,
)
c.scatter(
    terms[valid & ~off],
    worst[valid & ~off],
    s=20,
    marker="^",
    facecolor=mpl.colors.to_rgba(VERM, 0.25),
    edgecolor=VERM,
    linewidth=0.7,
    zorder=3,
)
c.scatter(
    terms[valid & off],
    worst[valid & off],
    s=20,
    marker="^",
    facecolor="#E2E5E8",
    edgecolor=GREY,
    linewidth=0.7,
    zorder=3,
)
c.text(
    0.05,
    0.31,
    r"adjoint ≤ 3.1 × 10$^{-15}$",
    transform=c.transAxes,
    fontsize=7.2,
    color=PURPLE,
)
symbol_key(69, 128.5, PURPLE, "Adjoint", "o")
symbol_key(89.5, 128.5, VERM, "FD", "^")
symbol_key(69, 124.7, GREY, "reactions off", "^", 6.7, False)


# d | two unaveraged heatmaps on one common logarithmic colour scale
panel(114, 134, "d", "FD error by merge tolerance")
sweep_map = {(d["case"], d["param"]): d for d in sweeps}
row_spec = [
    (("Jilin_kw", "kb"), r"Jilin · $K_b$"),
    (("Net1", "kb"), r"Net1 · $K_b$"),
    (("Net2_react", "kb"), r"N2-react · $K_b$"),
    (("Net2_react", "c0"), r"N2-react · $c_0$"),
    (("Net3_CHEM_kw_mixed_tanks", "kb"), r"N3-kw · $K_b$"),
    (("Net3_CHEM_kw_mixed_tanks", "kw"), r"N3-kw · $K_w$"),
    (("Net3_CHEM_mixed_tanks", "kb"), r"N3-mix · $K_b$"),
    (("Net2", "kb"), r"Net2 · $K_b$"),
    (("Net2", "kw"), r"Net2 · $K_w$"),
    (("Net2_MIX2", "kw"), r"N2-MIX2 · $K_w$"),
    (("Net3_CHEM_mixed_tanks", "kw"), r"N3-mix · $K_w$"),
]
ordered_sweeps = [sweep_map[key] for key, _ in row_spec]
hvals = sorted(
    {float(row["h"]) for row in ordered_sweeps[0]["runs"]["M1_ctol0"]["sweep"]},
    reverse=True,
)


def heat_matrix(mode):
    matrix = []
    for sweep in ordered_sweeps:
        values = {float(row["h"]): float(row["rel_err"]) for row in sweep["runs"][mode]["sweep"]}
        matrix.append([values[h] for h in hvals])
    return np.asarray(matrix)


mat_zero = heat_matrix("M1_ctol0")
mat_default = heat_matrix("M0_inp_ctol")
norm = LogNorm(vmin=1e-16, vmax=2)
cmap = mpl.colormaps["magma"]
d0 = axis(132, 84, 16.5, 37)
d1 = axis(154, 84, 16.5, 37)
for ax, matrix in [(d0, mat_zero), (d1, mat_default)]:
    ax.imshow(matrix, aspect="auto", origin="upper", interpolation="none", cmap=cmap, norm=norm)
    ax.set_xticks([0, 3, 6])
    ax.set_xticklabels([r"$10^{-1}$", r"$10^{-4}$", r"$10^{-7}$"], fontsize=7.2)
    ax.set_yticks(np.arange(len(row_spec)))
    ax.tick_params(axis="both", length=1.5, width=0.45, pad=1.1, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GREY)
        spine.set_linewidth(0.5)
    ax.axhline(6.5, color=WHITE, linewidth=1.3)
    ax.axhline(6.5, color=INK, linewidth=0.35)
d0.set_yticklabels([label for _, label in row_spec], fontsize=7.2)
for idx, label in enumerate(d0.get_yticklabels()):
    label.set_color(VERM if idx >= 7 else INK)
d1.set_yticklabels([])
ft(140.2, 123.4, r"$C_{\rm tol}=0$", 7.2, "bold", TEAL, ha="center")
ft(162.2, 123.4, r"$C_{\rm tol}=0.01$ mg L$^{-1}$", 7.2, "bold", GOLD, ha="center")
ft(113.2, 102.5, "Relative gradient error", 7.3, ha="center", rotation=90)
ft(151.2, 77.2, r"FD step $h$", 7.2, ha="center")
cax = axis(139, 129.0, 28, 1.35)
ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
cax.set_xticks([])
cax.set_yticks([])
for spine in cax.spines.values():
    spine.set_linewidth(0.4)
    spine.set_color(GREY)
ft(132.1, 129.7, "error", 6.6, color=GREY)
ft(139, 126.7, r"$10^{-16}$", 7.2, color=GREY)
ft(153, 126.7, r"$10^{-8}$", 7.2, color=GREY, ha="center")
ft(167, 126.7, "2", 6.3, color=GREY, ha="right")


# e | four small-multiple dumbbell plots
panel(4, 66, "e", "Batched gradient cost")
symbol_key(14, 60.4, GOLD, "Per-operator", "o", filled=False)
symbol_key(47, 60.4, PURPLE, "Fused", "D")
line_key(69, 60.4, VERM, "Projected FD", DASH)
facet_order = ["Net1", "Net2_react", "Jilin_kw", "Net3_CHEM_kw_mixed_tanks"]
facet_positions = [(14, 38), (69, 38), (14, 15), (69, 15)]
v2_map = {row["network"]: row for row in v2}
ecsv = []
for idx, (network, (x0, y0)) in enumerate(zip(facet_order, facet_positions)):
    row = v2_map[network]
    ax = axis(x0, y0, 41, 16)
    style(ax, "x")
    log_axes(ax, x=True, y=False)
    ax.grid(False)
    ax.grid(axis="x", which="major", color="#E7EAED", linewidth=0.35)
    ax.set_xlim(3e-6, 2)
    ax.set_xticks([1e-5, 1e-3, 1e-1])
    batches = [
        entry["B"]
        for entry in row["engines"]["fused"]["batches"]
        if "graph_forward_backward_s" in entry
    ]
    y_positions = np.arange(len(batches))
    per_operator = {
        entry["B"]: entry["graph_forward_backward_s"] / entry["B"]
        for entry in row["engines"]["adjoint"]["batches"]
        if "graph_forward_backward_s" in entry
    }
    fused = {
        entry["B"]: entry["graph_forward_backward_s"] / entry["B"]
        for entry in row["engines"]["fused"]["batches"]
        if "graph_forward_backward_s" in entry
    }
    for yi, batch in zip(y_positions, batches):
        ax.plot(
            [per_operator[batch], fused[batch]],
            [yi, yi],
            color=MID,
            linewidth=0.8,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(per_operator[batch], yi, s=14, marker="o", facecolor=WHITE, edgecolor=GOLD, linewidth=0.7, zorder=3)
        ax.scatter(fused[batch], yi, s=15, marker="D", facecolor=PURPLE, edgecolor=PURPLE, linewidth=0.5, zorder=4)
        ecsv.append({
            "network": network,
            "n_theta": row["n_theta"],
            "engine": "adjoint",
            "B": batch,
            "seconds_per_set": per_operator[batch],
            "projected": False,
        })
        ecsv.append({
            "network": network,
            "n_theta": row["n_theta"],
            "engine": "fused",
            "B": batch,
            "seconds_per_set": fused[batch],
            "projected": False,
        })
    ax.axvline(row["fd_gradient_cost_s"], color=VERM, linestyle=DASH, linewidth=0.9, zorder=0)
    ecsv.append({
        "network": network,
        "n_theta": row["n_theta"],
        "engine": "library_fd",
        "B": 1,
        "seconds_per_set": row["fd_gradient_cost_s"],
        "projected": True,
    })
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["1", "16", "128", "1,024", "4,096"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_title(
        rf"{NAMES[network]}  ($n_\theta={row['n_theta']}$)",
        fontsize=7.2,
        fontweight="bold",
        color=NET[network],
        pad=2,
    )
    if idx < 2:
        ax.tick_params(labelbottom=False)
ft(2.2, 34.0, r"Batch size $B$", 7.2, ha="center", rotation=90)
ft(62.0, 7.4, "Time per parameter set (s)", 7.2, ha="center")


# f | method dots and within-row implementation ranges, split by network
panel(114, 66, "f", "Jacobian-row throughput")
symbol_key(126, 60.4, CYAN, "Gather", "o", filled=False)
symbol_key(145, 60.4, GOLD, "SpMM", "s", filled=False)
symbol_key(161, 60.4, PURPLE, "Auto", "D")
fcsv = []
f_positions = [("LTown_kw", 38), ("BWSN2_kw", 15)]
engine_style = {
    "fused_gather": (CYAN, "o", -0.13),
    "fused_spmm": (GOLD, "s", 0.0),
    "fused": (PURPLE, "D", 0.13),
}
for idx, (network, y0) in enumerate(f_positions):
    row = shared[network]
    ax = axis(127, y0, 44, 16)
    style(ax, "x")
    log_axes(ax, x=True, y=False)
    ax.grid(False)
    ax.grid(axis="x", which="major", color="#E7EAED", linewidth=0.35)
    ax.set_xlim(10, 2e4)
    ax.set_xticks([1e1, 1e2, 1e3, 1e4])
    batch_sets = {
        engine: {
            entry["B"]: entry["jac_rows"]["rows_per_s"]
            for entry in row["engines"][engine]["batches"]
            if entry.get("jac_rows")
        }
        for engine in engine_style
    }
    batches = sorted(set.intersection(*(set(values) for values in batch_sets.values())))
    y_positions = np.arange(len(batches))
    for yi, batch in zip(y_positions, batches):
        values = [batch_sets[engine][batch] for engine in engine_style]
        ax.hlines(yi, min(values), max(values), color=MID, linewidth=0.8, zorder=1)
        for engine, (colour, marker, offset) in engine_style.items():
            value = batch_sets[engine][batch]
            ax.scatter(
                value,
                yi + offset,
                s=15,
                marker=marker,
                facecolor=WHITE if engine != "fused" else PURPLE,
                edgecolor=colour,
                linewidth=0.7,
                zorder=3,
            )
            fcsv.append({
                "network": network,
                "engine": engine,
                "B": batch,
                "rows_per_second": value,
                "prefix_rows": True,
            })
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{batch:,}" for batch in batches], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_title(NAMES[network], fontsize=7.2, fontweight="bold", color=NET[network], pad=2)
    if idx == 0:
        ax.tick_params(labelbottom=False)
ft(114.7, 34.0, r"Chunk size $B$", 7.2, ha="center", rotation=90)
ft(149.0, 7.4, r"Jacobian rows s$^{-1}$", 7.2, ha="center")


# Exact source-data exports and a provenance/encoding manifest
out = pf.OUT


def csvout(name, rows):
    with (out / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


bc = [
    {
        "network": row["network"],
        "quality_flag": int(row["qualflag"]),
        "fused_terms": int(terms[i]),
        "forward_rel": float(forward[i]),
        "adjoint_rel": float(adjoint[i]),
        "fd_rel": None if not np.isfinite(worst[i]) else float(worst[i]),
        "fd_worst_direction": directions[i],
        "reactflag_off": bool(off[i]),
    }
    for i, row in enumerate(g2)
]
dcsv = [
    {
        "case": sweep["case"],
        "parameter": sweep["param"],
        "mode": mode,
        "h": entry["h"],
        "relative_error": entry["rel_err"],
    }
    for sweep in sweeps
    for mode in ["M1_ctol0", "M0_inp_ctol"]
    for entry in sweep["runs"][mode]["sweep"]
]
for filename, rows in [
    ("fig1_source_bc.csv", bc),
    ("fig1_source_d.csv", dcsv),
    ("fig1_source_e.csv", ecsv),
    ("fig1_source_f.csv", fcsv),
]:
    csvout(filename, rows)

manifest = {
    "dimensions_mm": [W, H],
    "source_sha256": sources,
    "schematic": "Panel a is illustrative; its colours encode scientific roles. Parcel colours indicate record identity, and concentration-curve colours indicate parameter-set identity, not multiple chemical species.",
    "chart_encodings": {
        "b": "Marker colour/shape identifies CHEM, AGE and TRACE configurations.",
        "c": "Purple circles are fused-versus-per-operator adjoints; vermilion triangles are separate finite-difference checks; grey triangles have reactions off.",
        "d": "Two heatmaps contain all 154 errors on one LogNorm scale; no aggregation or interpolation.",
        "e": "Horizontal dumbbells pair per-operator and fused cost at each measured B; vermilion reference lines are projected library FD cost.",
        "f": "Each B row retains the three measured implementations; grey spans show the within-row range.",
    },
    "caveats": [
        "c: circles use fused verification; FD triangles use separately saved Ctol=0 per-operator verification and show the worst checked parameter-group direction.",
        "d: the first seven rows are reaction/affine directions; the last four vermilion-labelled rows are zero-coefficient or branch-boundary directions.",
        "e: CUDA-graph forward+backward divided by batch size; library FD is projected; setup excluded.",
        "f: prefix-row throughput; all BWSN2 rows timed here occur at t=0 and have zero Kw sensitivity; ordinary PIPE count excludes two CVPIPEs.",
    ],
    "g1_cases": len(g1),
    "g1_segments": sum(row["segments_created"] for row in g1),
    "g2_configs": len(g2),
    "fd_configs": int(valid.sum()),
    "fd_heatmap_values": len(dcsv),
}
(out / "fig1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
for extension in ["svg", "pdf", "png"]:
    fig.savefig(out / f"fig1_engine.{extension}", dpi=600)
pf._png_to_tif(out / "fig1_engine.png", out / "fig1_engine.tif")
fig.savefig(out / "fig1_engine_preview.png", dpi=220)
plt.close(fig)
print(
    f"Figure 1 v2: {W} x {H} mm; {len(g2)} configurations, "
    f"{valid.sum()} FD checks, {len(dcsv)} heatmap values."
)
