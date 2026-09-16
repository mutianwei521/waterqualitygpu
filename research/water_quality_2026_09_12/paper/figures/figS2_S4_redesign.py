"""Nature-style redesign of Supporting Figures S2--S4.

Figure contracts
----------------
S2  A quantitative heat-map pair makes the tolerance-dependent finite-difference
    behaviour legible without dropping any of the 154 reported errors.  Rows are
    the eleven configuration/parameter directions and columns are the seven
    tested steps; the outlined cell is the row minimum.
S3  An asymmetric quantitative composite separates the cost hierarchy (dot
    matrix) from routing scaling (six network facets), retaining all timings.
S4  A four-panel quantitative grid uses method-coded dot matrices to show that
    calibration accuracy is shared while execution-graph adjoints reduce fit
    time, and that predictive intervals approach nominal LOSO coverage.

The script reads the manuscript source tables directly.  It writes live-text
PDF/SVG masters, 600-dpi PNG/TIFF files, source-data CSV files, manifests and QA
reports.  No interpolation, smoothing or data filtering is used.

Run from any directory with::

    python paper/figures/figS2_S4_redesign.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paperfig as pf  # noqa: E402


TABLES = HERE.parent / "tables"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
ARXIV_TEXT_WIDTH_MM = 165.0

# Figure 1 v2 semantic palette.  Bright colours are reserved for identities and
# comparisons; pale tints carry guides/backgrounds so the page stays quiet.
INK = "#30363D"
GREY = "#707A84"
CYAN = "#56B4E9"
TEAL = "#009E73"
GOLD = "#E69F00"
VERM = "#D55E00"
PURPLE = "#7B61A8"
MAGENTA = "#CC79A7"
NAVY = "#2F5D7C"
WHITE = "#FFFFFF"
PALE_GREY = "#EDF0F2"
GRID = "#DDE2E5"


def mix(colour: str, amount: float = 0.72) -> str:
    """Mix a colour with white; amount is the white fraction."""
    from matplotlib.colors import to_hex, to_rgb

    rgb = np.array(to_rgb(colour))
    return to_hex((1 - amount) * rgb + amount * np.ones(3))


PALE_CYAN = mix(CYAN, 0.80)
PALE_TEAL = mix(TEAL, 0.82)
PALE_GOLD = mix(GOLD, 0.80)
PALE_VERM = mix(VERM, 0.82)
PALE_PURPLE = mix(PURPLE, 0.82)
PALE_MAGENTA = mix(MAGENTA, 0.82)

NET_COLOUR = {
    "BWSN2_kw": TEAL,
    "LTown_kw": MAGENTA,
    "Net1": GREY,
    "Net2_react": CYAN,
    "Jilin_kw": GOLD,
    "Net3_CHEM_kw_mixed_tanks": PURPLE,
    "Bellingham": CYAN,
    "Fairfield": PURPLE,
    "NorthPenn": TEAL,
}


def setup() -> None:
    """Shared journal typography and line geometry."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.fallback": "stixsans",
            # The arXiv build scales these ~177-mm masters to a 165-mm text
            # width.  An 8-pt floor keeps one-level math subscripts/exponents
            # (0.7x in mathtext) above 5 pt after that final scaling.
            "font.size": 8.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.55,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.minor.size": 1.3,
            "ytick.minor.size": 1.3,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.labelpad": 3.0,
            "axes.titlepad": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 1.05,
            "lines.markersize": 4.0,
            "lines.markeredgewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def md_tables(path: Path) -> list[list[list[str]]]:
    """Read each pipe table in a Markdown file, retaining its header."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def number(value: str) -> float:
    value = value.replace(",", "").strip()
    if value in {"", "-"}:
        return float("nan")
    if value == "inf":
        return float("inf")
    return float(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def colour_ticks(ax, mapping: dict[str, str]) -> None:
    for tick in ax.get_yticklabels():
        raw = tick.get_text().replace("North Penn", "NorthPenn")
        for key, colour in mapping.items():
            if key.replace("_kw", "").replace("NorthPenn", "North Penn") in tick.get_text() or key == raw:
                tick.set_color(colour)
                tick.set_fontweight("bold")
                break


def panel_label(fig, x: float, y: float, label: str) -> None:
    fig.text(x, y, label, fontsize=9.8, fontweight="bold", ha="left", va="bottom", color=INK)


def export(fig, stem: str) -> None:
    """Write tight masters with a real 1.5-mm safety margin around all glyphs."""
    from PIL import Image
    import tifffile

    pad = 1.5 * pf.MM
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=pad, dpi=600)
    rgb = np.asarray(Image.open(OUT / f"{stem}.png").convert("RGB"))
    tifffile.imwrite(
        OUT / f"{stem}.tif",
        rgb,
        compression="zlib",
        photometric="rgb",
        resolution=(600, 600),
        resolutionunit="INCH",
    )
    plt.close(fig)


def figure_s2() -> dict:
    source = TABLES / "T1b_fd_step_sweep.md"
    table = md_tables(source)[0]
    rows = table[1:]
    steps = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

    case_label = {
        "Jilin_kw": "Jilin",
        "Net1": "Net1",
        "Net2_MIX2": "Net2 · two-compartment tank",
        "Net2": "Net2",
        "Net2_react": "Net2 · first-order decay",
        "Net3_CHEM_kw_mixed_tanks": "Net3 · wall decay",
        "Net3_CHEM_mixed_tanks": "Net3",
    }
    parameter_label = {"kb": r"$K_b$", "kw": r"$K_w$", "c0": r"$c_0$"}
    parameter_colour = {"kb": PURPLE, "kw": TEAL, "c0": CYAN}

    # Preserve source order, which is identical for the two tolerance modes.
    mode_rows = {
        "input": [row for row in rows if "M0" in row[1]],
        "zero": [row for row in rows if "M1" in row[1]],
    }
    assert len(mode_rows["input"]) == len(mode_rows["zero"]) == 11

    labels: list[str] = []
    parameters: list[str] = []
    for row in mode_rows["input"]:
        case, tail = [part.strip() for part in row[0].split("/", 1)]
        parameter = tail.split()[0]
        labels.append(f"{case_label[case]}  {parameter_label[parameter]}")
        parameters.append(parameter)

    matrices = {
        mode: np.array([[number(value) for value in row[4:11]] for row in selected], dtype=float)
        for mode, selected in mode_rows.items()
    }
    assert all(matrix.shape == (11, 7) for matrix in matrices.values())
    assert all(np.all(np.isfinite(matrix)) and np.all(matrix > 0) for matrix in matrices.values())

    error_cmap = LinearSegmentedColormap.from_list(
        "fd_error",
        [NAVY, CYAN, PALE_CYAN, "#FFF7DD", GOLD, VERM],
        N=256,
    )
    norm = LogNorm(vmin=1e-15, vmax=2.0)

    fig = plt.figure(figsize=(pf.DOUBLE, 105 * pf.MM))
    gs = fig.add_gridspec(
        1,
        2,
        left=42 * pf.MM / pf.DOUBLE,
        right=0.982,
        top=0.925,
        bottom=0.205,
        wspace=0.12,
    )
    axes = [fig.add_subplot(gs[0, index]) for index in range(2)]
    image = None
    titles = [r"input tolerance, $C_{tol}=0.01$ mg/L", "zero tolerance"]
    for panel, (ax, mode, title) in enumerate(zip(axes, ("input", "zero"), titles)):
        image = ax.imshow(
            matrices[mode],
            cmap=error_cmap,
            norm=norm,
            aspect="auto",
            interpolation="none",
            rasterized=False,
        )
        ax.set_title(title, fontweight="bold", pad=5.5)
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels([rf"$10^{{-{power}}}$" for power in range(1, 8)])
        ax.set_xlabel(r"finite-difference step $h$")
        ax.set_yticks(np.arange(len(labels)))
        if panel == 0:
            ax.set_yticklabels(labels)
            for tick, parameter in zip(ax.get_yticklabels(), parameters):
                tick.set_color(parameter_colour[parameter])
                tick.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        # White cell separators and a white/ink double ring at the minimum of
        # each row.  The data layer itself remains uninterpolated.
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 11, 1), minor=True)
        ax.grid(which="minor", color=WHITE, linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_index, col_index in enumerate(np.argmin(matrices[mode], axis=1)):
            ax.scatter(col_index, row_index, s=27, facecolors="none", edgecolors=WHITE, linewidths=1.4, zorder=5)
            ax.scatter(col_index, row_index, s=12, facecolors="none", edgecolors=INK, linewidths=0.55, zorder=6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color(INK)

    panel_label(fig, 0.010, 0.935, "a")
    panel_label(fig, axes[1].get_position().x0 - 0.030, 0.935, "b")

    # One shared quantitative scale and a compact parameter-key legend.
    cax = fig.add_axes([0.47, 0.075, 0.40, 0.026])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_label("relative error of library central difference", labelpad=2)
    cbar.set_ticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1])
    cbar.set_ticklabels([r"$10^{-15}$", r"$10^{-12}$", r"$10^{-9}$", r"$10^{-6}$", r"$10^{-3}$", r"$1$"])
    cbar.ax.tick_params(length=1.8, pad=1.5)
    cbar.outline.set_linewidth(0.5)
    param_handles = [
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=parameter_colour[key], mec=INK, mew=0.4, label=label)
        for key, label in parameter_label.items()
    ]
    best_handle = Line2D([], [], marker="o", ls="none", ms=5.2, mfc="none", mec=INK, mew=0.7, label="minimum in row")
    fig.legend(
        handles=param_handles + [best_handle],
        loc="lower left",
        bbox_to_anchor=(0.060, 0.050),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.35,
        labelspacing=0.45,
    )

    csv_rows: list[dict] = []
    for mode, selected in mode_rows.items():
        for row in selected:
            case, tail = [part.strip() for part in row[0].split("/", 1)]
            parameter = tail.split()[0]
            for step, value in zip(steps, row[4:11]):
                csv_rows.append(
                    {
                        "mode": mode,
                        "case": case,
                        "parameter": parameter,
                        "step_h": f"{step:.0e}",
                        "relative_error": f"{number(value):.16g}",
                    }
                )
    write_csv(OUT / "figS2_source_sweep.csv", csv_rows)
    export(fig, "figS2_sweep")
    return {
        "source": str(source.relative_to(pf.RES)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "source_rows": len(csv_rows),
        "matrix_shape_each_panel": [11, 7],
        "data_range": [float(min(matrix.min() for matrix in matrices.values())), float(max(matrix.max() for matrix in matrices.values()))],
    }


METHODS_S3 = [
    ("library run", GREY, "o"),
    ("library FD gradient", GOLD, "s"),
    ("per-operator adjoint", VERM, "^"),
    (r"fused forward + adjoint, $B=1$", PURPLE, "D"),
    (r"fused, per set at $B=1{,}024$", TEAL, "P"),
]


def figure_s3() -> dict:
    source = TABLES / "T2_engine_cost.md"
    tables = md_tables(source)
    cost_rows = tables[0][1:]
    throughput_rows = tables[1][1:]
    name = {
        "Net1": "Net1",
        "Net2_react": "Net2 · decay",
        "Jilin_kw": "Jilin",
        "Net3_CHEM_kw_mixed_tanks": "Net3 · wall decay",
        "BWSN2_kw": "BWSN2",
        "LTown_kw": "L-Town",
    }

    fig = plt.figure(figsize=(170 * pf.MM, 110 * pf.MM))
    outer = fig.add_gridspec(
        1,
        2,
        left=0.115,
        right=0.985,
        top=0.91,
        bottom=0.245,
        width_ratios=[0.40, 0.60],
        wspace=0.24,
    )

    # Panel a: method-coded dot matrix on the native log-time axis.
    ax = fig.add_subplot(outer[0, 0])
    network_labels = [name[row[0]] for row in cost_rows]
    base = np.arange(len(cost_rows), dtype=float)
    values_by_method = [
        [number(row[5]) for row in cost_rows],
        [number(row[6]) for row in cost_rows],
        [number(row[7]) for row in cost_rows],
        [number(row[9]) for row in cost_rows],
        [number(row[11]) for row in cost_rows],
    ]
    offsets = np.linspace(-0.26, 0.26, len(METHODS_S3))
    ax.hlines(base, 3e-6, 1.3, color=PALE_GREY, lw=2.1, zorder=0)
    for offset, (method, colour, marker), values in zip(offsets, METHODS_S3, values_by_method):
        ax.scatter(
            values,
            base + offset,
            s=25,
            marker=marker,
            facecolor=colour,
            edgecolor=INK,
            linewidth=0.45,
            label=method,
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_xlim(3e-6, 1.5)
    ax.set_ylim(len(base) - 0.45, -0.45)
    ax.set_yticks(base)
    ax.set_yticklabels(network_labels)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_title("cost hierarchy", fontweight="bold")
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.45)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)

    for tick, row in zip(ax.get_yticklabels(), cost_rows):
        tick.set_color(NET_COLOUR[row[0]])
        tick.set_fontweight("bold")

    # Panel b: one network per facet; path identity is colour + marker + line.
    inner = outer[0, 1].subgridspec(2, 3, hspace=0.42, wspace=0.28)
    cases: list[str] = []
    for row in throughput_rows:
        if row[0] not in cases:
            cases.append(row[0])
    assert len(cases) == 6
    route_style = [
        ("gather path", 4, CYAN, "o", (0, (1.4, 1.4))),
        ("sparse (SpMM) path", 6, TEAL, "s", (0, (3.0, 1.6))),
        ("automatic choice", 8, PURPLE, "D", "-"),
    ]
    facet_axes = []
    for index, case in enumerate(cases):
        row_index, col_index = divmod(index, 3)
        facet = fig.add_subplot(inner[row_index, col_index])
        facet_axes.append(facet)
        selected = [row for row in throughput_rows if row[0] == case and row[3] != "-"]
        batch = np.array([number(row[2]) for row in selected])
        for route, column, colour, marker, linestyle in route_style:
            values = np.array([number(row[column]) for row in selected])
            facet.plot(
                batch,
                values,
                color=colour,
                marker=marker,
                linestyle=linestyle,
                markerfacecolor=WHITE if route == "automatic choice" else colour,
                markeredgecolor=INK,
                markeredgewidth=0.35,
                markersize=3.3,
                linewidth=1.30 if route != "automatic choice" else 0.80,
                label=route,
            )
        facet.set_xscale("log", base=2)
        facet.set_yscale("log")
        facet.set_xlim(0.75, 1400)
        facet.set_ylim(6, 5e4)
        facet.set_title(name[case], color=NET_COLOUR[case], fontweight="bold", pad=3.0)
        facet.set_xticks([1, 64, 1024])
        facet.set_xticklabels(["1", "64", "1,024"] if row_index == 1 else [])
        facet.tick_params(axis="x", which="minor", length=0)
        facet.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        facet.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        facet.yaxis.set_minor_formatter(NullFormatter())
        facet.tick_params(axis="y", which="minor", length=0)
        if col_index != 0:
            facet.set_yticklabels([])
            facet.tick_params(axis="y", length=0)
        facet.grid(True, axis="y", which="major", color=GRID, lw=0.4)
    fig.text(0.710, 0.175, r"batch size $B$", ha="center", va="top", fontsize=8.0)
    fig.text(0.430, 0.545, "Jacobian rows per second", ha="center", va="center", rotation=90, fontsize=8.0)

    panel_label(fig, 0.012, 0.925, "a")
    panel_label(fig, 0.410, 0.925, "b")

    method_handles = [
        Line2D([], [], marker=marker, ls="none", ms=5.0, mfc=colour, mec=INK, mew=0.45, label=method)
        for method, colour, marker in METHODS_S3
    ]
    fig.legend(
        handles=method_handles,
        loc="lower left",
        bbox_to_anchor=(0.025, 0.010),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.45,
    )
    route_handles = [
        Line2D([], [], color=colour, marker=marker, ls=linestyle, lw=1.0, ms=4.2, mec=INK, mew=0.35, label=route)
        for route, _, colour, marker, linestyle in route_style
    ]
    fig.legend(
        handles=route_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.030),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    cost_csv: list[dict] = []
    for row in cost_rows:
        for (method, _, _), value in zip(METHODS_S3, [row[5], row[6], row[7], row[9], row[11]]):
            cost_csv.append({"case": row[0], "method": re.sub(r"\$.*?\$", "", method), "wall_s": f"{number(value):.16g}"})
    throughput_csv: list[dict] = []
    for row in throughput_rows:
        if row[3] == "-":
            continue
        throughput_csv.append(
            {
                "case": row[0],
                "batch_B": f"{number(row[2]):.16g}",
                "gather_rows_per_s": f"{number(row[4]):.16g}",
                "spmm_rows_per_s": f"{number(row[6]):.16g}",
                "automatic_rows_per_s": f"{number(row[8]):.16g}",
            }
        )
    write_csv(OUT / "figS3_source_cost.csv", cost_csv)
    write_csv(OUT / "figS3_source_throughput.csv", throughput_csv)
    export(fig, "figS3_cost")
    return {
        "source": str(source.relative_to(pf.RES)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "source_rows": {"cost_points": len(cost_csv), "throughput_batches": len(throughput_csv)},
        "cost_range_s": [float(min(min(v) for v in values_by_method)), float(max(max(v) for v in values_by_method))],
        "facets": cases,
    }


METHODS_S4 = [
    ("nominal", "nominal(INP)", GREY, "o"),
    (r"FD, $C_{tol}=0$", "FD-LBFGSB", GOLD, "s"),
    (r"FD, $C_{tol}=0.01$", "FD-LBFGSB@ctol0.01", VERM, "^"),
    (r"FD, $C_{tol}=0.01$, step $10^{-3}$", "FD-LBFGSB@ctol0.01,eps1e-3", MAGENTA, "v"),
    ("adjoint, eager", "ADJ-LBFGSB", CYAN, "D"),
    ("adjoint, execution graph", "ADJ-LBFGSB-graph", PURPLE, "P"),
]


def figure_s4() -> dict:
    source = TABLES / "T3_field_calibration.md"
    tables = md_tables(source)
    calibration = tables[0][1:]
    loso = tables[2][1:]
    systems = [("Bellingham", "Bellingham"), ("Fairfield", "Fairfield"), ("NorthPenn", "North Penn")]

    def find(system: str, method: str, column: int, fixed: bool = False) -> float:
        key = system + (" (Kb fixed)" if fixed else "")
        for row in calibration:
            if row[0] == key and row[1] == method:
                return number(row[column])
        raise KeyError((key, method, column))

    fig = plt.figure(figsize=(174 * pf.MM, 118 * pf.MM))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.985,
        top=0.915,
        bottom=0.230,
        wspace=0.28,
        hspace=0.48,
    )
    axes = np.array([[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(2)])
    base = np.arange(len(systems), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(METHODS_S4))

    # a,b: all six methods remain visible even when their RMSE values coincide.
    for panel, (ax, fixed, title) in enumerate(
        [(axes[0, 0], False, "bulk coefficient free"), (axes[0, 1], True, "bulk coefficient fixed")]
    ):
        ax.hlines(base, 0.115, 0.275, color=PALE_GREY, lw=2.0, zorder=0)
        for offset, (label, method, colour, marker) in zip(offsets, METHODS_S4):
            values = [find(system, method, 3, fixed) for system, _ in systems]
            ax.scatter(
                values,
                base + offset,
                s=24,
                marker=marker,
                facecolor=colour,
                edgecolor=INK,
                linewidth=0.4,
                zorder=4,
                label=label,
            )
        ax.set_xlim(0.115, 0.275)
        ax.set_xticks([0.12, 0.16, 0.20, 0.24])
        ax.set_ylim(len(base) - 0.45, -0.45)
        ax.set_yticks(base)
        ax.set_yticklabels([label for _, label in systems])
        ax.set_xlabel("site RMSE (mg/L)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="x", color=GRID, lw=0.45)
        for tick, (system, _) in zip(ax.get_yticklabels(), systems):
            tick.set_color(NET_COLOUR[system])
            tick.set_fontweight("bold")

    # c: native log-time comparison.  Pale connectors and labels quantify the
    # graph-adjoint speed-up over the zero-tolerance FD fit.
    ax = axes[1, 0]
    ax.hlines(base, 0.12, 40, color=PALE_GREY, lw=2.0, zorder=0)
    timing_methods = METHODS_S4[1:]
    timing_offsets = np.linspace(-0.22, 0.22, len(timing_methods))
    for offset, (label, method, colour, marker) in zip(timing_offsets, timing_methods):
        values = [find(system, method, 6, False) for system, _ in systems]
        ax.scatter(values, base + offset, s=25, marker=marker, facecolor=colour, edgecolor=INK, linewidth=0.4, zorder=4)
    for row_index, (system, _) in enumerate(systems):
        fd_time = find(system, "FD-LBFGSB", 6, False)
        graph_time = find(system, "ADJ-LBFGSB-graph", 6, False)
        y = row_index + 0.34
        ax.plot([graph_time, fd_time], [y, y], color=PALE_PURPLE, lw=2.0, solid_capstyle="round", zorder=1)
        ax.vlines([graph_time, fd_time], y - 0.035, y + 0.035, color=PURPLE, lw=0.75, zorder=2)
        ax.text(
            math.sqrt(graph_time * fd_time),
            y - 0.035,
            rf"${fd_time / graph_time:.0f}\times$",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=PURPLE,
            fontweight="bold",
            bbox=dict(facecolor=WHITE, edgecolor="none", pad=0.25),
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.12, 40)
    ax.set_ylim(len(base) - 0.45, -0.45)
    ax.set_yticks(base)
    ax.set_yticklabels([label for _, label in systems])
    ax.set_xlabel("wall-clock time per fit (s)")
    ax.set_title("bulk coefficient free", fontweight="bold")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.45)
    for tick, (system, _) in zip(ax.get_yticklabels(), systems):
        tick.set_color(NET_COLOUR[system])
        tick.set_fontweight("bold")

    # d: coverage as distance to the nominal target; no bars or filled areas.
    ax = axes[1, 1]
    loso_order = [("Fairfield", "Fairfield"), ("Bellingham", "Bellingham"), ("NorthPenn", "North Penn")]
    pred_methods = [
        ("predictive, linearised", 6, PURPLE, "D"),
        ("predictive, sampled", 7, MAGENTA, "o"),
        ("parameter-only", 8, GREY, "s"),
    ]
    pred_offsets = [-0.17, 0.0, 0.17]
    base_d = np.arange(3, dtype=float)
    ax.hlines(base_d, 0.0, 1.02, color=PALE_GREY, lw=2.0, zorder=0)
    for offset, (label, column, colour, marker) in zip(pred_offsets, pred_methods):
        values = [number(next(row for row in loso if row[0] == system)[column]) for system, _ in loso_order]
        ax.scatter(values, base_d + offset, s=27, marker=marker, facecolor=colour, edgecolor=INK, linewidth=0.45, zorder=4, label=label)
    ax.axvline(0.95, color=GOLD, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(0.95, -0.48, "nominal 0.95", color=GOLD, ha="right", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(2.45, -0.55)
    ax.set_yticks(base_d)
    ax.set_yticklabels([label for _, label in loso_order])
    ax.set_xlabel("LOSO coverage of 95% band")
    ax.grid(True, axis="x", color=GRID, lw=0.45)
    for tick, (system, _) in zip(ax.get_yticklabels(), loso_order):
        tick.set_color(NET_COLOUR[system])
        tick.set_fontweight("bold")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.01),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.8,
        fontsize=6.7,
    )

    panel_label(fig, 0.013, 0.930, "a")
    panel_label(fig, 0.510, 0.930, "b")
    panel_label(fig, 0.013, 0.505, "c")
    panel_label(fig, 0.510, 0.505, "d")

    method_handles = [
        Line2D([], [], marker=marker, ls="none", ms=5.0, mfc=colour, mec=INK, mew=0.45, label=label)
        for label, _, colour, marker in METHODS_S4
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.015),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.45,
    )

    calibration_csv: list[dict] = []
    for row in calibration:
        calibration_csv.append(
            {
                "dataset": row[0],
                "method": row[1],
                "repeats": row[2],
                "median_RMSE_mg_L": row[3],
                "minimum_RMSE_mg_L": row[4],
                "maximum_RMSE_mg_L": row[5],
                "median_wall_s": row[6],
                "median_evaluations": row[7],
            }
        )
    loso_csv: list[dict] = []
    for row in loso:
        loso_csv.append(
            {
                "dataset": row[0],
                "n_obs": row[1],
                "fit_RMSE": row[2],
                "LOSO_RMSE": row[4],
                "parameter_share_predictive_variance": row[5],
                "coverage_linearised": row[6],
                "coverage_sampled": row[7],
                "coverage_parameter_only": row[8],
            }
        )
    write_csv(OUT / "figS4_source_calibration.csv", calibration_csv)
    write_csv(OUT / "figS4_source_loso.csv", loso_csv)
    export(fig, "figS4_calib")
    speedups = {
        system: find(system, "FD-LBFGSB", 6, False) / find(system, "ADJ-LBFGSB-graph", 6, False)
        for system, _ in systems
    }
    return {
        "source": str(source.relative_to(pf.RES)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "source_rows": {"calibration": len(calibration_csv), "loso": len(loso_csv)},
        "graph_adjoint_speedup_over_fd": speedups,
    }


def pdf_qa(stem: str) -> dict:
    """Check source masters and their final 165-mm arXiv placement."""
    import fitz
    from PIL import Image

    files = {suffix: OUT / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png", "tif")}
    assert all(path.exists() and path.stat().st_size > 0 for path in files.values())
    with fitz.open(files["pdf"]) as document:
        assert len(document) == 1
        page = document[0]
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
            if span.get("text", "").strip()
        ]
        assert spans
        outside = [
            span["text"]
            for span in spans
            if span["bbox"][0] < -0.1
            or span["bbox"][1] < -0.1
            or span["bbox"][2] > page.rect.width + 0.1
            or span["bbox"][3] > page.rect.height + 0.1
        ]
        fonts = page.get_fonts(full=True)
        unembedded = [font[3] for font in fonts if font[0] == 0]
        minimum_glyph = min(span["size"] for span in spans)
        assert not outside, outside
        assert minimum_glyph >= 4.9, minimum_glyph
        assert not unembedded, unembedded
        width_mm = page.rect.width / 72 * 25.4
        height_mm = page.rect.height / 72 * 25.4
        assert width_mm <= 183 and height_mm <= 220

        # Place the complete source PDF into an in-memory 165-mm-wide PDF and
        # inspect the transformed spans.  This mirrors \includegraphics at
        # text width and catches subscript/exponent shrinkage after placement.
        target_width_pt = ARXIV_TEXT_WIDTH_MM / 25.4 * 72
        scale = min(1.0, target_width_pt / page.rect.width)
        target_height_pt = page.rect.height * scale
        placed = fitz.open()
        placed_page = placed.new_page(width=page.rect.width * scale, height=target_height_pt)
        placed_page.show_pdf_page(placed_page.rect, document, 0)
        placed_bytes = placed.tobytes(garbage=4, deflate=True)
        placed.close()
    with fitz.open(stream=placed_bytes, filetype="pdf") as placed_document:
        placed_page = placed_document[0]
        placed_spans = [
            span
            for block in placed_page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
            if span.get("text", "").strip()
        ]
        assert placed_spans
        scaled_minimum_glyph = min(span["size"] for span in placed_spans)
        placed_outside = [
            span["text"]
            for span in placed_spans
            if span["bbox"][0] < -0.1
            or span["bbox"][1] < -0.1
            or span["bbox"][2] > placed_page.rect.width + 0.1
            or span["bbox"][3] > placed_page.rect.height + 0.1
        ]
        assert not placed_outside, placed_outside
        assert scaled_minimum_glyph >= 5.0, (stem, scaled_minimum_glyph, scale)
    with Image.open(files["tif"]) as image:
        dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
        assert all(abs(value - 600) < 0.5 for value in dpi), dpi
        pixels = list(image.size)
    svg_text = files["svg"].read_text(encoding="utf-8").count("<text")
    assert svg_text > 0
    return {
        "one_page": True,
        "dimensions_mm": [round(width_mm, 2), round(height_mm, 2)],
        "minimum_glyph_pt": round(minimum_glyph, 3),
        "arxiv_text_width_mm": ARXIV_TEXT_WIDTH_MM,
        "arxiv_width_scale": round(scale, 6),
        "arxiv_scaled_minimum_glyph_pt": round(scaled_minimum_glyph, 3),
        "text_outside_page": 0,
        "unembedded_fonts": [],
        "svg_text_nodes": svg_text,
        "tiff_dpi": list(dpi),
        "tiff_pixels": pixels,
    }


def main() -> None:
    setup()
    manifests = {
        "figS2_sweep": figure_s2(),
        "figS3_cost": figure_s3(),
        "figS4_calib": figure_s4(),
    }
    # Add output hashes only after all four representations exist.
    for stem, manifest in manifests.items():
        manifest["output_sha256"] = {suffix: sha256(OUT / f"{stem}.{suffix}") for suffix in ("pdf", "svg", "png", "tif")}
        manifest["script_sha256"] = sha256(Path(__file__))
        (OUT / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qa = {stem: pdf_qa(stem) for stem in manifests}
    qa["source_integrity"] = {
        stem: {
            "source": manifest["source"],
            "sha256_verified": sha256(pf.RES / manifest["source"]) == manifest["source_sha256"],
            "source_rows": manifest["source_rows"],
        }
        for stem, manifest in manifests.items()
    }
    assert all(item["sha256_verified"] for item in qa["source_integrity"].values())
    qa["result"] = "PASS"
    (OUT / "figS2_S4_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
