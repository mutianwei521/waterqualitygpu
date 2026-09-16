"""Nature-style redesign of Supporting Figures S5--S7.

The script reads the same manuscript tables as the legacy figure generator.  It changes
only visual encoding and exports source-data CSVs plus compact QA manifests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paperfig as pf
import fig23_style as st
from figS_si import TAB, md_tables, num


st.setup()

# The arXiv build places these 176-mm masters at a 165-mm text width.  Keep
# mathtext subscripts/superscripts above 5 pt after that final reduction:
# 7.8 pt x matplotlib's ~0.7 math-script factor x 165/176 = 5.12 pt.
ARXIV_WIDTH_MM = 165.0
MIN_FINAL_GLYPH_PT = 5.0
plt.rcParams.update({
    "font.size": 7.8,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
})

NET = {"Net3": st.MAGENTA, "North Penn": st.TEAL, "BWSN2": st.TEAL, "L-Town": st.MAGENTA}
FIELD = {"Fairfield": st.PURPLE, "Bellingham": st.CYAN, "North Penn": st.TEAL}
SET_MARK = {"random": "o", "random+dopt": "^", "random+count": "D", "random+random": "s"}
SET_LABEL = {"random": "Existing", "random+dopt": "+ D-optimal", "random+count": "+ Count-greedy",
             "random+random": "+ Random"}


def _assert_final_text_size(name: str) -> None:
    """Record and enforce the 165-mm arXiv text-width glyph floor."""
    qa_path = pf.OUT / f"{name}_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    source_width = float(qa["dimensions_mm"][0])
    source_min = float(qa["minimum_glyph_pt"])
    final_min = source_min * ARXIV_WIDTH_MM / source_width
    qa["arxiv_textwidth_mm"] = ARXIV_WIDTH_MM
    qa["arxiv_minimum_glyph_pt"] = round(final_min, 4)
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    manifest_path = pf.OUT / f"{name}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qa"] = qa
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{name}: source minimum {source_min:.3f} pt; "
          f"at {ARXIV_WIDTH_MM:.0f} mm = {final_min:.3f} pt")
    assert source_width <= 183.0, (name, source_width)
    assert final_min >= MIN_FINAL_GLYPH_PT, (name, source_min, source_width, final_min)


def _reset_source(table_name: str) -> Path:
    st.SOURCES.clear()
    source = TAB / table_name
    st.record_file(source)
    return source


def export_s5() -> None:
    source = _reset_source("T4_exp1_booster.md")
    tabs = md_tables(source)
    paths = [table for table in tabs if table[0][0] == "lam frac" and table[0][1] == "path stations"]
    checks = [table for table in tabs if table[0][0] == "lam frac" and table[0][1] == "obj matrix-free"]
    networks = [("Net3", paths[0], checks[0]), ("North Penn", paths[1], checks[1])]
    methods = [
        ("Penalised path", 1, 2, 3, "o", (0, (1.2, 1.2)), True),
        ("Support refit", 4, 5, 6, "s", "-", False),
        ("Constant rate", 7, 8, 9, "^", (0, (3, 1.5)), False),
    ]

    figure = plt.figure(figsize=(176 * pf.MM, 126 * pf.MM))
    axes = [st.axis(figure, 14, 74, 73, 38), st.axis(figure, 101, 74, 73, 38),
            st.axis(figure, 14, 13, 73, 33), st.axis(figure, 101, 13, 73, 33)]
    st.panel(figure, 3, 120, "a", "Net3: coverage–dose path")
    st.panel(figure, 90, 120, "b", "North Penn: coverage–dose path")
    st.panel(figure, 3, 58, "c", "Support size along the penalty path")
    st.panel(figure, 90, 58, "d", "Matrix-free objective agreement")

    source_rows, check_rows = [], []
    for panel, (network, table, check) in enumerate(networks):
        ax = axes[panel]
        color = NET[network]
        rows = table[1:]
        # Pale links compare the two doses on the same selected support; they are not intervals.
        for row in rows:
            ax.plot([num(row[6]), num(row[9])], [num(row[5]), num(row[8])], color=st.LIGHT, lw=.55, zorder=0)
        for label, station_i, coverage_i, mass_i, marker, line_style, filled in methods:
            mass = np.array([num(row[mass_i]) for row in rows])
            coverage = np.array([num(row[coverage_i]) for row in rows])
            ax.plot(mass, coverage, color=color, lw=.9, ls=line_style, marker=marker, ms=3.5,
                    mfc=color if filled else "white", mec=color, mew=.75, zorder=2, label=label)
            for row, x, y in zip(rows, mass, coverage):
                source_rows.append({"network": network, "lambda_fraction": row[0], "method": label,
                                    "stations": int(num(row[station_i])), "coverage": y, "mass_kg_d": x})
        anchor = next(row for row in rows if row[0] == "0.4")
        ax.annotate(r"$\lambda/\lambda_0=0.4$", (num(anchor[6]), num(anchor[5])), xytext=((-5 if panel else 5), 6),
                    textcoords="offset points", fontsize=7.8, color=color, ha=("right" if panel else "left"),
                    arrowprops=dict(arrowstyle="-", color=color, lw=.55))
        st.style(ax, pale=st.PALE_PURPLE if network == "Net3" else st.PALE_TEAL, grid="y")
        ax.set_xscale("log")
        ax.set_xlim(4e-4, 3.0)
        ax.set_ylim(.85, 1.004)
        ax.set_xlabel("Dosed chlorine mass (kg d$^{-1}$)")
        ax.set_ylabel("Demand-weighted coverage")
        ax.text(.97, .08, f"{len(rows)} penalty values", transform=ax.transAxes, ha="right", fontsize=7, color=st.GREY)

    method_handles = [Line2D([], [], color=st.PURPLE, marker=m, ls=ls, ms=3.5,
                             mfc=st.PURPLE if filled else "white", mec=st.PURPLE, label=label)
                      for label, _, _, _, m, ls, filled in methods]
    figure.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(.5, .946), ncol=3,
                  frameon=False, fontsize=7.1, handletextpad=.5, columnspacing=1.4)

    ax = axes[2]
    xi = np.arange(14)
    tick_idx = [0, 3, 5, 8, 11, 13]
    for network, table, _ in networks:
        rows = table[1:]
        color = NET[network]
        support = np.array([num(row[4]) for row in rows])
        active_const = np.array([num(row[7]) for row in rows])
        ax.step(xi, support, where="mid", color=color, lw=1.2, label=f"{network}: refit support")
        ax.scatter(xi, support, s=13, color=color, edgecolor="white", linewidth=.45, zorder=3)
        ax.plot(xi, active_const, color=color, ls=st.DASH, lw=.9, marker="^", ms=3.0,
                mfc="white", mec=color, label=f"{network}: constant-rate active")
    st.style(ax, pale="#FBFBFC", grid="y")
    ax.set_xticks(tick_idx, [paths[0][1:][i][0] for i in tick_idx])
    ax.set_xlabel(r"Penalty fraction $\lambda/\lambda_0$")
    ax.set_ylabel("Selected stations")
    ax.legend(loc="upper left", ncol=2, fontsize=6.6)

    ax = axes[3]
    for network, _, check in networks:
        rows = check[1:]
        rel = np.maximum([num(row[3]) for row in rows], 1e-17)
        color = NET[network]
        ax.plot(xi, rel, color=color, lw=.9, marker="o", ms=3.4, mfc=color, mec="white", mew=.4, label=network)
        for row, value in zip(rows, rel):
            check_rows.append({"network": network, "lambda_fraction": row[0],
                               "relative_objective_difference": value,
                               "stations_matrix_free": int(num(row[5])), "stations_dense": int(num(row[6]))})
    # Use a neutral audit band so it cannot be mistaken for the North Penn series.
    ax.axhspan(1e-17, 1e-10, color="#EEF1F4", zorder=-2)
    ax.axhline(1e-10, color=st.GREY, ls=st.DASH, lw=.6, zorder=-1)
    ax.axvspan(12.5, 13.5, color="#FFF0EB", zorder=-1)
    ax.text(.3, 4e-10, r"$\leq10^{-10}$ agreement", fontsize=7.8, color=st.GREY)
    ax.text(13, 2e-2, r"$\lambda=0$" + "\nnon-unique", ha="center", va="top", fontsize=7, color=st.VERM)
    st.style(ax, pale="#FBFBFC", grid="y")
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-1)
    ax.set_xticks(tick_idx, [checks[0][1:][i][0] for i in tick_idx])
    ax.set_xlabel(r"Penalty fraction $\lambda/\lambda_0$")
    ax.set_ylabel("Relative objective difference")
    ax.legend(loc="upper left", fontsize=6.8)

    st.source_csv("figS5_source_path.csv", source_rows)
    st.source_csv("figS5_source_objective.csv", check_rows)
    st.finish(figure, "figS5_booster",
              "Coverage and dose are plotted jointly for every saved penalty and method; support size and dense-solver agreement retain all values.",
              {"data_rows": {"path_designs": len(source_rows), "objective_checks": len(check_rows)},
               "encoding": {"Net3": st.MAGENTA, "North Penn": st.TEAL,
                            "method_shapes": {m[0]: m[4] for m in methods}},
               "limits": ["Pale links join refit and constant-rate doses on the same support; they are not intervals.",
                          "The group-lasso path is a continuous relaxation, not a combinatorial optimum certificate."]})
    _assert_final_text_size("figS5_booster")


def export_s6() -> None:
    source = _reset_source("T5_exp2_source.md")
    text = source.read_text(encoding="utf-8")
    tables = [table for table in md_tables(source) if table[0][0] == "window h"]
    networks = [("BWSN2", tables[0]), ("L-Town", tables[1])]
    windows = ["2 h", "6 h", "All data"]
    metrics = [("GLRT top-1", 2, st.PURPLE, "o", True), ("GLRT top-5", 3, st.MAGENTA, "s", False),
               ("Correct / equivalent", 4, st.TEAL, "D", True), ("NNLS top-1", 8, st.CYAN, "^", True)]
    # Add physical row separation without shrinking either row at publication width.  The upper row
    # moves with the enlarged canvas while panels c and d retain their bottom margin, so their titles
    # clear the upper-row x labels instead of competing for the same vertical band.
    figure = plt.figure(figsize=(176 * pf.MM, 132 * pf.MM))
    axes = [st.axis(figure, 14, 78, 72, 39), st.axis(figure, 101, 78, 72, 39),
            st.axis(figure, 32, 21, 80, 34), st.axis(figure, 124, 21, 49, 34)]
    st.panel(figure, 3, 125, "a", "BWSN2: identification accuracy")
    st.panel(figure, 90, 125, "b", "L-Town: identification accuracy")
    st.panel(figure, 3, 62, "c", "Measured pipeline versus projected library cost")
    st.panel(figure, 113, 62, "d", "Whole-study wall time")

    accuracy_rows, cost_rows = [], []
    for panel, (network, table) in enumerate(networks):
        ax = axes[panel]
        rows = table[1:]
        y = np.arange(3)[::-1]
        for label, index, color, marker, filled in metrics:
            values = np.array([num(row[index]) for row in rows])
            ax.plot(values, y, color=color, lw=.75, marker=marker, ms=4,
                    mfc=color if filled else "white", mec=color, mew=.75, label=label)
            for window, value in zip(windows, values):
                accuracy_rows.append({"network": network, "window": window, "criterion": label, "fraction": value,
                                      "detected_scenarios": int(num(rows[0][1]))})
        st.style(ax, pale=st.PALE_TEAL if network == "BWSN2" else st.PALE_PURPLE, grid="x")
        ax.set_xlim(0, 1.025)
        ax.set_yticks(y, windows)
        ax.set_xlabel("Fraction of detected scenarios")
        ax.text(.02, .06, f"n = {int(num(rows[0][1]))} detected", transform=ax.transAxes, fontsize=7,
                color=NET[network], fontweight="bold")
    metric_handles = [Line2D([], [], color=color, marker=marker, ls="-", ms=3.8,
                             mfc=color if filled else "white", mec=color, label=label)
                      for label, _, color, marker, filled in metrics]
    figure.legend(handles=metric_handles, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4,
                  frameon=False, fontsize=6.8, handletextpad=.45, columnspacing=1.1)

    ax = axes[2]
    row_labels, measured, projected = [], [], []
    for network, table in networks:
        rows = table[1:]
        for window, row in zip(windows, rows):
            row_labels.append(f"{network} · {window}")
            measured.append(num(row[10]))
            projected.append(num(row[11]))
            cost_rows.append({"network": network, "window": window, "measured_pipeline_s": num(row[10]),
                              "projected_library_s": num(row[11])})
    y = np.arange(len(row_labels))[::-1]
    for yi, m, p in zip(y, measured, projected):
        ax.plot([m, p], [yi, yi], color=st.LIGHT, lw=1.4, zorder=1)
    ax.scatter(measured, y, s=22, color=st.PURPLE, marker="D", edgecolor="white", linewidth=.5, zorder=3,
               label="Measured pipeline")
    ax.scatter(projected, y, s=27, facecolor="white", edgecolor=st.GOLD, marker="s", linewidth=.9, zorder=3,
               label="Library, projected")
    st.style(ax, pale="#FBFBFC", grid="x")
    ax.set_xscale("log")
    ax.set_xlim(.5, 2e7)
    ax.set_yticks(y, row_labels)
    for tick, label in zip(ax.get_yticklabels(), row_labels):
        tick.set_color(st.TEAL if label.startswith("BWSN2") else st.MAGENTA)
    ax.set_xlabel("Time per scenario (s)")
    ax.legend(loc="lower right", fontsize=6.8)

    timings = re.findall(r"timing s: ([^\n]*)", text)
    component_defs = [("record_s", "Record", st.CYAN), ("build_s", "Schedule", st.GOLD),
                      ("sweep_s", "Adjoint", st.PURPLE), ("screen_s", "Screen", st.TEAL),
                      ("refine_s", "Refine", st.MAGENTA), ("exhaustive_s", "NNLS check", st.NAVY)]
    timing_rows = []
    ax = axes[3]
    for yi, (network, timing) in enumerate(zip(["BWSN2", "L-Town"], timings)):
        values = {key: float(value) for key, value in re.findall(r"(\w+) ([\d.]+)", timing)}
        left = 0.0
        for key, label, color in component_defs:
            value = values.get(key, 0.0)
            ax.barh(1 - yi, value, left=left, height=.48, color=color, edgecolor="white", linewidth=.45)
            timing_rows.append({"network": network, "component": label, "seconds": value})
            left += value
        residual = max(0.0, values["total_s"] - left)
        ax.barh(1 - yi, residual, left=left, height=.48, color=st.LIGHT, edgecolor="white", linewidth=.45)
        timing_rows.append({"network": network, "component": "Other", "seconds": residual})
        ax.text(values["total_s"] + 15, 1 - yi, f'{values["total_s"]:.0f} s', va="center", ha="left",
                fontsize=7, color=st.INK)
    st.style(ax, pale="#FBFBFC", grid="x")
    ax.set_xlim(0, 1600)
    ax.set_yticks([1, 0], ["BWSN2", "L-Town"])
    ax.get_yticklabels()[0].set_color(st.TEAL); ax.get_yticklabels()[1].set_color(st.MAGENTA)
    ax.set_xlabel("Wall-clock time (s)")
    component_handles = [Patch(facecolor=color, edgecolor="white", label=label) for _, label, color in component_defs]
    component_handles.append(Patch(facecolor=st.LIGHT, edgecolor="white", label="Other"))
    figure.legend(handles=component_handles, loc="lower center", bbox_to_anchor=(.74, .025), ncol=4,
                  frameon=False, fontsize=6.4, handlelength=1.2, columnspacing=.8)

    st.source_csv("figS6_source_accuracy.csv", accuracy_rows)
    st.source_csv("figS6_source_cost.csv", cost_rows)
    st.source_csv("figS6_source_timing.csv", timing_rows)
    st.finish(figure, "figS6_source",
              "Accuracy is a dot-and-track comparison; measured and projected costs are paired endpoints; study time is a component stack.",
              {"data_rows": {"accuracy": len(accuracy_rows), "cost": len(cost_rows), "timing_components": len(timing_rows)},
               "encoding": {"criteria": {label: color for label, _, color, _, _ in metrics},
                            "networks": {key: NET[key] for key in ("BWSN2", "L-Town")}},
               "limits": ["Accuracy denominators are detected scenarios.",
                          "Library time is projected; the pipeline endpoint is measured."]})
    _assert_final_text_size("figS6_source")


def export_s7() -> None:
    source = _reset_source("T6_exp3_pipe_kw.md")
    tables = md_tables(source)
    field = tables[0][1:]
    sets = [("BWSN2", tables[1][1:]), ("L-Town", tables[2][1:])]
    # The two-line category labels in a and b need a larger inter-row band than a single-line axis.
    # Increase the canvas and carry only the upper row upward; c and d therefore sit visibly lower
    # while all data axes keep their original physical size and final text remains above 5 pt.
    figure = plt.figure(figsize=(176 * pf.MM, 140 * pf.MM))
    axes = [st.axis(figure, 14, 82, 72, 42), st.axis(figure, 101, 82, 72, 42),
            st.axis(figure, 14, 19, 72, 37), st.axis(figure, 101, 38, 72, 18), st.axis(figure, 101, 14, 72, 18)]
    st.panel(figure, 3, 132, "a", "Audit prediction versus re-estimation")
    st.panel(figure, 90, 132, "b", "Effective wall parameters")
    st.panel(figure, 3, 63, "c", "Pipe-level estimation gain")
    st.panel(figure, 90, 63, "d", "Field campaigns before and after prescription")

    audit_rows, field_rows = [], []
    ax = axes[0]
    for network, rows in sets:
        color = NET[network]
        for row in rows:
            sensor_set = row[0]
            x, y = num(row[3]), num(row[4])
            ax.scatter(x, y, s=25, color=color, marker=SET_MARK[sensor_set], edgecolor="white", linewidth=.55, zorder=3)
            audit_rows.append({"network": network, "sensor_set": sensor_set, "predicted_identifiable": x,
                               "realised_identifiable": y, "effective_parameters": num(row[5]),
                               "rmse_map": num(row[7]), "rmse_grouped_prior": num(row[8]),
                               "rmse_ratio": num(row[7]) / num(row[8])})
    ax.plot([0, 34], [0, 34], color=st.GREY, ls=st.DASH, lw=.8)
    ax.fill_between([0, 34], [0, 34], [3, 37], color=st.PALE_GOLD, alpha=.55, zorder=-2)
    ax.fill_between([0, 34], [-3, 31], [0, 34], color=st.PALE_GOLD, alpha=.55, zorder=-2)
    ax.text(.97, .05, "±3 pipes", transform=ax.transAxes, ha="right", fontsize=7, color=st.GOLD)
    st.style(ax, pale="#FBFBFC", grid="both")
    ax.set_xlim(0, 35); ax.set_ylim(0, 35)
    ax.set_xlabel("Identifiable pipes predicted")
    ax.set_ylabel("Identifiable pipes realised")
    handles = [Line2D([], [], marker=SET_MARK[key], ls="none", color=st.INK, mfc="white", label=label)
               for key, label in SET_LABEL.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=6.3, ncol=2)

    categories = [row[0] for row in sets[0][1]]
    labels = [SET_LABEL[key].replace("+ ", "+\n") for key in categories]
    x = np.arange(4)
    ax = axes[1]
    for offset, (network, rows) in zip((-.12, .12), sets):
        vals = [num(row[5]) for row in rows]
        color = NET[network]
        ax.vlines(x + offset, 0, vals, color=color, lw=1.0, alpha=.45)
        ax.scatter(x + offset, vals, s=25, color=color, edgecolor="white", linewidth=.55, zorder=3, label=network)
    st.style(ax, pale="#FBFBFC", grid="y")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Effective parameters, $p_{\rm eff}$")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper left", fontsize=6.8)

    ax = axes[2]
    for offset, (network, rows) in zip((-.12, .12), sets):
        ratios = [num(row[7]) / num(row[8]) for row in rows]
        color = NET[network]
        ax.vlines(x + offset, 0, ratios, color=color, lw=1.0, alpha=.45)
        ax.scatter(x + offset, ratios, s=25, color=color, edgecolor="white", linewidth=.55, zorder=3)
    ax.axhspan(0, 1, color=st.PALE_TEAL, alpha=.55, zorder=-2)
    ax.axhline(1, color=st.GREY, ls=st.DASH, lw=.75)
    ax.text(.98, .93, "Improvement", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=st.TEAL)
    st.style(ax, pale="#FBFBFC", grid="y")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"RMSE ratio, MAP / grouped prior")
    ax.set_ylim(0, 1.05)

    names = ["Fairfield", "Bellingham", "North Penn"]
    x_field = np.arange(3)
    before_ident = [num(row[6]) for row in field]
    after_ident = [num(row[11]) for row in field]
    before_eff = [num(row[7]) for row in field]
    after_eff = [num(row[12]) for row in field]
    for row in field:
        field_rows.append({"dataset": row[0], "identifiable_before": num(row[6]), "identifiable_after": num(row[11]),
                           "effective_before": num(row[7]), "effective_after": num(row[12]), "sites_added": num(row[10])})
    for ax, before, after, ylabel, ylim in ((axes[3], before_ident, after_ident, "Identifiable pipes", (0, 16)),
                                            (axes[4], before_eff, after_eff, r"$p_{\rm eff}$", (0, 31))):
        for i, (name, b, a) in enumerate(zip(names, before, after)):
            color = FIELD[name]
            ax.plot([i - .12, i + .12], [b, a], color=st.LIGHT, lw=1.3)
            ax.scatter(i - .12, b, s=22, facecolor="white", edgecolor=color, linewidth=.9, marker="o", zorder=3)
            ax.scatter(i + .12, a, s=25, color=color, edgecolor="white", linewidth=.5, marker="^", zorder=3)
        st.style(ax, pale="#FBFBFC", grid="y")
        ax.set_xlim(-.5, 2.5); ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
    axes[3].set_xticks(x_field, [])
    axes[4].set_xticks(x_field, names)
    for tick, name in zip(axes[4].get_xticklabels(), names):
        tick.set_color(FIELD[name])
    axes[3].legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=st.INK, label="Existing sites"),
                            Line2D([], [], marker="^", ls="none", mfc=st.INK, mec="white", label="+5 D-optimal sites")],
                   loc="upper left", ncol=2, fontsize=6.5)

    st.source_csv("figS7_source_audit.csv", audit_rows)
    st.source_csv("figS7_source_field.csv", field_rows)
    st.finish(figure, "figS7_ident",
              "Prediction accuracy, effective dimension and RMSE gain are encoded as points; field campaigns use paired before/after summaries.",
              {"data_rows": {"synthetic_sensor_sets": len(audit_rows), "field_datasets": len(field_rows)},
               "encoding": {"networks": {key: NET[key] for key in ("BWSN2", "L-Town")},
                            "field_datasets": FIELD, "sensor_set_shapes": SET_MARK},
               "limits": ["The ±3-pipe band is a visual audit tolerance, not a confidence interval.",
                          "RMSE ratios use identifiable pipes only, as defined in Table S7."]})
    _assert_final_text_size("figS7_ident")


if __name__ == "__main__":
    export_s5()
    export_s6()
    export_s7()
