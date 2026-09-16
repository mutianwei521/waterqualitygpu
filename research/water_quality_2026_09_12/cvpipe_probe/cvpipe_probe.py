"""Read-only regression probe for long CVPIPE quality behavior.

Uses the already built EPANET DLL. Creates only synthetic inputs/reports/results
in this directory. Does not modify EPANET sources or rebuild any library.
"""
from pathlib import Path
import ctypes as ct
import csv
import hashlib
import json
import math
import platform
import re
import time

HERE = Path(__file__).resolve().parent
DLL = HERE.parent / "build_epanet/bin/RelWithDebInfo/epanet2.dll"
LIB = ct.CDLL(str(DLL))
P, I, L, D = ct.c_void_p, ct.c_int, ct.c_long, ct.c_double
SIGNATURES = {
    "EN_createproject": [ct.POINTER(P)], "EN_deleteproject": [P],
    "EN_getversion": [ct.POINTER(I)],
    "EN_open": [P, ct.c_char_p, ct.c_char_p, ct.c_char_p],
    "EN_close": [P], "EN_solveH": [P], "EN_openQ": [P],
    "EN_initQ": [P, I], "EN_runQ": [P, ct.POINTER(L)],
    "EN_stepQ": [P, ct.POINTER(L)], "EN_closeQ": [P],
    "EN_getnodeindex": [P, ct.c_char_p, ct.POINTER(I)],
    "EN_getlinkindex": [P, ct.c_char_p, ct.POINTER(I)],
    "EN_getlinktype": [P, I, ct.POINTER(I)],
    "EN_getnodevalue": [P, I, I, ct.POINTER(D)],
    "EN_getlinkvalue": [P, I, I, ct.POINTER(D)],
    "EN_getstatistic": [P, I, ct.POINTER(D)],
}
for name, args in SIGNATURES.items():
    getattr(LIB, name).argtypes = args
    getattr(LIB, name).restype = I


def call(name, *args):
    status = getattr(LIB, name)(*args)
    if status != 0:
        raise RuntimeError(f"{name} returned EPANET status {status}")


def get_value(name, handle, index, prop):
    value = D()
    call(name, handle, index, prop, ct.byref(value))
    return value.value


def make_input(length_m, pipe_kind, bulk_per_day, quality_step):
    return f"""[TITLE]
Synthetic one reservoir / one junction / one long pipe CVPIPE probe
[JUNCTIONS]
J 0 1
[RESERVOIRS]
R 100
[PIPES]
P R J {length_m} 100 130 0 {pipe_kind}
[QUALITY]
R 1
J 0
[REACTIONS]
ORDER BULK 1
ORDER WALL 1
GLOBAL BULK {bulk_per_day}
GLOBAL WALL 0
[TIMES]
DURATION 5:00
HYDRAULIC TIMESTEP 1:00
QUALITY TIMESTEP 0:00:{quality_step:02d}
PATTERN TIMESTEP 1:00
REPORT TIMESTEP 1:00
[OPTIONS]
UNITS LPS
HEADLOSS H-W
QUALITY Chlorine mg/L
TOLERANCE 0.0000000001
[REPORT]
STATUS FULL
SUMMARY YES
[END]
"""


def run_case(length_m, kind, bulk, step, suffix=""):
    label = f"L{length_m}_{kind}_k{abs(bulk)}_dt{step}{suffix}"
    inp, rpt = HERE / (label + ".inp"), HERE / (label + ".rpt")
    inp.write_text(make_input(length_m, kind, bulk, step), encoding="ascii")
    ph = P()
    call("EN_createproject", ct.byref(ph))
    opened = qopened = False
    start = time.perf_counter()
    rows = []
    try:
        call("EN_open", ph, str(inp).encode(), str(rpt).encode(), b"")
        opened = True
        j, p, link_type = I(), I(), I()
        call("EN_getnodeindex", ph, b"J", ct.byref(j))
        call("EN_getlinkindex", ph, b"P", ct.byref(p))
        call("EN_getlinktype", ph, p, ct.byref(link_type))
        call("EN_solveH", ph)
        hydraulic = {
            "link_type_enum": link_type.value,
            "length_m": get_value("EN_getlinkvalue", ph, p, 1),
            "diameter_mm": get_value("EN_getlinkvalue", ph, p, 0),
            "bulk_coefficient_per_day": get_value("EN_getlinkvalue", ph, p, 6),
            "flow_L_per_s": get_value("EN_getlinkvalue", ph, p, 8),
            "velocity_m_per_s": get_value("EN_getlinkvalue", ph, p, 9),
            "headloss_m": get_value("EN_getlinkvalue", ph, p, 10),
            "open_status": get_value("EN_getlinkvalue", ph, p, 11),
            "junction_demand_L_per_s": get_value("EN_getnodevalue", ph, j, 9),
        }
        call("EN_openQ", ph)
        qopened = True
        call("EN_initQ", ph, 0)
        t, left = L(), L()
        while True:
            call("EN_runQ", ph, ct.byref(t))
            c = get_value("EN_getnodevalue", ph, j, 12)
            link_c = get_value("EN_getlinkvalue", ph, p, 14)
            q = get_value("EN_getlinkvalue", ph, p, 8)
            rows.append([t.value, c, link_c, q])
            if t.value >= 18000:
                break
            call("EN_stepQ", ph, ct.byref(left))
        balance = D()
        call("EN_getstatistic", ph, 4, ct.byref(balance))
    finally:
        if qopened:
            call("EN_closeQ", ph)
        if opened:
            call("EN_close", ph)
        call("EN_deleteproject", ph)

    csv_path = HERE / (label + ".csv")
    with csv_path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "junction_chlorine_mg_L", "link_mean_mg_L", "flow_L_s"])
        writer.writerows(rows)
    volume = math.pi * (hydraulic["diameter_mm"] / 1000) ** 2 / 4 * length_m
    travel = volume / (hydraulic["flow_L_per_s"] / 1000)
    expected = math.exp(bulk * travel / 86400)
    selected = sorted(set([0, step, 60, 600, 780, 790, 3600, 7200, 7850, 7860, 10800, 14400, 18000]))
    sampled = [row for row in rows if row[0] in selected]
    report_text = rpt.read_text(encoding="ascii", errors="replace")
    mass_fields = ["Initial Mass", "Mass Inflow", "Mass Outflow", "Mass Reacted", "Final Mass"]
    reported_mass = {}
    for field in mass_fields:
        matches = re.findall(r"^\s*" + re.escape(field) + r":\s*([+-]?[\d.eE+-]+)", report_text, re.MULTILINE)
        if not matches:
            raise AssertionError(f"Missing mass field {field} in {rpt}")
        reported_mass[field] = float(matches[-1])
    return {
        "id": label, "pipe_status_input": kind, "quality_step_s": step,
        "hydraulic": hydraulic, "geometric_volume_m3": volume,
        "analytic_advection_travel_time_s": travel,
        "analytic_first_order_steady_chlorine_mg_L": expected,
        "first_positive_time_s": next((r[0] for r in rows if r[1] > 1e-9), None),
        "first_above_0p01_time_s": next((r[0] for r in rows if r[1] > .01), None),
        "first_above_half_analytic_steady_time_s": next((r[0] for r in rows if r[1] > .5 * expected), None),
        "end_junction_chlorine_mg_L": rows[-1][1],
        "end_link_mean_mg_L": rows[-1][2],
        "mass_balance_ratio": balance.value,
        "reported_mass_balance_mg_rounded_by_report": reported_mass,
        "flow_min_L_s": min(r[3] for r in rows),
        "flow_max_L_s": max(r[3] for r in rows),
        "samples_time_node_link_flow": sampled,
        "full_trajectory_csv": csv_path.name,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
        "elapsed_seconds_including_hydraulics_and_python": time.perf_counter() - start,
        "all_api_return_codes_zero": True,
    }


def main():
    version = I()
    call("EN_getversion", ct.byref(version))
    output = []
    for length in [100, 1000]:
        for bulk in [0, -10]:
            for kind in ["OPEN", "CV"]:
                result = run_case(length, kind, bulk, 10)
                output.append(result)
                print(result["id"], result["first_positive_time_s"], result["end_junction_chlorine_mg_L"], flush=True)
    for bulk in [0, -10]:
        for kind in ["OPEN", "CV"]:
            result = run_case(1000, kind, bulk, 1)
            output.append(result)
            print(result["id"], result["first_positive_time_s"], result["end_junction_chlorine_mg_L"], flush=True)
    repeat = run_case(1000, "CV", -10, 10, "_repeat")
    original = next(r for r in output if r["id"] == "L1000_CV_k10_dt10")
    output.append(repeat)
    equality = original["csv_sha256"] == repeat["csv_sha256"]
    comparisons = []
    for pipe in [r for r in output if r["pipe_status_input"] == "OPEN"]:
        cv = next(r for r in output if r["id"] == pipe["id"].replace("OPEN", "CV"))
        comparisons.append({
            "pipe": pipe["id"], "cvpipe": cv["id"],
            "flow_difference_L_s": cv["hydraulic"]["flow_L_per_s"] - pipe["hydraulic"]["flow_L_per_s"],
            "headloss_difference_m": cv["hydraulic"]["headloss_m"] - pipe["hydraulic"]["headloss_m"],
            "junction_final_chlorine_difference_mg_L": cv["end_junction_chlorine_mg_L"] - pipe["end_junction_chlorine_mg_L"],
        })
    report = {
        "engine_version": version.value, "dll_path": str(DLL),
        "dll_sha256": hashlib.sha256(DLL.read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "protocol": "Synthetic single source R (1 mg/L), J (initial 0 mg/L), one 100 or 1000 m pipe, diameter 100 mm, constant demand 1 L/s, head 100 m; only OPEN vs CV changes within each pair. No wall reactions. Bulk zero or -10/day. All Q steps sampled using EN_runQ/EN_stepQ. Geometry and hydraulics independently queried through API. Five hours, dt10s and long-pipe dt1s confirmation. No engine modification.",
        "repeat_cv_reactive_dt10_identical_full_trajectory": equality,
        "comparisons": comparisons, "cases": output,
    }
    long_pipe = next(r for r in output if r["id"] == "L1000_OPEN_k10_dt1")
    long_cv = next(r for r in output if r["id"] == "L1000_CV_k10_dt1")
    report["diagnostic_checks"] = {
        "all_pairs_same_flow_and_headloss_to_1e_10": all(abs(p["flow_difference_L_s"]) < 1e-10 and abs(p["headloss_difference_m"]) < 1e-10 for p in comparisons),
        "long_cv_api_geometry_remains_1000m_100mm": long_cv["hydraulic"]["length_m"] == 1000 and long_cv["hydraulic"]["diameter_mm"] == 100,
        "long_pipe_arrival_close_to_geometric_travel_time": abs(long_pipe["first_positive_time_s"] - long_pipe["analytic_advection_travel_time_s"]) < 2,
        "long_cv_arrival_at_first_1_second_step": long_cv["first_positive_time_s"] == 1,
        "long_cv_reactive_final_concentration_equals_source": long_cv["end_junction_chlorine_mg_L"] == 1,
        "long_cv_reported_final_and_reacted_mass_zero": long_cv["reported_mass_balance_mg_rounded_by_report"]["Final Mass"] == 0 and long_cv["reported_mass_balance_mg_rounded_by_report"]["Mass Reacted"] == 0,
        "repeat_cv_full_trajectory_identical": equality,
    }
    euler_checks = []
    for step in [10, 1]:
        case = next(r for r in output if r["id"] == f"L1000_OPEN_k10_dt{step}")
        alpha, tau = 10 / 86400, case["analytic_advection_travel_time_s"]
        exact = case["analytic_first_order_steady_chlorine_mg_L"]
        bound_exponent = alpha * alpha * tau * step / (2 * (1 - alpha * step))
        euler_checks.append({
            "dt_s": step, "observed_pipe_difference_from_geometric_analytic": exact - case["end_junction_chlorine_mg_L"],
            "fixed_exposure_euler_absolute_error_bound": exact * (1 - math.exp(-bound_exponent)),
            "one_timestep_exposure_uncertainty_absolute_bound": math.exp(-alpha * max(0, tau - step)) - exact,
            "interpretation": "Fixed-exposure Euler bound is for y'= -alpha*y at equal elapsed time, excluding discrete parcel arrival/mixing and EPANET rounded unit constants. The separate timestep exposure bound illustrates that timing error is distinct. Neither is claimed as a theorem for the complete engine.",
        })
    report["ordinary_pipe_euler_error_context"] = euler_checks
    (HERE / "cvpipe_probe_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESULTS", len(output), "repeat_equal", equality, flush=True)


if __name__ == "__main__":
    main()
