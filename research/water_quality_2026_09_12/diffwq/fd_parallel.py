"""R2-4: measured wall time of a finite-difference gradient with the EPANET library on P worker processes.

One FD gradient of a bench.py case (the Table S3 set) is N_theta + 1 library quality runs on saved hydraulics,
N_theta = 2 pipes + tanks + nodes + sources as in bench.py; run j perturbs parameter j (pipe kb, pipe kw, tank kb,
node c0, source strength) and restores it afterwards. Each worker opens the case and solves the hydraulics once
(not timed). The wall time of M runs spread over P workers is timed: M = N_theta + 1 when that fits --budget-s on
one process, otherwise the first M columns (at least max(procs)), and the gradient time is scaled by (N_theta + 1) / M.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.fd_parallel --procs 1,2,4,8 --lib LIB --out fd_parallel.json
"""
import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

from . import epanet_api as api
from .network import extract
from .verify_torch import CASES, open_case
from .verify_m0 import CORE, ROOT
from .bench import DEFAULT_CASES, EXTRA_CASES, timed

_W = {}


def case_table():
    cases = {c[0]: (ROOT / CORE[c[1]], c[2], c[3]) for c in CASES}
    cases.update({k: (v[1], v[2], v[3]) for k, v in EXTRA_CASES.items()})
    return cases


def open_solved(lib, name, wd, tag):
    inp, setup, over = case_table()[name]
    pr = open_case(lib, name, inp, setup, over.get('qstep'), over.get('ctol'), wd)
    pr.solve_hydraulics(Path(wd) / f'{name}_{tag}.hyd')
    return pr


def columns(pr, net):
    """(kind, index, property) of the base run and of every FD column, in bench.py's N_theta count."""
    pipes = [k for k in range(1, net.nlinks + 1) if net.link_type[k] == api.EN_PIPE]
    tanks = [i for i in range(1, net.nnodes + 1) if pr.node_type(i) == api.EN_TANK]
    return ([('base', 0, None)] + [('link', k, api.EN_KBULK) for k in pipes] + [('link', k, api.EN_KWALL) for k in pipes]
            + [('node', i, api.EN_TANK_KBULK) for i in tanks] + [('node', i, api.EN_INITQUAL) for i in range(1, net.nnodes + 1)]
            + [('node', i, api.EN_SOURCEQUAL) for i in net.sources])


def _init(lib_path, name, wd, mode):
    wd = Path(wd) / f'w{os.getpid()}'; wd.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':                       # EPANET's Windows scratch-file names can collide across processes
        os.environ['TMP'] = os.environ['TEMP'] = str(wd)
    pr = open_solved(api.Library(lib_path), name, wd, 'w')
    _W.update(pr=pr, cols=columns(pr, extract(pr)), mode=mode)


def _ready(_):
    time.sleep(0.05)
    return os.getpid()


def _run(j):
    pr, (kind, i, prop) = _W['pr'], _W['cols'][j]
    if kind != 'base':
        get, put = (pr.link_value, pr.set_link_value) if kind == 'link' else (pr.node_value, pr.set_node_value)
        v = get(i, prop); put(i, prop, v + 1e-3 * max(abs(v), 1e-3))
    pr.run_quality_reference(_W['mode'])
    if kind != 'base':
        put(i, prop, v)
    return j


def run_case(args, name, wd):
    with open_solved(api.Library(args.lib), name, wd, 'main') as pr:
        net = extract(pr)
        cols = columns(pr, net)
        t_dll = timed(lambda: pr.run_quality_reference(args.mode), lambda: None, 3)
    n = len(cols)
    M = n if t_dll * n <= args.budget_s else min(n, max(max(args.procs), int(args.budget_s / t_dll)))
    repeat = args.repeat if t_dll * M <= 30 else 1
    rec = {'case': name, 'nodes': net.nnodes, 'links': net.nlinks, 'n_theta': n - 1, 'dll_quality_run_s': t_dll,
           'fd_gradient_projected_s': n * t_dll, 'runs_timed': M, 'repeat': repeat, 'procs': {}}
    ctx = get_context('spawn' if os.name == 'nt' else 'fork')
    for P in args.procs:
        with ctx.Pool(P, initializer=_init, initargs=(args.lib, name, wd, args.mode)) as pool:
            seen = set()
            while len(seen) < P:              # every worker has opened its project and solved the hydraulics
                seen |= set(pool.map(_ready, range(4 * P), chunksize=1))
            walls = []
            for _ in range(repeat):
                t0 = time.perf_counter(); done = sum(1 for _ in pool.imap_unordered(_run, range(M), chunksize=1))
                walls.append(time.perf_counter() - t0)
            assert done == M
        w = min(walls)
        rec['procs'][P] = {'wall_s': walls, 'fd_gradient_wall_s': w * n / M}
        print(f"  {name}: P={P} {M} runs in {w:.3f}s -> FD gradient {w * n / M:.3f}s", flush=True)
    w1 = rec['procs'][args.procs[0]]['fd_gradient_wall_s'] * args.procs[0]
    for P, r in rec['procs'].items():
        r['speedup_vs_1'] = w1 / r['fd_gradient_wall_s']; r['efficiency'] = r['speedup_vs_1'] / P
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=','.join(DEFAULT_CASES + list(EXTRA_CASES)))
    ap.add_argument('--procs', default='1,2,4,8')
    ap.add_argument('--mode', default='nextQ', choices=['nextQ', 'stepQ'])
    ap.add_argument('--budget-s', type=float, default=120.0, help='serial time budget of the timed runs per case')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    args.procs = [int(p) for p in args.procs.split(',')]
    wd = tempfile.mkdtemp(prefix='diffwq_fdpar_', dir=os.environ.get('TMPDIR'))
    try:
        cpu = subprocess.run(['lscpu'], capture_output=True, text=True).stdout
        cpu = next((l.split(':', 1)[1].strip() for l in cpu.splitlines() if l.startswith('Model name')), platform.processor())
    except OSError:
        cpu = platform.processor()
    report = {'engine_version': api.Library(args.lib).version(), 'platform': platform.platform(), 'cpu': cpu,
              'cpus_available': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
              'slurm_cpus': os.environ.get('SLURM_CPUS_PER_TASK'), 'mode': args.mode, 'budget_s': args.budget_s, 'results': []}
    for name in args.cases.split(','):
        print(name, flush=True)
        report['results'].append(run_case(args, name, wd))
        Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('report:', args.out)


if __name__ == '__main__':
    main()
