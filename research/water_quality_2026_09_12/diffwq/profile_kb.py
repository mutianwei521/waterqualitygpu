"""G4 / H1 probe: profile of the calibration objective along Kb.

For each value of Kb on a grid (bounds and the INP bottle-test value included) the Kw groups are
re-optimised with the exact adjoint gradient (L-BFGS-B, warm start); the profile SSE(Kb) shows how
much the field data constrain the bulk coefficient once the wall coefficients are free.

Usage (from research/water_quality_2026_09_12):
    python -m diffwq.profile_kb --datasets Fairfield,Bellingham,NorthPenn --out g4_profile_kb.json
"""
import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

from . import epanet_api as api
from .calib import CalibProblem, run_adj_lbfgsb


def run_dataset(name, lib, workdir, args):
    prob = CalibProblem(name, lib, workdir, device=args.device, fix_kb=True)
    try:
        kb_inp = prob.kb_fixed
        grid = sorted(set([float(v) for v in args.grid] + [kb_inp]), reverse=True)     # 0 -> -5
        out = {'dataset': name, 'kb_inp': kb_inp, 'param_names': prob.param_names, 'n_obs': len(prob.obs), 'profile': []}
        x0 = prob.theta_nominal
        for kb in grid:
            prob.kb_fixed = kb
            t0 = time.perf_counter()
            rec = run_adj_lbfgsb(prob, x0, label='ADJ-LBFGSB')
            x0 = np.asarray(rec['theta'])
            row = {'kb': kb, 'sse': rec['sse'], 'rmse': rec['rmse'], 'theta_kw': rec['theta'], 'n_grad': rec['n_eval'],
                   'wall_s': time.perf_counter() - t0, 'site_rmse': rec['site_rmse'], 'n_at_bound': int(sum(
                       abs(v - b) < 1e-6 for v, (b, _) in zip(rec['theta'], prob.bounds)))}
            out['profile'].append(row)
            print(f"[{name}] Kb={kb:+.3f}: rmse={rec['rmse']:.4f} sse={rec['sse']:.4f} Kw*={np.round(rec['theta'], 3).tolist()} "
                  f"({rec['n_eval']} gradients, {row['wall_s']:.0f}s)", flush=True)
        return out
    finally:
        prob.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='Fairfield,Bellingham,NorthPenn')
    ap.add_argument('--grid', default='0,-0.1,-0.25,-0.5,-0.75,-1,-1.5,-2,-3,-5')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.grid = [float(v) for v in args.grid.split(',')]
    lib = api.Library(args.lib)
    torch.set_num_threads(args.threads or max(1, os.cpu_count() // 2))
    results = {'platform': platform.platform(), 'torch': torch.__version__, 'epanet': lib.version(), 'args': vars(args), 'datasets': []}
    with tempfile.TemporaryDirectory(prefix='g4k_') as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        for name in args.datasets.split(','):
            results['datasets'].append(run_dataset(name, lib, workdir, args))
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=1))
    if args.out:
        print('wrote', args.out)


if __name__ == '__main__':
    main()
