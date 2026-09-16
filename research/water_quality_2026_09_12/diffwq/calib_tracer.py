"""Gate G4, transport-only part: the Net2-FL fluoride tracer study (55 h, 4 sampling nodes, no
reaction). The engine reproduces EPANET exactly, so the residual against the field data measures the
published hydraulic/transport model, not the engine. Three things are quantified:

  1. as-published fit (Ctol = 0.01 and 0; identical through the DLL and the replay engine);
  2. the tank mixing model as a discrete alternative (MIX1 / MIX2 / FIFO / LIFO), through the DLL;
  3. the only continuous transport parameters the map has, initial quality c0 (all nodes and the
     tank) and the source strength, fitted with the exact adjoint gradient (L-BFGS-B) and, for the
     cost comparison, with the DLL finite-difference gradient; the residual Jacobian gives the
     number of identifiable directions of that (affine) problem.

Usage (from research/water_quality_2026_09_12):
    python -m diffwq.calib_tracer --out g4_tracer.json
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
from scipy.optimize import minimize

from . import epanet_api as api
from .calib import obs_indices
from .calib_data import prepare_inp, load_observations
from .hydfile import read_hydfile
from .network import extract
from .replicate import Replicator
from .torch_wq import TorchWQ

MIX_MODELS = {'MIX1': (api.EN_MIX1, None), 'MIX2(f=0.5)': (api.EN_MIX2, 0.5), 'MIX2(f=0.2)': (api.EN_MIX2, 0.2),
              'FIFO': (api.EN_FIFO, None), 'LIFO': (api.EN_LIFO, None)}


def stats(sim, obs):
    r = sim - obs.y
    return {'rmse': float(np.sqrt(np.mean(r * r))), 'mae': float(np.abs(r).mean()), 'bias': float(r.mean()),
            'corr': float(np.corrcoef(sim, obs.y)[0, 1]),
            'site_rmse': {obs.site_node_id[s]: float(np.sqrt(np.mean(r[obs.site == s] ** 2)))
                          for s in range(len(obs.site_node_id))}}


def dll_sim(pr, obs, qstep, ctol):
    pr.set_option(api.EN_TOLERANCE, ctol)
    times, q, _ = pr.run_quality_reference('stepQ')
    i0, i1, w = obs_indices(obs, times, qstep, q.shape[1])
    flat = q.reshape(-1)
    return flat[i0] * (1 - w) + flat[i1] * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--lib', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    lib = api.Library(args.lib)
    name = 'Net2-FL'
    out = {'platform': platform.platform(), 'torch': torch.__version__, 'epanet': lib.version(), 'dataset': name}
    with tempfile.TemporaryDirectory(prefix='g4t_') as workdir:
        os.environ['TMP'] = os.environ['TEMP'] = os.environ['TMPDIR'] = workdir
        inp = prepare_inp(name, workdir)
        with api.Project(lib, inp, Path(workdir) / 'net2fl.rpt') as pr:
            obs = load_observations(name, pr)
            qstep = pr.time_param(api.EN_QUALSTEP)
            out['n_obs'], out['sites'], out['qstep'] = len(obs), obs.site_node_id, qstep
            hyd = Path(workdir) / 'net2fl.hyd'
            pr.solve_hydraulics(hyd)
            # 1. as published
            out['as_published'] = {f'ctol={c}': stats(dll_sim(pr, obs, qstep, c), obs) for c in (0.01, 0.0)}
            print('as published:', {k: round(v['rmse'], 4) for k, v in out['as_published'].items()}, flush=True)
            # 2. tank mixing model (tank 26), DLL at Ctol = 0
            tank = pr.node_index('26')
            out['mixing'] = {}
            for label, (model, frac) in MIX_MODELS.items():
                pr.set_node_value(tank, api.EN_MIXMODEL, model)
                if frac is not None:
                    pr.set_node_value(tank, api.EN_MIXFRACTION, frac)
                out['mixing'][label] = stats(dll_sim(pr, obs, qstep, 0.0), obs)
                print(f'  mixing {label:<12s} rmse={out["mixing"][label]["rmse"]:.4f}', flush=True)
            pr.set_node_value(tank, api.EN_MIXMODEL, api.EN_MIX1)
            pr.set_option(api.EN_TOLERANCE, 0.0)
            # 3. c0 + source strength through the replay engine (schedule at Ctol = 0, stepQ)
            net = extract(pr)
            trace = read_hydfile(hyd)
            rep = Replicator(net, trace, ctol=0.0, record=True)
            times, _, _ = rep.run('stepQ')
            eng = TorchWQ(net, rep.schedule, rep.reactflag, device=args.device)
            N1 = net.nnodes + 1
            i0, i1, w = obs_indices(obs, times, qstep, N1)
            dev = torch.device(args.device)
            i0t, i1t = torch.as_tensor(i0, device=dev), torch.as_tensor(i1, device=dev)
            wt, y = torch.as_tensor(w, device=dev), torch.as_tensor(obs.y, device=dev)
            src_node = next(iter(net.sources))                     # node 1, CONCEN with pattern 3
            c0_nom = net.c0 * net.ucf['QUALITY']
            src_nom = float(net.sources[src_node][1])
            # theta = [c0[1..N], src] (index 0 of c0 is padding)
            n_c0 = net.nnodes
            x_nom = np.concatenate((c0_nom[1:], [src_nom]))
            bounds = [(0.0, 1.5)] * n_c0 + [(0.5, 1.5)]

            def sim_torch(th):                                     # th [B, n_c0 + 1]
                B = th.shape[0]
                c0 = torch.cat([torch.zeros(B, 1, dtype=th.dtype, device=dev), th[:, :n_c0]], dim=1)
                src = torch.zeros(B, N1, dtype=th.dtype, device=dev)
                src[:, src_node] = th[:, n_c0]
                q = eng.forward(c0=c0, src=src).reshape(B, -1)
                return q[:, i0t] * (1 - wt) + q[:, i1t] * wt

            def sse_grad(x):
                th = torch.as_tensor(x, device=dev).view(1, -1).requires_grad_(True)
                r = sim_torch(th)[0] - y
                loss = r @ r; loss.backward()
                return loss.item(), th.grad[0].cpu().numpy().copy()

            def sse_dll(x):
                for i in range(1, N1):
                    pr.set_node_value(i, api.EN_INITQUAL, float(x[i - 1]))
                pr.set_node_value(src_node, api.EN_SOURCEQUAL, float(x[n_c0]))
                r = dll_sim(pr, obs, qstep, 0.0) - obs.y
                return float(r @ r)

            chk = abs(sse_grad(x_nom)[0] - sse_dll(x_nom)) / sse_dll(x_nom)
            print(f'  c0/src problem: {n_c0 + 1} parameters; torch vs DLL loss rel diff {chk:.1e}', flush=True)
            out['c0_src'] = {'n_params': n_c0 + 1, 'loss_rel_diff': chk, 'nominal': stats(sim_torch(
                torch.as_tensor(x_nom, device=dev).view(1, -1)).detach()[0].cpu().numpy(), obs)}
            n_g = [0]
            def fg(x):
                n_g[0] += 1; return sse_grad(x)
            t0 = time.perf_counter()
            res = minimize(fg, x_nom, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            t_adj = time.perf_counter() - t0
            sim_fit = sim_torch(torch.as_tensor(res.x, device=dev).view(1, -1)).detach()[0].cpu().numpy()
            out['c0_src']['adjoint_lbfgsb'] = {'n_grad': n_g[0], 'wall_s': t_adj, 'nit': int(res.nit), 'message': str(res.message),
                                               'theta': res.x.tolist(), 'fit': stats(sim_fit, obs), 'sse_dll_check': sse_dll(res.x)}
            print(f'  ADJ-LBFGSB: rmse {out["c0_src"]["nominal"]["rmse"]:.4f} -> {out["c0_src"]["adjoint_lbfgsb"]["fit"]["rmse"]:.4f} '
                  f'in {n_g[0]} gradients, {t_adj:.1f}s', flush=True)
            n_f = [0]
            def f_dll(x):
                n_f[0] += 1; return sse_dll(x)
            t0 = time.perf_counter()
            res2 = minimize(f_dll, x_nom, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'eps': 1e-8})
            t_fd = time.perf_counter() - t0
            out['c0_src']['fd_lbfgsb_dll'] = {'n_eval': n_f[0], 'wall_s': t_fd, 'nit': int(res2.nit), 'message': str(res2.message),
                                              'theta': res2.x.tolist(), 'sse': float(res2.fun), 'rmse': float(np.sqrt(res2.fun / len(obs)))}
            print(f'  FD-LBFGSB (DLL): rmse {out["c0_src"]["fd_lbfgsb_dll"]["rmse"]:.4f} in {n_f[0]} DLL runs, {t_fd:.1f}s', flush=True)
            # identifiability of the affine map: residual Jacobian by one batched adjoint sweep
            M = len(obs)
            th = torch.as_tensor(res.x, device=dev).repeat(M, 1).requires_grad_(True)
            sim_torch(th).diagonal().sum().backward()
            J = th.grad.cpu().numpy()
            sv = np.linalg.svd(J, compute_uv=False)
            out['c0_src']['jacobian_sv'] = sv.tolist()
            out['c0_src']['rank_1e-6'] = int(np.sum(sv > 1e-6 * sv[0]))
            col = np.linalg.norm(J, axis=0)
            out['c0_src']['unobserved_params'] = [net.node_ids[i + 1] for i in range(n_c0) if col[i] == 0.0]
            print(f'  Jacobian {J.shape}: rank(1e-6) = {out["c0_src"]["rank_1e-6"]}, parameters with zero sensitivity: '
                  f'{len(out["c0_src"]["unobserved_params"])}', flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print('wrote', args.out)


if __name__ == '__main__':
    main()
