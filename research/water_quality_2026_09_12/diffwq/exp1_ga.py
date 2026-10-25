"""R2-11, part 3: genetic-algorithm station selection with the exact inner schedule fit, on the nominal booster problem.

Chromosome: a set of m candidate stations. Fitness: the objective of the schedule refitted on that set (exp1_booster
program with lam = 0 and the nominal mass weight), every generation's new sets solved together in one batched solve
of fixed size (padded with an evaluated set, so that one CUDA graph serves all generations), each child warm-started
from its parents' schedules of the stations it inherits. Tournament selection, elitism, uniform crossover repaired to m
stations (stations common to both parents kept, the others inherited with probability 1/2, then removed or added from the
parents' remaining stations), mutation by swapping one station for a random candidate. The best set is finally refitted at
the full iteration budget (warm-started) and checked in the library.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp1_ga --network Net3 --m 3 --seed 0 --out ga_Net3_m3_s0.json
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

from . import epanet_api as api
from .exp1_booster import BoosterProblem
from .exp1_robust import library_metrics


def run(args):
    lib = api.Library(args.lib)
    wd = Path(tempfile.mkdtemp(prefix='diffwq_ga_', dir=os.environ.get('TMPDIR')))
    P = BoosterProblem(args.network, lib, wd, args)
    try:
        dt, dev, J = P.dt, P.dev, P.J
        mu = torch.full((args.pop,), P.mu, dtype=dt, device=dev)
        zero = torch.zeros(args.pop, dtype=dt, device=dev)
        rng = np.random.default_rng(args.seed)
        cache, sol, n_solves, t0 = {}, {}, 0, time.perf_counter()

        def evaluate(sets, warm):
            nonlocal n_solves
            new = [s for s in dict.fromkeys(sets) if s not in cache]
            for k in range(0, len(new), args.pop):
                chunk = new[k:k + args.pop]
                batch = chunk + [chunk[0]] * (args.pop - len(chunk))
                masks = torch.zeros(args.pop, J, dtype=dt, device=dev)
                u0 = torch.zeros(args.pop, J, P.K, dtype=dt, device=dev)
                for b, s in enumerate(batch):
                    masks[b, list(s)] = 1.0
                    for j, uj in warm.get(s, {}).items():              # warm start from the parents' schedules
                        u0[b, j] = uj.to(dev)
                res = P.solve(zero, mu, mask=masks, u0=u0, n_iter=args.search_iters)
                n_solves += len(chunk)
                for b, s in enumerate(chunk):
                    cache[s], sol[s] = res['stats'][b]['obj'], res['u'][b].cpu()
            return np.array([cache[s] for s in sets])

        def tournament(pop, fit):
            idx = rng.choice(len(pop), args.tour, replace=False)
            return pop[idx[np.argmin(fit[idx])]]

        pop = [tuple(sorted(rng.choice(J, args.m, replace=False).tolist())) for _ in range(args.pop)]
        hist, warm = [], {}
        for gen in range(args.gens):
            fit = evaluate(pop, warm)
            order = np.argsort(fit)
            hist.append({'gen': gen, 'best_obj': float(fit[order[0]]), 'median_obj': float(np.median(fit)), 'evaluated': len(cache),
                         'wall_s': time.perf_counter() - t0})
            if gen % 5 == 0 or gen == args.gens - 1:
                print(f"  gen {gen:3d} best {fit[order[0]]:.6e} median {np.median(fit):.6e} sets evaluated {len(cache)} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
            nxt, warm = [pop[i] for i in order[:args.elite]], {}
            while len(nxt) < args.pop:
                a, b = tournament(pop, fit), tournament(pop, fit)
                common, diff = set(a) & set(b), sorted(set(a) ^ set(b))
                got = [j for j in diff if rng.random() < 0.5]                # uniform crossover: common stations kept,
                rest = [j for j in diff if j not in got]                     # the others inherited with probability 1/2,
                while len(common) + len(got) > args.m:                       # then repaired to m stations with the
                    got.pop(int(rng.integers(len(got))))                     # parents' remaining stations
                while len(common) + len(got) < args.m:
                    got.append(rest.pop(int(rng.integers(len(rest)))))
                child = common | set(got)
                if rng.random() < args.pmut:
                    out_ = rng.choice(sorted(child)); in_ = int(rng.integers(J))
                    while in_ in child:
                        in_ = int(rng.integers(J))
                    child = (child - {out_}) | {in_}
                child = tuple(sorted(child))
                if child not in cache and child not in warm:
                    warm[child] = {j: sol[a if j in a else b][j] for j in child if j in a or j in b}
                nxt.append(child)
            pop = nxt
        fit = evaluate(pop, warm)
        best = pop[int(np.argmin(fit))]
        t_search = time.perf_counter() - t0
        P.free_graphs()
        mask = torch.zeros(1, J, dtype=dt, device=dev); mask[0, list(best)] = 1.0
        fin = P.solve(torch.zeros(1, dtype=dt, device=dev), mu[:1], mask=mask, u0=sol[best].view(1, J, P.K).to(dev),
                      label=f'GA best m={args.m} full refit')
        st = fin['stats'][0]
        lib_m = library_metrics(P, fin['u'][0].cpu().numpy(), '_ga')
        R = {'network': args.network, 'm': args.m, 'seed': args.seed, 'pop': args.pop, 'gens': args.gens, 'iters_search': args.search_iters,
             'elite': args.elite, 'tournament': args.tour, 'p_mutation': args.pmut, 'warm_start': 'parents', 'mu': P.mu,
             'best_set': [P.net.node_ids[P.cand[j]] for j in best], 'search_obj': float(min(fit)),
             'final': {kk: st[kk] for kk in ('obj', 'coverage', 'mean_short_mgL', 'min_c_mgL', 'mass_kg_d', 'n_stations', 'iters', 'converged')},
             'final_u': fin['u'][0].cpu().numpy().round(6).tolist(), 'library': lib_m,
             'sets_evaluated': len(cache), 'search_wall_s': t_search, 'history': hist,
             'record_s': P.t_record, 'build_s': P.t_build}
        Path(args.out).write_text(json.dumps(R, default=float), encoding='utf-8')
        print(f"[{args.network}] GA m={args.m} seed={args.seed}: obj {st['obj']:.6e} coverage {st['coverage']:.4f} "
              f"mass {st['mass_kg_d']:.3f} kg/d stations {R['best_set']} ({len(cache)} sets, {t_search:.0f}s)", flush=True)
    finally:
        P.free_graphs()
        P.pr.__exit__()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', required=True)
    ap.add_argument('--m', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--gens', type=int, default=60)
    ap.add_argument('--search-iters', type=int, default=1000, help='inner FISTA iterations during the search')
    ap.add_argument('--elite', type=int, default=4)
    ap.add_argument('--tour', type=int, default=3)
    ap.add_argument('--pmut', type=float, default=0.3)
    ap.add_argument('--theta-json', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lib', default=None)
    a = ap.parse_args()
    # settings of the nominal study (exp1_booster defaults); iters is the full budget of the final refit
    for k, v in dict(lo=0.2, margin=0.02, hi=4.0, mu_rel=0.1, iters=5000, tol=1e-7, chunk=256, no_graph=False, no_tanks=False).items():
        setattr(a, k, v)
    run(a)


if __name__ == '__main__':
    main()
