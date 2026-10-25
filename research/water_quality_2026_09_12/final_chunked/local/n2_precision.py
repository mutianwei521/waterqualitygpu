"""Which N2 is accurate? Box energies N2 = ||A_j 1_box||^2 of a few junctions on L-Town (48 h, v2 sensors, first detected
event, window 'inf'): (a) exp2_long.sweep_box (sum over rows of squared box sums), (b) exp2_source.box_glrt's differences
of 2-D prefix sums of G = A^T A, (c) the box block of G summed directly. Run from research/water_quality_2026_09_12."""
import argparse, json, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, '.')
from diffwq import epanet_api as api
from diffwq import exp2_source as E
from diffwq.exp2_long import SourceProblemLong

J = json.load(open('chunk_dev/exp2_v2_lt10.json'))
args = argparse.Namespace(**J['args'])
args.device, args.no_graph, args.exhaustive, args.chunk, args.n_chunks, args.chunk_terms, args.max_events = 'cuda', True, 0, 16, None, int(1.5e7), 4
args.n_out_ref = int(J['sizes']['snapshots'])
lib = api.Library(r'C:\wq_local\lib\epanet2.dll')
with tempfile.TemporaryDirectory() as wd:
    P = SourceProblemLong(J['network'], lib, wd, args)
    P.v2_sensors = [P.net.node_ids.index(s) for s in J['placement']['sensors']]
    P.set_sensors(P.v2_sensors)
    rng = np.random.default_rng(args.seed)
    P.place_sensors(args.n_train, rng, args.place_batch)
    scen, c, d, t_det, info = P.make_data(P.sample_scenarios(args.n_test, rng), rng, args.place_batch)
    j_true = scen[0][0]
    nodes = [j_true] + [j for j in P.junc if j != j_true][:7]
    n_rows = P.n_out * len(P.sens)
    taus = list(range(1, 1 + int(4 * 3600 // P.slot)))
    K = P.K
    N2a = [x for x in P.sweep_box(nodes, d[:1], [n_rows], taus)][-1][1].clone()          # [taus, nodes, K]
    G = [x for x in P.sweep_normal(nodes, d[:1], [n_rows])][-1][1].clone()                  # [nodes, K, K]
    C = torch.zeros(len(nodes), K + 1, K + 1, dtype=G.dtype, device=G.device)
    C[:, 1:, 1:] = G.cumsum(1).cumsum(2)
    diag = lambda X: torch.diagonal(X, dim1=1, dim2=2)
    out = []
    for ti, t in enumerate(taus[::5]):
        ti = taus.index(t)
        n = K + 1 - t
        b_ = diag(C[:, t:, t:]) - diag(C[:, :n, t:]) - diag(C[:, t:, :n]) + diag(C[:, :n, :n])   # (b)
        c_ = torch.stack([torch.stack([G[q, k0:k0 + t, k0:k0 + t].sum() for k0 in range(n)]) for q in range(len(nodes))])  # (c)
        a_ = N2a[ti, :, :n]
        m = c_.abs() > 1e-12 * c_.abs().max()
        rel = lambda x: float(((x - c_).abs()[m] / c_.abs()[m]).max())
        out.append({'tau_slots': t, 'sweep_box_vs_direct': rel(a_), 'prefix_vs_direct': rel(b_)})
        print(out[-1], flush=True)
    json.dump(out, open('chunk_dev/n2_precision.json', 'w'), indent=1)
    P.pr.__exit__()
