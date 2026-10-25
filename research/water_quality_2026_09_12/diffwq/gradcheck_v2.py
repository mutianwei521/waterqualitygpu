"""R1-3 / R2-3: independent validation of the replay gradient against central differences of the compiled EPANET library.
New module; no v1 file is modified. Reference value of every derivative: [L(x+hd) - L(x-hd)] / 2h from two library runs
(parameters written through the API, quality re-run on hydraulics solved once per worker), compared with g.d from the
fused engine's adjoint. Groups: A engine (G2 configurations x 3 parameter points), B field calibration maps (grouped and
per-pipe), C per-pipe wall maps (Exp3 setting), D slot-source maps (Exp1 / Exp2 setting), X reaction-factor boundary.
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.gradcheck_v2 --job A --lib LIB --out final_gradcheck --workers 10
"""
import argparse, json, os, tempfile, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from . import epanet_api as api
from .verify_m0 import CORE, ROOT, all_links, all_tanks_kb, init_qual, qual_type
from .verify_torch import CASES

H5 = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
KEYS = ('kb', 'kw', 'tank_kb', 'c0', 'src')
SETL = {'kb': api.EN_KBULK, 'kw': api.EN_KWALL}
CASE = {c[0]: c for c in CASES}


def scale_quality(f):
    def g(pr):
        for i in range(1, pr.count(api.EN_NODECOUNT) + 1):
            pr.set_node_value(i, api.EN_INITQUAL, f * pr.node_value(i, api.EN_INITQUAL))
            s = pr.node_value(i, api.EN_SOURCEQUAL)
            if s:
                pr.set_node_value(i, api.EN_SOURCEQUAL, f * s)
    return g


# parameter points of group A: (kb, kw, tank kb) set on all pipes / tanks, quality scaled; P0 = configuration as given
POINTS = {'P0': None, 'P1': (-0.1, -0.05, -0.1, 0.5), 'P2': (-2.0, -1.0, -2.0, 2.0)}


# ------------------------------------------------------------------ projects (built identically in main and workers)
def open_project(lib, spec, wd):
    k = spec['kind']
    rpt = str(Path(wd) / f"{spec['key']}_{os.getpid()}.rpt")
    if k == 'A':
        name, base, setup, over = CASE[spec['case']]
        pr = api.Project(lib, ROOT / CORE[base], rpt)
        for f in setup:
            f(pr)
        if over.get('qstep'):
            pr.set_time_param(api.EN_QUALSTEP, over['qstep'])
        pt = POINTS[spec['point']]
        if pt is not None:
            all_links(kb=pt[0], kw=pt[1])(pr); all_tanks_kb(pt[2])(pr); scale_quality(pt[3])(pr)
        if spec.get('kb_all') is not None:
            all_links(kb=spec['kb_all'], kw=0.0)(pr); all_tanks_kb(spec['kb_all'])(pr)
    elif k == 'B':
        pr = api.Project(lib, spec['inp'], rpt)
    elif k in ('C', 'D2'):
        from .exp3_pipe_kw import CONFIGS
        cfg = CONFIGS[spec['net']]
        pr = api.Project(lib, cfg['inp'], rpt)
        qual_type(api.EN_CHEM)(pr)
        if k == 'C':
            init_qual(0.5, reservoirs=1.0)(pr); all_links(kb=-0.5, kw=0.0)(pr); all_tanks_kb(-0.5)(pr)
        else:
            init_qual(0.0, reservoirs=0.0)(pr); all_links(kb=0.0, kw=0.0)(pr); all_tanks_kb(0.0)(pr)
        pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
        if cfg['unbalanced'] is not None:
            pr.set_option(api.EN_UNBALANCED, cfg['unbalanced'])
        if k == 'D2' and spec['slot_s'] < pr.time_param(api.EN_PATTERNSTEP):
            pr.resample_patterns(spec['slot_s'])
    elif k == 'D1':
        from .exp1_booster import CONFIGS
        cfg = CONFIGS['Net3']
        inp = cfg['inp_edit'](cfg['inp'], wd, f"net3_{os.getpid()}")
        pr = api.Project(lib, inp, rpt)
        for f in cfg['setup']:
            f(pr)
        pr.set_time_param(api.EN_DURATION, cfg['duration_h'] * 3600)
    pr.set_option(api.EN_TOLERANCE, 0.0)
    return pr


def loss_of(q, spec, cache):
    """Loss of one library run, summed in extended precision (np.longdouble, 64-bit mantissa on x86 Linux): a central
    difference subtracts two nearly equal losses, and rounding of a float64 sum (a few ulp of the loss) otherwise sets
    its floor for directions whose derivative is small relative to the loss (single pipes of large networks)."""
    L, ld = spec['loss'], np.longdouble
    if L[0] == 'wsum':
        if 'w' not in cache or cache['w'].shape != q.shape:
            cache['w'] = np.random.default_rng(L[1]).uniform(0.5, 1.5, size=q.shape)
        return (cache['w'].astype(ld) * q.astype(ld)).sum()
    i0, i1, w, y = (np.asarray(a) for a in L[1:])
    f = q.reshape(-1).astype(ld)
    r = f[i0] * (1 - w.astype(ld)) + f[i1] * w.astype(ld) - y.astype(ld)
    return (r * r).sum()


# ------------------------------------------------------------------ library workers
_W = {}


def _w_init(lib_path, wd):
    _W['lib'], _W['wd'], _W['pr'] = api.Library(lib_path), wd, {}


def _w_eval(task):
    spec, assign = task
    t0 = time.perf_counter()
    if spec['kind'] in ('D1', 'D2'):                       # sources installed per evaluation (patterns cannot be reset)
        pr = open_project(_W['lib'], spec, _W['wd'])
        try:
            for j, pat in assign:
                s = float(max(pat.max(), 1e-300))
                pr.set_source(int(j), api.EN_MASS, s, pr.add_pattern(f'v{j}', (pat / s).tolist()))
            pr.solve_hydraulics(Path(_W['wd']) / f'd_{os.getpid()}.hyd')
            _, q, _ = pr.run_quality_reference(spec['mode'])
        finally:
            pr.__exit__()
        cache = _W.setdefault(('c', spec['key']), {})
        return loss_of(q, spec, cache), time.perf_counter() - t0
    key = spec['key']
    if key not in _W['pr']:
        pr = open_project(_W['lib'], spec, _W['wd'])
        pr.solve_hydraulics(Path(_W['wd']) / f'{key}_{os.getpid()}.hyd')
        _W['pr'][key] = (pr, {})
    pr, cache = _W['pr'][key]
    for k, ent, val in assign:
        for e, v in zip(ent, val):
            if k in SETL:
                pr.set_link_value(int(e), SETL[k], float(v))
            elif k == 'tank_kb':
                pr.set_node_value(int(e), api.EN_TANK_KBULK, float(v))
            elif k == 'c0':
                pr.set_node_value(int(e), api.EN_INITQUAL, float(v))
            else:
                pr.set_node_value(int(e), api.EN_SOURCEQUAL, float(v))
    _, q, _ = pr.run_quality_reference(spec['mode'])
    return loss_of(q, spec, cache), time.perf_counter() - t0


# ------------------------------------------------------------------ replay side
class Replay:
    """Recorded map of one spec; x (flat parameter vector) -> loss, gradient; x <-> library assignments."""

    def __init__(self, lib, spec, wd, blocks, dev):
        import torch
        from .hydfile import read_hydfile
        from .network import extract, SECperDAY
        from .replicate import Replicator
        from .torch_wq import TorchWQ
        self.torch, self.spec, self.dev, T = torch, spec, dev, {}
        pr = open_project(lib, spec, wd)
        t0 = time.perf_counter(); pr.solve_hydraulics(Path(wd) / f"{spec['key']}_main.hyd"); T['hydraulics_s'] = time.perf_counter() - t0
        self.net = net = extract(pr)
        trace = read_hydfile(Path(wd) / f"{spec['key']}_main.hyd")
        t0 = time.perf_counter(); _, self.q_lib, _ = pr.run_quality_reference(spec['mode']); T['library_run_s'] = time.perf_counter() - t0
        d_kind = spec['kind'] in ('D1', 'D2')
        if d_kind:
            K = spec['K']
            vs = {j: api.EN_MASS for j in spec['cands']}
            rep = Replicator(net, trace, ctol=0.0, record=True, src_slots=K, virtual_sources=vs)
        else:
            rep = Replicator(net, trace, ctol=0.0, record=True)
        t0 = time.perf_counter(); rep.run(spec['mode']); T['record_s'] = time.perf_counter() - t0
        self.clamp_hits = int(rep.hits.get('react_clamp0', 0)); self.reactflag = bool(rep.reactflag)
        t0 = time.perf_counter()
        self.eng = TorchWQ(net, rep.schedule, rep.reactflag, device=dev, engine='fused')
        T['build_s'] = time.perf_counter() - t0
        pr.__exit__()
        f64 = torch.float64
        self.base = {'kb': net.kb * SECperDAY, 'kw': net.kw * SECperDAY, 'tank_kb': net.tank_kb * SECperDAY,
                     'c0': net.c0 * net.ucf['QUALITY'],
                     'src': np.array([net.sources.get(i, (0, 0.0, 0))[1] for i in range(net.nnodes + 1)])}
        self.tank_node = net.tank_node
        self._blocks = blocks                        # list of (key, entity indices [n_b], column of x [n_b])
        self.T = T
        self.f64 = f64

    @property
    def blocks(self):
        return self._blocks

    @blocks.setter
    def blocks(self, b):
        self._blocks = None if b is None else [t for t in b if len(t[1])]

    def x0(self):
        n = 1 + max(int(c.max()) for _, _, c in self.blocks)
        x = np.zeros(n)
        for k, ent, col in self.blocks:
            x[col] = self.base[k][ent]
        return x

    def full(self, x):
        torch = self.torch
        out = {}
        for k in KEYS:
            out[k] = torch.tensor(self.base[k], dtype=self.f64, device=self.dev).view(1, -1).clone()
        for k, ent, col in self.blocks:
            out[k][0, torch.as_tensor(ent, device=self.dev)] = x[torch.as_tensor(col, device=self.dev)]
        return out

    def forward_q(self, x):
        torch = self.torch
        if self.spec['kind'] in ('D1', 'D2'):
            return self.eng.forward(src=self.src_of(x))[0]
        return self.eng.forward(**self.full(x))[0]

    def src_of(self, x):
        torch, K, N1 = self.torch, self.spec['K'], self.net.nnodes + 1
        u = torch.tensor(self.spec['S0'], dtype=self.f64, device=self.dev).clone() if self.spec.get('S0') is not None \
            else torch.zeros(K, N1, dtype=self.f64, device=self.dev)
        sup = torch.as_tensor(self.spec['support'], device=self.dev)
        u = u.index_put((torch.arange(K, device=self.dev).repeat_interleave(len(sup)), sup.repeat(K)),
                        x.view(len(sup), K).t().reshape(-1), accumulate=True)
        return u.view(1, K, N1)

    def loss_grad(self, x_np):
        torch = self.torch
        x = torch.tensor(x_np, dtype=self.f64, device=self.dev).requires_grad_(True)
        q = self.forward_q(x)
        L = self.spec['loss']
        if L[0] == 'wsum':
            if not hasattr(self, '_w'):
                self._w = torch.tensor(np.random.default_rng(L[1]).uniform(0.5, 1.5, size=tuple(q.shape)), dtype=self.f64, device=self.dev)
            loss = (self._w * q).sum()
        else:
            i0, i1, w, y = (torch.as_tensor(np.asarray(a), device=self.dev) for a in L[1:])
            f = q.reshape(-1)
            r = f[i0] * (1.0 - w) + f[i1] * w - y
            loss = (r * r).sum()
        loss.backward()
        return float(loss.detach()), x.grad.cpu().numpy(), q.detach().cpu().numpy()

    def min_factor(self, x_np):
        if self.spec['kind'] in ('D1', 'D2') or not self.eng.chem:
            return None
        torch = self.torch
        with torch.no_grad():
            th = self.full(torch.tensor(x_np, dtype=self.f64, device=self.dev))
            lH, _ = self.eng._log_factors(th['kb'], th['kw'], th['tank_kb'])
            S1 = self.eng.S + 1
            lH = lH.view(1, S1, -1)
            return float(torch.exp(lH[:, 1:] - lH[:, :-1]).min())

    def assign(self, x):
        if self.spec['kind'] in ('D1', 'D2'):
            K, sup = self.spec['K'], self.spec['support']
            u = x.reshape(len(sup), K)
            S0 = self.spec.get('S0')
            return [(j, u[i] + (S0[:, j] if S0 is not None else 0.0)) for i, j in enumerate(sup) if u[i].max() > 0]
        out = []
        for k, ent, col in self.blocks:
            e = self.tank_node[ent] if k == 'tank_kb' else ent
            out.append((k, np.asarray(e), x[col]))
        return out


# ------------------------------------------------------------------ directions
def directions(x0, g, blocks, rng, n_dense, n_single, nonneg, zero_scale, kinds):
    """Scale-aware random dense directions and stratified single-coordinate directions (feasible: reaction coefficients stay
    <= 0 on the + side for zero-valued coefficients, nonnegative parameters at 0 are not moved)."""
    s = np.abs(x0).copy()
    zero_react = (s == 0) & ~nonneg
    s[zero_react] = zero_scale[zero_react]
    s[(x0 == 0) & nonneg] = 0.0
    out = []
    for r in range(n_dense):
        z = rng.standard_normal(x0.size)
        z[zero_react] = -np.abs(z[zero_react])
        if np.any(s * z):
            out.append(('dense', None, s * z))
    gi = np.abs(g) * s
    for kname in sorted(set(kinds)):
        idx = np.flatnonzero((np.asarray(kinds) == kname) & (s > 0) & (gi > 1e-12 * max(gi.max(), 1e-300)))
        if idx.size == 0:
            continue
        o = idx[np.argsort(-gi[idx])]
        q = [o[:max(1, o.size // 10)], o[o.size // 10: o.size // 2], o[o.size // 2:]]
        take = [min(max(1, round(0.4 * n_single)), q[0].size), min(round(0.3 * n_single), q[1].size), min(round(0.3 * n_single), q[2].size)]
        pick = list(rng.choice(q[0], take[0], replace=False)) + (list(rng.choice(q[1], take[1], replace=False)) if take[1] else []) \
            + (list(rng.choice(q[2], take[2], replace=False)) if take[2] else [])
        for i in list(dict.fromkeys(int(t) for t in pick))[:n_single]:
            d = np.zeros(x0.size); d[i] = s[i] * (-1.0 if zero_react[i] else 1.0)
            out.append(('single', f'{kname}#{int(i)}', d))
    return out


def evaluate(pool, R, spec, x0, dirs, H, label, cls_fn):
    L0, g, q_rep = R.loss_grad(x0)
    rows, tasks, meta = [], [(spec, R.assign(x0))], []
    for di, (dk, tag, d) in enumerate(dirs):
        for h in H:
            tasks.append((spec, R.assign(x0 + h * d))); tasks.append((spec, R.assign(x0 - h * d))); meta.append((di, h))
    t0 = time.perf_counter()
    res = pool.map(_w_eval, tasks, chunksize=max(1, len(tasks) // (8 * pool._processes)))
    t_lib = time.perf_counter() - t0
    Lb = res[0][0]
    for di, (dk, tag, d) in enumerate(dirs):
        ad = float(g @ d)
        nrm = float(np.linalg.norm(g) * np.linalg.norm(d))
        st = []
        for m, (dj, h) in enumerate(meta):
            if dj != di:
                continue
            lp, lm = res[1 + 2 * m][0], res[2 + 2 * m][0]
            fd, fw, bw = (float((lp - lm) / (2 * h)), float((lp - Lb) / h), float((Lb - lm) / h))   # differences in long double
            den = max(abs(ad), abs(fd)) if max(abs(ad), abs(fd)) >= 1e-10 * nrm else max(nrm, 1e-300)
            st.append({'h': h, 'fd': fd, 'rel_err': abs(fd - ad) / den, 'fwd_err': abs(fw - ad) / max(den, 1e-300),
                       'bwd_err': abs(bw - ad) / max(den, 1e-300)})
        e = {s['h']: s['rel_err'] for s in st}
        slope = (np.log10(max(e[1e-2], 1e-300) / max(e[1e-4], 1e-300)) / 2.0) if (1e-2 in e and 1e-4 in e) else None
        slope1 = np.log10(max(e[1e-2], 1e-300) / max(e[1e-3], 1e-300)) if (1e-2 in e and 1e-3 in e) else None
        cls = cls_fn(dk, tag, d)
        mn = min(e.values())
        if cls == 'affine':
            ok = max(e.values()) <= 1e-9
        elif cls == 'smooth':
            ok = mn <= 1e-6 and ((slope1 is not None and 1.7 <= slope1 <= 2.3) or e.get(1e-2, 1.0) <= 1e-9)
            ok_pre = mn <= 1e-6 and ((slope is not None and 1.7 <= slope <= 2.3) or e.get(1e-2, 1.0) <= 1e-9)
        else:
            ok = None
        rows.append({'kind': dk, 'tag': tag, 'class': cls, 'n_nonzero': int((d != 0).sum()), 'adjoint': ad, 'steps': st,
                     'min_err': mn, 'slope_1e-2_1e-4': slope, 'slope_1e-2_1e-3': slope1, 'pass': ok,
                     'pass_preregistered': (ok_pre if cls == 'smooth' else ok)})
    lib_t = [r[1] for r in res]
    same = R.blocks is not None and spec['kind'] not in ('D1', 'D2') and np.array_equal(x0, R.x0())
    rel_fwd = float(np.abs(q_rep - R.q_lib).max() / max(np.abs(R.q_lib).max(), 1e-300)) if same else None
    return {'label': label, 'n_params': int(x0.size), 'loss_replay': L0, 'loss_library': float(Lb), 'loss_rel_diff': float(abs(L0 - Lb) / max(abs(Lb), 1e-300)),
            'max_rel_replay_vs_library_base': rel_fwd, 'min_reaction_factor': R.min_factor(x0), 'replicator_clamp_hits': R.clamp_hits,
            'reactflag': R.reactflag, 'n_library_runs': len(tasks), 'library_wall_s': t_lib, 'library_run_median_s': float(np.median(lib_t)),
            'timing': R.T, 'directions': rows}


def summarize(maps):
    out = {}
    for m in maps:
        for r in m['directions']:
            c = out.setdefault(r['class'], {'n': 0, 'pass': 0, 'min_errs': []})
            c['n'] += 1; c['pass'] += bool(r['pass']); c['min_errs'].append(r['min_err'])
            c['pass_preregistered'] = c.get('pass_preregistered', 0) + bool(r.get('pass_preregistered'))
    for c in out.values():
        e = np.array(c.pop('min_errs')); c.update(median=float(np.median(e)), p95=float(np.quantile(e, 0.95)), max=float(e.max()))
    return out


# ------------------------------------------------------------------ jobs
def blocks_full(net, kinds_out):
    """Group A: every pipe kb / kw, every real tank kb, every node c0, every source strength."""
    pipes = np.array([k for k in range(1, net.nlinks + 1) if net.link_type[k] == api.EN_PIPE], dtype=int)
    tanks = np.array([i for i in range(1, net.ntanks + 1) if not net.tank_is_reservoir[i]], dtype=int)
    nodes = np.arange(1, net.nnodes + 1)
    srcs = np.array(sorted(net.sources), dtype=int)
    blocks, c = [], 0
    for k, ent in (('kb', pipes), ('kw', pipes), ('tank_kb', tanks), ('c0', nodes), ('src', srcs)):
        if ent.size:
            blocks.append((k, ent, np.arange(c, c + ent.size))); kinds_out += [k] * ent.size; c += ent.size
    return blocks


def job_A(pool, a, dev, wd, save):
    maps = []
    for name in [c[0] for c in CASES]:
        for pt in ('P0', 'P1', 'P2'):
            spec = {'kind': 'A', 'key': f'A_{name}_{pt}', 'case': name, 'point': pt, 'mode': 'nextQ', 'loss': ('wsum', 0)}
            lib = api.Library(a.lib)
            R = Replay(lib, spec, wd, None, dev)
            if R.net.qualflag == api.EN_TRACE or (pt != 'P0' and R.net.qualflag != api.EN_CHEM):
                continue
            kinds = []
            R.blocks = blocks_full(R.net, kinds)
            if R.net.qualflag == api.EN_AGE:                       # AGE: initial age only
                R.blocks = [b for b in R.blocks if b[0] == 'c0']; R.blocks = [(k, e, np.arange(e.size)) for k, e, _ in R.blocks]
                kinds = ['c0'] * R.blocks[0][1].size
            x0 = R.x0()
            nonneg = np.isin(np.array(kinds), ['c0', 'src'])
            zs = np.where(np.array(kinds) == 'kw', 0.1, 0.5)
            _, g, _ = R.loss_grad(x0)
            dirs = directions(x0, g, R.blocks, np.random.default_rng([a.seed, len(maps)]), 20, 10, nonneg, zs, kinds)
            setpoint = any(t == api.EN_SETPOINT for t, _, _ in R.net.sources.values())
            K_ = np.array(kinds)

            def cls_fn(dk, tag, d, R=R, K_=K_, x0=x0, setpoint=setpoint):
                touched = set(K_[d != 0])
                if not R.reactflag and touched & {'kb', 'kw', 'tank_kb'}:
                    return 'switch_all_zero_reaction'
                if 'kw' in touched and np.any(x0[(d != 0) & (K_ == 'kw')] == 0):
                    return 'kink_kw0'
                if setpoint and 'src' in touched:
                    return 'setpoint'
                return 'smooth'
            maps.append(evaluate(pool, R, spec, x0, dirs, H5, spec['key'], cls_fn))
            print(f"[A] {spec['key']}: {sum(bool(r['pass']) for r in maps[-1]['directions'])}/"
                  f"{sum(r['pass'] is not None for r in maps[-1]['directions'])} pass, fwd {maps[-1]['max_rel_replay_vs_library_base']:.1e}", flush=True)
            save(maps)
    # X: reaction-factor boundary (Net1, kw = 0, all kb equal): replay vs library and kb direction derivative
    for kb in (-10.0, -100.0, -250.0, -280.0, -300.0):
        spec = {'kind': 'A', 'key': f'X_Net1_kb{int(-kb)}', 'case': 'Net1', 'point': 'P0', 'kb_all': kb, 'mode': 'nextQ', 'loss': ('wsum', 0)}
        lib = api.Library(a.lib)
        R = Replay(lib, spec, wd, None, dev)
        kinds = []
        R.blocks = [b for b in blocks_full(R.net, kinds) if b[0] == 'kb']
        x0 = R.x0()
        rec = {'label': spec['key'], 'kb': kb, 'replicator_clamp_hits': R.clamp_hits}
        try:
            mf = R.min_factor(x0); rec['min_reaction_factor'] = mf
            d = np.abs(x0)
            r = evaluate(pool, R, spec, x0, [('dense', 'all_kb', d)], [1e-3, 1e-5], spec['key'], lambda *z: 'boundary')
            rec.update(loss_rel_diff=r['loss_rel_diff'], max_rel_replay_vs_library_base=r['max_rel_replay_vs_library_base'],
                       steps=r['directions'][0]['steps'])
        except Exception as e:
            rec['replay_error'] = f'{type(e).__name__}: {e}'[:200]
        maps.append({'label': spec['key'], 'boundary': rec, 'directions': []})
        print(f"[X] {rec}", flush=True)
        save(maps)
    return maps


def job_B(pool, a, dev, wd, save):
    from .calib import CalibProblem
    maps = []
    for ds in ('Fairfield', 'Bellingham', 'NorthPenn'):
        lib = api.Library(a.lib)
        P = CalibProblem(ds, lib, wd, device='cpu', verbose=False)
        loss = ('sse', P._i0, P._i1, P._w, P.obs.y)
        pipes = np.flatnonzero(P.is_pipe)
        tanks = np.array([i for i in range(1, P.net.ntanks + 1) if not P.net.tank_is_reservoir[i]], dtype=int)
        inp = str(P.inp)
        for param in ('grouped', 'perpipe'):
            for pt, (kbv, kwv) in {'P1': (-0.5, -0.2), 'P2': (-1.5, -1.0), 'Pcal': (None, None)}.items():
                spec = {'kind': 'B', 'key': f'B_{ds}_{param}_{pt}', 'inp': inp, 'mode': 'stepQ', 'loss': loss}
                R = Replay(lib, spec, wd, None, dev)
                if param == 'grouped':
                    G = P.G
                    R.blocks = [('kb', pipes, np.zeros(pipes.size, dtype=int)), ('tank_kb', tanks, np.zeros(tanks.size, dtype=int))] + \
                               [('kw', pipes[P.link_group[pipes] == gg], np.full(int((P.link_group[pipes] == gg).sum()), 1 + gg)) for gg in range(G)]
                    kinds = ['kb'] + ['kw'] * G
                    cal = {'Fairfield': [0.0, -5.0, -0.156, -0.573, 0.0], 'Bellingham': [0.0, -0.176, 0.0, -5.0, -5.0],
                           'NorthPenn': [-0.089, -5.0, -2.343, 0.0, 0.0]}[ds]
                    x0 = np.array(cal[:1 + G]) if pt == 'Pcal' else np.array([kbv] + [kwv] * G)
                else:
                    R.blocks = [('kw', pipes, np.arange(pipes.size))]
                    kinds = ['kw'] * pipes.size
                    if pt == 'Pcal':
                        x0 = np.array({'Fairfield': [-5.0, -0.156, -0.573, 0.0], 'Bellingham': [-0.176, 0.0, -5.0, -5.0],
                                       'NorthPenn': [-5.0, -2.343, 0.0, 0.0]}[ds])[P.link_group[pipes]]
                        R.base['kb'][pipes] = {'Fairfield': 0.0, 'Bellingham': 0.0, 'NorthPenn': -0.089}[ds]
                    else:
                        x0 = np.full(pipes.size, kwv); R.base['kb'][pipes] = kbv
                    R.base['tank_kb'][tanks] = R.base['kb'][pipes[0]]
                    fixed = [('kb', pipes, R.base['kb'][pipes]), ('tank_kb', R.tank_node[tanks], R.base['tank_kb'][tanks])]
                    orig = R.assign
                    R.assign = (lambda x, orig=orig, fixed=fixed: fixed + orig(x))
                x0 = x0.astype(float)
                _, g, _ = R.loss_grad(x0)
                K_ = np.array(kinds)
                dirs = directions(x0, g, R.blocks, np.random.default_rng([a.seed, 100 + len(maps)]), 20, 10 if param == 'perpipe' else 0,
                                  np.zeros(x0.size, bool), np.where(K_ == 'kw', 0.1, 0.5), kinds)
                if param == 'grouped':
                    for i in range(x0.size):
                        d = np.zeros(x0.size); d[i] = abs(x0[i]) if x0[i] != 0 else -0.1
                        dirs.append(('single', f'{kinds[i]}#{i}', d))

                def cls_fn(dk, tag, d, x0=x0, K_=K_):
                    on = d != 0
                    if np.any(x0[on] == 0) or np.any(x0[on] <= -5.0 + 1e-12):
                        return 'kink_kw0' if np.any(x0[on & (K_ == 'kw')] == 0) else 'bound'
                    return 'smooth'
                maps.append(evaluate(pool, R, spec, x0, dirs, H5, spec['key'], cls_fn))
                print(f"[B] {spec['key']}: {sum(bool(r['pass']) for r in maps[-1]['directions'])}/"
                      f"{sum(r['pass'] is not None for r in maps[-1]['directions'])} pass, loss {maps[-1]['loss_rel_diff']:.1e}", flush=True)
                save(maps)
    return maps


def job_C(pool, a, dev, wd, save, net_name, H, n_dense, n_single, maps=None):
    maps = [] if maps is None else maps
    spec = {'kind': 'C', 'key': f'C_{net_name}', 'net': net_name, 'mode': 'stepQ', 'loss': ('wsum', 0)}
    lib = api.Library(a.lib)
    R = Replay(lib, spec, wd, None, dev)
    pipes = np.array([k for k in range(1, R.net.nlinks + 1) if R.net.link_type[k] == api.EN_PIPE])
    R.blocks = [('kw', pipes, np.arange(pipes.size))]
    med = 0.08 if net_name == 'LTown' else 0.4
    x0 = -np.exp(np.log(med) + 0.5 * np.random.default_rng([a.seed, 7]).standard_normal(pipes.size))
    _, g, _ = R.loss_grad(x0)
    dirs = directions(x0, g, R.blocks, np.random.default_rng([a.seed, 8]), n_dense, n_single, np.zeros(x0.size, bool),
                      np.full(x0.size, 0.1), ['kw'] * x0.size)
    maps.append(evaluate(pool, R, spec, x0, dirs, H, spec['key'], lambda *z: 'smooth'))
    print(f"[C] {spec['key']}: {sum(bool(r['pass']) for r in maps[-1]['directions'])}/{len(maps[-1]['directions'])} pass, "
          f"loss {maps[-1]['loss_rel_diff']:.1e}, lib {maps[-1]['library_run_median_s']:.2f}s", flush=True)
    save(maps)
    return maps


def job_D(pool, a, dev, wd, save, which, n_sup, n_dense, n_single, maps=None):
    maps = [] if maps is None else maps
    lib = api.Library(a.lib)
    if which == 'Net3':
        from .exp1_booster import CONFIGS
        spec = {'kind': 'D1', 'key': 'D_Net3', 'mode': 'nextQ', 'loss': ('wsum', 0), 'K': CONFIGS['Net3']['slots']}
    else:
        from .exp2_source import CONFIGS
        spec = {'kind': 'D2', 'key': f'D_{which}', 'net': which, 'mode': 'nextQ', 'loss': ('wsum', 0), 'slot_s': CONFIGS[which]['slot_s']}
    pr = open_project(lib, dict(spec, K=1), wd)
    nn = pr.count(api.EN_NODECOUNT)
    junc = [i for i in range(1, nn + 1) if pr.node_type(i) == api.EN_JUNCTION and pr.node_value(i, api.EN_SOURCEQUAL) in (None, 0, 0.0)]
    if spec['kind'] == 'D2':
        spec['K'] = int(pr.time_param(api.EN_DURATION) // pr.time_param(api.EN_PATTERNSTEP))
    pr.__exit__()
    rng = np.random.default_rng([a.seed, 9])
    spec['cands'] = junc
    spec['support'] = sorted(int(j) for j in rng.choice(junc, min(n_sup, len(junc)), replace=False))
    R = Replay(lib, spec, wd, None, dev)
    K, S = spec['K'], len(spec['support'])
    x0 = rng.uniform(2e4, 2e5, size=S * K)                     # mg/min in every slot of every support node
    _, g, _ = R.loss_grad(x0)
    dirs = [('dense', None, x0 * rng.uniform(-1, 1, x0.size)) for _ in range(n_dense)]
    for i in rng.choice(x0.size, n_single, replace=False):
        d = np.zeros(x0.size); d[i] = x0[i]; dirs.append(('single', f'src_slot#{int(i)}', d))
    maps.append(evaluate(pool, R, spec, x0, dirs, [1e-2, 0.5], spec['key'], lambda *z: 'affine'))
    print(f"[D] {spec['key']}: {sum(bool(r['pass']) for r in maps[-1]['directions'])}/{len(maps[-1]['directions'])} pass, "
          f"loss {maps[-1]['loss_rel_diff']:.1e}", flush=True)
    save(maps)
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True, choices=['A', 'B', 'C_LTown', 'C_BWSN2', 'BL', 'BW', 'smoke'])
    ap.add_argument('--lib', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=10); ap.add_argument('--seed', type=int, default=20261022)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wd = tempfile.mkdtemp(prefix=f'gc_{a.job}_', dir=str(out))
    pool = get_context('fork').Pool(a.workers, initializer=_w_init, initargs=(a.lib, wd))   # forked before CUDA starts
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = {'job': a.job, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu',
            'epanet': api.Library(a.lib).version(), 'seed': a.seed, 'workers': a.workers,
            'loss_sum_eps': float(np.finfo(np.longdouble).eps)}
    t_start = time.perf_counter()

    def save(maps):
        (out / f'gradcheck_{a.job}.json').write_text(json.dumps({'meta': dict(meta, wall_s=time.perf_counter() - t_start),
                                                                  'summary': summarize([m for m in maps if m['directions']]),
                                                                  'maps': maps}, default=float), encoding='utf-8')
    if a.job == 'smoke':
        global CASES
        CASES[:] = [c for c in CASES if c[0] in ('Net1', 'Net2_react')]
        job_A(pool, a, dev, wd, save)
    elif a.job == 'A':
        job_A(pool, a, dev, wd, save)
    elif a.job == 'BL':                                      # field maps + L-Town per-pipe + slot-source maps of Net3 and L-Town
        m = job_B(pool, a, dev, wd, save)
        m = job_C(pool, a, dev, wd, save, 'LTown', H5, 20, 40, m)
        m = job_D(pool, a, dev, wd, save, 'Net3', 50, 10, 20, m)
        job_D(pool, a, dev, wd, save, 'LTown', 50, 10, 20, m)
    elif a.job == 'BW':                                      # BWSN2: per-pipe wall map and slot-source map
        m = job_C(pool, a, dev, wd, save, 'BWSN2', [1e-2, 1e-3, 1e-5], 10, 20)
        job_D(pool, a, dev, wd, save, 'BWSN2', 50, 5, 10, m)
    print('done', a.job, f'{time.perf_counter() - t_start:.0f}s', flush=True)
    pool.close(); pool.join()


if __name__ == '__main__':
    main()
