"""Exp2 v2.0 (reviewer 1, comment 7). The v1 module exp2_source.py and its results are left untouched; this file
subclasses its SourceProblem and reuses its _run unchanged except for three methods:
  option 2  sample_scenarios: a drawn source junction whose outflow is stagnant (<= Q_STAGNANT, EPANET skips the source)
            during any part of the injection is redrawn from a separate stream; the main stream stays aligned with v1, so
            the 200 test scenarios equal v1's except for the redrawn junctions.
  option 4  place_sensors: greedy detection-likelihood placement (v1 rule, ties by mean alarm time) on --n-train-v2
            training scenarios from a separate stream, nested sensor sets of --counts sizes (or the v1 sensor list).
  option 5  make_data: alarm threshold --thr; noise drawn per sensor node from its own stream (the same noise for a node
            in every configuration); noise-induced first alarms flagged; pure-noise false-alarm rate on 1,000 windows.
Configurations are given as --configs tag:placement:thr:count,...  (placement v1 | v2).
Usage (from research/water_quality_2026_09_12):
    python -m diffwq.exp2_v2 --ref final_5090/exp2/exp2_BWSN2.json --configs c1:v1:0.5:20,c2_20:v2:0.5:20 --out-dir final_v2
"""
import argparse, json, os, tempfile, time
from pathlib import Path
import numpy as np
import torch
from . import epanet_api as api
from . import exp2_source as E
from .replicate import Q_STAGNANT


class SourceProblemV2(E.SourceProblem):
    def __init__(self, name, lib, workdir, args):
        super().__init__(name, lib, workdir, args)
        n, per = self.net, self.trace.periods
        out = np.zeros((len(per), n.nnodes + 1))
        for i, p in enumerate(per):                              # outflow volume rate per node (cfs): links leaving + demand
            q = np.asarray(p.flow[1:])
            np.add.at(out[i], np.where(q >= 0, n.n1[1:], n.n2[1:]), np.abs(q))
            out[i] += np.maximum(np.asarray(p.demand), 0.0)
        self._out, self._pt = out, np.array([p.t for p in per])
        self._ph = np.array([p.hydstep for p in per])
        self._rng_fix = np.random.default_rng([args.seed, 2])
        self.n_resampled, self.v2_sensors, self.v2_place = 0, None, None

    def injected_ok(self, j, k0, d):
        a, b = k0 * self.slot, (k0 + d) * self.slot
        idx = np.flatnonzero((self._pt < b) & (self._pt + np.maximum(self._ph, 1) > a))
        return bool(np.all(self._out[idx, j] > Q_STAGNANT))

    def sample_scenarios(self, n, rng):
        sc = super().sample_scenarios(n, rng)
        if self.args.scen_rule != 'full':
            return sc
        out = []
        for j, k0, d, m in sc:
            if not self.injected_ok(j, k0, d):
                self.n_resampled += 1
                while not self.injected_ok(j, k0, d):
                    j = int(self._rng_fix.choice(self.junc))
            out.append((j, k0, d, m))
        return out

    def greedy_v2(self, n_train, n_max, batch):
        """Nested greedy order on n_train training scenarios (v2 rule) and the train coverage curve."""
        scen = self.sample_scenarios(n_train, np.random.default_rng([self.args.seed, 1]))
        det, tdet = [], []
        with torch.no_grad():
            for s in range(0, n_train, batch):
                f = self.first_alarm(self.c_full(self.src_of(scen[s:s + batch]), eager=True))
                det.append(torch.isfinite(f)); tdet.append(f)
        det = torch.cat(det) & self.is_junc.bool().view(1, -1)
        tdet = torch.cat(tdet)
        cov, chosen, curve = torch.zeros(n_train, dtype=torch.bool, device=self.dev), [], []
        for _ in range(n_max):
            new = det & ~cov.view(-1, 1)
            gain = new.sum(0).to(self.dt)
            mean_t = torch.where(new, tdet, torch.zeros_like(tdet)).sum(0) / gain.clamp_min(1.0)
            score = gain - 1e-9 * mean_t
            if chosen:
                score[torch.as_tensor(chosen, device=self.dev)] = -1.0
            j = int(score.argmax()); chosen.append(j); cov |= det[:, j]; curve.append(float(cov.float().mean()))
        self.v2_place = {'n_train': n_train, 'order': chosen, 'curve': curve, 'detectable_fraction': float(det.any(1).float().mean())}

    def place_sensors(self, n_train, rng, batch):
        E.SourceProblem.sample_scenarios(self, n_train, rng)       # keep the main stream aligned with v1
        self.set_sensors(self.v2_sensors)
        info = {'sensors': [self.net.node_ids[j] for j in self.v2_sensors]}
        if self.v2_place and self.v2_sensors == self.v2_place['order'][:len(self.v2_sensors)]:
            info.update(n_train=self.v2_place['n_train'], coverage_curve=self.v2_place['curve'][:len(self.v2_sensors)],
                        detectable_fraction=self.v2_place['detectable_fraction'])
        return info

    def noise(self, n, n_out):
        cols = [np.random.default_rng([self.args.seed, 3, int(j)]).normal(0.0, self.args.noise, size=(n, n_out)) for j in self.sens]
        return torch.as_tensor(np.stack(cols, 2), dtype=self.dt, device=self.dev)

    def make_data(self, scen, rng, batch):
        c = self.truth_series(scen, batch)
        d = c + self.noise(len(scen), c.shape[1])
        fa = self.first_alarm(d)                                                   # [n, n_s]
        first, s_star = fa.min(1)
        start = torch.as_tensor([k0 * self.slot for _, k0, _, _ in scen], dtype=self.dt, device=self.dev)
        false_alarm = torch.isfinite(first) & (first <= start)
        keep = torch.isfinite(first) & ~false_alarm
        t_idx = torch.searchsorted(self.times_t, torch.where(torch.isfinite(first), first, self.times_t[-1]))
        nf_at_alarm = c[torch.arange(len(scen), device=self.dev), t_idx.clamp(max=c.shape[1] - 1), s_star]
        noise_alarm = keep & (nf_at_alarm < 0.5 * self.args.thr)
        nf_det = torch.isfinite(self.first_alarm(c).amin(1))
        idx = torch.nonzero(keep).flatten().tolist()
        self.v2_data = {'n_resampled_test': self.n_resampled, 'n_noise_induced_first_alarm': int(noise_alarm.sum()),
                        'noise_free_detected': int(nf_det.sum()), 'detected_flags': keep.tolist(),
                        'scenarios_all': [(self.net.node_ids[j], k0 * self.slot / 3600, dd * self.slot / 3600, m) for j, k0, dd, m in scen]}
        return ([scen[i] for i in idx], c[keep], d[keep], first[keep],
                {'n_sampled': len(scen), 'n_detected': len(idx), 'n_false_alarm': int(false_alarm.sum()),
                 'n_undetected': int((~torch.isfinite(first)).sum()),
                 'detection_delay_h': ((first[keep] - start[keep]) / 3600).tolist(),
                 'first_detection_h': (first[keep] / 3600).tolist()})

    def false_alarm_rate(self, n_win=1000):
        z = torch.as_tensor(np.random.default_rng([self.args.seed, 4]).normal(0.0, self.args.noise, size=(n_win, self.n_out, len(self.sens))),
                            dtype=self.dt, device=self.dev)
        return float(torch.isfinite(self.first_alarm(z).amin(1)).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='v1 result JSON (settings and v1 sensor list)')
    ap.add_argument('--configs', required=True)
    ap.add_argument('--n-train-v2', type=int, default=4096)
    ap.add_argument('--scen-rule', default='full', choices=['full', 'uniform'])
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--lib', default=None)
    ap.add_argument('--exhaustive', type=int, default=0, help='scenarios refined over every junction (v1 validation)')
    a = ap.parse_args()
    J = json.load(open(a.ref)); n0 = J['networks'][0]
    args = argparse.Namespace(**J['args'])
    args.device, args.no_graph, args.exhaustive, args.chunk, args.scen_rule = 'cuda', True, a.exhaustive, a.chunk, a.scen_rule
    cfgs = [dict(zip(('tag', 'place', 'thr', 'count'), c.split(':'))) for c in a.configs.split(',')]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    P = SourceProblemV2(n0['network'], api.Library(a.lib), tempfile.mkdtemp(prefix='exp2v2_'), args)
    setup_s = time.perf_counter() - t0
    v1_sens = [P.net.node_ids.index(s) for s in n0['placement']['sensors']]
    n_max = max(int(c['count']) for c in cfgs if c['place'] == 'v2') if any(c['place'] == 'v2' for c in cfgs) else 0
    if n_max:
        t1 = time.perf_counter(); P.greedy_v2(a.n_train_v2, n_max, args.place_batch); place_s = time.perf_counter() - t1
    for c in cfgs:
        args.thr = float(c['thr'])
        P.n_resampled = 0
        P.v2_sensors = v1_sens if c['place'] == 'v1' else P.v2_place['order'][:int(c['count'])]
        t1 = time.perf_counter()
        R = E._run(P, args, t1)
        R['v2'] = {'config': c, 'scen_rule': a.scen_rule, 'n_train_v2': a.n_train_v2 if c['place'] == 'v2' else 512,
                   'setup_s_shared': setup_s, 'placement_v2_s': place_s if n_max else None,
                   'false_alarm_rate_noise_only_48h': P.false_alarm_rate(), **P.v2_data}
        R['args'] = vars(args).copy()
        (out / f"exp2_v2_{c['tag']}.json").write_text(json.dumps(R, default=float), encoding='utf-8')
        s = R['summary']; dd = R['data']
        print(f"[{c['tag']}] detected {dd['n_detected']}/200 (noise-free {P.v2_data['noise_free_detected']}), resampled "
              f"{P.v2_data['n_resampled_test']}, noise-induced {P.v2_data['n_noise_induced_first_alarm']}, FAR "
              f"{R['v2']['false_alarm_rate_noise_only_48h']:.3f}, end-to-end top1 "
              f"{[round(w['glrt_top1'] * dd['n_detected'] / 200, 3) for w in s]}, {time.perf_counter() - t1:.0f}s", flush=True)
    P.pr.__exit__()


if __name__ == '__main__':
    main()
