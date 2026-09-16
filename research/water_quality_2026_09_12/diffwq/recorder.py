"""Transport schedule recorder (phase A output of the trace-replay design).

While the Python replicator runs (mode M0 or M1), every concentration assignment is
logged as a *linear combination* of earlier values. Segment concentrations are
immutable *versions*: each write to a segment (creation, Ctol merge, tank update)
creates a new version id. Reads reference the version current at read time. Reaction
is not logged: with first-order kinetics (or AGE) it is a closed-form per-chain factor
that the replay engine applies when a version is read (torch_wq.py).

Term kinds (kind, index, weight):
  'ver'      version index, weight
  'u'        node index, weight              node quality at this step (or carried over)
  'uflow'    node index, weight              node quality used for outflows (tank + source)
  'c0'       node index, weight              initial quality parameter
  'src'      node index, multiplier          source strength parameter x multiplier
  'setpoint' node index, multiplier          max(strength x multiplier - u[node], 0)
  'const'    0, value
Targets: ('ver', new version), ('u', node), ('uflow', node).
With nsrc_slots = K > 1 (Replicator(src_slots=K)) the 'src' / 'setpoint' index is
slot * (nnodes + 1) + node, the multiplier excludes the pattern factor, and the parameter is the
strength x pattern value active in that slot (EPANET pattern grid, period K).
"""
from dataclasses import dataclass, field
import numpy as np

TERM_KINDS = ('ver', 'u', 'uflow', 'c0', 'src', 'setpoint', 'const')


@dataclass
class Op:
    step: int
    level: int
    tkind: str
    tgt: np.ndarray                  # [n] target indices
    terms: dict = field(default_factory=dict)   # kind -> (row[m], idx[m], w[m])


@dataclass
class Schedule:
    nsteps: int
    dt: np.ndarray                   # [S+1] seconds, dt[0] = 0 (init pseudo-step)
    period: np.ndarray               # [S+1] hydraulic period index of each step
    kf: np.ndarray                   # [P, nlinks+1] wall mass-transfer coefficient (hydraulics only)
    sc_zero: bool
    nver: int
    ver_chain: np.ndarray            # [V] chain index: link k, or nlinks + tank i
    ver_birth: np.ndarray            # [V] step at which the version was written
    ver_death: np.ndarray            # [V] step at which it stopped being current (S+1 if never)
    snapshots: np.ndarray            # [n_out] number of completed steps at each output
    ops: list                        # Op in execution order
    nnodes: int
    nchains: int
    kinds_seen: dict
    nsrc_slots: int = 1


class ScheduleRecorder:
    def __init__(self, nnodes, nlinks, ntanks, nsrc_slots=1):
        self.nnodes, self.nlinks = nnodes, nlinks
        self.nchains = nlinks + ntanks + 1
        self.nsrc_slots = int(nsrc_slots)
        self.step = 0
        self.dt, self.period = [0.0], [0]
        self.kf = {}
        self.sc_zero = False
        self.cur_ver = []            # segment id -> current version
        self.ver_chain, self.ver_birth, self.ver_death = [], [], []
        self.assign = []             # (step, tkind, tidx, terms)
        self.snapshots = []

    # ---- lifecycle ----
    def begin_step(self, dt, period):
        self.step += 1
        self.dt.append(float(dt))
        self.period.append(int(period))

    def snapshot(self):
        self.snapshots.append(self.step)

    def set_period_kf(self, p, kf, sc_zero):
        self.kf[p] = np.array(kf, dtype=np.float64)
        self.sc_zero = bool(sc_zero)

    # ---- versions ----
    def ver(self, seg_id):
        return self.cur_ver[seg_id]

    def _new_version(self, chain):
        v = len(self.ver_chain)
        self.ver_chain.append(chain)
        self.ver_birth.append(self.step)
        self.ver_death.append(-1)
        return v

    def new_seg(self, seg, chain, terms):
        v = self._new_version(chain)
        assert seg.id == len(self.cur_ver)
        self.cur_ver.append(v)
        self.assign.append((self.step, 'ver', v, list(terms)))

    def set_seg(self, seg, terms):
        old = self.cur_ver[seg.id]
        self.ver_death[old] = self.step
        v = self._new_version(self.ver_chain[old])
        self.cur_ver[seg.id] = v
        self.assign.append((self.step, 'ver', v, list(terms)))

    def kill_seg(self, seg_id):
        self.ver_death[self.cur_ver[seg_id]] = self.step

    def set_u(self, node, terms):
        self.assign.append((self.step, 'u', int(node), list(terms)))

    def set_uflow(self, node, terms):
        self.assign.append((self.step, 'uflow', int(node), list(terms)))

    # ---- post-processing: dependency levels within each step, then CSR groups ----
    def build(self):
        S = self.step
        V = len(self.ver_chain)
        ver_step = np.asarray(self.ver_birth, dtype=np.int64)
        ver_level = np.zeros(V, dtype=np.int64)
        u_step = np.full(self.nnodes + 1, -1, dtype=np.int64); u_level = np.zeros(self.nnodes + 1, dtype=np.int64)
        f_step = np.full(self.nnodes + 1, -1, dtype=np.int64); f_level = np.zeros(self.nnodes + 1, dtype=np.int64)
        groups = {}
        kinds_seen = {}
        for (s, tkind, tidx, terms) in self.assign:
            level = 0
            for kind, idx, w in terms:
                if kind == 'ver':
                    if ver_step[idx] == s:
                        level = max(level, ver_level[idx] + 1)
                elif kind in ('u', 'setpoint'):
                    if u_step[idx] == s:
                        level = max(level, u_level[idx] + 1)
                elif kind == 'uflow':
                    if f_step[idx] == s:
                        level = max(level, f_level[idx] + 1)
                kinds_seen[kind] = kinds_seen.get(kind, 0) + 1
            if tkind == 'ver':
                ver_level[tidx] = level
            elif tkind == 'u':
                u_step[tidx], u_level[tidx] = s, level
            else:
                f_step[tidx], f_level[tidx] = s, level
            groups.setdefault((s, level, tkind), []).append((tidx, terms))
        ops = []
        for (s, level, tkind) in sorted(groups):
            items = groups[(s, level, tkind)]
            tgt = np.array([t for t, _ in items], dtype=np.int64)
            if np.unique(tgt).size != tgt.size:
                raise RuntimeError(f'duplicate {tkind} target within step {s} level {level}')
            per_kind = {}
            for row, (_, terms) in enumerate(items):
                for kind, idx, w in terms:
                    per_kind.setdefault(kind, ([], [], []))
                    per_kind[kind][0].append(row); per_kind[kind][1].append(idx); per_kind[kind][2].append(w)
            terms = {k: (np.array(r, dtype=np.int64), np.array(i, dtype=np.int64), np.array(w, dtype=np.float64))
                     for k, (r, i, w) in per_kind.items()}
            ops.append(Op(s, level, tkind, tgt, terms))
        death = np.asarray(self.ver_death, dtype=np.int64)
        death[death < 0] = S + 1
        P = max(self.kf) + 1 if self.kf else 0
        kf = np.zeros((P, self.nlinks + 1))
        for p, arr in self.kf.items():
            kf[p] = arr
        return Schedule(nsteps=S, dt=np.asarray(self.dt), period=np.asarray(self.period, dtype=np.int64),
                        kf=kf, sc_zero=self.sc_zero, nver=V, ver_chain=np.asarray(self.ver_chain, dtype=np.int64),
                        ver_birth=ver_step, ver_death=death, snapshots=np.asarray(self.snapshots, dtype=np.int64),
                        ops=ops, nnodes=self.nnodes, nchains=self.nchains, kinds_seen=kinds_seen,
                        nsrc_slots=self.nsrc_slots)
