"""Fused replay schedule: one state vector, one gather / scatter op per (step, level) or per step.

Slots of the state vector st[M] (batch-minor in the engine: st[M, B]):
    [0, V)                  segment versions (each written at most once)
    [V, V+N1)               u      (N1 = nnodes + 1)
    [V+N1, V+2 N1)          uflow
    [V+2 N1, V+3 N1)        c0     (parameter, copied in before the sweep)
    [V+3 N1, V+(3+K) N1)    src    (parameter, K time slots, copied in before the sweep)
    M-1                     constant 1.0
A fused op writes st[tgt[r]] = sum_{terms with row r} w * F * st[idx] (+ setpoint terms), where the
reaction factor F = exp(logH[fs] - logH[fb]) (CHEM) or the additive T[fs] - T[fb] (AGE) applies to
version reads and fs = fb = DUMMY (F = 1) to every other slot. 'setpoint' terms stay nonlinear,
max(st[sp_src] * w - st[sp_u], 0), and block substitution.
Kind-merge: the ops of one (step, level) become one fused op (targets are disjoint slots).
Flatten: every value written and read inside the same step is substituted (sparse products of the
per-level maps), leaving one op per step; steps holding setpoint terms are cut at those levels.
Versions born and consumed inside one step are never materialised after flattening. Weights of
merged paths are summed in double precision; the replayed values differ from the level-by-level
replay by floating-point reassociation only.
"""
import numpy as np
import scipy.sparse as sps


class FusedOp:
    __slots__ = ('step', 'tgt', 'row', 'idx', 'w', 'fs', 'fb', 'sp')

    def __init__(self, step, tgt, row, idx, w, fs, fb, sp=None):
        self.step, self.tgt, self.row, self.idx, self.w, self.fs, self.fb, self.sp = step, tgt, row, idx, w, fs, fb, sp


class FusedSchedule:
    def __init__(self, sch, flatten=True):
        N1, V, C, S = sch.nnodes + 1, sch.nver, sch.nchains, sch.nsteps
        K = int(getattr(sch, 'nsrc_slots', 1))
        self.N1, self.V, self.C, self.S, self.K = N1, V, C, S, K
        self.base = {'ver': 0, 'u': V, 'uflow': V + N1, 'c0': V + 2 * N1, 'src': V + 3 * N1}
        self.const_slot = V + (3 + K) * N1
        self.M = self.const_slot + 1
        self.DUMMY = (S + 1) * C                                   # extended logH / T index with F = 1
        self.chain, self.birth, self.death = sch.ver_chain, sch.ver_birth, sch.ver_death
        self.flatten = bool(flatten)
        self.ops = []
        by_step = {}
        for op in sch.ops:
            by_step.setdefault(op.step, []).append(op)
        for s in sorted(by_step):
            self._fuse_step(s, by_step[s])
        self.nterms = int(sum(len(op.idx) for op in self.ops))
        self.snapshots = sch.snapshots

    # ---- factors of a leaf slot read at step s ----
    def _fsfb(self, s, idx):
        if self.V == 0:
            d = np.full(len(idx), self.DUMMY, dtype=np.int64)
            return d, d.copy()
        isv = idx < self.V
        vi = np.minimum(idx, self.V - 1)
        ch = self.chain[vi]
        fs = np.where(isv, s * self.C + ch, self.DUMMY)
        fb = np.where(isv, self.birth[vi] * self.C + ch, self.DUMMY)
        return fs.astype(np.int64), fb.astype(np.int64)

    # ---- one recorded op -> (targets, rows, global slots, weights, setpoint) ----
    def _globalize(self, op):
        tg = op.tgt + self.base[op.tkind]
        rows, cols, ws, sp = [], [], [], None
        for kind, (row, idx, w) in op.terms.items():
            if kind == 'setpoint':
                sp = (row.astype(np.int64), self.base['src'] + idx, self.base['u'] + idx % self.N1, w.astype(np.float64))
                continue
            rows.append(row)
            cols.append(np.full(len(row), self.const_slot, dtype=np.int64) if kind == 'const' else idx + self.base[kind])
            ws.append(w)
        cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros(0, dtype=dt)
        return tg.astype(np.int64), cat(rows, np.int64), cat(cols, np.int64), cat(ws, np.float64), sp

    def _emit(self, s, tgt, row, idx, w, sp=None):
        fs, fb = self._fsfb(s, idx)
        self.ops.append(FusedOp(s, tgt, row, idx, w, fs, fb, sp))

    def _emit_merged(self, s, ops):
        """Ops of one (step, level): concatenate (disjoint targets), rows offset."""
        parts = [self._globalize(op) for op in ops]
        off = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        tgt = np.concatenate([p[0] for p in parts])
        row = np.concatenate([p[1] + o for p, o in zip(parts, off)])
        idx = np.concatenate([p[2] for p in parts]); w = np.concatenate([p[3] for p in parts])
        sps_ = [(p[4][0] + o, p[4][1], p[4][2], p[4][3]) for p, o in zip(parts, off) if p[4] is not None]
        sp = tuple(np.concatenate([q[i] for q in sps_]) for i in range(4)) if sps_ else None
        self._emit(s, tgt, row, idx, w, sp)

    def _fuse_step(self, s, ops):
        levels = {}
        for op in ops:
            levels.setdefault(op.level, []).append(op)
        if not self.flatten:
            for lv in sorted(levels):
                self._emit_merged(s, levels[lv])
            return
        parts = {lv: [self._globalize(op) for op in levels[lv]] for lv in sorted(levels)}
        slots = np.unique(np.concatenate([np.concatenate((p[0], p[2])) for ps in parts.values() for p in ps]))
        n = len(slots)
        loc = lambda g: np.searchsorted(slots, g)
        W = sps.identity(n, format='csr', dtype=np.float64)
        written = np.zeros(n, dtype=bool)

        def flush(final):
            nonlocal W, written
            rows = np.flatnonzero(written)
            if final:                                              # versions consumed inside this step: never read again
                g = slots[rows]
                rows = rows[~((g < self.V) & (self.death[np.minimum(g, max(self.V - 1, 0))] == s))]
            if len(rows):
                Wr = W[rows].tocsr()
                Wr.eliminate_zeros()
                Wr = Wr.tocoo()
                self._emit(s, slots[rows], Wr.row.astype(np.int64), slots[Wr.col], Wr.data.astype(np.float64))
            W = sps.identity(n, format='csr', dtype=np.float64)
            written = np.zeros(n, dtype=bool)

        for lv, ps in parts.items():
            if any(p[4] is not None for p in ps):                  # nonlinear level: barrier, replayed as recorded
                flush(final=False)
                self._emit_merged(s, levels[lv])
                continue
            for tg, row, col, w, _ in ps:                          # P = identity with the target rows replaced by A
                lt = loc(tg)
                keep = np.ones(n, dtype=bool); keep[lt] = False
                kd = np.flatnonzero(keep)
                P = sps.coo_matrix((np.ones(len(kd)), (kd, kd)), shape=(n, n)).tocsr()
                A = sps.coo_matrix((w, (lt[row], loc(col))), shape=(n, n)).tocsr()
                W = (P + A) @ W
                written[lt] = True
        flush(final=True)
