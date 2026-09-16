"""Phase B: differentiable replay of a recorded transport Schedule in PyTorch.

State: x[B, V] segment-version values (as written), u[B, N+1] node quality, uflow[B, N+1]
node quality carried by outflows. Reading version j at step s applies the closed-form
reaction of its chain between birth and s:
    CHEM, first order:  x_j * exp(logH[s, chain] - logH[birth, chain]),
                        logH = cumsum(log(1 + (kb + rc(kw)) dt))  (pipes), log(1 + kb_tank dt) (tanks)
    AGE:                x_j + (T[s] - T[birth]) / 3600 on pipes and tanks
    TRACE / no reaction: x_j
Everything else in the schedule is a fixed linear combination decided by the hydraulics
(and, in mode M0, by the recorded Ctol merge decisions), so autograd gives exact
derivatives of the replayed EPANET map with respect to
    theta = {kb[L+1], kw[L+1], tank_kb[T+1], c0[N+1], src[N+1]}   (EPANET user units).
Parameters may carry a leading batch dimension B (shared schedule, B parameter sets).
Supported kinetics: bulk/tank/wall order 1 with Climit = 0, AGE, TRACE. Other orders
need the stepwise path (not implemented here).
For CHEM the factors are applied even when EPANET's Reactflag is off (all kb = kw = 0):
they are then exactly 1.0, so the forward values are unchanged, while the derivative with
respect to kb/kw at zero is the one of the (continuous) EPANET map rather than 0.
Engines: 'adjoint' (default) runs the sweep without a graph and applies a hand-written
reverse sweep (O(terms) time and memory); 'autograd' records every in-place write, which is
exact but costs O(ops x versions) in backward. Both give identical gradients.
'fused' replays the FusedSchedule (fuse.py): one batch-minor state vector st[M, B] holding
versions, u, uflow, c0, src and a constant, one gather / scatter op per (step, level) or, with
flatten=True, per step. Reaction factors of all terms are precomputed once per forward
(Fw = exp(logH[fs] - logH[fb]) * w) when B * terms fits `budget` bytes, else per op. Same map
and same adjoint as the other engines up to floating-point reassociation.
Sources: src may carry K time slots (Schedule.nsrc_slots, see replicate.Replicator(src_slots=K)):
shape [K, N+1] or [B, K, N+1]; slot k of node i is the strength x pattern value active while
((Qtime + Pstart) // Pstep) % K == k. set_fixed_reaction() freezes kb/kw/tank_kb and reuses
their factors, so repeated forward(c0=..., src=...) calls are a fixed affine map.
SpMM path (fused engine, CUDA): when one parameter point serves B columns (shared factors
Fw[T_total, 1]: fixed reaction, B = 1 parameter set with batched c0 / src, or the unit cotangents
of jac_rows_kw), a fused op with m terms is applied as a cuSPARSE sparse x dense product whenever
B >= spmm_min_batch and m x B >= spmm_min_work (the gather / index_add_ kernels win for small,
launch-bound ops; the default 2^22 term-columns is the eager crossover measured on an RTX 5090,
CUDA-graph replay moves it lower): st[tgt] = W_op st (forward) and gst[U_op] += W_op^T g (adjoint, U_op = the op's
distinct source slots), with the kw-row contraction of jac_rows_kw as a third product over the
op's chains. Same float64 arithmetic as the gather / index_add_ path (differences at the
summation-order level, ~1e-15 relative) with fewer, bandwidth-efficient kernels, no atomics and
no [T_total, B] buffer.
"""
import warnings
import numpy as np
import torch

from . import epanet_api as api
from .fuse import FusedSchedule
from .network import SECperDAY

warnings.filterwarnings('ignore', message='Sparse CSR tensor support is in beta state')


def default_src(net, K):
    """[K, N+1] source strengths (user units) reproducing the recorded run: strength x pattern value
    per slot for junction / tank sources, strength in slot 0 (pattern folded into the schedule) for
    reservoirs and for K = 1."""
    out = np.zeros((K, net.nnodes + 1))
    for i, (stype, c0, pat) in net.sources.items():
        if K == 1 or net.node_type[i] == api.EN_RESERVOIR:
            out[0, i] = c0
        elif pat == 0:
            out[:, i] = c0
        else:
            f = net.patterns[pat]
            out[:, i] = [c0 * f[k % len(f)] for k in range(K)]
    return out


class TorchWQ:
    def __init__(self, net, schedule, reactflag, device='cpu', dtype=torch.float64, engine='adjoint',
                 flatten=True, budget=2 << 30, spmm_min_batch=16, spmm_min_work=1 << 22):
        self.net, self.sch = net, schedule
        self.dev, self.dtype = torch.device(device), dtype
        if engine not in ('adjoint', 'autograd', 'fused'):
            raise ValueError(engine)
        self.engine = engine
        self.K = int(getattr(schedule, 'nsrc_slots', 1))
        self.budget = int(budget)
        self.spmm_min_batch = int(spmm_min_batch or 0)             # 0: gather / index_add_ path only
        self.spmm_min_work = int(spmm_min_work)                    # per-op terms x B below which the gather path is kept
        self._spmm, self._fixed_T = None, None
        self.fixed = None
        n = net
        self.chem = n.qualflag == api.EN_CHEM
        if self.chem:
            if not (n.bulkorder == 1.0 and n.tankorder == 1.0 and n.wallorder == 1.0 and n.climit == 0.0):
                raise NotImplementedError('closed-form replay needs first-order kinetics with Climit = 0')
        self.reactflag = bool(reactflag)
        S, C = schedule.nsteps, schedule.nchains
        self.S, self.C, self.V, self.N = S, C, schedule.nver, n.nnodes
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), dtype=dt or dtype, device=self.dev)
        self.dt = t(schedule.dt)                                   # [S+1], dt[0] = 0
        self.period = t(schedule.period, torch.long)
        self.kf = t(schedule.kf)                                   # [P, L+1]
        self.diam = t(n.diam)
        self.ucf_elev = float(n.ucf['ELEV'])
        self.ucf_qual = float(n.ucf['QUALITY'])
        L = n.nlinks
        self.is_pipe = t(np.concatenate(([False], n.link_type[1:] == api.EN_PIPE)), torch.bool)      # [L+1]
        tank_reacts = np.concatenate(([False], ~n.tank_is_reservoir[1:]))
        self.chain_reacts = t(np.concatenate((self.is_pipe.cpu().numpy(), tank_reacts[1:])), torch.bool)  # [C]
        if engine == 'fused':
            self._build_fused(schedule, flatten)
            return
        # per-op index/weight tensors are views into one concatenated buffer per dtype (one transfer each)
        ibuf, fbuf = [], []
        I_ = lambda a: (ibuf.append(np.asarray(a, dtype=np.int64)), len(ibuf) - 1)[1]
        F_ = lambda a: (fbuf.append(np.asarray(a, dtype=np.float64)), len(fbuf) - 1)[1]
        spec = []
        for op in schedule.ops:
            terms = {}
            for kind, (row, idx, w) in op.terms.items():
                d = {'row': I_(row), 'idx': I_(idx), 'w': F_(w)}
                if kind == 'ver':
                    ch = schedule.ver_chain[idx]
                    d['flat_s'] = I_(op.step * C + ch)
                    d['flat_b'] = I_(schedule.ver_birth[idx] * C + ch)
                terms[kind] = d
            spec.append((op.step, op.tkind, I_(op.tgt), len(op.tgt), terms))
        def split(bufs, dt):
            big = torch.as_tensor(np.concatenate(bufs), dtype=dt, device=self.dev)
            return torch.split(big, [len(a) for a in bufs])
        IV, FV = split(ibuf, torch.long), split(fbuf, dtype)
        self.ops = [(s, tk, IV[tg], n_, {k: {kk: (FV[v] if kk == 'w' else IV[v]) for kk, v in d.items()}
                                        for k, d in terms.items()}) for s, tk, tg, n_, terms in spec]
        self.snap_after_op = self._snapshot_map([op.step for op in schedule.ops], schedule.snapshots)

    @staticmethod
    def _snapshot_map(op_steps, snapshots):
        """snapshot k = state after all ops of steps <= k (several snapshots may share an op index)"""
        steps = np.asarray(op_steps)
        out = {}
        for k in snapshots:
            j = int(np.searchsorted(steps, k, side='right')) - 1
            out[j] = out.get(j, 0) + 1
        return out

    def _build_fused(self, schedule, flatten):
        F = self.fsch = FusedSchedule(schedule, flatten=flatten)
        dev, dt = self.dev, self.dtype
        for op in F.ops:                                           # row-major terms: the per-op CSR order of the SpMM path
            p = np.argsort(op.row, kind='stable')
            op.row, op.idx, op.w, op.fs, op.fb = (a[p] for a in (op.row, op.idx, op.w, op.fs, op.fb))
        cat = lambda key, dtp: torch.as_tensor(np.concatenate([getattr(op, key) for op in F.ops]) if F.ops
                                               else np.zeros(0), dtype=dtp, device=dev)
        lens = [len(op.idx) for op in F.ops]
        self.IDX, self.FS, self.FB = cat('idx', torch.long), cat('fs', torch.long), cat('fb', torch.long)
        self.W = cat('w', dt)
        TGT = cat('tgt', torch.long)
        TGT_ROW = torch.as_tensor(np.concatenate([op.tgt[op.row] for op in F.ops]), dtype=torch.long, device=dev)
        ROW = torch.as_tensor(np.concatenate([op.row for op in F.ops]), dtype=torch.long, device=dev)
        idx_v, fs_v, fb_v, w_v = (torch.split(a, lens) for a in (self.IDX, self.FS, self.FB, self.W))
        tr_v, row_v = torch.split(TGT_ROW, lens), torch.split(ROW, lens)
        tgt_v = torch.split(TGT, [len(op.tgt) for op in F.ops])
        offs = np.concatenate(([0], np.cumsum(lens)))
        self.fops = []
        for j, op in enumerate(F.ops):
            sp = None
            if op.sp is not None:
                sp = tuple(torch.as_tensor(a, dtype=(dt if i == 3 else torch.long), device=dev) for i, a in enumerate(op.sp))
            self.fops.append((tgt_v[j], tr_v[j], row_v[j], idx_v[j], w_v[j], fs_v[j], fb_v[j], sp,
                              int(offs[j]), int(offs[j + 1])))
        self.T_total = int(offs[-1])
        self.max_op_terms = int(max(lens)) if lens else 0
        self.snap_after_op = self._snapshot_map([op.step for op in F.ops], schedule.snapshots)

    # ---- reaction factors from parameters (user units) ----
    def _log_factors(self, kb, kw, tank_kb):
        """logH[B, S+1, C] for CHEM (first order) and T[B, S+1, C] for AGE (additive)."""
        B = kb.shape[0]
        S, C, L = self.S, self.C, self.net.nlinks
        n = self.net
        if self.chem:
            kb_i = kb / SECperDAY                                      # [B, L+1] 1/s
            kwu = kw / SECperDAY / self.ucf_elev                       # [B, L+1] ft/s
            if self.sch.sc_zero:
                rc = kwu * (4.0 / self.diam.clamp_min(1e-300))         # [B, L+1]
                rc = rc.unsqueeze(1).expand(B, S + 1, L + 1)
            else:
                kf = self.kf[self.period]                              # [S+1, L+1]
                den = kf + kwu.abs().unsqueeze(1)
                rc = torch.where(kf > 0, (4.0 / self.diam.clamp_min(1e-300)) * kwu.unsqueeze(1) * kf
                                 / den.clamp_min(1e-300), torch.zeros_like(den))
            r_link = torch.where(self.is_pipe, kb_i.unsqueeze(1) + rc, torch.zeros_like(rc))   # [B, S+1, L+1]
            r_tank = (tank_kb / SECperDAY).unsqueeze(1).expand(B, S + 1, -1)                  # [B, S+1, T+1]
            r = torch.cat([r_link, r_tank[:, :, 1:]], dim=2)          # chains: 0..L links, L+1..L+T tanks
            f = 1.0 + r * self.dt.view(1, S + 1, 1)
            if not (self.dev.type == 'cuda' and torch.cuda.is_current_stream_capturing()) and bool((f <= 0).any()):
                raise ValueError('reaction factor <= 0: step too large for closed-form replay')
            logH = torch.cumsum(torch.log(f), dim=1)
            return logH.reshape(B, -1), None
        if n.qualflag == api.EN_AGE:
            T = torch.cumsum(self.dt, 0).view(1, S + 1, 1) / 3600.0 * self.chain_reacts.view(1, 1, C)
            return None, T.expand(B, -1, -1).reshape(B, -1)
        return None, None

    def _default(self, key):
        """Device copies of the recorded parameter values (built once: no host transfer inside a CUDA graph)."""
        if not hasattr(self, '_defaults'):
            n = self.net
            vals = {'kb': n.kb * SECperDAY, 'kw': n.kw * SECperDAY, 'tank_kb': n.tank_kb * SECperDAY,
                    'c0': n.c0 * n.ucf['QUALITY'], 'src': default_src(n, self.K)}
            self._defaults = {k: torch.as_tensor(np.asarray(v), dtype=self.dtype, device=self.dev) for k, v in vals.items()}
        return self._defaults[key]

    def _prep(self, v, key):
        if v is None:
            v = self._default(key)
        v = v.to(self.dev, self.dtype)
        return v.unsqueeze(0) if v.dim() == 1 else v

    def _prep_src(self, src):
        """-> [B, K*(N+1)] from [N+1] / [K, N+1] / [B, N+1] (K = 1) / [B, K, N+1] / [B, K*(N+1)]."""
        K, N1 = self.K, self.N + 1
        if src is None:
            src = self._default('src')
        src = src.to(self.dev, self.dtype)
        if src.dim() == 1:
            src = src.view(1, -1)
        elif src.dim() == 3:
            src = src.reshape(src.shape[0], -1)
        elif K > 1 and src.shape == (K, N1):
            src = src.reshape(1, -1)
        if src.shape[1] != K * N1:
            raise ValueError(f'src shape {tuple(src.shape)}: expected {K} slot(s) x {N1} nodes')
        return src

    def set_fixed_reaction(self, kb=None, kw=None, tank_kb=None):
        """Freeze the reaction parameters (fused engine): later forward(c0=, src=) calls reuse their
        factors, so the map is affine in (c0, src) and its adjoint needs no logH work. None = unfreeze."""
        self.fixed = self._fixed_T = None
        if kb is None and kw is None and tank_kb is None:
            return
        kb, kw, tank_kb = self._prep(kb, 'kb'), self._prep(kw, 'kw'), self._prep(tank_kb, 'tank_kb')
        if max(v.shape[0] for v in (kb, kw, tank_kb)) != 1:
            raise ValueError('set_fixed_reaction takes a single parameter set')
        with torch.no_grad():
            lH, T = self._fused_factors(*self._log_factors(kb, kw, tank_kb))
            Fw = self._fw_all(lH) if T is None else None
        self.fixed = (lH, T, Fw)

    def forward(self, kb=None, kw=None, tank_kb=None, c0=None, src=None):
        """Returns node quality snapshots [B, n_out, N+1] in user units."""
        c0 = self._prep(c0, 'c0')
        src = self._prep_src(src)
        if self.engine == 'fused' and self.fixed is not None and kb is None and kw is None and tank_kb is None:
            B = max(c0.shape[0], src.shape[0])
            c0i = c0.expand(B, -1) / self.ucf_qual
            lH, T, Fw = self.fixed
            out = _ReplayFused.apply(self, lH, T, c0i, src.expand(B, -1), Fw)
            return out * self.ucf_qual
        kb, kw, tank_kb = self._prep(kb, 'kb'), self._prep(kw, 'kw'), self._prep(tank_kb, 'tank_kb')
        B = max(v.shape[0] for v in (kb, kw, tank_kb, c0, src))
        if self.engine == 'fused':                                      # theta may stay at batch 1 (broadcast)
            Bt = max(v.shape[0] for v in (kb, kw, tank_kb))
            kb, kw, tank_kb = (v.expand(Bt, -1) for v in (kb, kw, tank_kb))
            c0, src = c0.expand(B, -1), src.expand(B, -1)
            lH, T = self._fused_factors(*self._log_factors(kb, kw, tank_kb))
            out = _ReplayFused.apply(self, lH, T, c0 / self.ucf_qual, src, None)
            return out * self.ucf_qual
        kb, kw, tank_kb, c0, src = (v.expand(B, -1) for v in (kb, kw, tank_kb, c0, src))
        logH, T = self._log_factors(kb, kw, tank_kb)
        c0i = c0 / self.ucf_qual                                       # internal units
        if self.engine == 'adjoint':
            out = _Replay.apply(self, logH, T, c0i, src)
        else:
            out, _, _ = self._sweep(logH, T, c0i, src, B)
        return out * self.ucf_qual

    # ---- fused engine: batch-minor state st[M, B], factors on an extended (dummy-row) logH ----
    def _fused_factors(self, logH, T):
        """[B, (S+1)C] -> [(S+1)C + 1, B] with a trailing zero row (index DUMMY, factor 1)."""
        ext = lambda a: torch.cat([a.t(), torch.zeros(1, a.shape[0], dtype=a.dtype, device=a.device)], 0)
        return (ext(logH) if logH is not None else None), (ext(T) if T is not None else None)

    def _fw_all(self, lH):
        """Fw[T_total, Bt] = exp(lH[fs] - lH[fb]) * w for every term of the fused schedule (None: F = 1)."""
        if lH is None:
            return self.W.unsqueeze(1)
        return torch.exp(lH[self.FS] - lH[self.FB]) * self.W.unsqueeze(1)

    def _use_big(self, lH):
        """Materialise the per-term factors Fw[T_total, Bt] (Bt parameter sets) when they fit the budget."""
        Bt = 1 if lH is None else lH.shape[1]
        return self.T_total * 8 * Bt <= self.budget

    # ---- SpMM path: per-op CSR views of the fused ops (CUDA, shared factors, B >= spmm_min_batch) ----
    def _spmm_on(self, Fw, B):
        """Per-op threshold on the term count (None: SpMM path off for this sweep, also when no op reaches it)."""
        if not (self.spmm_min_batch and self.dev.type == 'cuda' and Fw is not None and Fw.shape[1] == 1
                and B >= self.spmm_min_batch):
            return None
        m_min = -(-self.spmm_min_work // max(int(B), 1))
        return m_min if self.max_op_terms >= m_min else None

    @staticmethod
    def _crow(counts):
        return torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).to(torch.int32)

    @staticmethod
    def _csr(crow, col, vals, shape):
        return torch.sparse_csr_tensor(crow, col, vals, size=shape, check_invariants=False)

    def _spmm_struct(self):
        """Built once, on first use: forward W_op[n_tgt, M] in the stored (row-major) term order; transposed W_op^T
        restricted to the op's distinct source slots U_op, [n_U, n_tgt], in source-major term order perm_t.
        int32 indices (cuSPARSE 32-bit path)."""
        if self._spmm is None:
            if self.dev.type == 'cuda' and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('SpMM structures are built on the first eager pass: warm up before CUDA graph capture')
            if max(self.fsch.M, self.T_total) >= 2 ** 31:
                raise NotImplementedError('SpMM path: slots and terms must be int32-addressable')
            i32 = torch.int32
            crow_f, crow_t, U = [], [], []
            perm_t, col_t = [self.IDX.new_zeros(0, dtype=i32)], [self.IDX.new_zeros(0, dtype=i32)]
            for tgt, tgt_row, row, idx, w, fs, fb, sp, a, b in self.fops:
                crow_f.append(self._crow(torch.bincount(row, minlength=len(tgt))))
                srt, p = torch.sort(idx, stable=True)
                u, cnt = torch.unique_consecutive(srt, return_counts=True)
                U.append(u); crow_t.append(self._crow(cnt)); perm_t.append((p + a).to(i32)); col_t.append(row[p].to(i32))
            self._spmm = {'crow_f': crow_f, 'col_f': self.IDX.to(i32), 'crow_t': crow_t, 'U': U,
                          'perm_t': torch.cat(perm_t), 'col_t': torch.cat(col_t), 'chains': None}
        return self._spmm

    def _spmm_chains(self):
        """jac_rows_kw: P_op[n_ch, n_tgt] gathers the op's terms by chain fs % C (dummy terms: chain 0, value 0)."""
        S_ = self._spmm_struct()
        if S_['chains'] is None:
            i32, C = torch.int32, self.C
            crow_c, ch_u = [], []
            perm_c, col_c = [self.IDX.new_zeros(0, dtype=i32)], [self.IDX.new_zeros(0, dtype=i32)]
            for tgt, tgt_row, row, idx, w, fs, fb, sp, a, b in self.fops:
                srt, p = torch.sort(fs % C, stable=True)
                u, cnt = torch.unique_consecutive(srt, return_counts=True)
                ch_u.append(u); crow_c.append(self._crow(cnt)); perm_c.append((p + a).to(i32)); col_c.append(row[p].to(i32))
            S_['chains'] = (crow_c, ch_u, torch.cat(perm_c), torch.cat(col_c))
        return S_['chains']

    def _fw_T(self, Fw):
        """Fw[:, 0] in the source-major term order of the transposed ops (kept for the frozen reaction factors)."""
        S_ = self._spmm_struct()
        frozen = self.fixed is not None and self.fixed[2] is not None and Fw.data_ptr() == self.fixed[2].data_ptr()
        if frozen and self._fixed_T is not None:
            return self._fixed_T
        v = Fw[:, 0].index_select(0, S_['perm_t'])
        if frozen and not (self.dev.type == 'cuda' and torch.cuda.is_current_stream_capturing()):
            self._fixed_T = v
        return v

    def _sweep_fused(self, lH, T, c0i, src, B, Fw=None, keep_masks=False):
        """Returns (snapshots [B, n_out, N+1] internal units, final state st[M, B], setpoint masks)."""
        F, dev, dt = self.fsch, self.dev, self.dtype
        N1, V = F.N1, F.V
        st = torch.zeros(F.M, B, dtype=dt, device=dev)
        st[F.base['c0']:F.base['c0'] + N1] = c0i.t()
        st[F.base['src']:F.const_slot] = src.t()
        st[F.const_slot] = 1.0
        snaps, snap_at, masks = [], self.snap_after_op, {}
        m_min = self._spmm_on(Fw, B)
        if m_min is not None:
            S_, fw = self._spmm_struct(), Fw[:, 0]
        for j, (tgt, tgt_row, row, idx, w, fs, fb, sp, a, b) in enumerate(self.fops):
            spmm = m_min is not None and b - a >= m_min
            if not spmm:
                vals = st[idx]                                              # [m, B]
                if Fw is not None:
                    vals = vals * Fw[a:b]
                else:
                    if lH is not None:
                        vals = vals * torch.exp(lH[fs] - lH[fb])
                    elif T is not None:
                        vals = vals + (T[fs] - T[fb])
                    vals = vals * w.unsqueeze(1)
            if sp is not None:
                sp_row, sp_src, sp_u, sp_w = sp
                arg = st[sp_src] * sp_w.unsqueeze(1) - st[sp_u]
                if keep_masks:
                    masks[j] = (arg > 0).to(dt)
                spv = torch.clamp(arg, min=0.0)
            if spmm:                                                        # st[tgt] = W_op st (product taken before the write)
                st.index_copy_(0, tgt, torch.sparse.mm(self._csr(S_['crow_f'][j], S_['col_f'][a:b], fw[a:b], (len(tgt), F.M)), st))
            else:
                st.index_fill_(0, tgt, 0.0)
                st.index_add_(0, tgt_row, vals)
            if sp is not None:
                st.index_add_(0, tgt[sp_row], spv)
            if j in snap_at:
                snaps.extend([st[V:V + N1].clone()] * snap_at[j])
        out = torch.stack(snaps, 0).permute(2, 0, 1).contiguous()          # [B, n_out, N+1]
        return out, st, masks

    def _adjoint_fused(self, g_out, st, lH, T, masks, Fw, need_glH):
        F, dev, dt = self.fsch, self.dev, self.dtype
        N1, V, B = F.N1, F.V, st.shape[1]
        g_out = g_out.permute(1, 2, 0)                                      # [n_out, N+1, B]
        gst = torch.zeros_like(st)
        glH = torch.zeros_like(lH) if (lH is not None and need_glH) else None
        gT = torch.zeros_like(T) if T is not None else None
        sum_b = (lambda g: g.sum(1, keepdim=True)) if (lH is not None and lH.shape[1] == 1 and B > 1) else (lambda g: g)
        gFw = (torch.empty(self.T_total, B, dtype=dt, device=dev)
               if (Fw is not None and glH is not None and self.T_total * 8 * B <= self.budget) else None)
        m_min = self._spmm_on(Fw, B)
        if m_min is not None:
            S_, fwT = self._spmm_struct(), self._fw_T(Fw)
        k = g_out.shape[0]
        for j in range(len(self.fops) - 1, -1, -1):
            for _ in range(self.snap_after_op.get(j, 0)):
                k -= 1
                gst[V:V + N1] += g_out[k]
            tgt, tgt_row, row, idx, w, fs, fb, sp, a, b = self.fops[j]
            g_acc = gst[tgt]
            gst.index_fill_(0, tgt, 0.0)
            if sp is not None:
                sp_row, sp_src, sp_u, sp_w = sp
                g = g_acc[sp_row] * masks[j]
                gst.index_add_(0, sp_src, g * sp_w.unsqueeze(1))
                gst.index_add_(0, sp_u, -g)
            if b == a:
                continue
            spmm = m_min is not None and b - a >= m_min
            if spmm:                                                        # gst[U_op] += W_op^T g_acc
                U = S_['U'][j]
                gst.index_add_(0, U, torch.sparse.mm(self._csr(S_['crow_t'][j], S_['col_t'][a:b], fwT[a:b], (len(U), len(tgt))), g_acc))
                if glH is None:
                    continue
            gr = g_acc[row]
            if Fw is not None:
                if not spmm:
                    gst.index_add_(0, idx, gr * Fw[a:b])
                if gFw is not None:
                    torch.mul(gr, st[idx], out=gFw[a:b])
                elif glH is not None:
                    gl = sum_b(gr * Fw[a:b] * st[idx])
                    glH.index_add_(0, fs, gl)
                    glH.index_add_(0, fb, -gl)
            elif lH is not None:
                g = gr * (torch.exp(lH[fs] - lH[fb]) * w.unsqueeze(1))
                gst.index_add_(0, idx, g)
                if glH is not None:
                    gl = sum_b(g * st[idx])
                    glH.index_add_(0, fs, gl)
                    glH.index_add_(0, fb, -gl)
            elif T is not None:
                g = gr * w.unsqueeze(1)
                gst.index_add_(0, idx, g)
                gT.index_add_(0, fs, g)
                gT.index_add_(0, fb, -g)
            else:
                gst.index_add_(0, idx, gr * w.unsqueeze(1))
        assert k == 0
        if gFw is not None:
            gl = sum_b(gFw * Fw)
            glH.index_add_(0, self.FS, gl)
            glH.index_add_(0, self.FB, -gl)
        gc0 = gst[F.base['c0']:F.base['c0'] + N1].t()
        gsrc = gst[F.base['src']:F.const_slot].t()
        return glH, gT, gc0, gsrc

    # ---- rows of the Jacobian d out / d kw at one parameter point (fused engine) ----
    def _dlogH_dkw(self, kb, kw, tank_kb):
        """A[(S+1)C + 1, 1] = d logH[s, c] / d kw_c (a chain depends on its own kw only), zero dummy row: forward-mode
        derivative of _log_factors along the direction of ones."""
        import torch.autograd.forward_ad as fwAD
        with fwAD.dual_level():
            logH_d, _ = self._log_factors(kb, fwAD.make_dual(kw, torch.ones_like(kw)), tank_kb)
            tangent = fwAD.unpack_dual(logH_d).tangent
        return self._fused_factors(tangent, None)[0]

    def _adjoint_fused_rows(self, snap, node, st, lH, masks, Fw, A, pre=None):
        """Adjoint of a B = 1 fused forward (state st[M, 1]) for m unit cotangents, cotangent b = 1 at snapshot
        snap[b], node node[b]. The parameter gradient is contracted on the fly with A = d logH / d p (per chain):
        every term contributes gl x (A[fs] - A[fb]) to the chain of fs. Returns gP[C + 1, m].
        pre = (Fw^T values, chain-contraction values, per-op term threshold) selects the SpMM path (see jac_rows_kw)."""
        F, dev, dt = self.fsch, self.dev, self.dtype
        V, C, m = F.V, self.C, len(snap)
        order = torch.argsort(snap)
        bounds = np.searchsorted(snap[order].cpu().numpy(), np.arange(sum(self.snap_after_op.values()) + 1))
        cols, nodes = order, node[order] + V
        one = torch.ones((), dtype=dt, device=dev)
        gst = torch.zeros(F.M, m, dtype=dt, device=dev)
        gP = torch.zeros(C + 1, m, dtype=dt, device=dev)
        big = Fw is not None and self.T_total * 8 * m <= self.budget
        gFw = torch.empty(self.T_total, m, dtype=dt, device=dev) if big else None
        if pre is not None:
            S_, (fwT, Pv, m_min) = self._spmm, pre
            crow_c, ch_u, _, col_c = S_['chains']
        k = len(bounds) - 1
        for j in range(len(self.fops) - 1, -1, -1):
            for _ in range(self.snap_after_op.get(j, 0)):
                k -= 1
                a0, b0 = bounds[k], bounds[k + 1]
                if b0 > a0:
                    gst.index_put_((nodes[a0:b0], cols[a0:b0]), one, accumulate=True)
            tgt, tgt_row, row, idx, w, fs, fb, sp, a, b = self.fops[j]
            g_acc = gst[tgt]
            gst.index_fill_(0, tgt, 0.0)
            if sp is not None:
                sp_row, sp_src, sp_u, sp_w = sp
                g = g_acc[sp_row] * masks[j]
                gst.index_add_(0, sp_src, g * sp_w.unsqueeze(1))
                gst.index_add_(0, sp_u, -g)
            if b == a:
                continue
            if pre is not None and b - a >= m_min:
                U = S_['U'][j]
                gst.index_add_(0, U, torch.sparse.mm(self._csr(S_['crow_t'][j], S_['col_t'][a:b], fwT[a:b], (len(U), len(tgt))), g_acc))
                gP.index_add_(0, ch_u[j], torch.sparse.mm(self._csr(crow_c[j], col_c[a:b], Pv[a:b], (len(ch_u[j]), len(tgt))), g_acc))
                if big:                                        # this op's chain contribution is complete
                    gFw[a:b].zero_()
                continue
            gr = g_acc[row]
            fac = Fw[a:b] if Fw is not None else torch.exp(lH[fs] - lH[fb]) * w.unsqueeze(1)
            gst.index_add_(0, idx, gr * fac)
            if big:
                torch.mul(gr, st[idx], out=gFw[a:b])
            else:
                gP.index_add_(0, fs % C, gr * (fac * st[idx] * (A[fs] - A[fb])))
        assert k == 0
        if big:
            gP.index_add_(0, self.FS % C, gFw * (Fw * (A[self.FS] - A[self.FB])))
        return gP

    def jac_rows_kw(self, rows, kb=None, kw=None, tank_kb=None, c0=None, src=None, chunk=64):
        """Fused engine, one parameter point: J[r] = d out[rows[r]] / d kw (user units) for the flat indices `rows`
        into [n_out, N+1]. One B = 1 forward is shared by all rows; each adjoint sweep carries `chunk` unit
        cotangents and accumulates d/d kw per chain (no [(S+1)C, B] logH gradient). Returns [n_rows, L+1]."""
        if self.engine != 'fused':
            raise NotImplementedError('jac_rows_kw needs the fused engine')
        kb, kw, tank_kb = self._prep(kb, 'kb'), self._prep(kw, 'kw'), self._prep(tank_kb, 'tank_kb')
        c0, src = self._prep(c0, 'c0'), self._prep_src(src)
        if not self.chem:
            raise NotImplementedError('no reaction parameters')
        with torch.no_grad():
            lH, T = self._fused_factors(*self._log_factors(kb, kw, tank_kb))
            Fw = self._fw_all(lH) if self._use_big(lH) else None
            _, st, masks = self._sweep_fused(lH, T, c0 / self.ucf_qual, src, 1, Fw=Fw, keep_masks=True)
        A = self._dlogH_dkw(kb.detach(), kw.detach(), tank_kb.detach())
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.dev)
        L1, N1 = self.net.nlinks + 1, self.N + 1
        J = torch.empty(len(rows), L1, dtype=self.dtype, device=self.dev)
        with torch.no_grad():
            pre, m_min = None, self._spmm_on(Fw, chunk)
            if m_min is not None:                              # per-term values of W_op^T and of the chain contraction, once
                chains = self._spmm_chains()
                Pv = (Fw[:, 0] * st[self.IDX, 0] * (A[self.FS, 0] - A[self.FB, 0])).index_select(0, chains[2])
                pre = (self._fw_T(Fw), Pv, m_min)
            for r0 in range(0, len(rows), chunk):
                rr = rows[r0:r0 + chunk]
                J[r0:r0 + len(rr)] = self._adjoint_fused_rows(rr // N1, rr % N1, st, lH, masks, Fw, A, pre)[:L1].t()
        return J * self.ucf_qual

    # ---- forward sweep over the recorded ops (shared by both engines) ----
    def _sweep(self, logH, T, c0i, src, B, keep_masks=False):
        """Returns (snapshots [B, n_out, N+1] internal units, final version vector x, setpoint masks)."""
        dev, dt = self.dev, self.dtype
        x = torch.zeros(B, self.V, dtype=dt, device=dev)
        u = torch.zeros(B, self.N + 1, dtype=dt, device=dev)
        uflow = torch.zeros(B, self.N + 1, dtype=dt, device=dev)
        snaps, snap_at, masks = [], self.snap_after_op, {}
        for j, (step, tkind, tgt, ntgt, terms) in enumerate(self.ops):
            acc = torch.zeros(B, ntgt, dtype=dt, device=dev)
            if 'ver' in terms:
                d = terms['ver']
                vals = x[:, d['idx']]
                if logH is not None:
                    vals = vals * torch.exp(logH[:, d['flat_s']] - logH[:, d['flat_b']])
                elif T is not None:
                    vals = vals + (T[:, d['flat_s']] - T[:, d['flat_b']])
                acc.index_add_(1, d['row'], vals * d['w'])
            for kind, vec in (('u', u), ('uflow', uflow), ('c0', c0i), ('src', src)):
                if kind in terms:
                    d = terms[kind]
                    acc.index_add_(1, d['row'], vec[:, d['idx']] * d['w'])
            if 'setpoint' in terms:
                d = terms['setpoint']
                arg = src[:, d['idx']] * d['w'] - u[:, d['idx'] % (self.N + 1)]
                if keep_masks:
                    masks[j] = (arg > 0).to(dt)
                acc.index_add_(1, d['row'], torch.clamp(arg, min=0.0))
            if 'const' in terms:
                d = terms['const']
                acc.index_add_(1, d['row'], d['w'].unsqueeze(0).expand(B, -1))
            target = x if tkind == 'ver' else (u if tkind == 'u' else uflow)
            target.index_copy_(1, tgt, acc)
            if j in snap_at:
                snaps.extend([u.clone()] * snap_at[j])
        return torch.stack(snaps, dim=1), x, masks

    # ---- reverse sweep: exact adjoint of _sweep, O(terms) time, no autograd graph ----
    def _adjoint(self, g_out, x, logH, T, masks):
        B = x.shape[0]
        dev, dt = self.dev, self.dtype
        gx = torch.zeros_like(x)
        gu = torch.zeros(B, self.N + 1, dtype=dt, device=dev)
        guf = torch.zeros_like(gu); gc0 = torch.zeros_like(gu)
        gsrc = torch.zeros(B, self.K * (self.N + 1), dtype=dt, device=dev)
        glogH = torch.zeros_like(logH) if logH is not None else None
        gT = torch.zeros_like(T) if T is not None else None
        k = g_out.shape[1]                                             # next snapshot slot (from the end)
        for j in range(len(self.ops) - 1, -1, -1):
            for _ in range(self.snap_after_op.get(j, 0)):              # snapshots read u after op j
                k -= 1
                gu += g_out[:, k, :]
            step, tkind, tgt, ntgt, terms = self.ops[j]
            G = gx if tkind == 'ver' else (gu if tkind == 'u' else guf)
            g_acc = G[:, tgt].clone()
            G.index_fill_(1, tgt, 0.0)                                 # value overwritten here: older copy gets nothing
            if 'ver' in terms:
                d = terms['ver']
                g = g_acc[:, d['row']] * d['w']
                if logH is not None:
                    F = torch.exp(logH[:, d['flat_s']] - logH[:, d['flat_b']])
                    gx.index_add_(1, d['idx'], g * F)
                    gl = g * F * x[:, d['idx']]
                    glogH.index_add_(1, d['flat_s'], gl)
                    glogH.index_add_(1, d['flat_b'], -gl)
                elif T is not None:
                    gx.index_add_(1, d['idx'], g)
                    gT.index_add_(1, d['flat_s'], g)
                    gT.index_add_(1, d['flat_b'], -g)
                else:
                    gx.index_add_(1, d['idx'], g)
            for kind, Gv in (('u', gu), ('uflow', guf), ('c0', gc0), ('src', gsrc)):
                if kind in terms:
                    d = terms[kind]
                    Gv.index_add_(1, d['idx'], g_acc[:, d['row']] * d['w'])
            if 'setpoint' in terms:
                d = terms['setpoint']
                g = g_acc[:, d['row']] * masks[j]
                gsrc.index_add_(1, d['idx'], g * d['w'])
                gu.index_add_(1, d['idx'] % (self.N + 1), -g)
        assert k == 0
        return glogH, gT, gc0, gsrc


class _ReplayFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eng, lH, T, c0i, src, Fw):
        B = c0i.shape[0]
        if Fw is None and T is None and eng._use_big(lH):
            Fw = eng._fw_all(lH)
        out, st, masks = eng._sweep_fused(lH, T, c0i, src, B, Fw=Fw, keep_masks=True)
        ctx.eng, ctx.masks = eng, masks
        ctx.has = (lH is not None, T is not None, Fw is not None)
        ctx.need_glH = bool(ctx.needs_input_grad[1])
        ctx.save_for_backward(st, *(t for t in (lH, T, Fw) if t is not None))
        return out

    @staticmethod
    def backward(ctx, g_out):
        st, *rest = ctx.saved_tensors
        lH = rest.pop(0) if ctx.has[0] else None
        T = rest.pop(0) if ctx.has[1] else None
        Fw = rest.pop(0) if ctx.has[2] else None
        glH, gT, gc0, gsrc = ctx.eng._adjoint_fused(g_out.contiguous(), st, lH, T, ctx.masks, Fw, ctx.need_glH)
        return None, glH, gT, gc0, gsrc, None


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eng, logH, T, c0i, src):
        B = c0i.shape[0]
        out, x, masks = eng._sweep(logH, T, c0i, src, B, keep_masks=True)
        ctx.eng, ctx.masks = eng, masks
        ctx.has = (logH is not None, T is not None)
        ctx.save_for_backward(x, *(t for t in (logH, T) if t is not None))
        return out

    @staticmethod
    def backward(ctx, g_out):
        x, *rest = ctx.saved_tensors
        logH = rest.pop(0) if ctx.has[0] else None
        T = rest.pop(0) if ctx.has[1] else None
        glogH, gT, gc0, gsrc = ctx.eng._adjoint(g_out.contiguous(), x, logH, T, ctx.masks)
        return None, glogH, gT, gc0, gsrc
