"""Time-chunked replay of the fused schedule (reviewer 1, comment 5). New module; torch_wq.py and its results untouched.

The fused engine (torch_wq.TorchWQ, engine='fused') keeps one state slot per segment version written over the whole
horizon, so its state st[M, B] grows with the horizon, and an adjoint pass with B columns holds two such arrays. Here the
recorded ops are cut into chunks of about equal term counts (cuts between steps only). Chunk c holds
    L_c = live-in(c) + written(c),   live-in(c) = versions written before chunk c and read in chunk c or later,
in that order, followed by the fixed block of u, uflow, c0, src and the constant (the layout of fuse.py), so the state has
max_c |L_c| + (3 + K) N1 + 1 slots: it no longer grows with the horizon once the chunks are short. The forward runs the
chunks in time order and hands live-in(c+1) on; the adjoint runs them in reverse and hands the cotangents of live-in(c+1)
back. Reaction gradients need the values of the versions a chunk reads: the forward then checkpoints live-in(c), u and
uflow at the start of every chunk, and the adjoint recomputes each chunk from its checkpoint before reversing it (one
extra forward). The rows of the sensor response of an affine source map (source identification, frozen reaction
factors) need no forward at all (rows_src), and a batch of rows starts its reverse sweep at the op of its last snapshot.
Per-op arithmetic is that of the fused engine's gather / index_add_ path with the terms in the same order, so the results
agree with it to the floating-point reassociation of the atomic accumulation (and with its SpMM path to summation order).
"""
import numpy as np
import torch

from .torch_wq import TorchWQ
from .fuse import FusedSchedule


class ChunkedWQ(TorchWQ):
    """Fused engine with a time-chunked state. n_chunks, or chunk_terms (terms per chunk), sets the cut; one chunk replays
    the fused engine's arithmetic on a compacted version state."""

    def __init__(self, net, schedule, reactflag, device='cpu', dtype=torch.float64, engine='fused', flatten=True,
                 n_chunks=None, chunk_terms=None, **kw):
        if engine != 'fused':
            raise ValueError('ChunkedWQ replays the fused schedule')
        self.n_chunks_req, self.chunk_terms = n_chunks, chunk_terms
        kw['spmm_min_batch'] = 0                                   # gather / index_add_ path only
        super().__init__(net, schedule, reactflag, device=device, dtype=dtype, engine='fused', flatten=flatten, **kw)

    # ---- layout ----
    @staticmethod
    def _remap(a, loc, V, gofs):
        out = a + gofs                                             # non-version slots: fixed block after the versions
        v = a < V
        if v.any():
            lv = loc[a[v]]
            if (lv < 0).any():
                raise RuntimeError('a version is read or written outside the layout of its chunk')
            out[v] = lv
        return out

    def _build_fused(self, schedule, flatten):
        F = self.fsch = FusedSchedule(schedule, flatten=flatten)
        dev, dt = self.dev, self.dtype
        V = F.V
        ops = F.ops
        for op in ops:                                             # row-major terms, as in TorchWQ._build_fused
            p = np.argsort(op.row, kind='stable')
            op.row, op.idx, op.w, op.fs, op.fb = (a[p] for a in (op.row, op.idx, op.w, op.fs, op.fb))
        n_ops = len(ops)
        lens = np.array([len(op.idx) for op in ops], dtype=np.int64)
        offs = np.concatenate(([0], np.cumsum(lens)))
        self.T_total = int(offs[-1])
        self.max_op_terms = int(lens.max()) if n_ops else 0
        # ---- chunks: contiguous op ranges, cut between steps at about equal term counts ----
        steps = np.array([op.step for op in ops], dtype=np.int64)
        first = np.flatnonzero(np.r_[True, steps[1:] != steps[:-1]]) if n_ops else np.zeros(0, np.int64)
        if self.n_chunks_req:
            nc = int(self.n_chunks_req)
        elif self.chunk_terms:
            nc = -(-self.T_total // int(self.chunk_terms))
        else:
            nc = 1
        nc = max(1, min(nc, len(first)))
        cuts = sorted({int(first[np.argmin(np.abs(offs[first] - self.T_total * c / nc))]) for c in range(1, nc)} - {0})
        bounds = [0] + cuts + [n_ops]
        nc = len(bounds) - 1
        self.chunk_ops = [(bounds[c], bounds[c + 1]) for c in range(nc)]
        chunk_of = np.repeat(np.arange(nc), np.diff(bounds))
        # ---- versions: chunk (and op) writing each one, first and last reads ----
        w_op = np.full(V, -1, dtype=np.int64)
        r_first = np.full(V, n_ops, dtype=np.int64)
        r_c = np.full(V, -1, dtype=np.int64)
        for j, op in enumerate(ops):
            w_op[op.tgt[op.tgt < V]] = j
            r_c[op.idx[op.idx < V]] = chunk_of[j]                  # chunks are non-decreasing in j: the last read wins
        for j in range(n_ops - 1, -1, -1):                          # reverse order: the first read wins
            r_first[ops[j].idx[ops[j].idx < V]] = j
        read = r_c >= 0
        if np.any(read & (w_op >= 0) & (r_first <= w_op)):
            raise RuntimeError('a version is read before it is written')
        w_c = np.where(w_op >= 0, chunk_of[np.maximum(w_op, 0)], -1)
        live_in = [np.flatnonzero((w_c < c) & (r_c >= c)) for c in range(nc)]
        written = [np.flatnonzero(w_c == c) for c in range(nc)]
        sizes = [len(a) + len(b) for a, b in zip(live_in, written)]
        Vb = max(sizes) if sizes else 0
        gofs = Vb - V
        self.Vb, self.M_loc = Vb, Vb + (F.M - V)
        self.live_n, self.L_n = [len(a) for a in live_in], sizes
        loc = np.full(V, -1, dtype=np.int64)
        tgt_l, tr_l, idx_l, sp_l, hand = [], [], [], [], []
        for c in range(nc):
            L = np.concatenate((live_in[c], written[c]))
            loc[L] = np.arange(len(L))
            for j in range(bounds[c], bounds[c + 1]):
                op = ops[j]
                t = self._remap(op.tgt, loc, V, gofs)
                tgt_l.append(t); tr_l.append(t[op.row]); idx_l.append(self._remap(op.idx, loc, V, gofs))
                sp_l.append(None if op.sp is None else (op.sp[0], op.sp[1] + gofs, op.sp[2] + gofs, op.sp[3]))
            if c + 1 < nc:
                hand.append(loc[live_in[c + 1]])
            loc[L] = -1
        cat = lambda xs, dtp: torch.as_tensor(np.concatenate(xs) if xs else np.zeros(0), dtype=dtp, device=dev)
        self.IDX = cat(idx_l, torch.long)
        self.FS, self.FB = cat([op.fs for op in ops], torch.long), cat([op.fb for op in ops], torch.long)
        self.W = cat([op.w for op in ops], dt)
        TGT, TR, ROW = cat(tgt_l, torch.long), cat(tr_l, torch.long), cat([op.row for op in ops], torch.long)
        ls = lens.tolist()
        idx_v, fs_v, fb_v, w_v, tr_v, row_v = (torch.split(x, ls) for x in (self.IDX, self.FS, self.FB, self.W, TR, ROW))
        tgt_v = torch.split(TGT, [len(op.tgt) for op in ops])
        self.fops = []
        for j in range(n_ops):
            sp = None
            if sp_l[j] is not None:
                sp = tuple(torch.as_tensor(a, dtype=(dt if i == 3 else torch.long), device=dev) for i, a in enumerate(sp_l[j]))
            self.fops.append((tgt_v[j], tr_v[j], row_v[j], idx_v[j], w_v[j], fs_v[j], fb_v[j], sp, int(offs[j]), int(offs[j + 1])))
        self.hand = [torch.as_tensor(h, dtype=torch.long, device=dev) for h in hand]
        self.snap_after_op = self._snapshot_map(steps.tolist(), schedule.snapshots)
        cnt = np.zeros(n_ops, dtype=np.int64)
        for j, k in self.snap_after_op.items():
            if j >= 0:
                cnt[j] = k
        self.snap_cum = np.cumsum(cnt)                             # snapshots taken after ops 0..j
        self.snap_op = np.repeat(np.arange(n_ops), cnt)            # op after which snapshot k is taken
        self.chunk_of_op = chunk_of
        self.lb = {k: v + gofs for k, v in F.base.items() if k != 'ver'}
        self.l_const = F.const_slot + gofs
        self.has_sp = any(s is not None for s in sp_l)

    def chunk_info(self):
        return {'n_chunks': len(self.chunk_ops), 'version_slots_max': int(self.Vb), 'state_slots': int(self.M_loc),
                'state_slots_unchunked': int(self.fsch.M), 'live_in_max': int(max(self.live_n)),
                'terms': int(self.T_total), 'ops': len(self.fops)}

    # ---- forward ----
    def forward(self, kb=None, kw=None, tank_kb=None, c0=None, src=None):
        """Returns node quality snapshots [B, n_out, N+1] in user units (as TorchWQ.forward)."""
        c0 = self._prep(c0, 'c0')
        src = self._prep_src(src)
        if self.fixed is not None and kb is None and kw is None and tank_kb is None:
            B = max(c0.shape[0], src.shape[0])
            lH, T, Fw = self.fixed
            return _ReplayChunked.apply(self, lH, T, c0.expand(B, -1) / self.ucf_qual, src.expand(B, -1), Fw) * self.ucf_qual
        kb, kw, tank_kb = self._prep(kb, 'kb'), self._prep(kw, 'kw'), self._prep(tank_kb, 'tank_kb')
        B = max(v.shape[0] for v in (kb, kw, tank_kb, c0, src))
        Bt = max(v.shape[0] for v in (kb, kw, tank_kb))
        kb, kw, tank_kb = (v.expand(Bt, -1) for v in (kb, kw, tank_kb))
        lH, T = self._fused_factors(*self._log_factors(kb, kw, tank_kb))
        return _ReplayChunked.apply(self, lH, T, c0.expand(B, -1) / self.ucf_qual, src.expand(B, -1), None) * self.ucf_qual

    def _put_params(self, st, c0i, src):
        N1 = self.fsch.N1
        st[self.lb['c0']:self.lb['c0'] + N1] = c0i.t()
        st[self.lb['src']:self.l_const] = src.t()
        st[self.l_const] = 1.0

    def _run_op(self, st, j, lH, T, Fw, masks=None):
        tgt, tgt_row, row, idx, w, fs, fb, sp, a, b = self.fops[j]
        vals = st[idx]                                             # [m, B]
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
            if masks is not None:
                masks[j] = (arg > 0).to(self.dtype)
            spv = torch.clamp(arg, min=0.0)
        st.index_fill_(0, tgt, 0.0)
        st.index_add_(0, tgt_row, vals)
        if sp is not None:
            st.index_add_(0, tgt[sp_row], spv)

    def _sweep_chunked(self, lH, T, c0i, src, B, Fw=None, keep_masks=False, checkpoint=False):
        """Returns (snapshots [B, n_out, N+1] internal units, final state, setpoint masks, checkpoints)."""
        N1, u0 = self.fsch.N1, self.lb['u']
        st = torch.zeros(self.M_loc, B, dtype=self.dtype, device=self.dev)
        self._put_params(st, c0i, src)
        snaps, masks, ckpt = [], ({} if keep_masks else None), []
        for c, (j0, j1) in enumerate(self.chunk_ops):
            if c:
                h = self.hand[c - 1]
                st[:len(h)] = st[h]                                # live-in(c) to the front of chunk c's layout
            if checkpoint:
                ckpt.append((st[:self.live_n[c]].clone(), st[u0:u0 + 2 * N1].clone()))
            for j in range(j0, j1):
                self._run_op(st, j, lH, T, Fw, masks)
                if j in self.snap_after_op:
                    snaps.extend([st[u0:u0 + N1].clone()] * self.snap_after_op[j])
        out = torch.stack(snaps, 0).permute(2, 0, 1).contiguous()
        return out, st, (masks if masks is not None else {}), ckpt

    # ---- adjoint ----
    def _op_adjoint(self, gst, j, g_acc, st_f, lH, T, Fw, masks, glH, gT, sum_b):
        tgt, tgt_row, row, idx, w, fs, fb, sp, a, b = self.fops[j]
        if sp is not None:
            sp_row, sp_src, sp_u, sp_w = sp
            g = g_acc[sp_row] * masks[j]
            gst.index_add_(0, sp_src, g * sp_w.unsqueeze(1))
            gst.index_add_(0, sp_u, -g)
        if b == a:
            return
        gr = g_acc[row]
        if Fw is not None:
            gst.index_add_(0, idx, gr * Fw[a:b])
            if glH is not None:
                gl = sum_b(gr * Fw[a:b] * st_f[idx])
                glH.index_add_(0, fs, gl)
                glH.index_add_(0, fb, -gl)
        elif lH is not None:
            g = gr * (torch.exp(lH[fs] - lH[fb]) * w.unsqueeze(1))
            gst.index_add_(0, idx, g)
            if glH is not None:
                gl = sum_b(g * st_f[idx])
                glH.index_add_(0, fs, gl)
                glH.index_add_(0, fb, -gl)
        elif T is not None:
            g = gr * w.unsqueeze(1)
            gst.index_add_(0, idx, g)
            gT.index_add_(0, fs, g)
            gT.index_add_(0, fb, -g)
        else:
            gst.index_add_(0, idx, gr * w.unsqueeze(1))

    def _hand_back(self, gst, c):
        """Cotangents of live-in(c+1) (front of chunk c+1's layout) to their slots in chunk c's layout; every other
        version slot is zero at this point (written versions are cleared when their write is reversed)."""
        h = self.hand[c]
        tmp = gst[:len(h)].clone()
        gst[:len(h)] = 0.0
        gst.index_copy_(0, h, tmp)

    def _adjoint_chunked(self, g_out, lH, T, masks, Fw, need_glH, ckpt, c0i, src):
        N1, u0 = self.fsch.N1, self.lb['u']
        g_out = g_out.permute(1, 2, 0)                             # [n_out, N+1, B]
        B = g_out.shape[2]
        gst = torch.zeros(self.M_loc, B, dtype=self.dtype, device=self.dev)
        glH = torch.zeros_like(lH) if (lH is not None and need_glH) else None
        gT = torch.zeros_like(T) if T is not None else None
        sum_b = (lambda g: g.sum(1, keepdim=True)) if (lH is not None and lH.shape[1] == 1 and B > 1) else (lambda g: g)
        st_f = None
        if glH is not None:                                        # forward values of the versions each chunk reads
            st_f = torch.zeros(self.M_loc, max(B, c0i.shape[0]), dtype=self.dtype, device=self.dev)
            self._put_params(st_f, c0i, src)
        k = g_out.shape[0]
        nc = len(self.chunk_ops)
        for c in range(nc - 1, -1, -1):
            j0, j1 = self.chunk_ops[c]
            if c < nc - 1:
                self._hand_back(gst, c)
            if st_f is not None:
                lv, uu = ckpt[c]
                st_f[:len(lv)] = lv
                st_f[u0:u0 + 2 * N1] = uu
                for j in range(j0, j1):
                    self._run_op(st_f, j, lH, T, Fw)
            for j in range(j1 - 1, j0 - 1, -1):
                for _ in range(self.snap_after_op.get(j, 0)):
                    k -= 1
                    gst[u0:u0 + N1] += g_out[k]
                tgt = self.fops[j][0]
                g_acc = gst[tgt]
                gst.index_fill_(0, tgt, 0.0)
                self._op_adjoint(gst, j, g_acc, st_f, lH, T, Fw, masks, glH, gT, sum_b)
        assert k == 0
        gc0 = gst[self.lb['c0']:self.lb['c0'] + N1].t()
        gsrc = gst[self.lb['src']:self.l_const].t()
        return glH, gT, gc0, gsrc

    # ---- rows of the sensor response of the affine source map ----
    def rows_src(self, snap, node):
        """J[b] = d out[snap[b], node[b]] / d src [m, K, N+1] (user units) for the affine source map (frozen reaction
        factors by set_fixed_reaction, or no reaction): adjoint only, no forward; the reverse sweep starts at the op
        of the last snapshot of the batch (later ops carry no cotangent)."""
        if self.has_sp:
            raise NotImplementedError('setpoint sources make the source map nonlinear')
        if self.chem and self.fixed is None:
            raise RuntimeError('rows_src needs set_fixed_reaction')
        lH, T, Fw = self.fixed if self.fixed is not None else (None, None, None)
        if T is not None:
            raise NotImplementedError('AGE has no source map')
        N1, K, dev, dt = self.fsch.N1, self.K, self.dev, self.dtype
        snap = torch.as_tensor(snap, dtype=torch.long, device=dev)
        node = torch.as_tensor(node, dtype=torch.long, device=dev)
        m = len(snap)
        gst = getattr(self, '_rows_buf', None)
        if gst is None or gst.shape[1] != m:
            self._rows_buf = gst = None
            self._rows_buf = gst = torch.empty(self.M_loc, m, dtype=dt, device=dev)
        gst.zero_()
        order = torch.argsort(snap)
        s_sorted = snap[order].cpu().numpy()
        bounds = np.searchsorted(s_sorted, np.arange(len(self.snap_op) + 1))
        cols, rows_u = order, node[order] + self.lb['u']
        one = torch.ones((), dtype=dt, device=dev)
        j_start = int(self.snap_op[int(s_sorted[-1])])
        c_start = int(self.chunk_of_op[j_start])
        k = int(self.snap_cum[j_start])
        ident = lambda g: g
        for c in range(c_start, -1, -1):
            j0, j1 = self.chunk_ops[c]
            if c < c_start:
                self._hand_back(gst, c)
            for j in range(j_start if c == c_start else j1 - 1, j0 - 1, -1):
                for _ in range(self.snap_after_op.get(j, 0)):
                    k -= 1
                    a0, b0 = bounds[k], bounds[k + 1]
                    if b0 > a0:
                        gst.index_put_((rows_u[a0:b0], cols[a0:b0]), one, accumulate=True)
                tgt = self.fops[j][0]
                g_acc = gst[tgt]
                gst.index_fill_(0, tgt, 0.0)
                self._op_adjoint(gst, j, g_acc, None, lH, None, Fw, None, None, None, ident)
        assert k == 0
        return gst[self.lb['src']:self.l_const].t().reshape(m, K, N1) * self.ucf_qual


class _ReplayChunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eng, lH, T, c0i, src, Fw):
        B = max(c0i.shape[0], src.shape[0])
        if Fw is None and T is None and eng._use_big(lH):
            Fw = eng._fw_all(lH)
        need = lH is not None and bool(ctx.needs_input_grad[1])
        out, _, masks, ckpt = eng._sweep_chunked(lH, T, c0i, src, B, Fw=Fw, keep_masks=True, checkpoint=need)
        ctx.eng, ctx.masks, ctx.ckpt, ctx.need_glH = eng, masks, ckpt, need
        ctx.has = (lH is not None, T is not None, Fw is not None)
        ctx.save_for_backward(c0i, src, *(t for t in (lH, T, Fw) if t is not None))
        return out

    @staticmethod
    def backward(ctx, g_out):
        c0i, src, *rest = ctx.saved_tensors
        lH = rest.pop(0) if ctx.has[0] else None
        T = rest.pop(0) if ctx.has[1] else None
        Fw = rest.pop(0) if ctx.has[2] else None
        glH, gT, gc0, gsrc = ctx.eng._adjoint_chunked(g_out.contiguous(), lH, T, ctx.masks, Fw, ctx.need_glH, ctx.ckpt, c0i, src)
        ctx.ckpt = None
        return None, glH, gT, gc0, gsrc, None
