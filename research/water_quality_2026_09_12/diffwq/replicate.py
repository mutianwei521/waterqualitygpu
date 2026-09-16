"""Pure-Python replication of EPANET 2.3's water quality engine
(quality.c, qualroute.c, qualreact.c), operation for operation.

Hydraulics come from the engine's binary hydraulics file (float32 values, flows of
closed links zeroed), exactly as quality.c runqual() -> output.c readhyd() sees them.
Tank volume `Tank[i].V` is NOT updated by the quality engine in that mode; it keeps
the value left by the hydraulic run (used only by FIFO/LIFO overflow tests).

Modes: 'nextQ' reproduces EN_solveQ / EN_runQ+EN_nextQ (dt = min(Qstep, remaining
hydraulic period)); 'stepQ' reproduces EN_runQ+EN_stepQ (fixed Qstep, split at
hydraulic events).
"""
from collections import Counter, deque
import math
import platform
import numpy as np
from . import epanet_api as api
from .network import LperFT3, SECperDAY, GPMperCFS
from .recorder import ScheduleRecorder

Q_STAGNANT = 0.005 / GPMperCFS
BIG, TINY = 1.0e10, 1.0e-6
CLOSED = 2
# types.h: PI = M_PI when math.h defines it (gcc/glibc), else 3.141592654 (MSVC build)
PI_MSVC = 3.141592654


def SGN(x):
    return -1 if x < 0 else 1


class Seg:
    __slots__ = ('v', 'c', 'id')

    def __init__(self, v, c, sid):
        self.v = v
        self.c = c
        self.id = sid


class Replicator:
    def __init__(self, net, trace, qstep=None, ctol=None, pi=None, record=False, src_slots=None, virtual_sources=None):
        """src_slots = K: record 'src' terms per time slot k = ((Qtime + Pstart) // Pstep) % K with the
        pattern factor left to the parameter (junction / tank sources; reservoir sources stay folded).
        virtual_sources: {node: source type} (or iterable of nodes, MASS) of junctions / tanks without a
        source: their zero-strength 'src' terms are recorded so the replay is differentiable in them.
        Neither option changes the replicated values."""
        self.net = net
        self.trace = trace
        self.qstep = int(qstep) if qstep is not None else int(net.qstep)
        self.ctol = float(ctol) if ctol is not None else float(net.ctol)
        if pi is None:
            pi = PI_MSVC if platform.system() == 'Windows' else math.pi
        self.pi = pi
        self.record = bool(record)    # when True, run() logs a transport Schedule (self.rec)
        self.rec = None
        n = net
        self.K = int(src_slots) if src_slots is not None else None
        if virtual_sources is None:
            virtual_sources = {}
        elif not isinstance(virtual_sources, dict):
            virtual_sources = {int(i): api.EN_MASS for i in virtual_sources}
        self.virtual_sources = {int(i): int(t) for i, t in virtual_sources.items()}
        if self.virtual_sources and (not self.record or self.K is None):
            raise ValueError('virtual_sources need record=True and src_slots')
        for i, t in self.virtual_sources.items():
            if i in n.sources or n.node_type[i] == api.EN_RESERVOIR or t not in (api.EN_MASS, api.EN_CONCEN, api.EN_FLOWPACED):
                raise ValueError(f'virtual source at node {i}: must be a junction / tank without a source, type MASS/CONCEN/FLOWPACED')
        if self.K is not None:
            for i, (stype, c0, pat) in n.sources.items():
                if pat != 0 and n.node_type[i] != api.EN_RESERVOIR and self.K % len(n.patterns[pat]) != 0:
                    raise ValueError(f'src_slots={self.K} is not a multiple of the pattern length {len(n.patterns[pat])} (node {i})')
        self.nn, self.nl, self.nt, self.nj = n.nnodes, n.nlinks, n.ntanks, n.njuncs
        self.nchains = self.nl + self.nt + 1
        self.linkvol = np.zeros(self.nl + 1)
        for k in range(1, self.nl + 1):
            self.linkvol[k] = 0.785398 * n.length[k] * (n.diam[k] * n.diam[k])
        self.bucf = self.getucf(n.bulkorder)
        self.tucf = self.getucf(n.tankorder)
        self.sc = n.viscos / n.diffus if n.diffus > 0.0 else 0.0
        self.reactflag = self.setreactflag()
        self.reset()

    # ---------------- qualreact.c helpers ----------------
    @staticmethod
    def getucf(order):
        if order < 0.0:
            order = 0.0
        if order == 1.0:
            return 1.0
        return 1.0 / math.pow(LperFT3, order - 1.0)

    def setreactflag(self):
        n = self.net
        if n.qualflag == api.EN_TRACE:
            return 0
        if n.qualflag == api.EN_AGE:
            return 1
        for k in range(1, self.nl + 1):
            if n.link_type[k] <= api.EN_PIPE and (n.kb[k] != 0.0 or n.kw[k] != 0.0):
                return 1
        for i in range(1, self.nt + 1):
            if n.tank_kb[i] != 0.0:
                return 1
        return 0

    # ---------------- quality.c initqual / initsegs ----------------
    def reset(self):
        n = self.net
        self.nodequal = np.array(n.c0, dtype=np.float64)
        if n.qualflag == api.EN_TRACE:
            self.nodequal[:] = 0.0
        self.tank_c = np.zeros(self.nt + 1)
        for i in range(1, self.nt + 1):
            self.tank_c[i] = self.nodequal[n.tank_node[i]]
        if n.qualflag == api.EN_TRACE:
            self.nodequal[n.tracenode] = 100.0
        self.tank_v = np.array(n.tank_v_hyd_final, dtype=np.float64)   # stale, see module doc
        self.chains = [deque() for _ in range(self.nchains)]
        self.next_id = 0
        self.seg_link = []            # segment id -> chain index (link k or nlinks + tank i)
        self.segcount = 0
        self.flowdir = np.zeros(self.nl + 1, dtype=np.int64)
        self.sorted_nodes = np.zeros(self.nn + 1, dtype=np.int64)
        self.rc = np.zeros(self.nl + 1)
        self.kf = np.zeros(self.nl + 1)
        self.linkflow = np.zeros(self.nl + 1)
        self.linkstatus = np.zeros(self.nl + 1, dtype=np.int64)
        self.demand = np.zeros(self.nn + 1)
        self.sourcequal = 0.0
        self.qtime = 0
        self.htime = 0
        self.period_index = 0
        self.cur_period = 0
        self.src_term = None
        self.mb_inflow = self.mb_outflow = self.mb_reacted = 0.0
        self.mb_ratio = 0.0
        self.hits = Counter()         # data-dependent branch coverage (see verify_m0)
        self.rec = ScheduleRecorder(self.nn, self.nl, self.nt, self.K or 1) if self.record else None
        self.flow_ref = None
        if self.rec is not None:      # initial node quality (step 0 of the schedule)
            for i in range(1, self.nn + 1):
                if n.qualflag == api.EN_TRACE:
                    self.rec.set_u(i, [('const', 0, 100.0 if i == n.tracenode else 0.0)])
                else:
                    self.rec.set_u(i, [('c0', i, 1.0)])
        # initsegs
        for k in range(1, self.nl + 1):
            if n.link_type[k] == api.EN_PIPE:
                self.addseg(k, self.linkvol[k], self.nodequal[n.n2[k]], [('u', int(n.n2[k]), 1.0)])
        for j in range(1, self.nt + 1):
            if n.tank_is_reservoir[j]:
                continue
            k = n.tank_node[j]
            c = n.c0[k]
            v = n.tank_v0[j]
            terms = [('c0', int(k), 1.0)]
            k = self.nl + j
            self.addseg(k, v, c, terms)
            if n.tank_mix[j] == api.EN_MIX2:
                v1 = max(0.0, v - n.tank_v1frac[j] * n.tank_vmax[j])
                self.chains[k][0].v = v1
                v = v - v1
                self.addseg(k, v, c, terms)
        self.mb_initial = self.findstoredmass()

    def addseg(self, k, v, c, terms=None):
        seg = Seg(v, c, self.next_id)
        self.next_id += 1
        self.seg_link.append(k)
        self.chains[k].append(seg)
        self.segcount += 1
        if self.rec is not None:
            self.rec.new_seg(seg, k, terms)
        return seg

    def reversesegs(self, k):
        self.chains[k].reverse()

    def findstoredmass(self):
        total = 0.0
        for k in range(1, self.nl + 1):
            for seg in self.chains[k]:
                total += seg.c * seg.v
        for i in range(1, self.nt + 1):
            if self.net.tank_is_reservoir[i]:
                continue
            for seg in self.chains[self.nl + i]:
                total += seg.c * seg.v
        return total

    # ---------------- quality.c runqual: load hydraulics for a period ----------------
    def linkflow_of(self, k):
        return 0.0 if self.linkstatus[k] <= CLOSED else self.linkflow[k]

    def load_period(self, p):
        per = self.trace.periods[p]
        self.cur_period = p
        self.linkflow = per.flow.tolist()
        self.linkstatus = per.status.tolist()
        self.demand = per.demand.tolist()
        if self.net.qualflag != api.EN_NONE and self.qtime < self.net.duration:
            if self.reactflag and self.net.qualflag != api.EN_AGE:
                self.ratecoeffs()
            if self.flowdirchanged():
                self.sortnodes()
        if self.rec is not None:      # hydraulics-only wall mass-transfer coefficients, all pipes
            n = self.net
            kf = [0.0] + [self.masstransfer(k) if (n.link_type[k] <= api.EN_PIPE and n.diam[k] > 0.0) else 0.0
                          for k in range(1, self.nl + 1)]
            self.rec.set_period_kf(p, kf, self.sc == 0.0)

    def ratecoeffs(self):
        n = self.net
        for k in range(1, self.nl + 1):
            kw = n.kw[k]
            if kw != 0.0:
                kw = self.piperate(k)
            self.rc[k] = kw

    def masstransfer(self, k):
        """Hydraulics-only part of piperate(): the wall mass-transfer coefficient kf."""
        n = self.net
        d = n.diam[k]
        if self.sc == 0.0:
            return BIG
        q = self.linkflow_of(k)
        a = self.pi * d * d / 4.0
        u = abs(q) / a
        Re = u * d / n.viscos
        if Re < 1.0:
            Sh = 2.0
        elif Re >= 2300.0:
            Sh = 0.0149 * math.pow(Re, 0.88) * math.pow(self.sc, 0.333)
        else:
            y = d / n.length[k] * Re * self.sc
            Sh = 3.65 + 0.0668 * y / (1.0 + 0.04 * math.pow(y, 0.667))
        return Sh * n.diffus / d

    def piperate(self, k):
        n = self.net
        d = n.diam[k]
        kf = self.masstransfer(k)
        self.kf[k] = kf
        if self.sc == 0.0:
            if n.wallorder == 0.0:
                return BIG
            return n.kw[k] * (4.0 / d) / n.ucf['ELEV']
        if n.wallorder == 0.0:
            return kf
        kw = n.kw[k] / n.ucf['ELEV']
        kw = (4.0 / d) * kw * kf / (kf + abs(kw))
        return kw

    def flowdirchanged(self):
        result = False
        fd = self.flowdir
        for k in range(1, self.nl + 1):
            olddir = fd[k]
            q = self.linkflow_of(k)
            newdir = SGN(q)
            if abs(q) < Q_STAGNANT:
                newdir = 0
            if newdir * olddir < 0:
                self.reversesegs(k)
            if newdir != olddir:
                result = True
            fd[k] = newdir
        return result

    def sortnodes(self):
        n = self.net
        nn = self.nn
        indegree = [0] * (nn + 1)
        stack = []
        for k in range(1, self.nl + 1):
            d = self.flowdir[k]
            if d == 1:
                indegree[n.n2[k]] += 1
            elif d == -1:
                indegree[n.n1[k]] += 1
        for i in range(1, nn + 1):
            if indegree[i] == 0:
                stack.append(i)
        numsorted = 0
        order = self.sorted_nodes
        order[:] = 0
        while numsorted < nn:
            if not stack:
                j = self.selectnonstacknode(numsorted, indegree)
                if j == 0:
                    break
                indegree[j] = 0
                stack.append(j)
            i = stack.pop()
            numsorted += 1
            order[numsorted] = i
            for k, _ in n.adj[i]:
                d = self.flowdir[k]
                if d == 0:
                    continue
                m = n.n2[k] if d > 0 else n.n1[k]
                if m != i and indegree[m] > 0:
                    indegree[m] -= 1
                    if indegree[m] == 0:
                        stack.append(m)
        if numsorted < nn:
            raise RuntimeError('sortnodes failed (EPANET error 120)')

    def selectnonstacknode(self, numsorted, indegree):
        n = self.net
        for i in range(numsorted, 0, -1):
            m = self.sorted_nodes[i]
            for _, other in n.adj[m]:
                if indegree[other] > 0:
                    return other
        for i in range(1, self.nn + 1):
            if indegree[i] > 0:
                return i
        return 0

    # ---------------- qualreact.c: reactions ----------------
    def bulkrate(self, c, kb, order):
        n = self.net
        if order == 0.0:
            c = 1.0
        elif order < 0.0:
            c1 = n.climit + SGN(kb) * c
            if abs(c1) < TINY:
                c1 = SGN(c1) * TINY
            c = c / c1
        else:
            if n.climit == 0.0:
                c1 = c
            else:
                c1 = max(0.0, SGN(kb) * (n.climit - c))
            if order == 1.0:
                c = c1
            elif order == 2.0:
                c = c1 * c
            else:
                c = c1 * math.pow(max(0.0, c), order - 1.0)
        if c < 0:
            c = 0
        return kb * c

    def wallrate(self, c, d, kw, kf):
        n = self.net
        if kw == 0.0 or d == 0.0:
            return 0.0
        if n.wallorder == 0.0:
            kf = SGN(kw) * c * kf
            kw = kw * (n.ucf['ELEV'] * n.ucf['ELEV'])
            if abs(kf) < abs(kw):
                kw = kf
            return kw * 4.0 / d
        return c * kf

    def pipereact(self, k, c, dt):
        n = self.net
        if n.qualflag == api.EN_AGE:
            return max(0.0, c + float(dt) / 3600.0)
        rbulk = self.bulkrate(c, n.kb[k], n.bulkorder) * self.bucf
        rwall = self.wallrate(c, n.diam[k], n.kw[k], self.rc[k])
        dcbulk = rbulk * float(dt)
        dcwall = rwall * float(dt)
        dc = dcbulk + dcwall
        if c + dc < 0.0:
            self.hits['react_clamp0'] += 1
        return max(0.0, c + dc)

    def tankreact(self, c, kb, dt):
        n = self.net
        if n.qualflag == api.EN_AGE:
            dc = float(dt) / 3600.0
        else:
            rbulk = self.bulkrate(c, kb, n.tankorder) * self.tucf
            dc = rbulk * float(dt)
        return max(0.0, c + dc)

    def reactpipes(self, dt):
        n = self.net
        for k in range(1, self.nl + 1):
            if n.link_type[k] != api.EN_PIPE:
                continue
            for seg in self.chains[k]:
                cseg = seg.c
                seg.c = self.pipereact(k, cseg, dt)
                self.mb_reacted += (cseg - seg.c) * seg.v

    def reacttanks(self, dt):
        n = self.net
        for i in range(1, self.nt + 1):
            if n.tank_is_reservoir[i]:
                continue
            for seg in self.chains[self.nl + i]:
                c = seg.c
                seg.c = self.tankreact(c, n.tank_kb[i], dt)
                self.mb_reacted += (c - seg.c) * seg.v


    # ---------------- qualroute.c: transport over one quality step ----------------
    def transport(self, tstep, pattern_time):
        n = self.net
        if self.reactflag:
            self.reactpipes(tstep)
            self.reacttanks(tstep)
        if self.rec is not None:
            self.rec.begin_step(tstep, self.cur_period)
        adj, n1, n2, fd = n.adj, n.n1, n.n2, self.flowdir
        for j in range(1, self.nn + 1):
            node = self.sorted_nodes[j]
            if node == 0:
                continue
            volin = massin = volout = 0.0
            pieces = [] if self.rec is not None else None
            for k, _ in adj[node]:
                m = n2[k] if fd[k] >= 0 else n1[k]
                if m == node:
                    volin, massin = self.evalnodeinflow(k, tstep, volin, massin, pieces)
                else:
                    volout += abs(self.linkflow_of(k))
            if n.node_type[node] == api.EN_JUNCTION:
                volout += max(0.0, self.demand[node])
            volout *= tstep
            nodequal = self.findnodequal(node, volin, massin, volout, tstep, pattern_time, pieces)
            for k, _ in adj[node]:
                m = n1[k] if fd[k] >= 0 else n2[k]
                if m == node:
                    self.evalnodeoutflow(k, nodequal, tstep, node)
            self.updatemassbalance(node, massin, volout, tstep)

    def evalnodeinflow(self, k, tstep, volin, massin, pieces):
        q = self.linkflow_of(k)
        v = abs(q) * tstep
        chain = self.chains[k]
        while v > 0.0:
            if not chain:
                break
            seg = chain[0]
            vseg = seg.v
            vseg = min(vseg, v)
            volin += vseg
            massin += vseg * seg.c
            v -= vseg
            if pieces is not None:
                pieces.append((self.rec.ver(seg.id), vseg))
            if v >= 0.0 and vseg >= seg.v:
                chain.popleft()
                self.segcount -= 1
                if pieces is not None:
                    self.rec.kill_seg(seg.id)
            else:
                seg.v -= vseg
        return volin, massin

    def findnodequal(self, node, volin, massin, volout, tstep, pattern_time, pieces):
        n = self.net
        ntype = n.node_type[node]
        nq = self.nodequal
        rec = self.rec
        if rec is not None:
            self.flow_ref = ('u', int(node), 1.0)     # value carried by this node's outflows
        if ntype == api.EN_JUNCTION:
            if self.demand[node] < 0.0:
                self.hits['junction_negative_demand'] += 1
            volin -= min(0.0, self.demand[node]) * tstep
            if volin > 0.0:
                nq[node] = massin / volin
                if rec is not None:
                    rec.set_u(node, [('ver', ver, w / volin) for ver, w in pieces])
            elif self.reactflag:
                self.hits['noflowqual'] += 1
                nq[node] = self.noflowqual(node)
        elif ntype == api.EN_TANK:
            nq[node] = self.mixtank(node, volin, massin, volout, pieces)
        self.sourcequal = 0.0
        if n.qualflag == api.EN_TRACE:
            if node == n.tracenode:
                if ntype == api.EN_RESERVOIR:
                    self.sourcequal = 100.0
                else:
                    self.sourcequal = max(100.0 - nq[node], 0.0)
                nq[node] = 100.0
                if rec is not None:
                    rec.set_u(node, [('const', 0, 100.0)])
            return nq[node]
        self.src_term = None
        self.sourcequal = self.findsourcequal(node, volout, tstep, pattern_time)
        if self.sourcequal == 0.0:
            # EPANET adds nothing; a slotted / virtual term (value 0 here) keeps the map affine in src[slot]
            if self.K is not None and self.src_term is not None and self.src_term[0] == 'src' and ntype != api.EN_RESERVOIR:
                if ntype == api.EN_JUNCTION:
                    rec.set_u(node, [('u', int(node), 1.0), self.src_term])
                else:
                    rec.set_uflow(node, [('u', int(node), 1.0), self.src_term])
                    self.flow_ref = ('uflow', int(node), 1.0)
            return nq[node]
        if ntype == api.EN_JUNCTION:
            nq[node] += self.sourcequal
            if rec is not None:
                rec.set_u(node, [('u', int(node), 1.0), self.src_term])
            return nq[node]
        if ntype == api.EN_TANK:
            if rec is not None:
                rec.set_uflow(node, [('u', int(node), 1.0), self.src_term])
                self.flow_ref = ('uflow', int(node), 1.0)
            return nq[node] + self.sourcequal
        nq[node] = self.sourcequal          # reservoir
        if rec is not None:
            rec.set_u(node, [self.src_term])
        return self.sourcequal

    def noflowqual(self, node):
        n = self.net
        c = 0.0
        kount = 0
        ids = []
        for k, _ in n.adj[node]:
            d = self.flowdir[k]
            if n.n2[k] == node and d >= 0:
                inflow = True
            elif n.n1[k] == node and d < 0:
                inflow = True
            else:
                inflow = False
            chain = self.chains[k]
            if inflow and chain:
                c += chain[0].c
                kount += 1
                ids.append(chain[0].id)
            elif (not inflow) and chain:
                c += chain[-1].c
                kount += 1
                ids.append(chain[-1].id)
        if kount > 0:
            c = c / float(kount)
        if self.rec is not None:
            self.rec.set_u(node, [('ver', self.rec.ver(i), 1.0 / kount) for i in ids])
        return c

    def evalnodeoutflow(self, k, c, tstep, node):
        v = abs(self.linkflow_of(k)) * tstep
        if v == 0.0:
            return
        chain = self.chains[k]
        if chain:
            seg = chain[-1]
            if abs(seg.c - c) < self.ctol:
                vold = seg.v
                seg.c = (seg.c * seg.v + c * v) / (seg.v + v)
                seg.v += v
                self.hits['outflow_merge'] += 1
                if self.rec is not None:
                    fr = self.flow_ref
                    self.rec.set_seg(seg, [('ver', self.rec.ver(seg.id), vold / (vold + v)),
                                           (fr[0], fr[1], v / (vold + v))])
                return
        seg = self.addseg(k, v, c, [self.flow_ref] if self.rec is not None else None)
        self.hits['outflow_newseg'] += 1

    def findsourcequal(self, node, volout, tstep, pattern_time):
        n = self.net
        if n.qualflag != api.EN_CHEM:
            return 0.0
        src = n.sources.get(node)
        virtual = False
        if src is None:
            if self.rec is None or node not in self.virtual_sources:
                return 0.0
            src, virtual = (self.virtual_sources[node], 0.0, 0), True
        stype, c0, pat = src
        # slotted: junction / tank source recorded per time slot, pattern factor left to the parameter
        slotted = self.rec is not None and self.K is not None and n.node_type[node] != api.EN_RESERVOIR
        if c0 == 0.0 and not slotted:
            return 0.0
        if tstep <= 0 or volout / tstep <= Q_STAGNANT:
            self.hits['source_stagnant_skip'] += 1
            return 0.0
        self.hits['virtual_source' if virtual else ('source', int(stype), bool(n.node_type[node] == api.EN_JUNCTION), pat != 0)] += 1
        c = c0
        if stype == api.EN_MASS:
            c /= 60.0
        else:
            c /= n.ucf['QUALITY']
        if pat != 0:
            f = n.patterns[pat]
            c = c * f[((pattern_time + n.pstart) // n.pstep) % len(f)]
        if stype == api.EN_CONCEN:
            if n.node_type[node] == api.EN_JUNCTION:
                if self.demand[node] < 0.0:
                    self.hits['source_concen_junction_inflow'] += 1
                    c = -c * self.demand[node] * tstep / volout
                else:
                    c = 0.0
        elif stype == api.EN_MASS:
            c = c * tstep / volout
        if self.rec is not None:                  # c is linear in the strength c0 up to here
            kind = 'setpoint' if stype == api.EN_SETPOINT else 'src'
            if slotted:                           # multiplier per unit (strength x pattern value)
                m = (1.0 / 60.0) if stype == api.EN_MASS else 1.0 / n.ucf['QUALITY']
                if stype == api.EN_CONCEN and n.node_type[node] == api.EN_JUNCTION:
                    m = -m * self.demand[node] * tstep / volout if self.demand[node] < 0.0 else 0.0
                elif stype == api.EN_MASS:
                    m = m * tstep / volout
                slot = ((pattern_time + n.pstart) // n.pstep) % self.K
                self.src_term = (kind, int(slot * (n.nnodes + 1) + node), m)
            else:
                self.src_term = (kind, int(node), c / c0)
        if stype == api.EN_SETPOINT:
            c = max(c - self.nodequal[node], 0.0)
        return c

    def updatemassbalance(self, node, massin, volout, tstep):
        ntype = self.net.node_type[node]
        masslost = massadded = 0.0
        if ntype == api.EN_JUNCTION:
            masslost = max(0.0, self.demand[node]) * tstep * self.nodequal[node]
            massadded = self.sourcequal * volout
        elif ntype == api.EN_RESERVOIR:
            masslost = massin
            if self.sourcequal > 0.0:
                massadded = self.sourcequal * volout
            else:
                massadded = self.nodequal[node] * volout
        else:
            massadded = self.sourcequal * volout
        self.mb_outflow += masslost
        self.mb_inflow += massadded


    # ---------------- qualreact.c: tank mixing models ----------------
    def mixtank(self, node, volin, massin, volout, pieces):
        n = self.net
        i = node - self.nj
        vnet = volin - volout
        mix = n.tank_mix[i]
        if mix == api.EN_MIX1:
            self.tankmix1(i, volin, massin, vnet, pieces)
        elif mix == api.EN_MIX2:
            self.tankmix2(i, volin, massin, vnet, pieces)
        elif mix == api.EN_FIFO:
            self.tankmix3(i, volin, massin, vnet, pieces)
        elif mix == api.EN_LIFO:
            self.tankmix4(i, volin, massin, vnet, pieces)
        return self.tank_c[i]

    def _rec_tank_u(self, i, terms):
        self.rec.set_u(self.nj + i, terms)

    def tankmix1(self, i, vin, win, vnet, pieces=None):
        n = self.net
        rec = self.rec
        chain = self.chains[self.nl + i]
        if chain:
            seg = chain[0]
            vold = seg.v
            vnew = seg.v + vin
            if vnew > 0.0:
                if rec is not None:
                    rec.set_seg(seg, [('ver', rec.ver(seg.id), vold / vnew)]
                                + [('ver', p, w / vnew) for p, w in pieces])
                seg.c = (seg.c * seg.v + win) / vnew
            seg.v += vnet
            seg.v = max(0.0, seg.v)
            self.tank_c[i] = seg.c
            self.hits['mix1'] += 1
            if seg.v > n.tank_vmax[i]:
                self.hits['mix1_overflow'] += 1
                self.mb_outflow += (seg.v - n.tank_vmax[i]) * self.tank_c[i]
                seg.v = n.tank_vmax[i]
            if rec is not None:
                self._rec_tank_u(i, [('ver', rec.ver(seg.id), 1.0)])

    def tankmix2(self, i, vin, win, vnet, pieces=None):
        n = self.net
        rec = self.rec
        chain = self.chains[self.nl + i]
        if not chain:
            return
        mixzone = chain[-1]
        stagzone = chain[0]
        vmz = n.tank_v1frac[i] * n.tank_vmax[i]
        vt = 0.0
        if vnet > 0.0:
            vt = max(0.0, mixzone.v + vnet - vmz)
            if vin > 0.0:
                den = mixzone.v + vin
                if rec is not None:
                    rec.set_seg(mixzone, [('ver', rec.ver(mixzone.id), mixzone.v / den)]
                                + [('ver', p, w / den) for p, w in pieces])
                mixzone.c = (mixzone.c * mixzone.v + win) / den
            if vt > 0.0:
                den = stagzone.v + vt
                if rec is not None:
                    rec.set_seg(stagzone, [('ver', rec.ver(stagzone.id), stagzone.v / den),
                                           ('ver', rec.ver(mixzone.id), vt / den)])
                stagzone.c = (stagzone.c * stagzone.v + mixzone.c * vt) / den
        elif vnet < 0.0:
            if stagzone.v > 0.0:
                vt = min(stagzone.v, -vnet)
            if vin + vt > 0.0:
                den = mixzone.v + vin + vt
                if rec is not None:
                    rec.set_seg(mixzone, [('ver', rec.ver(mixzone.id), mixzone.v / den)]
                                + [('ver', p, w / den) for p, w in pieces]
                                + [('ver', rec.ver(stagzone.id), vt / den)])
                mixzone.c = (mixzone.c * mixzone.v + win + stagzone.c * vt) / den
        if vt > 0.0:
            if vnet > 0.0:
                self.hits['mix2_fill_to_stagnant'] += 1
                mixzone.v = vmz
                stagzone.v += vt
                vsz = n.tank_vmax[i] - vmz
                if stagzone.v > vsz:
                    self.hits['mix2_overflow'] += 1
                    self.mb_outflow += (stagzone.v - vsz) * stagzone.c
                    stagzone.v = vsz
            else:
                self.hits['mix2_drain_from_stagnant'] += 1
                stagzone.v = max(0.0, stagzone.v - vt)
                mixzone.v = vmz + vt + vnet
        else:
            self.hits['mix2_mixzone_only'] += 1
            mixzone.v += vnet
            mixzone.v = min(mixzone.v, vmz)
            mixzone.v = max(0.0, mixzone.v)
            if vmz - mixzone.v > 0.0:
                stagzone.v = 0.0
        self.tank_c[i] = mixzone.c
        if rec is not None:
            self._rec_tank_u(i, [('ver', rec.ver(mixzone.id), 1.0)])

    def _withdraw(self, chain, vout):
        """Shared leading-segment withdrawal loop of tankmix3/tankmix4.
        Returns (volume, mass, [(version, volume)]) of the pieces taken."""
        rec = self.rec
        vsum = wsum = 0.0
        taken = []
        while vout > 0.0:
            if not chain:
                break
            seg = chain[0]
            vseg = seg.v
            vseg = min(vseg, vout)
            if seg is chain[-1]:
                vseg = vout
            vsum += vseg
            wsum += seg.c * vseg
            if rec is not None:
                taken.append((rec.ver(seg.id), vseg))
            vout -= vseg
            if vout >= 0.0 and vseg >= seg.v:
                if len(chain) > 1:
                    chain.popleft()
                    if rec is not None:
                        rec.kill_seg(seg.id)
            else:
                seg.v -= vseg
        return vsum, wsum, taken

    def tankmix3(self, i, vin, win, vnet, pieces=None):
        n = self.net
        rec = self.rec
        k = self.nl + i
        chain = self.chains[k]
        if not chain:
            return
        if vin > 0.0:
            cin = win / vin
            seg = chain[-1]
            if abs(seg.c - cin) < self.ctol:
                self.hits['fifo_inflow_merge'] += 1
                seg.v += vin                      # EPANET: volume only, concentration kept
            else:
                self.hits['fifo_inflow_newseg'] += 1
                terms = [('ver', p, w / vin) for p, w in pieces] if rec is not None else None
                seg = self.addseg(k, vin, cin, terms)
        vout = vin - vnet
        if self.tank_v[i] >= n.tank_vmax[i] and vnet > 0.0:
            self.hits['fifo_overflow_staleV'] += 1
            vout = vin
        vsum, wsum, taken = self._withdraw(chain, vout)
        self.hits['fifo'] += 1
        if vsum > 0.0:
            self.tank_c[i] = wsum / vsum
            if rec is not None:
                self._rec_tank_u(i, [('ver', v, vseg / vsum) for v, vseg in taken])
        elif not chain:
            self.tank_c[i] = 0.0
            if rec is not None:
                self._rec_tank_u(i, [('const', 0, 0.0)])
        else:
            self.tank_c[i] = chain[0].c
            if rec is not None:
                self._rec_tank_u(i, [('ver', rec.ver(chain[0].id), 1.0)])
        if self.tank_v[i] >= n.tank_vmax[i] and vnet > 0.0:
            self.mb_outflow += vnet * self.tank_c[i]

    def tankmix4(self, i, vin, win, vnet, pieces=None):
        n = self.net
        rec = self.rec
        k = self.nl + i
        chain = self.chains[k]
        if not chain:
            return
        cin = win / vin if vin > 0.0 else 0.0
        self.tank_c[i] = chain[-1].c
        seg = chain[-1]
        if vnet > 0.0:
            if abs(seg.c - cin) < self.ctol:
                self.hits['lifo_inflow_merge'] += 1
                seg.v += vnet                     # EPANET: volume only, concentration kept
            else:
                self.hits['lifo_inflow_newseg'] += 1
                terms = None
                if rec is not None:
                    terms = [('ver', p, w / vin) for p, w in pieces] if vin > 0.0 else [('const', 0, 0.0)]
                seg = self.addseg(k, vnet, cin, terms)
            self.tank_c[i] = chain[-1].c
            if rec is not None:
                self._rec_tank_u(i, [('ver', rec.ver(chain[-1].id), 1.0)])
            if self.tank_v[i] >= n.tank_vmax[i]:
                self.hits['lifo_overflow_staleV'] += 1
                _, wsum, _ = self._withdraw(chain, vnet)
                self.mb_outflow += wsum
        elif vnet < 0.0:
            self.hits['lifo_drain'] += 1
            self.reversesegs(k)
            vsum, wsum, taken = self._withdraw(chain, -vnet)
            self.reversesegs(k)
            self.tank_c[i] = (wsum + win) / (vsum + vin)
            if rec is not None:
                den = vsum + vin
                self._rec_tank_u(i, [('ver', v, vseg / den) for v, vseg in taken]
                                 + [('ver', p, w / den) for p, w in pieces])
        else:
            self.hits['lifo_static'] += 1
            if rec is not None:
                self._rec_tank_u(i, [('ver', rec.ver(chain[-1].id), 1.0)])

    # ---------------- quality.c: mass balance & drivers ----------------
    def evalmassbalance(self):
        final = self.findstoredmass()
        massin = self.mb_initial + self.mb_inflow
        massout = self.mb_outflow + final
        if self.mb_reacted > 0.0:
            massout += self.mb_reacted
        else:
            massin -= self.mb_reacted
        self.mb_ratio = 1.0 if massin == 0.0 else massout / massin
        return self.mb_ratio

    def runqual(self):
        if self.qtime == self.htime:
            p = self.period_index
            per = self.trace.periods[p]
            if per.t != self.qtime:
                raise RuntimeError(f'hydraulics file out of sync: record t={per.t} at Qtime={self.qtime}')
            self.period_index += 1
            self.load_period(p)
            self.htime = per.t + per.hydstep

    def run(self, mode='nextQ', observe=None):
        """Drive the engine; returns (times, quality[nt, nnodes+1] internal units, mass balance ratio).
        With record=True the transport Schedule is available afterwards as self.schedule."""
        self.reset()
        self.schedule = None
        dur = self.net.duration
        times, rows = [], []
        # the mass balance ratio is recomputed from scratch at every call (stored mass over all segments, a Python
        # loop): keep the per-period / per-step calls only for an observer, the final value is evaluated below
        mb_each = observe is not None

        def record():
            times.append(self.qtime)
            rows.append(self.nodequal.copy())
            if self.rec is not None:
                self.rec.snapshot()
            if observe is not None:
                observe(self)

        if mode == 'nextQ':
            while True:
                self.runqual()
                record()
                hydstep = self.htime - self.qtime if self.htime <= dur else 0
                if hydstep > 0:
                    qt = 0
                    while qt < hydstep:
                        dt = min(self.qstep, hydstep - qt)
                        qt += dt
                        self.transport(dt, self.qtime)
                if mb_each:
                    self.evalmassbalance()
                self.qtime += hydstep
                if hydstep == 0:
                    break
        elif mode == 'stepQ':
            while True:
                self.runqual()
                record()
                if self.qtime >= dur:
                    break
                tstep = self.qstep
                while True:
                    dt = tstep
                    hstep = self.htime - self.qtime
                    if hstep < dt:
                        dt = hstep
                        self.transport(dt, self.qtime)
                        self.qtime += dt
                        self.runqual()
                    else:
                        self.transport(dt, self.qtime)
                        self.qtime += dt
                    tstep -= dt
                    if tstep <= 0:
                        break
                if mb_each:
                    self.evalmassbalance()
        else:
            raise ValueError(mode)
        self.evalmassbalance()
        if self.rec is not None:
            self.schedule = self.rec.build()
        return np.asarray(times, dtype=np.int64), np.asarray(rows), self.mb_ratio
