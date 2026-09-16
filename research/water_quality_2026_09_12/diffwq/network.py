"""Network data needed by the quality engine, extracted through the API and
converted back to EPANET's internal units (ft, ft3, cfs, mass/ft3, 1/s)."""
from dataclasses import dataclass, field
import numpy as np
from . import epanet_api as api

# constants from types.h
GPMperCFS, AFDperCFS, MGDperCFS, IMGDperCFS = 448.831, 1.9837, 0.64632, 0.5382
LPSperCFS, LPMperCFS, CMHperCFS, CMDperCFS, MLDperCFS, CMSperCFS = 28.317, 1699.0, 101.94, 2446.6, 2.4466, 0.028317
LperFT3, MperFT, SECperDAY = 28.317, 0.3048, 86400
DIFFUS, VISCOS = 1.3e-8, 1.1e-5
US_FLOW_UNITS = {0: 1.0, 1: GPMperCFS, 2: MGDperCFS, 3: IMGDperCFS, 4: AFDperCFS}
SI_FLOW_UNITS = {5: LPSperCFS, 6: LPMperCFS, 7: MLDperCFS, 8: CMHperCFS, 9: CMDperCFS, 10: CMSperCFS}


def unit_factors(flow_flag, qualflag):
    """Replicates input1.c initunits() for the factors the quality engine needs."""
    if flow_flag in SI_FLOW_UNITS:
        dcf, qcf, hcf = 1000.0 * MperFT, SI_FLOW_UNITS[flow_flag], MperFT
    else:
        dcf, qcf, hcf = 12.0, US_FLOW_UNITS[flow_flag], 1.0
    ccf = 1.0 / LperFT3 if qualflag == api.EN_CHEM else 1.0
    return {'ELEV': hcf, 'LENGTH': hcf, 'DIAM': dcf, 'FLOW': qcf, 'QUALITY': ccf, 'VOLUME': hcf ** 3}


@dataclass
class NetworkData:
    nnodes: int
    nlinks: int
    ntanks: int
    njuncs: int
    node_type: np.ndarray
    link_type: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    length: np.ndarray
    diam: np.ndarray
    kb: np.ndarray
    kw: np.ndarray
    c0: np.ndarray
    tank_node: np.ndarray
    tank_is_reservoir: np.ndarray
    tank_v0: np.ndarray
    tank_vmin: np.ndarray
    tank_vmax: np.ndarray
    tank_mix: np.ndarray
    tank_v1frac: np.ndarray
    tank_kb: np.ndarray
    tank_v_hyd_final: np.ndarray
    sources: dict
    patterns: dict
    qualflag: int
    tracenode: int
    ctol: float
    climit: float
    diffus: float
    viscos: float
    bulkorder: float
    wallorder: float
    tankorder: float
    duration: int
    qstep: int
    pstep: int
    pstart: int
    ucf: dict
    adj: list
    node_ids: list = field(default_factory=list)
    link_ids: list = field(default_factory=list)


def extract(pr: api.Project, snap_rel=1e-9) -> NetworkData:
    """Call after pr.solve_hydraulics() so that tank volumes reflect the engine state."""
    nn, nl, nt = pr.count(api.EN_NODECOUNT), pr.count(api.EN_LINKCOUNT), pr.count(api.EN_TANKCOUNT)
    nj = nn - nt
    qualflag, _, _, tracenode = pr.qual_info()
    ucf = unit_factors(pr.flow_units(), qualflag)
    node_type = np.zeros(nn + 1, dtype=np.int64)
    c0 = np.zeros(nn + 1)
    sources = {}
    for i in range(1, nn + 1):
        node_type[i] = pr.node_type(i)
        c0[i] = pr.node_value(i, api.EN_INITQUAL) / ucf['QUALITY']
        sq = pr.node_value(i, api.EN_SOURCEQUAL)
        if sq is not None:
            sources[i] = (int(pr.node_value(i, api.EN_SOURCETYPE)), sq, int(pr.node_value(i, api.EN_SOURCEPAT)))
    link_type = np.zeros(nl + 1, dtype=np.int64)
    n1 = np.zeros(nl + 1, dtype=np.int64); n2 = np.zeros(nl + 1, dtype=np.int64)
    length = np.zeros(nl + 1); diam = np.zeros(nl + 1); kb = np.zeros(nl + 1); kw = np.zeros(nl + 1)
    for k in range(1, nl + 1):
        link_type[k] = pr.link_type(k)
        n1[k], n2[k] = pr.link_nodes(k)
        length[k] = pr.link_value(k, api.EN_LENGTH) / ucf['LENGTH']
        diam[k] = pr.link_value(k, api.EN_DIAMETER) / ucf['DIAM']
        kb[k] = pr.link_value(k, api.EN_KBULK) / SECperDAY
        kw[k] = pr.link_value(k, api.EN_KWALL) / SECperDAY
    tank_node = np.zeros(nt + 1, dtype=np.int64)
    is_res = np.zeros(nt + 1, dtype=bool)
    v0 = np.zeros(nt + 1); vmin = np.zeros(nt + 1); vmax = np.zeros(nt + 1); vfin = np.zeros(nt + 1)
    mix = np.zeros(nt + 1, dtype=np.int64); v1frac = np.ones(nt + 1); tkb = np.zeros(nt + 1)
    for i in range(1, nt + 1):
        n = nj + i
        tank_node[i] = n
        is_res[i] = node_type[n] == api.EN_RESERVOIR
        if is_res[i]:
            continue
        v0[i] = pr.node_value(n, api.EN_INITVOLUME) / ucf['VOLUME']
        vmin[i] = pr.node_value(n, api.EN_MINVOLUME) / ucf['VOLUME']
        vmax[i] = pr.node_value(n, api.EN_MAXVOLUME) / ucf['VOLUME']
        mix[i] = int(pr.node_value(n, api.EN_MIXMODEL))
        v1frac[i] = pr.node_value(n, api.EN_MIXFRACTION)
        tkb[i] = pr.node_value(n, api.EN_TANK_KBULK) / SECperDAY
        v = pr.node_value(n, api.EN_TANKVOLUME) / ucf['VOLUME']
        # hydraul.c tanklevels() clamps V exactly to Vmax/Vmin; EN_TANKVOLUME returns
        # tankvolume(tankgrade(V)) whose round trip can miss the clamp by an ulp.
        if abs(v - vmax[i]) <= snap_rel * vmax[i]:
            v = vmax[i]
        elif abs(v - vmin[i]) <= snap_rel * max(vmin[i], vmax[i]):
            v = vmin[i]
        vfin[i] = v
    patterns = {idx: pr.pattern(idx) for idx in range(1, pr.count(api.EN_PATCOUNT) + 1)}
    adj = [[] for _ in range(nn + 1)]
    for k in range(nl, 0, -1):           # project.c buildadjlists prepends: list order = descending k
        adj[n1[k]].append((k, int(n2[k])))
        adj[n2[k]].append((k, int(n1[k])))
    return NetworkData(
        nnodes=nn, nlinks=nl, ntanks=nt, njuncs=nj, node_type=node_type, link_type=link_type,
        n1=n1, n2=n2, length=length, diam=diam, kb=kb, kw=kw, c0=c0,
        tank_node=tank_node, tank_is_reservoir=is_res, tank_v0=v0, tank_vmin=vmin, tank_vmax=vmax,
        tank_mix=mix, tank_v1frac=v1frac, tank_kb=tkb, tank_v_hyd_final=vfin,
        sources=sources, patterns=patterns, qualflag=qualflag, tracenode=tracenode,
        ctol=pr.option(api.EN_TOLERANCE) / ucf['QUALITY'], climit=pr.option(api.EN_CONCENLIMIT) / ucf['QUALITY'],
        diffus=pr.option(api.EN_SP_DIFFUS) * DIFFUS, viscos=pr.option(api.EN_SP_VISCOS) * VISCOS,
        bulkorder=pr.option(api.EN_BULKORDER), wallorder=pr.option(api.EN_WALLORDER), tankorder=pr.option(api.EN_TANKORDER),
        duration=pr.time_param(api.EN_DURATION), qstep=pr.time_param(api.EN_QUALSTEP),
        pstep=pr.time_param(api.EN_PATTERNSTEP), pstart=pr.time_param(api.EN_PATTERNSTART),
        ucf=ucf, adj=adj,
        node_ids=[''] + [pr.node_id(i) for i in range(1, nn + 1)],
        link_ids=[''] + [pr.link_id(k) for k in range(1, nl + 1)])
