"""Minimal ctypes binding to the EPANET 2.3 shared library (epanet2.dll / libepanet2.so)."""
import ctypes as ct
import os
import platform
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def default_library_path():
    env = os.environ.get('EPANET_LIB')
    if env:
        return Path(env)
    if platform.system() == 'Windows':
        return HERE.parent / 'build_epanet' / 'bin' / 'RelWithDebInfo' / 'epanet2.dll'
    return Path.home() / 'wdn_quantum' / 'EPANET-dev' / 'build' / 'lib' / 'libepanet2.so'


# --- enum values copied from include/epanet2_enums.h ---
EN_NODECOUNT, EN_TANKCOUNT, EN_LINKCOUNT, EN_PATCOUNT = 0, 1, 2, 3
EN_ELEVATION, EN_BASEDEMAND, EN_INITQUAL, EN_SOURCEQUAL, EN_SOURCEPAT, EN_SOURCETYPE = 0, 1, 4, 5, 6, 7
EN_DEMAND, EN_QUALITY, EN_INITVOLUME, EN_MIXMODEL, EN_MINVOLUME = 9, 12, 14, 15, 18
EN_MINLEVEL, EN_MAXLEVEL, EN_MIXFRACTION, EN_TANK_KBULK, EN_TANKVOLUME, EN_MAXVOLUME = 20, 21, 22, 23, 24, 25
EN_CANOVERFLOW = 26
EN_DIAMETER, EN_LENGTH, EN_KBULK, EN_KWALL, EN_FLOW, EN_STATUS = 0, 1, 6, 7, 8, 11
EN_TRIALS, EN_ACCURACY, EN_TOLERANCE, EN_SP_VISCOS, EN_UNBALANCED, EN_SP_DIFFUS = 0, 1, 2, 13, 14, 18
EN_BULKORDER, EN_WALLORDER, EN_TANKORDER, EN_CONCENLIMIT = 19, 20, 21, 22
EN_DURATION, EN_HYDSTEP, EN_QUALSTEP, EN_PATTERNSTEP, EN_PATTERNSTART, EN_REPORTSTART = 0, 1, 2, 3, 4, 6
EN_MASSBALANCE = 4
EN_JUNCTION, EN_RESERVOIR, EN_TANK = 0, 1, 2
EN_CVPIPE, EN_PIPE, EN_PUMP = 0, 1, 2
EN_NONE, EN_CHEM, EN_AGE, EN_TRACE = 0, 1, 2, 3
EN_CONCEN, EN_MASS, EN_SETPOINT, EN_FLOWPACED = 0, 1, 2, 3
EN_MIX1, EN_MIX2, EN_FIFO, EN_LIFO = 0, 1, 2, 3
EN_SAVE = 1

P, I, L, D = ct.c_void_p, ct.c_int, ct.c_long, ct.c_double
_SIG = {
    'EN_createproject': [ct.POINTER(P)], 'EN_deleteproject': [P], 'EN_getversion': [ct.POINTER(I)],
    'EN_open': [P, ct.c_char_p, ct.c_char_p, ct.c_char_p], 'EN_close': [P],
    'EN_solveH': [P], 'EN_savehydfile': [P, ct.c_char_p],
    'EN_openQ': [P], 'EN_initQ': [P, I], 'EN_runQ': [P, ct.POINTER(L)], 'EN_nextQ': [P, ct.POINTER(L)],
    'EN_stepQ': [P, ct.POINTER(L)], 'EN_closeQ': [P],
    'EN_getcount': [P, I, ct.POINTER(I)], 'EN_getnodetype': [P, I, ct.POINTER(I)],
    'EN_getlinktype': [P, I, ct.POINTER(I)], 'EN_getlinknodes': [P, I, ct.POINTER(I), ct.POINTER(I)],
    'EN_getnodevalue': [P, I, I, ct.POINTER(D)], 'EN_getlinkvalue': [P, I, I, ct.POINTER(D)],
    'EN_getnodevalues': [P, I, ct.POINTER(D)],
    'EN_getoption': [P, I, ct.POINTER(D)], 'EN_setoption': [P, I, D],
    'EN_gettimeparam': [P, I, ct.POINTER(L)], 'EN_settimeparam': [P, I, L],
    'EN_getqualinfo': [P, ct.POINTER(I), ct.c_char_p, ct.c_char_p, ct.POINTER(I)],
    'EN_getflowunits': [P, ct.POINTER(I)], 'EN_getpatternlen': [P, I, ct.POINTER(I)],
    'EN_getpatternvalue': [P, I, I, ct.POINTER(D)], 'EN_getstatistic': [P, I, ct.POINTER(D)],
    'EN_getnodeid': [P, I, ct.c_char_p], 'EN_getlinkid': [P, I, ct.c_char_p],
    'EN_getnodeindex': [P, ct.c_char_p, ct.POINTER(I)], 'EN_getlinkindex': [P, ct.c_char_p, ct.POINTER(I)],
    'EN_setnodevalue': [P, I, I, D], 'EN_setlinkvalue': [P, I, I, D],
    'EN_setqualtype': [P, I, ct.c_char_p, ct.c_char_p, ct.c_char_p],
    'EN_addpattern': [P, ct.c_char_p], 'EN_getpatternindex': [P, ct.c_char_p, ct.POINTER(I)],
    'EN_setpattern': [P, I, ct.POINTER(D), I],
}


class EpanetError(RuntimeError):
    pass


class Library:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_library_path()
        self.lib = ct.CDLL(str(self.path))
        for name, args in _SIG.items():
            f = getattr(self.lib, name)
            f.argtypes = args
            f.restype = I

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code > 100:        # codes <= 100 are warnings (e.g. 1..9 hydraulic warnings)
            raise EpanetError(f'{name}: EPANET error {code}')
        return code

    def version(self):
        v = I(); self.call('EN_getversion', ct.byref(v)); return v.value


class Project:
    """One opened EPANET project. Use as a context manager."""

    def __init__(self, library: Library, inp, rpt='', out=''):
        self.L = library
        self.h = P()
        self.L.call('EN_createproject', ct.byref(self.h))
        self.L.call('EN_open', self.h, str(inp).encode(), str(rpt).encode(), str(out).encode())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.L.call('EN_close', self.h)
        self.L.call('EN_deleteproject', self.h)

    # --- generic getters ---
    def count(self, what):
        v = I(); self.L.call('EN_getcount', self.h, what, ct.byref(v)); return v.value

    def node_type(self, i):
        v = I(); self.L.call('EN_getnodetype', self.h, i, ct.byref(v)); return v.value

    def link_type(self, k):
        v = I(); self.L.call('EN_getlinktype', self.h, k, ct.byref(v)); return v.value

    def link_nodes(self, k):
        a, b = I(), I(); self.L.call('EN_getlinknodes', self.h, k, ct.byref(a), ct.byref(b)); return a.value, b.value

    def node_value(self, i, prop):
        v = D(); code = self.L.lib.EN_getnodevalue(self.h, i, prop, ct.byref(v))
        if code == 240:       # node has no source
            return None
        if code > 100:
            raise EpanetError(f'EN_getnodevalue({i},{prop}): {code}')
        return v.value

    def node_values(self, prop):
        n = self.count(EN_NODECOUNT); arr = (D * n)()
        self.L.call('EN_getnodevalues', self.h, prop, arr)
        return np.concatenate(([0.0], np.ctypeslib.as_array(arr).astype(np.float64)))

    def link_value(self, k, prop):
        v = D(); self.L.call('EN_getlinkvalue', self.h, k, prop, ct.byref(v)); return v.value

    def option(self, opt):
        v = D(); self.L.call('EN_getoption', self.h, opt, ct.byref(v)); return v.value

    def set_option(self, opt, value):
        self.L.call('EN_setoption', self.h, opt, D(value))

    def time_param(self, prm):
        v = L(); self.L.call('EN_gettimeparam', self.h, prm, ct.byref(v)); return v.value

    def set_time_param(self, prm, value):
        self.L.call('EN_settimeparam', self.h, prm, L(int(value)))

    def qual_info(self):
        qt, tn = I(), I(); name = ct.create_string_buffer(64); units = ct.create_string_buffer(64)
        self.L.call('EN_getqualinfo', self.h, ct.byref(qt), name, units, ct.byref(tn))
        return qt.value, name.value.decode(errors='replace'), units.value.decode(errors='replace'), tn.value

    def flow_units(self):
        v = I(); self.L.call('EN_getflowunits', self.h, ct.byref(v)); return v.value

    def pattern(self, idx):
        n = I(); self.L.call('EN_getpatternlen', self.h, idx, ct.byref(n))
        v = D(); out = []
        for p in range(1, n.value + 1):
            self.L.call('EN_getpatternvalue', self.h, idx, p, ct.byref(v)); out.append(v.value)
        return np.asarray(out, dtype=np.float64)

    def node_id(self, i):
        b = ct.create_string_buffer(64); self.L.call('EN_getnodeid', self.h, i, b); return b.value.decode()

    def link_id(self, k):
        b = ct.create_string_buffer(64); self.L.call('EN_getlinkid', self.h, k, b); return b.value.decode()

    def statistic(self, what):
        v = D(); self.L.call('EN_getstatistic', self.h, what, ct.byref(v)); return v.value

    # --- setters (user units, as EN_set* expects) ---
    def node_index(self, node_id):
        v = I(); self.L.call('EN_getnodeindex', self.h, str(node_id).encode(), ct.byref(v)); return v.value

    def link_index(self, link_id):
        v = I(); self.L.call('EN_getlinkindex', self.h, str(link_id).encode(), ct.byref(v)); return v.value

    def set_node_value(self, i, prop, value):
        self.L.call('EN_setnodevalue', self.h, i, prop, D(value))

    def set_link_value(self, k, prop, value):
        self.L.call('EN_setlinkvalue', self.h, k, prop, D(value))

    def set_qual_type(self, qualtype, chem_name='Chemical', chem_units='mg/L', trace_node=''):
        self.L.call('EN_setqualtype', self.h, qualtype, chem_name.encode(), chem_units.encode(),
                    str(trace_node).encode())

    def set_pattern(self, idx, values):
        vals = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        self.L.call('EN_setpattern', self.h, idx, vals.ctypes.data_as(ct.POINTER(D)), len(vals))

    def add_pattern(self, pattern_id, values):
        """Create a time pattern with the given multipliers; returns its index."""
        self.L.call('EN_addpattern', self.h, str(pattern_id).encode())
        idx = I(); self.L.call('EN_getpatternindex', self.h, str(pattern_id).encode(), ct.byref(idx))
        self.set_pattern(idx.value, values)
        return idx.value

    def resample_patterns(self, new_step):
        """Refine the pattern time step to new_step (a divisor of the current one) by repeating every
        multiplier, so demands and existing sources are unchanged while slots become finer."""
        old = self.time_param(EN_PATTERNSTEP)
        if old % new_step:
            raise ValueError(f'pattern step {old} not a multiple of {new_step}')
        r = old // new_step
        for idx in range(1, self.count(EN_PATCOUNT) + 1):
            self.set_pattern(idx, np.repeat(self.pattern(idx), r))
        self.set_time_param(EN_PATTERNSTEP, new_step)
        return r

    def set_source(self, i, stype, strength, pattern_index=0):
        """Install / overwrite a quality source at node i (user units)."""
        self.set_node_value(i, EN_SOURCETYPE, stype)
        self.set_node_value(i, EN_SOURCEQUAL, strength)
        self.set_node_value(i, EN_SOURCEPAT, pattern_index)

    # --- simulation ---
    def solve_hydraulics(self, hydfile_out):
        self.L.call('EN_solveH', self.h)
        self.L.call('EN_savehydfile', self.h, str(hydfile_out).encode())

    def run_quality_reference(self, mode='nextQ'):
        """Run the engine's own quality solver on the saved hydraulics.

        mode 'nextQ': EN_runQ/EN_nextQ loop (what EN_solveQ and the CLI do);
        results recorded at every hydraulic event time.
        mode 'stepQ': EN_runQ/EN_stepQ loop; results recorded every quality step.
        Returns (times, quality[nt, nnodes+1] in user units, mass balance ratio).
        Set EN_QUALSTEP / EN_TOLERANCE on the project beforehand if needed.
        """
        dur = self.time_param(EN_DURATION)
        n = self.count(EN_NODECOUNT)
        arr = (D * n)()
        t, dt = L(), L()
        times, rows = [], []
        self.L.call('EN_openQ', self.h)
        try:
            self.L.call('EN_initQ', self.h, 0)
            while True:
                self.L.call('EN_runQ', self.h, ct.byref(t))
                self.L.call('EN_getnodevalues', self.h, EN_QUALITY, arr)
                times.append(t.value)
                rows.append(np.concatenate(([0.0], np.ctypeslib.as_array(arr).astype(np.float64))))
                if mode == 'nextQ':
                    self.L.call('EN_nextQ', self.h, ct.byref(dt))
                    if dt.value <= 0:
                        break
                else:
                    if t.value >= dur:
                        break
                    self.L.call('EN_stepQ', self.h, ct.byref(dt))
            ratio = self.statistic(EN_MASSBALANCE)
        finally:
            self.L.call('EN_closeQ', self.h)
        return np.asarray(times, dtype=np.int64), np.asarray(rows), ratio
