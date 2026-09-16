"""Reader for EPANET's binary hydraulics file (project.c openhydfile, output.c savehyd).

Layout (little-endian, INT4 = int32, REAL4 = float32):
  header : magic, version, Nnodes, Nlinks, Ntanks, Npumps, Nvalves, Dur   (8 x int32)
  record : t | demand[Nnodes] | head[Nnodes] | flow[Nlinks] | status[Nlinks] |
           setting[Nlinks] | hydstep                                       (repeated)
The last record has hydstep == 0 and is followed by an EOF mark byte.

The quality engine (quality.c runqual -> output.c readhyd) consumes exactly these
float32 values widened to double, with flows of closed links already zeroed.
"""
from dataclasses import dataclass
import numpy as np

MAGICNUMBER = 516114521
ENGINE_VERSION = 201


@dataclass
class HydPeriod:
    t: int
    hydstep: int
    demand: np.ndarray   # float64 view of float32 data, 1-based (index 0 unused)
    head: np.ndarray
    flow: np.ndarray
    status: np.ndarray   # int, EPANET StatusType codes (CLOSED = 2, OPEN = 3, ...)
    setting: np.ndarray


@dataclass
class HydTrace:
    nnodes: int
    nlinks: int
    ntanks: int
    npumps: int
    nvalves: int
    duration: int
    periods: list


def read_hydfile(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    header = buf[:32].view('<i4')
    magic, version = int(header[0]), int(header[1])
    if magic != MAGICNUMBER or version != ENGINE_VERSION:
        raise ValueError(f'not an EPANET 2.3 hydraulics file: magic={magic} version={version}')
    nnodes, nlinks, ntanks, npumps, nvalves, dur = (int(v) for v in header[2:8])
    off = 32
    periods = []
    rec_floats = 2 * nnodes + 3 * nlinks

    def take_f4(n):
        nonlocal off
        arr = buf[off:off + 4 * n].view('<f4').astype(np.float64)
        off += 4 * n
        return np.concatenate(([0.0], arr))   # 1-based like the C arrays

    while off + 4 <= buf.size:
        t = int(buf[off:off + 4].view('<i4')[0]); off += 4
        if off + 4 * rec_floats + 4 > buf.size:
            raise ValueError('truncated hydraulics file')
        demand = take_f4(nnodes)
        head = take_f4(nnodes)
        flow = take_f4(nlinks)
        status = take_f4(nlinks).astype(np.int64)   # (char)x[i] in readhyd
        setting = take_f4(nlinks)
        hydstep = int(buf[off:off + 4].view('<i4')[0]); off += 4
        periods.append(HydPeriod(t, hydstep, demand, head, flow, status, setting))
        if hydstep == 0:
            break
    return HydTrace(nnodes, nlinks, ntanks, npumps, nvalves, dur, periods)
