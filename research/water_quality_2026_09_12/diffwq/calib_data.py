"""G4 field data: the four measured water-quality data sets shipped with the ASCE TF-WDST
network corpus (`networks/asce-tf-wdst/`), read into (node index, time, value) triples.

  Net2-FL     fluoride tracer, Cherry Hill / Brushy Plains (Rossman et al. 1994); node ids in the file
  Fairfield   free chlorine, Fairfield CA Zone III, AWWARF 1993; site -> node map in SITES.LST
  Bellingham  free chlorine, Bellingham WA Dakin-Yew, 1993;      site -> node map in SITES.LST
  NorthPenn   total chlorine, North Penn / Lansdale PA, 1993;    site -> node map in SITES.LST

Measurement files not listed in SITES.LST (Bellingham CL2-1/7/10, Fairfield FWP-CL, North Penn
N1/N8) have no node assignment and are not used. Times are hours from the start of the
simulation; points beyond the run duration are dropped, as the EPANET calibration report does.
BELL_CL1.INP is written in EPANET 1.x syntax ([CONTROL], [REACTION]); EPANET 2.3 rejects those
section names (error 200), so a copy with the 2.x names is written to a work directory. The
original INP files are never modified.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

from . import epanet_api as api

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORPUS = ROOT / 'networks' / 'asce-tf-wdst'

DATASETS = {
    'Net2-FL': {'inp': CORPUS / 'Net2-FL' / 'Net2.inp', 'dat': [CORPUS / 'Net2-FL' / 'Net2-FL.dat'], 'sites': None},
    'Fairfield': {'inp': CORPUS / 'Fairfield' / 'FFCL-1.INP', 'sites': CORPUS / 'Fairfield' / 'SITES.LST'},
    'Bellingham': {'inp': CORPUS / 'Bellingham' / 'BELL_CL1.INP', 'sites': CORPUS / 'Bellingham' / 'SITES.LST',
                   'legacy': True},
    'NorthPenn': {'inp': CORPUS / 'North Penn Water Authority System' / 'NPCL-1.INP',
                  'sites': CORPUS / 'North Penn Water Authority System' / 'SITES.LST'},
}
LEGACY_SECTIONS = {'[CONTROL]': '[CONTROLS]', '[REACTION]': '[REACTIONS]'}


@dataclass
class Observations:
    dataset: str
    site_node_id: list          # per site: EPANET node id
    site_file: list             # per site: measurement file name
    node: np.ndarray            # [M] node index (1-based, EPANET order)
    t: np.ndarray               # [M] seconds from simulation start
    y: np.ndarray               # [M] measured concentration, user units (mg/L)
    site: np.ndarray            # [M] site index
    n_dropped: int              # points beyond the run duration

    def __len__(self):
        return int(self.y.size)


def prepare_inp(name, workdir):
    """Path of the INP to open: the original, or (legacy syntax) a fixed copy in workdir."""
    ds = DATASETS[name]
    src = Path(ds['inp'])
    if not ds.get('legacy'):
        return src
    text = src.read_text(errors='replace')
    for old, new in LEGACY_SECTIONS.items():
        text = re.sub(r'^(\s*)' + re.escape(old), lambda m: m.group(1) + new, text, flags=re.M | re.I)
    dst = Path(workdir) / (src.stem + '_en2' + src.suffix)
    dst.write_text(text)
    return dst


def _read_sites(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        parts = line.split()
        out.append((parts[0], parts[1]))
    return out


def _read_dat(path):
    """Two-column (time, value) file with optional ';' comments."""
    rows = []
    for line in Path(path).read_text(errors='replace').splitlines():
        line = line.split(';')[0].strip()
        if not line:
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1])))
    return rows


def _read_net2fl(path):
    """Three columns (node, hour, value); a blank node field repeats the previous node."""
    rows, node = [], None
    for line in Path(path).read_text(errors='replace').splitlines():
        if not line.strip() or line.lstrip().startswith(';'):
            continue
        parts = line.split()
        if len(parts) == 3:
            node = parts[0]; hour, val = parts[1], parts[2]
        elif len(parts) == 2 and node is not None:
            hour, val = parts
        else:
            raise ValueError(f'{path}: cannot parse line {line!r}')
        rows.append((node, float(hour), float(val)))
    return rows


def load_observations(name, pr: api.Project) -> Observations:
    """Observations of data set `name` mapped onto the node indices of the opened project."""
    ds = DATASETS[name]
    dur = pr.time_param(api.EN_DURATION)
    site_ids, site_files, node, t, y, site = [], [], [], [], [], []
    if name == 'Net2-FL':
        by_node = {}
        for nid, hour, val in _read_net2fl(ds['dat'][0]):
            by_node.setdefault(nid, []).append((hour, val))
        entries = [(nid, Path(ds['dat'][0]).name, rows) for nid, rows in by_node.items()]
    else:
        folder = Path(ds['inp']).parent
        entries = []
        for nid, fname in _read_sites(ds['sites']):
            matches = [p for p in folder.iterdir() if p.name.lower() == fname.lower()]
            if not matches:
                raise FileNotFoundError(f'{name}: {fname} listed in SITES.LST not found')
            entries.append((nid, matches[0].name, _read_dat(matches[0])))
    dropped = 0
    for nid, fname, rows in entries:
        idx = pr.node_index(nid)
        s = len(site_ids)
        site_ids.append(nid); site_files.append(fname)
        for hour, val in rows:
            ts = hour * 3600.0
            if ts > dur:
                dropped += 1
                continue
            node.append(idx); t.append(ts); y.append(val); site.append(s)
    return Observations(name, site_ids, site_files, np.asarray(node, dtype=np.int64), np.asarray(t),
                        np.asarray(y), np.asarray(site, dtype=np.int64), dropped)
