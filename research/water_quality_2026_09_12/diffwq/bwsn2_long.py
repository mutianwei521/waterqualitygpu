"""R1-5: measured size and memory of the BWSN2 Exp2 schedule beyond 48 h, on one GPU. New module; v1 files untouched.
The Exp2 setting (conservative contaminant, virtual MASS sources at every junction, 900-s slots, nextQ, Ctol 0) is recorded
for --hours; measured: recording / build time, host peak RSS, schedule size, GPU peak of one B = 1 forward + adjoint and of
one adjoint row pass (identification sweep). The normal equations of the identification are sized from the recorded schedule
(N_junctions x K^2 x 8 B), not allocated.
Usage (from research/water_quality_2026_09_12):  python -m diffwq.bwsn2_long --hours 96 --ref exp2_BWSN2.json --lib LIB --out f.json
"""
import argparse, json, resource, tempfile, time
import numpy as np
import torch
from . import epanet_api as api
from . import exp2_source as E

ap = argparse.ArgumentParser()
ap.add_argument('--hours', type=int, required=True); ap.add_argument('--ref', required=True)
ap.add_argument('--lib', required=True); ap.add_argument('--out', required=True); ap.add_argument('--chunk', type=int, default=32)
a = ap.parse_args()
J = json.load(open(a.ref)); n0 = J['networks'][0]
args = argparse.Namespace(**J['args']); args.device, args.no_graph, args.chunk = 'cuda', True, a.chunk
E.CONFIGS['BWSN2']['duration_h'] = a.hours
t0 = time.perf_counter()
P = E.SourceProblem('BWSN2', api.Library(a.lib), tempfile.mkdtemp(prefix='bw_long_', dir='.'), args)
R = {'hours': a.hours, 'setup_s': time.perf_counter() - t0, 'record_s': P.t_record, 'build_s': P.t_build, 'library_run_s': P.t_dll,
     'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, **P.sch_info, 'K_slots': P.K, 'junctions': len(P.junc),
     'state_slots_M': int(P.eng.fsch.M), 'host_peak_rss_GB_after_setup': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20}
N1 = P.net.nnodes + 1
sync = torch.cuda.synchronize
sync(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
x = torch.zeros(1, P.K, N1, dtype=torch.float64, device='cuda', requires_grad=True)
t1 = time.perf_counter(); P.eng.forward(src=x).sum().backward(); sync(); R['fwd_adj_B1_s'] = time.perf_counter() - t1
R['gpu_resident_after_build_GB'] = base / 2 ** 30
R['gpu_peak_fwd_adj_B1_GB'] = torch.cuda.max_memory_allocated() / 2 ** 30
del x; torch.cuda.empty_cache()
P.set_sensors([P.net.node_ids.index(s) for s in n0['placement']['sensors']])          # the 20 sensors of the paper
sync(); torch.cuda.reset_peak_memory_stats()
try:
    t1 = time.perf_counter(); A = P.rows(0, a.chunk); sync(); R['adjoint_row_pass_s'] = time.perf_counter() - t1
    R['gpu_peak_row_pass_GB'] = torch.cuda.max_memory_allocated() / 2 ** 30; del A
except Exception as e:
    R['row_pass_error'] = f'{type(e).__name__}: {str(e)[:200]}'
R['sized_not_allocated'] = {'normal_equations_all_junctions_GB': len(P.junc) * P.K ** 2 * 8 / 2 ** 30,
                            'adjoint_state_and_cotangent_chunk_GB': 2 * P.eng.fsch.M * a.chunk * 8 / 2 ** 30,
                            'one_row_block_GB': a.chunk * P.K * N1 * 8 / 2 ** 30}
R['host_peak_rss_GB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
R['gpu_total_GB'] = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
json.dump(R, open(a.out, 'w'), indent=1, default=float)
print(json.dumps(R, default=float))
