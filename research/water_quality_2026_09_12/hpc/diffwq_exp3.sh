#!/bin/bash
#SBATCH -J diffwq_exp3
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_exp3_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_exp3_%j.out
# diffwq_exp3.sh - Exp3 (pipe-by-pipe wall-decay calibration, Laplace identifiability audit, D-optimal sensor
# prescription) on one RTX 5090 node. Field data sets (Fairfield, Bellingham, North Penn) audited at the G4
# grouped estimate theta* of job diffwq_g4g_1587008; synthetic studies on L-Town (905 pipes, 10 + 5 sensors, all
# junctions as candidates) and BWSN Network 2 (14,822 pipes, 20 + 10 sensors, 256 candidates): 48 h chlorine,
# one snapshot per 5-min quality step, sensors every 15 min, noise 0.02 mg/L. Jacobian rows by the shared-forward
# adjoint with `chunk` unit cotangents per sweep: L-Town 256; BWSN2 128 (adjoint state 8.8 M slots -> ~9 GB per
# sweep next to the 5.9 GB candidate Jacobian), automatic retry at chunk 64 if the first attempt fails.
# Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_exp3.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_exp3_$SLURM_JOB_ID
THETA=$WQ/results/diffwq_g4g_1587008
export TMPDIR=/tmp/diffwq_$SLURM_JOB_ID
mkdir -p "$OUT" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== job $SLURM_JOB_ID on $(hostname)  $(date)"
echo "partition=$SLURM_JOB_PARTITION cpus=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread' ; free -g 2>/dev/null | head -2
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh || exit 1
conda activate dhn_gpu || exit 1
cd "$WQ/research/water_quality_2026_09_12" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python - <<'EOF' || exit 2
import torch, platform, scipy
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda, "scipy", scipy.__version__)
p = torch.cuda.get_device_properties(0)
print("device", p.name, "capability", f"{p.major}.{p.minor}", "mem_GB", round(p.total_memory / 2**30, 1))
EOF
ls "$THETA"/g4_graph_*_cuda.json || echo "WARNING: no G4 theta files in $THETA"

RC=0
run() {  # run <label> <command...> : time one step; a failing step never aborts the job (exit code in RC)
  echo; echo "=== $1  $(date)"; shift
  local t0=$SECONDS
  "$@"; RC=$?
  echo "=== exit=$RC elapsed=$((SECONDS - t0))s"
}

run "Exp3 field audits at theta* (Fairfield, Bellingham, North Penn)" \
  python -m diffwq.exp3_pipe_kw --field Fairfield,Bellingham,NorthPenn --theta-dir "$THETA" --device cuda \
         --out "$OUT/exp3_field.json"

run "Exp3 L-Town (stepQ, chunk 256)" \
  python -m diffwq.exp3_pipe_kw --network LTown --device cuda --chunk 256 --out "$OUT/exp3_LTown.json"

run "Exp3 BWSN2 (stepQ, chunk 128)" \
  python -m diffwq.exp3_pipe_kw --network BWSN2 --device cuda --chunk 128 --out "$OUT/exp3_BWSN2.json"
if [ $RC -ne 0 ]; then
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>&1
  run "Exp3 BWSN2 retry (stepQ, chunk 64)" \
    python -m diffwq.exp3_pipe_kw --network BWSN2 --device cuda --chunk 64 --design-chunk 16 --out "$OUT/exp3_BWSN2.json"
fi

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
