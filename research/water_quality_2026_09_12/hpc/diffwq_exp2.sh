#!/bin/bash
#SBATCH -J diffwq_exp2
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_exp2_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_exp2_%j.out
# diffwq_exp2.sh - Exp2 (contamination source identification: node, start, duration, mass from sparse sensor
# series) on one RTX 5090 node. L-Town (782 junctions, 5-min slots, K=576, 10 sensors) and BWSN Network 2
# (12,523 junctions, 15-min slots, K=192, 20 sensors): greedy detection-likelihood placement on 512 training
# scenarios, 200 test scenarios (noise 0.1 mg/L, alarm 0.5 mg/L confirmed twice), one adjoint row sweep per
# batch of 8 scenarios (normal equations of every junction on the 2 h / 6 h / full windows), exact box GLRT of
# every junction (parametric), NNLS + BIC refinement of the top-30 (nonparametric), exhaustive refinement of the
# first 20 scenarios, DLL check of the truth series. Memory (measured eager peaks): L-Town 23 MB/sample -> CUDA
# graph at chunk 128 (~14 GB); BWSN2 3.3 GB static + 218 MB/sample -> eager (no graph) at chunk 32 (chunk 64 ran
# out of memory in job 1587794 next to the 3.7 GB normal-equation accumulator), automatic retry at chunk 16.
# Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_exp2.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_exp2_$SLURM_JOB_ID
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
import torch, platform
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda)
p = torch.cuda.get_device_properties(0)
print("device", p.name, "capability", f"{p.major}.{p.minor}", "mem_GB", round(p.total_memory / 2**30, 1))
EOF

RC=0
run() {  # run <label> <command...> : time one step; a failing step never aborts the job (exit code in RC)
  echo; echo "=== $1  $(date)"; shift
  local t0=$SECONDS
  "$@"; RC=$?
  echo "=== exit=$RC elapsed=$((SECONDS - t0))s"
}

COMMON="--device cuda --n-train 512 --n-test 200 --batch 8 --exhaustive 20 --gpu-budget-gb 24 --windows 2,6,inf"

run "Exp2 L-Town (chunk 128)" \
  python -m diffwq.exp2_source --network LTown $COMMON --place-batch 64 --chunk 128 --out "$OUT/exp2_LTown.json"

run "Exp2 BWSN2 (eager, chunk 32)" \
  python -m diffwq.exp2_source --network BWSN2 $COMMON --no-graph --place-batch 16 --chunk 32 --out "$OUT/exp2_BWSN2.json"
if [ $RC -ne 0 ]; then
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>&1
  run "Exp2 BWSN2 retry (eager, chunk 16)" \
    python -m diffwq.exp2_source --network BWSN2 $COMMON --no-graph --place-batch 8 --chunk 16 --out "$OUT/exp2_BWSN2.json"
fi

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
