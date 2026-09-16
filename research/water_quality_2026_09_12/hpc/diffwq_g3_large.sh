#!/bin/bash
#SBATCH -J diffwq_g3L
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_g3L_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_g3L_%j.out
# diffwq_g3_large.sh - Gate G3 at large batch sizes on one RTX 5090 (32 GB), one process per case so that
# no memory from a previous case or CUDA-graph pool is carried over. Follows diffwq_g123.sh (job 1586077),
# where B=8192 ran out of memory only because earlier graph pools were still held.
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_g3L_$SLURM_JOB_ID
export TMPDIR=/tmp/diffwq_$SLURM_JOB_ID
mkdir -p "$OUT" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== job $SLURM_JOB_ID on $(hostname)  $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh || exit 1
conda activate dhn_gpu || exit 1
cd "$WQ/research/water_quality_2026_09_12" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

run() {  # run <case> <batches>
  echo; echo "=== $1 B=$2  $(date)"
  local t0=$SECONDS
  python -m diffwq.bench --device cuda --cases "$1" --batches "$2" --graph --repeat 5 --out "$OUT/g3_${1}.json"
  echo "=== exit=$? elapsed=$((SECONDS - t0))s"
}
run Net1                      8192,16384,32768,65536
run Net2_react                6144,8192
run Jilin_kw                  6144,8192
run Net3_CHEM_kw_mixed_tanks  6144,8192

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
