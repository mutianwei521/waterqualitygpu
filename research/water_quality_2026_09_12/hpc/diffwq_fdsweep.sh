#!/bin/bash
#SBATCH -J diffwq_fds
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_fds_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_fds_%j.out
# diffwq_fdsweep.sh - step-size sweep of EPANET central differences against the replay gradient (G2 §3.1 table),
# at the INP Ctol (M0) and at Ctol = 0 (M1), on the cluster-built libepanet2.so. CPU-only DLL runs; the GPU
# allocation is only what the partition requires. Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_fdsweep.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_fds_$SLURM_JOB_ID
export TMPDIR=/tmp/diffwq_$SLURM_JOB_ID
mkdir -p "$OUT" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== job $SLURM_JOB_ID on $(hostname)  $(date)"
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)'
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh || exit 1
conda activate dhn_gpu || exit 1
cd "$WQ/research/water_quality_2026_09_12" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

run() {  # run <label> <command...> : time one step; a failing step never aborts the job
  echo; echo "=== $1  $(date)"; shift
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== exit=$rc elapsed=$((SECONDS - t0))s"
}

HS=${HS:-1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7}
# SPECS overrides the case:param list, e.g. the directions that straddle kw = 0 / the reaction-off branch:
#   sbatch --export=ALL,SPECS="Net2:kw Net2:kb Net3_CHEM_mixed_tanks:kw Net3_CHEM_mixed_tanks:kb" diffwq_fdsweep.sh
SPECS=${SPECS:-"Net3_CHEM_kw_mixed_tanks:kb Net3_CHEM_kw_mixed_tanks:kw Net2_react:kb Net2_react:c0 Jilin_kw:kb Net1:kb"}
for spec in $SPECS; do
  case=${spec%%:*}; param=${spec##*:}
  run "FD sweep $case / $param" \
    python -m diffwq.fd_sweep --case "$case" --param "$param" --hs "$HS" --out "$OUT/g2_fd_sweep_${case}_${param}.json"
done

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
