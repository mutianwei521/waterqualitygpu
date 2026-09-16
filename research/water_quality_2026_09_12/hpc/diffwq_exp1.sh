#!/bin/bash
#SBATCH -J diffwq_exp1
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_exp1_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_exp1_%j.out
# diffwq_exp1.sh - Exp1 (booster siting + dosing schedule as one convex program on the exact EPANET map) on one
# RTX 5090 node: Net3 (chlorine system, 7-day horizon, 24-slot periodic schedule, 95 candidates) and North Penn
# (field total-chlorine model with the G4-calibrated Kb / Kw, one-shot 36-slot schedule, 328 candidates).
# Group-lasso path (14 lam), refit / constant-rate baselines, greedy forward selection (8 stages) with a batched
# refit, mass-weight sweep, dense response-matrix cross-check, DLL check of every reported design.
# Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_exp1.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_exp1_$SLURM_JOB_ID
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

run() {  # run <label> <command...> : time one step; a failing step never aborts the job
  echo; echo "=== $1  $(date)"; shift
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== exit=$rc elapsed=$((SECONDS - t0))s"
}

THETA=g4_graph_5090/g4_graph_NorthPenn_cuda.json
ls -la "$THETA" || echo "WARNING: $THETA missing (North Penn runs with the INP kinetics)"

run "Exp1 Net3 (matrix cross-check)" \
  python -m diffwq.exp1_booster --network Net3 --device cuda --matrix --chunk 256 --out "$OUT/exp1_Net3.json"
if [ -f "$THETA" ]; then
  run "Exp1 NorthPenn, G4 theta (matrix cross-check)" \
    python -m diffwq.exp1_booster --network NorthPenn --theta-json "$THETA" --device cuda --matrix --chunk 256 \
           --out "$OUT/exp1_NorthPenn.json"
else
  run "Exp1 NorthPenn, INP kinetics (matrix cross-check)" \
    python -m diffwq.exp1_booster --network NorthPenn --device cuda --matrix --chunk 256 --out "$OUT/exp1_NorthPenn.json"
fi

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
