#!/bin/bash
#SBATCH -J diffwq_g3s
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_g3s_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_g3s_%j.out
# diffwq_g3_spmm.sh - engine v3 (per-op cuSPARSE CSR SpMM for shared-parameter sweeps) on one RTX 5090 node:
# (1) G2 exactness of the fused engine with the SpMM path forced at every batch size (forward vs DLL, adjoint vs
# FD, fused vs v1 adjoint); (2) G3 shared-theta cost table (B initial-quality / source columns, gradients w.r.t.
# c0 and src, Jacobian rows with chunk = B) for gather-only, SpMM-only and the automatic per-op choice, with
# CUDA-graph replay, on the G2 benchmark set and on the experiment networks (L-Town, BWSN2; one snapshot per
# quality step as in Exp3). Every engine after the first is checked against the first on identical inputs.
# Package layout on the cluster mirrors the workspace (~/wdn_quantum/research/water_quality_2026_09_12).
# Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_g3_spmm.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_g3s_$SLURM_JOB_ID
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

ENG=fused_gather,fused_spmm,fused
run "G2 exactness, fused engine, SpMM path forced" \
  python -m diffwq.verify_torch --engine fused_spmm --device cuda --out "$OUT/g2_fused_spmm.json"
run "G3 shared-theta, G2 benchmark set (gather / spmm / auto, jac rows, graph)" \
  python -m diffwq.bench --device cuda --shared-theta --jac-rows --graph --engines $ENG \
         --batches 1,16,64,256,1024 --repeat 3 --out "$OUT/g3_shared_core.json"
run "G3 shared-theta, L-Town (stepQ)" \
  python -m diffwq.bench --device cuda --shared-theta --jac-rows --graph --engines $ENG --mode stepQ \
         --cases LTown_kw --batches 1,16,64,256,1024 --repeat 3 --out "$OUT/g3_shared_LTown_stepQ.json"
run "G3 shared-theta, BWSN2 (stepQ, eager)" \
  python -m diffwq.bench --device cuda --shared-theta --jac-rows --engines $ENG --mode stepQ \
         --cases BWSN2_kw --batches 1,16,64,128 --repeat 3 --out "$OUT/g3_shared_BWSN2_stepQ.json"

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
