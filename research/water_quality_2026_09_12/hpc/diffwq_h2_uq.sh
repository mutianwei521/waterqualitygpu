#!/bin/bash
#SBATCH -J diffwq_h2
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_h2_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_h2_%j.out
# diffwq_h2_uq.sh - H2 probe (linearised covariance + Laplace sampling + LOSO coverage) of the three
# chlorine field data sets on one RTX 5090, then the G4 calibration comparison (calib.py) on the same
# node. Package layout on the cluster mirrors the workspace: ~/wdn_quantum/{research/water_quality_2026_09_12/diffwq,
# networks/asce-tf-wdst, EPANET-dev}. Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_h2_uq.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_h2_$SLURM_JOB_ID
export TMPDIR=/tmp/diffwq_$SLURM_JOB_ID          # node-local scratch for .hyd/.rpt files
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
# H2: fits through the CUDA-graph B=1 replay (or CPU if measured faster), sampling in batches of 1024 on the GPU,
# CPU (8 threads) reference of the sampled field for the device cross-check and the wall-time comparison.
run "H2 free Kb (S=4096 map draws, 1024 per LOSO fold)" \
  python -m diffwq.uq --datasets Fairfield,Bellingham,NorthPenn --device cuda --samples 4096 --loso-samples 1024 \
         --chunk 1024 --cpu-chunk 256 --out "$OUT/h2_uq.json"
run "H2 Kb fixed at the INP value" \
  python -m diffwq.uq --datasets Fairfield,Bellingham,NorthPenn --device cuda --samples 4096 --loso-samples 1024 \
         --chunk 1024 --cpu-chunk 256 --fix-kb --out "$OUT/h2_uq_fixkb.json"
# G4 on the same node (calib.py has no graph capture: its B=1 adjoint runs eagerly on the GPU, the DE/NM/FD baselines
# call libepanet2.so on the node's CPU); one process per data set, sequential.
for ds in Fairfield Bellingham NorthPenn; do
  run "G4 calib $ds (de x5, nm, fd, adj; cuda eager)" \
    python -m diffwq.calib --datasets $ds --methods de,nm,fd,adj --seeds 5 --device cuda --out "$OUT/g4_calib_${ds}_cuda.json"
done

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
