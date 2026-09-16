#!/bin/bash
#SBATCH -J diffwq_v2
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_v2_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_v2_%j.out
# diffwq_v2.sh - engine v2 (fused single-state replay, step flattening, time-slot / virtual sources) on one
# RTX 5090 node: (1) exactness of the fused engines on the G2 case set (forward vs DLL, adjoint vs FD, fused vs
# v1 adjoint); (2) time-slot + virtual sources vs DLL runs with patterned MASS sources; (3) G3 cost table for the
# three engines with CUDA-graph replay; (4) the G4 FD / ADJ / ADJ-graph wall-clock comparison with the fused
# engine. Package layout on the cluster mirrors the workspace (~/wdn_quantum/research/water_quality_2026_09_12).
# Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_v2.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_v2_$SLURM_JOB_ID
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

run() {  # run <label> <command...> : time one step; a failing step never aborts the job
  echo; echo "=== $1  $(date)"; shift
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== exit=$rc elapsed=$((SECONDS - t0))s"
}

run "G2 exactness, fused engine (flatten)" \
  python -m diffwq.verify_torch --engine fused --device cuda --out "$OUT/g2_fused.json"
run "G2 exactness, fused engine (kind-merge only)" \
  python -m diffwq.verify_torch --engine fused_nf --device cuda --out "$OUT/g2_fused_nf.json"
run "time-slot + virtual sources vs DLL" \
  python -m diffwq.verify_src --cases Net1,Net2_react,Jilin_kw,Net3_CHEM_kw_mixed_tanks --device cuda \
         --out "$OUT/verify_src.json"
run "G3 cost table, three engines, CUDA graph" \
  python -m diffwq.bench --device cuda --engines adjoint,fused_nf,fused --batches 1,16,128,1024,4096 \
         --repeat 3 --graph --out "$OUT/g3_bench_v2_cuda.json"
for ds in Fairfield Bellingham NorthPenn; do
  run "G4 fused $ds free Kb (fd, adj, adjg x3)" \
    python -m diffwq.calib --datasets $ds --methods fd,adj,adjg --repeat 3 --device cuda --engine fused \
           --out "$OUT/g4_fused_${ds}_cuda.json"
  run "G4 fused $ds Kb fixed (fd, adj, adjg x3)" \
    python -m diffwq.calib --datasets $ds --methods fd,adj,adjg --repeat 3 --device cuda --engine fused --fix-kb \
           --out "$OUT/g4_fused_${ds}_fixkb_cuda.json"
done

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
