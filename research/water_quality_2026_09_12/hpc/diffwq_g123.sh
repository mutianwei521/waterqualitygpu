#!/bin/bash
#SBATCH -J diffwq_g123
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_g123_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_g123_%j.out
# diffwq_g123.sh - Gates G1/G2/G3 of the differentiable water-quality engine on one RTX 5090.
# Package layout on the cluster mirrors the workspace: ~/wdn_quantum/{research/water_quality_2026_09_12/diffwq,
# EPANET-dev/example-networks, networks}. Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_g123.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_g123_$SLURM_JOB_ID
export TMPDIR=/tmp/diffwq_$SLURM_JOB_ID          # node-local scratch for .hyd/.rpt files
mkdir -p "$OUT" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== job $SLURM_JOB_ID on $(hostname)  $(date)"
echo "partition=$SLURM_JOB_PARTITION cpus=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh || exit 1
conda activate dhn_gpu || exit 1
cd "$WQ/research/water_quality_2026_09_12" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python - <<'EOF' || exit 2
import torch, platform
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda.is_available", torch.cuda.is_available(), "arch_list", torch.cuda.get_arch_list())
p = torch.cuda.get_device_properties(0)
print("device", p.name, "capability", f"{p.major}.{p.minor}", "mem_GB", round(p.total_memory / 2**30, 1))
EOF

run() {  # run <label> <command...> : time one step; a failing step never aborts the job
  echo; echo "=== $1  $(date)"; shift
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== exit=$rc elapsed=$((SECONDS - t0))s"
}
run "G1 core nextQ (gcc/M_PI build of libepanet2)" python -m diffwq.verify_m0 --set core --out "$OUT/g1_core_nextQ.json"
run "G1 variants nextQ" python -m diffwq.verify_m0 --set variants --out "$OUT/g1_variants_nextQ.json"
run "G1 core stepQ" python -m diffwq.verify_m0 --set core --mode stepQ --out "$OUT/g1_core_stepQ.json"
run "G2 cuda adjoint vs autograd vs central differences" python -m diffwq.verify_torch --device cuda --engine both --out "$OUT/g2_nextQ_ctolinp_cuda_both.json"
run "G2 cuda Ctol=0 (mode M1)" python -m diffwq.verify_torch --device cuda --engine adjoint --ctol 0 --out "$OUT/g2_nextQ_ctol0_cuda.json"
run "G3 cuda batch sweep + CUDA graphs" python -m diffwq.bench --device cuda --batches 1,16,128,1024,4096,8192 --graph --repeat 5 --out "$OUT/g3_bench_cuda.json"
run "G3 cpu reference (8 threads)" python -m diffwq.bench --device cpu --batches 1,16,128,1024 --out "$OUT/g3_bench_cpu.json"
run "G1 corpus nextQ (56 networks, ~10 min in Python)" python -m diffwq.verify_m0 --set corpus --out "$OUT/g1_corpus_nextQ.json"

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
