#!/bin/bash
#SBATCH -J diffwq_g4g
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/diffwq_g4g_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/diffwq_g4g_%j.out
# diffwq_g4_graph.sh - closes the G4 wall-clock comparison on one RTX 5090 node: L-BFGS-B with the
# forward-difference DLL gradient (FD, node CPU), with the eager adjoint gradient (ADJ, GPU) and with the
# adjoint gradient replayed from a CUDA-graph capture of the B=1 map (ADJ-graph, GPU; the capture uq.py
# uses for its fits). Each of the three is repeated 3 times per data set (free Kb, and Kb fixed at the
# INP value). Package layout on the cluster mirrors the workspace: ~/wdn_quantum/{research/
# water_quality_2026_09_12/diffwq, networks/asce-tf-wdst, EPANET-dev}.
# Submit from the login node: sbatch ~/wdn_quantum/jobs/diffwq_g4_graph.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/diffwq_g4g_$SLURM_JOB_ID
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
# one process per data set and Kb treatment, sequential (the DLL runs share the node's CPU with torch's
# host side; nothing else runs on the node). --threads 0 -> SLURM_CPUS_PER_TASK torch CPU threads.
for ds in Fairfield Bellingham NorthPenn; do
  run "G4 graph $ds free Kb (fd, adj, adjg x3)" \
    python -m diffwq.calib --datasets $ds --methods fd,adj,adjg --repeat 3 --device cuda \
           --out "$OUT/g4_graph_${ds}_cuda.json"
  run "G4 graph $ds Kb fixed at the INP value (fd, adj, adjg x3)" \
    python -m diffwq.calib --datasets $ds --methods fd,adj,adjg --repeat 3 --device cuda --fix-kb \
           --out "$OUT/g4_graph_${ds}_fixkb_cuda.json"
done

echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
