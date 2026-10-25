#!/bin/bash
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /data/home/scxi690/wdn_quantum/logs/%x_%j.out
#SBATCH -e /data/home/scxi690/wdn_quantum/logs/%x_%j.out
# r1_job.sh - npj revision timings on one RTX 5090 (5090补测执行计划.md). STEP selects the job:
#   A1 BWSN2 200 sensors (T1)  A2 L-Town 10 sensors (T2)  B ctol_study part 3 (T3 + T5)
#   C BWSN2 96 h / 168 h (T6)  T4 parallel library finite differences  G gradient validation (R1-3 / R2-3)  B2 R2-5 / R1-8 / R1-6 extras
#   R2-11: R11S smoke  R11R robust designs (ROB_NET)  R11G GA (GA_NET, GA_M, GA_SEED)  R11M MILP certificate (Net3, m = 3)
#   R2-8: MC MCMC posterior + LOSO (MC_DATA)
#   R1-5 chunked replay: L48 checks + two-level identification at 48 h (validation)  L168 BWSN2 168 h
# Submit from the login node: sbatch -J r1_A1 -t 06:00:00 --export=ALL,STEP=A1 ~/wdn_quantum/jobs/r1_job.sh
set -o pipefail
WQ=/data/home/scxi690/wdn_quantum
OUT=$WQ/results/r1_${STEP}_$SLURM_JOB_ID
export TMPDIR=/tmp/diffwq_$SLURM_JOB_ID          # node-local scratch for .hyd/.rpt files
mkdir -p "$OUT" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== job $SLURM_JOB_ID step $STEP on $(hostname)  $(date)"
echo "partition=$SLURM_JOB_PARTITION cpus=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh || exit 1
conda activate dhn_gpu || exit 1
cd "$WQ/research/water_quality_2026_09_12" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -c "import torch, platform; print('python', platform.python_version(), 'torch', torch.__version__, torch.cuda.get_device_name(0))" || exit 2
LIB=$WQ/EPANET-dev/build/lib/libepanet2.so
REF=$WQ/results/diffwq_exp2_1588990

run() {  # run <label> <command...> : time one step; a failing step never aborts the job
  echo; echo "=== $1  $(date)"; shift
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== exit=$rc elapsed=$((SECONDS - t0))s"
}
case $STEP in
  A1) run "T1 BWSN2 200 sensors" python -m diffwq.exp2_v2 --ref $REF/exp2_BWSN2.json --configs bw200x:v2:0.5:200 \
        --exhaustive 20 --chunk 16 --out-dir "$OUT" --lib $LIB ;;
  A2) run "T2 L-Town 10 sensors" python -m diffwq.exp2_v2 --ref $REF/exp2_LTown.json --configs lt10:v2:0.5:10 \
        --exhaustive 20 --chunk 64 --out-dir "$OUT" --lib $LIB ;;
  B)  run "T3+T5 part 3, 8 networks" python -m diffwq.ctol_study --part 3 --lib $LIB --out "$TMPDIR/out" \
        --nets Net1,Net2,Net3,Jilin,Fairfield,Bellingham,NorthPenn,LTown
      cp "$TMPDIR"/out/*.json "$OUT"/
      run "T3+T5 part 3, BWSN2" python -m diffwq.ctol_study --part 3 --lib $LIB --out "$TMPDIR/out" --nets BWSN2
      cp "$TMPDIR"/out/*.json "$OUT"/ ;;
  C)  run "T6 BWSN2 96 h" python -m diffwq.bwsn2_long --hours 96 --ref $REF/exp2_BWSN2.json --lib $LIB --out "$OUT/bwsn2_96h.json"
      run "T6 BWSN2 168 h" python -m diffwq.bwsn2_long --hours 168 --ref $REF/exp2_BWSN2.json --lib $LIB --out "$OUT/bwsn2_168h.json" ;;
  G)  export OMP_NUM_THREADS=1             # R1-3 / R2-3 gradient validation (rerun of the V100 jobs A, BL, BW)
      mkdir -p "$TMPDIR/gc"
      for j in smoke A BL BW; do
        run "gradcheck $j" python -m diffwq.gradcheck_v2 --job $j --lib $LIB --out "$TMPDIR/gc" --workers 8
        cp "$TMPDIR"/gc/gradcheck_$j.json "$OUT"/
      done ;;
  B2) THETA=$WQ/results/diffwq_g4g_1587008      # R2-5 events, R1-8 prior s.d. scan, R1-6 site-level error term
      run "R2-5 hydraulic events" python -m diffwq.hyd_events --lib $LIB --out "$OUT/hyd_events.json"
      for tau in 0.1 0.3 1.0; do
        run "R1-8 field audit tau=$tau" python -m diffwq.exp3_pipe_kw --field Fairfield,Bellingham,NorthPenn --theta-dir "$THETA"           --device cuda --field-tau $tau --lib $LIB --out "$OUT/exp3_field_tau$tau.json"
      done
      run "R1-6 site-level error LOSO" python -m diffwq.uq_site --datasets Fairfield,Bellingham,NorthPenn --device cuda --lib $LIB         --out "$OUT/uq_site.json" ;;
  R11S) D1=$WQ/results/diffwq_exp1_1588989       # R2-11 smoke test (small settings)
      run "R2-11 smoke: robust" python -m diffwq.exp1_robust --network Net3 --designs $D1/exp1_Net3.json --scenarios "nominal,demand x1.2" --lib $LIB --out "$OUT/robust_smoke.json"
      run "R2-11 smoke: GA" python -m diffwq.exp1_ga --network Net3 --m 3 --pop 8 --gens 2 --search-iters 100 --lib $LIB --out "$OUT/ga_smoke.json"
      run "R2-11 smoke: MILP" python -m diffwq.exp1_milp --designs $D1/exp1_Net3.json --m 3 --time-limit 60 --lib $LIB --out "$OUT/milp_smoke.json" ;;
  R11R) D1=$WQ/results/diffwq_exp1_1588989       # R2-11 parts 1-2: hydraulic and chlorine-demand sensitivity, ROB_NET = Net3 | NorthPenn
      TH=""; [ "$ROB_NET" = "NorthPenn" ] && TH="--theta-json g4_graph_5090/g4_graph_NorthPenn_cuda.json"
      run "R2-11 robust $ROB_NET" python -m diffwq.exp1_robust --network $ROB_NET --designs $D1/exp1_$ROB_NET.json $TH ${ROB_RESUME:+--resume $ROB_RESUME} --lib $LIB --out "$OUT/robust_$ROB_NET.json" ;;
  R11G) TH=""; [ "$GA_NET" = "NorthPenn" ] && TH="--theta-json g4_graph_5090/g4_graph_NorthPenn_cuda.json"
      run "R2-11 GA $GA_NET m=$GA_M seed=$GA_SEED" python -m diffwq.exp1_ga --network $GA_NET --m $GA_M --seed $GA_SEED $TH --lib $LIB --out "$OUT/ga_${GA_NET}_m${GA_M}_s${GA_SEED}.json" ;;
  R11M) D1=$WQ/results/diffwq_exp1_1588989       # R2-11 MILP certificate (Net3, m = 3), GA supports from the R11G jobs
      run "R2-11 MILP Net3 m=3" python -m diffwq.exp1_milp --designs $D1/exp1_Net3.json --m 3 --ga "$WQ/results/r1_R11G_*/ga_Net3_m3_s*.json" --time-limit 1800 --lib $LIB --out "$OUT/milp_Net3_m3.json" ;;
  L48) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True       # R1-5 chunked replay: checks, 48-h validation
      REFV2=$WQ/results/r1_A1_1723390/exp2_v2_bw200x.json
      run "R1-5 chunked replay checks (small networks, GPU)" python -m diffwq.verify_chunked --device cuda --lib $LIB --out "$OUT/chunked_check_cuda.json"
      run "R1-5 two-level identification BWSN2 48 h" python -m diffwq.exp2_long --ref $REFV2 --hours 48 --max-events 16 --chunk 16         --validate --lib $LIB --out "$OUT/exp2_long_48h.json" ;;
  L48b) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True      # R1-5 two-level identification at 48 h only
      REFV2=$WQ/results/r1_A1_1723390/exp2_v2_bw200x.json
      run "R1-5 two-level identification BWSN2 48 h" python -m diffwq.exp2_long --ref $REFV2 --hours 48 --max-events 16 --chunk 16         --validate --lib $LIB --out "$OUT/exp2_long_48h.json" ;;
  L168) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True      # R1-5 chunked replay: BWSN2 168 h
      REFV2=$WQ/results/r1_A1_1723390/exp2_v2_bw200x.json
      run "R1-5 chunked replay BWSN2 168 h" python -m diffwq.exp2_long --ref $REFV2 --hours 168 --max-events 16 --chunk 32         --validate --lib $LIB --out "$OUT/exp2_long_168h.json" ;;
  T4) export OMP_NUM_THREADS=1
      run "T4 parallel library FD" python -m diffwq.fd_parallel --procs 1,2,4,8 --lib $LIB --out "$OUT/fd_parallel.json" ;;
  MC) run "R2-8 MCMC $MC_DATA" python -m diffwq.uq_mcmc --datasets $MC_DATA --device cuda --chains 16 --warmup 1000 --draws 1000 --lib $LIB --out "$OUT/mcmc_$MC_DATA.json" ;;
  *)  echo "unknown STEP=$STEP"; exit 3 ;;
esac
echo; echo "=== results in $OUT"; ls -la "$OUT"
echo "=== done $(date)"
