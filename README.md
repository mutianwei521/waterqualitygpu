# waterqualitygpu

**Update history**

| Date | Release |
|---|---|
| 2026-09-16 | First release: code and results of the original submission |
| 2026-10-25 | Update: revision analyses requested by the reviewers (see [Revision analyses](#revision-analyses-october-2026)) |

**更新记录**

| 日期 | 内容 |
|---|---|
| 2026-09-16 | 首次发布：原稿的代码与结果 |
| 2026-10-25 | 更新：按审稿意见新增的返修分析（见下方 Revision analyses） |

Reproduction code for **"Exact Model Gradients for Chlorine Management and Contamination Source
Identification in Drinking Water Distribution Networks."**

EPANET predicts drinking-water quality but provides no derivatives. This repository makes EPANET
2.3's Lagrangian water-quality calculation exactly differentiable: a Python replicator reproduces
the engine's parcel transport operation by operation and **records** the parcel-concentration
dependency graph with concentration merging disabled; PyTorch then **replays** that recorded
schedule on a GPU, so reverse-mode differentiation yields exact derivatives with respect to bulk
and wall decay coefficients, initial qualities, and source strengths.

The code covers everything reported in the paper:

| Stage | Module | What it establishes |
|---|---|---|
| G1 | `diffwq.verify_m0` | the replicator matches the compiled engine bit for bit (91 cases) |
| G2 | `diffwq.verify_torch`, `diffwq.fd_sweep`, `diffwq.verify_src` | GPU replay and adjoint vs. autograd vs. central differences |
| G3 | `diffwq.bench` | cost of one exact gradient, batch scaling, CUDA graphs, sparse variants |
| G4 | `diffwq.calib`, `diffwq.perpipe`, `diffwq.uq`, `diffwq.site_influence` | field chlorine calibration, per-pipe estimation, uncertainty |
| Exp 1 | `diffwq.exp1_booster` | booster siting and dosing as one program on the exact map |
| Exp 2 | `diffwq.exp2_source` | contamination-source ranking |
| Exp 3 | `diffwq.exp3_pipe_kw` | pipe-level wall-decay identifiability |

The revision (October 2026) added the analyses requested by the reviewers; they are listed in
[Revision analyses](#revision-analyses-october-2026) below.

## Repository layout

The scripts resolve every path relative to their own location, so the directory nesting below
matters. Keep it as it is; two of the three top-level directories are third-party material you
download yourself.

```
<repo root>/
├── EPANET-dev/                             OWA EPANET 2.3 source (you clone it, see below)
├── networks/                               benchmark and field networks (you download, see below)
└── research/water_quality_2026_09_12/
    ├── diffwq/                             the differentiable engine (record -> replay -> adjoint)
    ├── hpc/                                the SLURM job scripts that produced every result
    │   └── results_5090/                   G1-G3 result JSONs from those jobs
    ├── final_5090/                         final-engine result JSONs (G2 sweeps, G3 shared, Exp 1-3)
    ├── v2_5090/, g4_graph_5090/, h2_5090/   calibration, per-pipe and uncertainty result JSONs
    ├── final_5090_v2/                      revision runs on the RTX 5090 (one directory per Slurm job)
    ├── final_v2/, final_ctol/, final_long/, final_chunked/, final_mcmc/
    │                                        further revision results (see the table below)
    ├── paper/
    │   ├── build_tables.py                 rebuilds every table and every number of the first submission
    │   ├── tables/                          T1-T6 plus numbers.json (141 numbers)
    │   └── figures/                         figure scripts and the shared style module paperfig.py
    ├── paper_v2/                            the same for the revised manuscript, plus the new SI tables
    │   ├── build_tables.py, *_tables.py, ident_scan.py, perf_ledger.py, resid_diagnostics.py
    │   ├── tables/                          T1-T6, T_S_*.md and their numbers JSONs
    │   └── figures/                         revised figure scripts (Fig. 4, Fig. S4, S6, S8, S9, ...)
    ├── audit_inputs.py, pilot_quality.py    input inventory and the step/tolerance sensitivity audit
    ├── cvpipe_probe/                        13-case check-valve pipe probe of the released engine
    └── REPRODUCE.md                         notes for the audit stage (in Chinese)
```

## Prerequisites

* Python 3.11 or newer. `pip install -r requirements.txt`. PyTorch must be a CUDA build:
  the published runs used torch 2.8.0+cu128 on one NVIDIA RTX 5090 (compute capability 12.0).
  Everything also runs on CPU (`--device cpu`), only slower.
* **EPANET 2.3 shared library.** The replicator reads hydraulics from the engine's own binary
  `.hyd` file and calls the engine through `ctypes`, so a real EPANET build is required; the
  reported runs used engine version 20306.

  ```bash
  git clone https://github.com/OpenWaterAnalytics/EPANET.git EPANET-dev
  cmake -S EPANET-dev -B EPANET-dev/build -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build EPANET-dev/build --config Release --target epanet2
  export EPANET_LIB=$PWD/EPANET-dev/build/lib/libepanet2.so     # .dll on Windows
  ```

  `diffwq/epanet_api.py` looks at `$EPANET_LIB` first, then
  `research/water_quality_2026_09_12/build_epanet/bin/RelWithDebInfo/epanet2.dll` on Windows, then
  `~/wdn_quantum/EPANET-dev/build/lib/libepanet2.so`.
* **Networks and field data.** Not redistributed here; see [`networks/README.md`](networks/README.md)
  for the sources and the exact directory names the code expects.

## Reproducing the paper

Run everything from `research/water_quality_2026_09_12`. The GPU stages are long; the SLURM scripts
in `hpc/` are the exact recipes that produced the shipped JSONs, including the CLI flags,
partition, and wall-time limits. Their `#SBATCH` paths are from our cluster and need editing for
yours.

```bash
cd research/water_quality_2026_09_12

# G1  operation-level exactness against the compiled engine
python -m diffwq.verify_m0 --set core     --out g1_core_nextQ.json
python -m diffwq.verify_m0 --set variants --out g1_variants_nextQ.json
python -m diffwq.verify_m0 --set corpus   --out g1_corpus_nextQ.json      # 56 networks, ~10 min

# G2  adjoint vs autograd vs central differences, and the finite-difference step sweep
python -m diffwq.verify_torch --device cuda --engine both --out g2_nextQ_cuda_both.json
python -m diffwq.fd_sweep --case Net2_react --param kb --hs 1e-2,1e-3,1e-4,1e-5,1e-6 --out g2_fd_Net2_kb.json

# G3  gradient cost, batch scaling, CUDA graphs
python -m diffwq.bench --device cuda --batches 1,16,128,1024,4096,8192 --graph --repeat 5 --out g3_cuda.json

# G4  field chlorine calibration, per-pipe wall decay, uncertainty
python -m diffwq.calib --datasets Fairfield --methods de,nm,fd,adj --seeds 5 --device cuda --out g4_calib_Fairfield.json
python -m diffwq.uq --datasets Fairfield,Bellingham,NorthPenn --device cuda --samples 4096 --out h2_uq.json

# Applications
python -m diffwq.exp1_booster  --network Net3   --device cuda --matrix --chunk 256 --out exp1_Net3.json
python -m diffwq.exp2_source   --network LTown  --device cuda --place-batch 64 --chunk 128 --out exp2_LTown.json
python -m diffwq.exp3_pipe_kw  --network BWSN2  --device cuda --chunk 128 --out exp3_BWSN2.json
```

Tables and figures are built from the result JSONs already in this repository, so they need no GPU
and no EPANET build:

```bash
python paper/build_tables.py          # writes paper/tables/T1-T6.md and numbers.json
python paper/figures/fig4_exp2.py     # each figure script writes pdf/svg/png/tif into paper/figures/out
```

`build_tables.py` regenerates all seven table files and all 141 numbers byte for byte from the
shipped JSONs. Figure scripts that draw network maps additionally need `networks/`.

## Revision analyses (October 2026)

Every revision run used one RTX 5090 of the same Slurm cluster. `hpc/r1_job.sh` is the job script;
its `STEP` variable selects the run, and the result directories in `final_5090_v2/` are named
`r1_<STEP>_<jobid>`, each with the job's log next to it (`r1_<STEP>_<jobid>.out`). The script reads
earlier results from `$WQ/results/...` on our cluster; the same files are in this repository
(`final_5090/`, `g4_graph_5090/`, `final_5090_v2/`), so adjust those paths before reuse.

| Topic (reviewer comment) | Module | `STEP` | Results | Table / figure code |
|---|---|---|---|---|
| Effect of the zero merging tolerance (R1-1, R2-2) | `diffwq.ctol_study` | `B` | `final_ctol/`, `final_5090_v2/r1_B_*` | `paper_v2/ctol_tables.py`, `paper_v2/figures/figS_ctol.py` |
| Independent finite-difference check of the gradients (R1-3, R2-3) | `diffwq.gradcheck_v2` | `G` | `final_5090_v2/r1_G_1723889` | `paper_v2/gradcheck_tables.py` |
| Parallel library finite differences, crossover (R1-4, R2-4) | `diffwq.fd_parallel` | `T4` | `final_5090_v2/r1_T4_*` | `paper_v2/perf_ledger.py` |
| Horizons of 96 h and 168 h (R1-5, R2-6) | `diffwq.bwsn2_long` | `C` | `final_5090_v2/r1_C_*`, `final_long/` | `paper_v2/perf_ledger.py`, `paper_v2/figures/figS_scaling.py` |
| Time-chunked replay and two-level source identification (R1-5) | `diffwq.chunked`, `diffwq.exp2_long`, `diffwq.verify_chunked` | `L48`, `L48b`, `L168` | `final_chunked/` | — |
| Residual diagnostics, site-level error term (R1-6, R2-8) | `diffwq.uq_site` | `B2` | `final_5090_v2/r1_B2_*` | `paper_v2/resid_diagnostics.py` |
| MCMC posterior of the calibration (R2-8) | `diffwq.uq_mcmc` | `MC` | `final_mcmc/` | — |
| End-to-end source identification, sensor sets (R1-7) | `diffwq.exp2_v2` | `A1`, `A2` | `final_5090_v2/r1_A*`, `final_v2/` | `paper_v2/build_tables.py`, `paper_v2/figures/fig4_exp2.py` |
| Identifiability vs. threshold and prior (R1-8) | `diffwq.exp3_pipe_kw --field-tau` | `B2` | `final_5090_v2/r1_B2_*` | `paper_v2/ident_scan.py` |
| Hydraulic events in the recorded trajectories (R2-5) | `diffwq.hyd_events` | `B2` | `final_5090_v2/r1_B2_*` | `paper_v2/events_table.py` |
| Booster design sensitivity, GA and MILP comparison (R2-11) | `diffwq.exp1_robust`, `diffwq.exp1_ga`, `diffwq.exp1_milp` | `R11R`, `R11G`, `R11M` | `final_5090_v2/r1_R11*` | `paper_v2/r211_tables.py` |

Examples (from `research/water_quality_2026_09_12`; the full set of flags is in `hpc/r1_job.sh`):

```bash
python -m diffwq.ctol_study --part 3 --nets Net1,Net2,Net3 --out ctol_out
python -m diffwq.gradcheck_v2 --job A --workers 8 --out gc
python -m diffwq.verify_chunked --device cuda --out chunked_check.json        # chunked vs unchunked replay
python -m diffwq.exp2_long --ref final_5090_v2/r1_A1_1723390/exp2_v2_bw200x.json --hours 168        --max-events 16 --chunk 32 --validate --out exp2_long_168h.json
python -m diffwq.uq_mcmc --datasets NorthPenn --device cuda --chains 16 --warmup 1000 --draws 1000 --out mcmc_NorthPenn.json
python -m diffwq.exp1_ga --network Net3 --m 3 --seed 0 --out ga_Net3_m3_s0.json
```

The revised tables are rebuilt from the shipped JSONs without a GPU or an EPANET build:

```bash
python paper_v2/build_tables.py
python paper_v2/ctol_tables.py
python paper_v2/gradcheck_tables.py
python paper_v2/events_table.py
python paper_v2/ident_scan.py
python paper_v2/perf_ledger.py
python paper_v2/r211_tables.py
python paper_v2/resid_diagnostics.py
```

Each writes into `paper_v2/tables/`; on the shipped data all of them reproduce the committed files
(the last digit of two fitted exponents in `perf_ledger_points.json` can differ between platforms).
On Windows set `PYTHONIOENCODING=utf-8` first, since the scripts print Unicode.

Notes on the revision results:

* `final_ctol/ctol_part1f_merged.jsonl` (all 2,970 tasks of part 1, computed on an RTX 3090 node)
  is the file the tables use; `ctol_part1f_3090.jsonl` is the raw log of that run.
* `final_v2/exp2/` holds the R1-7 source-identification results in the layout of the first
  submission (see its `README.txt`); `final_v2/exp2_v2_*.json` are the sensor-set study.
* `final_chunked/local/` are CPU checks of the chunked replay; `final_chunked/c5090/` the GPU runs at
  48 h and 168 h. The first 168-h attempt (`log_cancelled_after_batch1.txt`) used a different random
  stream and is kept for reference only.

## What is and is not here

Included: the engine, the experiment drivers, the cluster job scripts, the table and figure code,
and the result JSONs behind every published table and number, for the first submission and for the
revision.

Not included: the manuscript text; rendered figure files; the third-party networks and the EPANET
source (both fetched as above); the compiled EPANET build directory; and earlier superseded runs of
the same experiments (the shipped `final_5090/` runs are the ones the paper reports).

## Citation

Mu, T.; Wang, Y.; Yuan, M.; Luo, Q.; Xiao, M.; Wang, W.; Li, J.; Huang, M. Exact Model Gradients
for Chlorine Management and Contamination Source Identification in Drinking Water Distribution
Networks. Preprint, 2026.

## License

MIT, see [LICENSE](LICENSE). EPANET and the benchmark networks keep their own licenses.

---

## 中文说明

本仓库是论文《Exact Model Gradients for Chlorine Management and Contamination Source
Identification in Drinking Water Distribution Networks》的复现代码。核心思路：先用 Python 逐操作复现
EPANET 2.3 的拉格朗日水质输运并在关闭浓度合并的条件下**记录**水团浓度依赖关系，再用 PyTorch 在 GPU 上
**重放**该记录，于是反向模式微分给出关于衰减系数、初始水质与源强度的精确导数。

* 目录结构必须保持 README 中的三层布局，因为所有脚本按自身位置解析相对路径。
* 运行前需要两样外部材料：自行编译的 EPANET 2.3 动态库（放好后设 `EPANET_LIB` 环境变量），以及
  `networks/` 下的基准与实测管网数据（来源见 `networks/README.md`，本仓库不转发布）。
* 论文的全部表格与 141 个数字可以直接由仓库内的结果 JSON 重建，不需要 GPU：
  `python paper/build_tables.py`。图件脚本在 `paper/figures/` 下，绘制管网底图的脚本额外需要
  `networks/`。
* GPU 实验的完整命令、分区与时限见 `hpc/*.sh`，其中的 `#SBATCH` 路径来自我们的集群，需按自己的环境修改。
* **返修（2026 年 10 月）新增内容：**
  - 新模块：零合并容差研究 `ctol_study`、独立梯度验证 `gradcheck_v2`、并行有限差分 `fd_parallel`、
    长时长 `bwsn2_long`、分段回放 `chunked` / `exp2_long` / `verify_chunked`、站级误差项 `uq_site`、
    MCMC 后验 `uq_mcmc`、源识别 v2 `exp2_v2`、水力事件 `hyd_events`、加氯设计敏感性与方法对比
    `exp1_robust` / `exp1_ga` / `exp1_milp`。
  - 作业脚本 `hpc/r1_job.sh`（用 `STEP` 选择运行内容），结果在 `final_5090_v2/` 等目录，对应关系见上方英文表格。
  - 返修版表格由 `paper_v2/` 下的脚本从仓库内的 JSON 重建，不需要 GPU；Windows 上请先设 `PYTHONIOENCODING=utf-8`。
