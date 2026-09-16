# waterqualitygpu

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
    ├── paper/
    │   ├── build_tables.py                 rebuilds every table and every number quoted in the paper
    │   ├── tables/                          T1-T6 plus numbers.json (141 numbers)
    │   └── figures/                         figure scripts and the shared style module paperfig.py
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

## What is and is not here

Included: the engine, the experiment drivers, the cluster job scripts, the table and figure code,
and the result JSONs behind every published table and number.

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
