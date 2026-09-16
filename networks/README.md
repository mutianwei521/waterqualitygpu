# networks/ — where the benchmark and field data go

These are third-party network files and are not redistributed here. Download them into this
directory, keeping the folder names below: the code addresses several of them by exact path
(`diffwq/verify_m0.py`, `diffwq/calib_data.py`, `paper/figures/paperfig.py`).

| Directory | Contents used by this code | Source |
|---|---|---|
| `asce-tf-wdst/` | The ASCE Task Committee / University of Kentucky research database. Supplies the three field chlorine data sets used for calibration (`Bellingham/`, `Fairfield/`, `North Penn Water Authority System/`: INP plus `SITES.LST` and the measurement files), `Battle of the Water Sensor Networks/BWSN_Network_2.inp` (12,527 nodes), `Jilin including water quality/`, and `Net2-FL/`. | Water Distribution System Research Database, https://uknowledge.uky.edu/wdsrd (DOI 10.13023/kwrri.wdsrd); older listing at https://www.uky.edu/WDST/database.html |
| `L-Town/` | `L-TOWN.inp`, the BattLeDIM 2020 network (785 nodes, 909 links). | https://zenodo.org/records/4017659 (DOI 10.5281/zenodo.4017659); competition site https://battledim.ucy.ac.cy/ |
| `epanet-example-networks/` | OWA example and multi-species networks; part of the corpus scan of gate G1. | https://github.com/OpenWaterAnalytics/epanet-example-networks |
| `exeter-benchmarks/` | University of Exeter Centre for Water Systems benchmarks (EXNET, D-Town BWN-II, Anytown and others); also part of the corpus scan. | https://www.exeter.ac.uk/research/centres/cws/resources/benchmarks/ |
| `epanet-desktop/`, `pangaea/` | Further INP files picked up by the corpus scan (`Net1`, `Net3`, `NYT.inp`, `PacificCity.inp`). | https://github.com/OpenWaterAnalytics/epanet-desktop; PANGAEA data publisher |

`Net1.inp`, `Net2.inp` and `Net3.inp` are read from `EPANET-dev/example-networks/`, that is from the
EPANET source checkout, not from here.

Gate G1's corpus scan (`python -m diffwq.verify_m0 --set corpus`) walks every `*.inp` under this
directory and keeps the ones whose `[OPTIONS] QUALITY` is not `NONE`; the paper reports the 56 that
qualified in our copy of the collections above. A different snapshot of the same sources may yield a
different count without contradicting the per-case results, which are all reported individually.

Each collection carries its own license and citation requirements. Please honour them.
