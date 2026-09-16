# Exp2: contamination source identification from sparse sensors

## BWSN2: 12527 nodes / 14831 links, 48 h, 192 slots of 900 s, 2,404,416 unknowns (12523 candidate nodes), 59,856,795 fused terms

sensors 20 (greedy detection coverage on 512 training scenarios, detectable fraction 0.967); test scenarios 200, detected 77, false alarms 0, median detection delay 14.25 h


### Identification by observation window after first detection (GLRT = parametric box-constrained likelihood ratio; NNLS = non-parametric l1 profile)

| window h | scenarios | GLRT top-1 | GLRT top-5 | GLRT correct or equivalent | median equivalent set | median |start err| h | median mass rel err | NNLS top-1 | NNLS correct or equiv | refine wall s per scenario | projected exhaustive DLL s | exhaustive n | same best as top-M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 77 | 0.675 | 0.857 | 0.987 | 1 | 0 | 0.0347 | 0.169 | 0.571 | 0.957 | 6.21e+06 | 20 | 0.4 |
| 6 | 77 | 0.831 | 0.935 | 1 | 1 | 0 | 0.00151 | 0.273 | 0.675 | 0.927 | 7.06e+06 | 20 | 0.7 |
| inf | 77 | 0.87 | 0.948 | 1 | 1 | 0 | 0.00134 | 0.325 | 0.636 | 0.959 | 1.05e+07 | 20 | 0.7 |


DLL check of the truth runs: 3 scenarios, max rel err 4.78e-16


timing s: record_s 251.2, build_s 90.9, setup_s 366.4, dll_quality_run_s 4.4, placement_s 1.7, data_s 0.6, sweep_s 210.2, screen_s 9.7, refine_s 218.9, exhaustive_s 591.5, identification_s 1030.6, adjoint_row_passes 1330.0, graph_capture_s 0.0, total_s 1438.0

## LTown: 785 nodes / 909 links, 48 h, 576 slots of 300 s, 450,432 unknowns (782 candidate nodes), 8,003,505 fused terms

sensors 10 (greedy detection coverage on 512 training scenarios, detectable fraction 1.000); test scenarios 200, detected 174, false alarms 0, median detection delay 3.67 h


### Identification by observation window after first detection (GLRT = parametric box-constrained likelihood ratio; NNLS = non-parametric l1 profile)

| window h | scenarios | GLRT top-1 | GLRT top-5 | GLRT correct or equivalent | median equivalent set | median |start err| h | median mass rel err | NNLS top-1 | NNLS correct or equiv | refine wall s per scenario | projected exhaustive DLL s | exhaustive n | same best as top-M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 174 | 0.937 | 0.977 | 1 | 1 | 0 | 0.0631 | 0.276 | 0.523 | 1.17 | 9.24e+03 | 20 | 0.7 |
| 6 | 174 | 0.96 | 0.989 | 1 | 1 | 0 | 7.47e-05 | 0.466 | 0.69 | 1.17 | 1.12e+04 | 20 | 0.95 |
| inf | 174 | 0.977 | 0.989 | 1 | 1 | 0 | 5.09e-05 | 0.448 | 0.678 | 1.17 | 2.32e+04 | 20 | 0.95 |


DLL check of the truth runs: 3 scenarios, max rel err 4.89e-16


timing s: record_s 15.3, build_s 28.3, setup_s 45.5, dll_quality_run_s 0.1, placement_s 0.3, data_s 0.1, sweep_s 97.7, screen_s 28.6, refine_s 609.9, exhaustive_s 244.8, identification_s 981.1, adjoint_row_passes 1012.0, graph_capture_s 1.4, total_s 1032.4
