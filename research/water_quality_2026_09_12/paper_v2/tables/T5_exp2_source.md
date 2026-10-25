# Exp2: contamination source identification from sparse sensors

## BWSN2: 12527 nodes / 14831 links, 48 h, 192 slots of 900 s, 2,404,416 unknowns (12523 candidate nodes), 59,856,795 fused terms

sensors 200 (greedy detection coverage on 4096 training scenarios, detectable fraction 1.000); test scenarios 200, detected 126, false alarms 0, median detection delay 11.00 h


### Identification by observation window after first detection (GLRT = parametric box-constrained likelihood ratio; NNLS = non-parametric l1 profile)

| window h | scenarios | GLRT top-1 | GLRT top-5 | GLRT correct or equivalent | median equivalent set | median |start err| h | median mass rel err | NNLS top-1 | NNLS correct or equiv | refine wall s per scenario | projected exhaustive DLL s | exhaustive n | same best as top-M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 126 | 0.937 | 0.992 | 1 | 1 | 0 | 0.00211 | 0.389 | 0.667 | 1.04 | 5.84e+06 | 20 | 0.8 |
| 6 | 126 | 0.929 | 1 | 1 | 1 | 0 | 3.54e-04 | 0.571 | 0.746 | 0.891 | 6.71e+06 | 20 | 0.95 |
| inf | 126 | 0.976 | 1 | 1 | 1 | 0 | 2.45e-04 | 0.675 | 0.754 | 0.832 | 1.06e+07 | 20 | 0.9 |


DLL check of the truth runs: 3 scenarios, max rel err 4.78e-16


timing s: record_s 292.1, build_s 102.6, setup_s 413.4, dll_quality_run_s 4.4, placement_s 15.3, data_s 2.0, sweep_s 5263.7, screen_s 15.8, refine_s 348.2, exhaustive_s 508.1, identification_s 6136.2, adjoint_row_passes 42400.0, graph_capture_s 0.0, total_s 6566.8

## LTown: 785 nodes / 909 links, 48 h, 576 slots of 300 s, 450,432 unknowns (782 candidate nodes), 8,003,505 fused terms

sensors 10 (greedy detection coverage on 4096 training scenarios, detectable fraction 1.000); test scenarios 200, detected 178, false alarms 0, median detection delay 4.42 h


### Identification by observation window after first detection (GLRT = parametric box-constrained likelihood ratio; NNLS = non-parametric l1 profile)

| window h | scenarios | GLRT top-1 | GLRT top-5 | GLRT correct or equivalent | median equivalent set | median |start err| h | median mass rel err | NNLS top-1 | NNLS correct or equiv | refine wall s per scenario | projected exhaustive DLL s | exhaustive n | same best as top-M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 178 | 0.916 | 0.955 | 1 | 1 | 0 | 0.00384 | 0.258 | 0.449 | 0.819 | 6.42e+03 | 20 | 0.55 |
| 6 | 178 | 0.983 | 0.994 | 1 | 1 | 0 | 6.41e-05 | 0.416 | 0.64 | 0.814 | 7.75e+03 | 20 | 0.8 |
| inf | 178 | 0.994 | 0.994 | 1 | 1 | 0 | 5.91e-05 | 0.427 | 0.635 | 0.813 | 1.59e+04 | 20 | 1 |


DLL check of the truth runs: 3 scenarios, max rel err 4.89e-16


timing s: record_s 11.3, build_s 27.3, setup_s 40.8, dll_quality_run_s 0.0, placement_s 3.6, data_s 0.7, sweep_s 135.7, screen_s 21.4, refine_s 435.5, exhaustive_s 237.5, identification_s 830.2, adjoint_row_passes 2093.0, graph_capture_s 0.0, total_s 875.2
