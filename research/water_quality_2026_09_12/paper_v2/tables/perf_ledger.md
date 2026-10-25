# Performance and memory ledger (read-only summary of existing results)
Times are only comparable within one hardware column. 5090 = paper; the tolerance study part 3 (B), the BWSN2 long horizons (C) and the parallel finite differences (T4) were re-measured on the 5090 node on 2026-10-22 (final_5090_v2/).

## A. Scale and preprocessing
| case | hardware | nodes | steps | versions | recorded terms | fused terms | record s | build s |
|---|---|---|---|---|---|---|---|---|
| G2 Net1 | RTX 5090 |  | 290 | 3950 | 11356 | 14478 | 0.185 | 0.9 |
| G2 Net2 | RTX 5090 |  | 660 | 27101 | 81062 | 207140 | 0.654 | 5.54 |
| G2 Net3 | RTX 5090 |  | 290 | 35089 | 103444 | 654928 | 0.694 | 4.85 |
| G2 Jilin | RTX 5090 |  | 1152 | 39202 | 117481 | 156345 | 0.905 | 1.99 |
| G2 Net1_MIX2 | RTX 5090 |  | 290 | 4118 | 11811 | 16031 | 0.175 | 0.67 |
| G2 Net1_FIFO | RTX 5090 |  | 290 | 3690 | 10880 | 13547 | 0.166 | 0.496 |
| G2 Net1_LIFO | RTX 5090 |  | 290 | 3690 | 10908 | 13652 | 0.134 | 0.497 |
| G2 Net1_MIX1_overflow | RTX 5090 |  | 289 | 4059 | 11395 | 15044 | 0.183 | 0.617 |
| G2 Net1_FIFO_overflow | RTX 5090 |  | 289 | 3808 | 10746 | 13301 | 0.145 | 0.534 |
| G2 Net1_LIFO_overflow | RTX 5090 |  | 289 | 3808 | 10746 | 13367 | 0.146 | 0.549 |
| G2 Net1_src_mass | RTX 5090 |  | 290 | 3950 | 11672 | 15487 | 0.186 | 0.709 |
| G2 Net1_src_setpoint | RTX 5090 |  | 290 | 3950 | 11655 | 14877 | 0.17 | 0.693 |
| G2 Net1_src_flowpaced | RTX 5090 |  | 290 | 3950 | 11655 | 15538 | 0.168 | 0.702 |
| G2 Net1_src_concen_reservoir_pat | RTX 5090 |  | 290 | 3950 | 11503 | 15351 | 0.171 | 0.649 |
| G2 Net1_src_concen_junction_inflow | RTX 5090 |  | 291 | 3958 | 11883 | 15576 | 0.197 | 0.656 |
| G2 Net1_src_mass_tank | RTX 5090 |  | 290 | 3950 | 11578 | 15325 | 0.147 | 0.688 |
| G2 Net1_ctol0 | RTX 5090 |  | 290 | 3950 | 11039 | 15610 | 0.255 | 0.582 |
| G2 Net2_MIX2 | RTX 5090 |  | 660 | 27462 | 81992 | 195948 | 0.639 | 5.26 |
| G2 Net2_LIFO | RTX 5090 |  | 660 | 26530 | 79998 | 185795 | 0.575 | 4.37 |
| G2 Net3_CHEM_mixed_tanks | RTX 5090 |  | 290 | 34655 | 103746 | 760693 | 0.795 | 4.77 |
| G2 Net3_AGE_mixed_tanks | RTX 5090 |  | 290 | 34709 | 102355 | 339992 | 0.865 | 2.28 |
| G2 Net3_TRACE_River_mixed_tanks | RTX 5090 |  | 290 | 34679 | 102774 | 625651 | 0.625 | 4.46 |
| G2 Net2_react | RTX 5090 |  | 660 | 27101 | 81328 | 186676 | 0.874 | 5.22 |
| G2 Jilin_kw | RTX 5090 |  | 1152 | 39202 | 117324 | 172210 | 1.11 | 1.72 |
| G2 Net3_CHEM_kw_mixed_tanks | RTX 5090 |  | 290 | 34662 | 103654 | 439946 | 0.819 | 3.3 |
| L-Town stepQ 48 h | RTX 5090 | 785 | 580 | 528365 |  | 4822038 | 19.3 | 18.7 |
| BWSN2 stepQ 48 h | RTX 5090 | 12527 | 595 | 8835091 |  | 40921757 | 881 | 57.7 |
| Exp1 Net3 168 h | RTX 5090 | 97 | 2030 | 244903 |  | 1785271 | 14.8 | 14.4 |
| Exp1 North Penn 36 h | RTX 5090 | 339 | 360 | 144761 |  | 900842 | 9.34 | 2.43 |
| Exp2 L-Town 48 h | RTX 5090 | 785 | 580 | 528365 |  | 8003505 | 15.3 | 28.3 |
| Exp2 BWSN2 48 h | RTX 5090 | 12527 | 595 | 8835017 |  | 59856795 | 251 | 90.9 |
| Exp3 L-Town 48 h | RTX 5090 | 785 | 580 | 528365 |  | 4822038 | 17.9 | 19.9 |
| Exp3 BWSN2 48 h | RTX 5090 | 12527 | 595 | 8835091 |  | 40921757 | 866 | 59.3 |
| Calibration Fairfield | RTX 5090 |  | 240 | 30607 |  |  | 1.05 | 0.111 |
| Calibration Bellingham | RTX 5090 |  | 350 | 60020 |  |  | 1.77 | 0.117 |
| Calibration NorthPenn | RTX 5090 |  | 360 | 144761 |  |  | 6.4 | 0.0442 |
| BWSN2 48 h Ctol 0.0 | RTX 5090 | 12527 | 593 | 8805439 | 25676194 | 40826301 | 656 | 39.8 |
| BWSN2 48 h Ctol 0.01 | RTX 5090 | 12527 | 593 | 8805439 | 26416675 | 462636160 | 158 | 604 |
| Net1 48 h Ctol 0.0 | RTX 5090 | 11 | 579 | 7873 | 22272 | 31164 | 0.385 | 1.2 |
| Net1 48 h Ctol 0.01 | RTX 5090 | 11 | 579 | 7873 | 22869 | 27771 | 0.198 | 0.865 |
| Net1 168 h Ctol 0.0 | RTX 5090 | 11 | 2029 | 27594 | 79742 | 113185 | 1.34 | 2.88 |
| Net1 168 h Ctol 0.01 | RTX 5090 | 11 | 2029 | 27594 | 80273 | 96424 | 0.644 | 2.94 |
| Net2 48 h Ctol 0.0 | RTX 5090 | 36 | 576 | 23657 | 68092 | 98236 | 1.68 | 1.16 |
| Net2 48 h Ctol 0.01 | RTX 5090 | 36 | 576 | 23657 | 71079 | 166728 | 0.543 | 3.1 |
| Net2 168 h Ctol 0.0 | RTX 5090 | 36 | 2021 | 82768 | 246870 | 359273 | 6.76 | 4.04 |
| Net2 168 h Ctol 0.01 | RTX 5090 | 36 | 2021 | 82768 | 249198 | 616429 | 1.83 | 11.1 |
| Net3 48 h Ctol 0.0 | RTX 5090 | 97 | 583 | 70243 | 206671 | 361309 | 2.41 | 1.93 |
| Net3 48 h Ctol 0.01 | RTX 5090 | 97 | 583 | 70243 | 209071 | 548339 | 1.22 | 3.74 |
| Net3 168 h Ctol 0.0 | RTX 5090 | 97 | 2053 | 246598 | 732887 | 1287667 | 10.9 | 6.76 |
| Net3 168 h Ctol 0.01 | RTX 5090 | 97 | 2053 | 246598 | 736756 | 1752216 | 4.62 | 12.9 |
| Jilin 48 h Ctol 0.0 | RTX 5090 | 28 | 576 | 19618 | 58558 | 85473 | 0.412 | 0.56 |
| Jilin 48 h Ctol 0.01 | RTX 5090 | 28 | 576 | 19618 | 58590 | 84382 | 0.372 | 0.637 |
| Jilin 168 h Ctol 0.0 | RTX 5090 | 28 | 2016 | 68578 | 205438 | 300288 | 1.38 | 2.03 |
| Jilin 168 h Ctol 0.01 | RTX 5090 | 28 | 2016 | 68578 | 205470 | 294257 | 1.37 | 2.12 |
| Fairfield 48 h Ctol 0.0 | RTX 5090 | 112 | 480 | 61077 | 177382 | 290607 | 2.31 | 1.96 |
| Fairfield 48 h Ctol 0.01 | RTX 5090 | 112 | 480 | 61077 | 182288 | 380349 | 1.28 | 2.84 |
| Fairfield 168 h Ctol 0.0 | RTX 5090 | 112 | 1033 | 131133 | 391073 | 645485 | 5.81 | 4.26 |
| Fairfield 168 h Ctol 0.01 | RTX 5090 | 112 | 1033 | 131133 | 393036 | 729396 | 2.94 | 5.86 |
| Bellingham 48 h Ctol 0.0 | RTX 5090 | 123 | 480 | 82250 | 243878 | 505421 | 2.2 | 2.34 |
| Bellingham 48 h Ctol 0.01 | RTX 5090 | 123 | 480 | 82250 | 245447 | 767594 | 1.41 | 3.72 |
| Bellingham 168 h Ctol 0.0 | RTX 5090 | 123 | 1680 | 287450 | 855885 | 1766417 | 9.14 | 8.39 |
| Bellingham 168 h Ctol 0.01 | RTX 5090 | 123 | 1680 | 287450 | 859756 | 2563956 | 5.14 | 13 |
| NorthPenn 48 h Ctol 0.0 | RTX 5090 | 339 | 480 | 192881 | 576750 | 853232 | 8.61 | 1.6 |
| NorthPenn 48 h Ctol 0.01 | RTX 5090 | 339 | 480 | 192881 | 588747 | 1575992 | 3.65 | 5.86 |
| NorthPenn 168 h Ctol 0.0 | RTX 5090 | 339 | 1680 | 674081 | 2039879 | 3021338 | 45.1 | 5.47 |
| NorthPenn 168 h Ctol 0.01 | RTX 5090 | 339 | 1680 | 674081 | 2063409 | 5776170 | 14.9 | 23.3 |
| LTown 48 h Ctol 0.0 | RTX 5090 | 785 | 580 | 528365 | 1575597 | 4822038 | 13.6 | 12.7 |
| LTown 48 h Ctol 0.01 | RTX 5090 | 785 | 580 | 528365 | 1580206 | 5677137 | 10.5 | 14.8 |
| LTown 168 h Ctol 0.0 | RTX 5090 | 785 | 2030 | 1847046 | 5523726 | 16638154 | 52.4 | 46.3 |
| LTown 168 h Ctol 0.01 | RTX 5090 | 785 | 2030 | 1847046 | 5529666 | 18261357 | 38.3 | 50.8 |

## B. Engine execution, B = 1 and largest batch
| case | engine | hardware | n_theta | library run s | fwd+adj B=1 eager s | fwd+adj B=1 graph s | peak MB B=1 | largest B | per-set s at largest B | peak MB at largest B / note |
|---|---|---|---|---|---|---|---|---|---|---|
| Net1 | adjoint | RTX 5090 | 36 | 0.000606 | 0.393 | 0.0688 | 0.783 | 4096 | 3e-05 | 1.25e+03 |
| Net1 | fused | RTX 5090 | 36 | 0.000606 | 0.0256 | 0.00362 | 1.51 | 4096 | 5.06e-06 | 3.05e+03 |
| Net2_react | adjoint | RTX 5090 | 118 | 0.00243 | 3.52 | 0.475 | 5.25 | 4096 | 0.000259 | 7.82e+03 |
| Net2_react | fused | RTX 5090 | 118 | 0.00243 | 0.0568 | 0.00941 | 16.8 | 4096 | 3.84e-05 | 8.77e+03 |
| Jilin_kw | adjoint | RTX 5090 | 96 | 0.00369 | 0.664 | 0.115 | 7.64 | 4096 | 7.33e-05 | 1.17e+04 |
| Jilin_kw | fused | RTX 5090 | 96 | 0.00369 | 0.0972 | 0.0154 | 16.9 | 4096 | 4.76e-05 | 1.3e+04 |
| Net3_CHEM_kw_mixed_tanks | adjoint | RTX 5090 | 334 | 0.00265 | 2.19 | 0.375 | 6.72 | 4096 | 0.000186 | 1.02e+04 |
| Net3_CHEM_kw_mixed_tanks | fused | RTX 5090 | 334 | 0.00265 | 0.0261 | 0.00518 | 36.9 | 4096 | 8.9e-05 | 1.14e+04 |
| L-Town shared | fused_gather | RTX 5090 | 2596 | 0.11 | 0.0525 | 0.0136 | 295 | 1024 |  |  |
| L-Town shared | fused_spmm | RTX 5090 | 2596 | 0.11 | 0.0725 | 0.0146 | 5.2e+03 | 1024 |  |  |
| L-Town shared | fused | RTX 5090 | 2596 | 0.11 | 0.0534 | 0.0136 | 5.06e+03 | 1024 |  |  |
| BWSN2 shared | fused_gather | RTX 5090 | 42169 | 20.5 | 0.0544 |  | 2.7e+03 | 64 |  | OOM at B=[128] |
| BWSN2 shared | fused_spmm | RTX 5090 | 42169 | 20.5 | 0.0759 |  | 8.44e+03 | 64 |  | OOM at B=[128] |
| BWSN2 shared | fused | RTX 5090 | 42169 | 20.5 | 0.0553 |  | 7.22e+03 | 64 |  | OOM at B=[128] |

## B2. Calibration: graph capture, one gradient, complete fits (all measured)
| data set | hardware | graph capture s | gradient eager s | gradient graph s | library run s | FD fit median s | adjoint eager fit s | adjoint graph fit s |
|---|---|---|---|---|---|---|---|---|
| Bellingham | RTX 5090 | 5.48 | 0.733 |  | 0.00854 | 3.24 | 19.6 | 5.33 |
| Bellingham_fixkb | RTX 5090 | 5.19 | 0.736 |  | 0.00858 | 2.47 | 34.7 | 9.43 |
| Fairfield | RTX 5090 | 45.5 | 0.446 |  | 0.00582 | 1.59 | 16.2 | 4.11 |
| Fairfield_fixkb | RTX 5090 | 3.8 | 0.447 |  | 0.0058 | 0.75 | 9.33 | 2.32 |
| NorthPenn | RTX 5090 | 3.91 | 0.478 |  | 0.0351 | 9.14 | 18.5 | 5.15 |
| NorthPenn_fixkb | RTX 5090 | 4.13 | 0.474 |  | 0.0355 | 6.76 | 16 | 4.46 |

## B3. Applications: complete-task wall clock
| task | hardware | record s | build s | graph capture s | total s | main parts (s) |
|---|---|---|---|---|---|---|
| Exp1 Net3 | RTX 5090 | 14.8 | 14.4 | 2.39 | 1.42e+03 | lipschitz_s 4, sweep_s 159, refit_s 158, const_s 54, greedy_s 828, mu_sweep_s 148, matrix_s 20, dll_check_s 13 |
| Exp1 North Penn | RTX 5090 | 9.34 | 2.43 | 1.28 | 633 | lipschitz_s 2, sweep_s 30, refit_s 30, const_s 30, greedy_s 327, mu_sweep_s 27, matrix_s 162, dll_check_s 10 |
| Exp2 L-Town (v1, 10 sensors) | RTX 5090 | 15.3 | 28.3 | 1.43 | 1.03e+03 | sweep_s 98, screen_s 29, refine_s 610, exhaustive_s 245, identification_s 981 |
| Exp2 BWSN2 (v1, 20 sensors) | RTX 5090 | 251 | 90.9 | 0 | 1.44e+03 | dll_quality_run_s 4, placement_s 2, sweep_s 210, screen_s 10, refine_s 219, exhaustive_s 591, identification_s 1031 |
| Exp3 L-Town | RTX 5090 | 17.9 | 19.9 |  | 150 | jac_check_s 2, estimate_random_s 12, candidate_jacobian_s 50, design_logdet_s 2, design_count_s 1, estimate_random+dopt_s 14, estimate_random+count_s 14, estimate_random+random_s 13 |
| Exp3 BWSN2 | RTX 5090 | 866 | 59.3 |  | 1.46e+03 | dll_quality_run_s 20, dll_check_s 21, jac_check_s 4, estimate_random_s 26, candidate_jacobian_s 52, design_logdet_s 132, design_count_s 132, estimate_random+dopt_s 33, estimate_random+count_s 31, estimate_random+random_s 30 |
| Exp2 v2 bw200x (200 sensors) | RTX 5090 | 292 | 103 | 0 | 6.57e+03 | sweep 5264, screen 16, refine 348, NNLS check 508 |
| Exp2 v2 lt10 (10 sensors) | RTX 5090 | 11.3 | 27.3 | 0 | 875 | sweep 136, screen 21, refine 435, NNLS check 238 |
| Exp2 v2 bw200x (200 sensors) | RTX 3090 | 250 | 83.6 | 0 | 1.33e+04 | sweep 11461, refine 572 |
| Exp2 v2 c1 (20 sensors) | RTX 3090 | 257 | 93 | 0 | 1.1e+03 | sweep 801, refine 256 |
| Exp2 v2 c2_100 (100 sensors) | RTX 3090 | 257 | 93 | 0 | 6.03e+03 | sweep 5512, refine 458 |
| Exp2 v2 c2_20 (20 sensors) | RTX 3090 | 257 | 93 | 0 | 1.33e+03 | sweep 1001, refine 272 |
| Exp2 v2 c2_200 (200 sensors) | RTX 3090 | 257 | 93 | 0 | 1.12e+04 | sweep 10686, refine 477 |
| Exp2 v2 c2_50 (50 sensors) | RTX 3090 | 257 | 93 | 0 | 2.85e+03 | sweep 2364, refine 436 |
| Exp2 v2 c3_100 (100 sensors) | RTX 3090 | 257 | 93 | 0 | 5.82e+03 | sweep 5289, refine 469 |
| Exp2 v2 c3_20 (20 sensors) | RTX 3090 | 257 | 93 | 0 | 1.29e+03 | sweep 949, refine 286 |
| Exp2 v2 c3_200 (200 sensors) | RTX 3090 | 257 | 93 | 0 | 1.29e+04 | sweep 12376, refine 442 |
| Exp2 v2 c3_50 (50 sensors) | RTX 3090 | 257 | 93 | 0 | 3.48e+03 | sweep 3005, refine 415 |
| Exp2 v2 lt10 (10 sensors) | RTX 3090 | 15.9 | 29.7 | 0 | 1.89e+03 | sweep 548, refine 829 |

## C. Provenance of finite-difference costs
| quantity | label | how obtained | source |
|---|---|---|---|
| Calibration FD L-BFGS-B fits (Ctol 0 and 0.01) | measured | library called N times inside the optimiser | g4_graph_5090, h2_5090 |
| FD step sweep 1e-1 to 1e-7 (Table S2) | measured | two library runs per step and direction | final_5090/fds |
| Engine benchmark FD gradient 22-887 ms (Table S3) | projected from one run | (n_theta + 1) x measured library run | bench.py line 100 |
| Ratio to FD at B_max 2,000-6,100 (Table S3, TOC) | projected from one run | projected FD gradient / measured per-set replay cost | bench.py |
| Exp1 library cost of the response columns | projected from one run | candidates x slots x library run | exp1_booster.py line 558 |
| Exp2 exhaustive library cost 9e3-2e4 s / 6e6-1e7 s | projected from one run | junctions x admissible slots x library run | exp2_source.py line 538 |
| Exp3 FD gradient with the library 3e5 s, 4.3e6-fold | projected from one run | (P + 1) x library run | exp3_pipe_kw.py line 399 |
| Exp3 FD gradient with the batched replay 89 s | projected from batched replay | passes x measured batched forward | exp3_pipe_kw.py line 400 |
| Parallel FD (R2-4) | measured | 1, 2, 4 and 8 worker processes on the 8 cores of one 5090 allocation; large cases timed on a column subset and scaled | final_5090_v2/r1_T4_*/fd_parallel.json |

## D2. One-gradient time follows the number of quality steps (RTX 5090, B = 1, eager, median of 3 warm runs)
| network | Ctol | steps 48 h | fwd+adj 48 h s | steps ratio 168/48 h | time ratio 168/48 h |
|---|---|---|---|---|---|
| BWSN2 | 0 | 593 | 0.042 |  |  |
| BWSN2 | 0.01 | 593 | 0.153 |  |  |
| Net1 | 0 | 579 | 0.0359 | 3.5 | 3.38 |
| Net1 | 0.01 | 579 | 0.0355 | 3.5 | 3.41 |
| Net2 | 0 | 576 | 0.0358 | 3.51 | 3.39 |
| Net2 | 0.01 | 576 | 0.0359 | 3.51 | 3.38 |
| Net3 | 0 | 583 | 0.0364 | 3.52 | 3.43 |
| Net3 | 0.01 | 583 | 0.0366 | 3.52 | 3.4 |
| Jilin | 0 | 576 | 0.0355 | 3.5 | 3.39 |
| Jilin | 0.01 | 576 | 0.0357 | 3.5 | 3.38 |
| Fairfield | 0 | 480 | 0.0303 | 2.15 | 2.09 |
| Fairfield | 0.01 | 480 | 0.0304 | 2.15 | 2.09 |
| Bellingham | 0 | 480 | 0.0305 | 3.5 | 3.33 |
| Bellingham | 0.01 | 480 | 0.0305 | 3.5 | 3.36 |
| NorthPenn | 0 | 480 | 0.0303 | 3.5 | 3.37 |
| NorthPenn | 0.01 | 480 | 0.0304 | 3.5 | 3.39 |
| LTown | 0 | 580 | 0.0452 | 3.5 | 3.45 |
| LTown | 0.01 | 580 | 0.0454 | 3.5 | 3.42 |

## D3. BWSN2 Exp2 schedule beyond 48 h (measured on one RTX 5090 32 GB; normal equations sized from the recorded schedule, not allocated; fwd+adj is a single cold run)
| hours | slots K | versions | fused terms | host peak RSS GB | record s | build s | GPU peak fwd+adj B=1 GB | fwd+adj B=1 s | GPU peak adjoint row pass (32 rows) GB | normal equations GB (sized) |
|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 384 | 17625800 | 119697606 | 27.3 | 400 | 130 | 7.12 | 0.436 | 23.3 | 13.8 |
| 168 | 672 | 30922977 | 212431253 | 47.5 | 714 | 243 | 12.6 | 0.226 | OOM | 42.1 |

## D. Scaling fits (B = 1, log-log; memory: gather path only)
| hardware | points | memory exponent vs fused terms | MB per million terms at 1e7 terms | time exponent | s at 1e7 terms |
|---|---|---|---|---|---|
| RTX 5090 | 40 | 0.953 | 79.7 | 0.109 | 0.0794 |

## Limits that must be stated
- Time-chunked replay is not implemented and has no benchmark; it must not be described as a validated capability.
- Largest validated problem for the full source identification: BWSN2, 48 h, about 6e7 fused terms, 32 GB (Exp2 needed 32 rows per adjoint pass after an OOM at 64).
- Measured beyond 48 h (D3): one B = 1 gradient of the 168 h BWSN2 schedule (2.1e8 terms) fits in 12.6 GB, but the identification does not: the adjoint row pass runs out of memory at 32 rows and the normal equations alone would need 42 GB. Longer identification needs node-split normal equations (several GPUs or host memory) or time-chunked replay, neither of which is implemented.
- Any statement about longer horizons or finer steps is an extrapolation of the fits in D, labelled as such.
- 5090 BWSN2 48 h, Ctol 0: 40,826,301 fused terms, B = 1 forward+adjoint peak 3880 MB.