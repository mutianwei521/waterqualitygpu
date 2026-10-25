# Booster siting methods on the nominal problem (R2-11)

Same candidates, objective (squared shortfall and excess plus the nominal mass weight) and inner schedule fit for every method; coverage and mass from the library run of the design. GPU times on one RTX 5090; the MILP runs on 8 CPU cores. The MILP solves the L1 version of the objective; its stations are then refitted with the squared objective, and the L1 column gives each support's optimal L1 value relative to the MILP optimum. The objective has no station cost, so objectives are comparable only at equal station count.

| network | method | stations | objective (×10⁻⁵) | coverage % | mass kg/d | wall s | L1 gap to MILP | notes |
|---|---|---|---|---|---|---|---|---|
| Net3 | greedy forward selection, m = 3 | 3: 127, 131, 211 | 4.887 | 99.37 | 4.40 | 302 | +0.58% | 3 stages, 282 sets |
| Net3 | genetic algorithm, m = 3, seed 2 | 3: 131, 211, 3 | 4.871 | 99.34 | 4.41 | 2026 | +0.00% | pop 64, 60 gen, 888 sets |
| Net3 | genetic algorithm, m = 3, seed 1 | 3: 131, 211, 3 | 4.871 | 99.34 | 4.41 | 2015 | +0.00% | pop 64, 60 gen, 957 sets |
| Net3 | genetic algorithm, m = 3, seed 0 | 3: 127, 131, 211 | 4.887 | 99.37 | 4.40 | 2060 | +0.58% | pop 64, 60 gen, 806 sets |
| Net3 | MILP (HiGHS, L1 objective), m = 3 | 3: 131, 211, 3 | 4.871 | 99.34 | 4.41 | 182 | 0 (optimum) | rel. gap 0.0e+00, 2424 nodes, CPU |
| Net3 | greedy forward selection, m = 8 | 8: 127, 131, 211, 217, 231, 243, 1, 2 | 1.456 | 99.83 | 1.80 | 778 |  | 8 stages, 732 sets |
| Net3 | genetic algorithm, m = 8, seed 0 | 8: 127, 131, 139, 153, 213, 243, 255, 1 | 1.206 | 99.83 | 1.57 | 2093 |  | pop 64, 60 gen, 1560 sets |
| Net3 | group lasso, lambda = 0.1 lambda0, refitted | 21 | 3.768 | 99.37 | 0.43 | 318 |  | 14 penalties in one batch |
| Net3 | constant rate on the same stations | 20 | 4.283 | 99.31 | 1.10 | 54 |  | 14 supports in one batch |
| North Penn | greedy forward selection, m = 8 | 8: 266, 292, 296, 308, 318, 429, 615, 616 | 21.041 | 97.24 | 1.33 | 297 |  | 8 stages, 2,596 sets |
| North Penn | genetic algorithm, m = 8, seed 17 (rank 19 of 21) | 8: 268, 275, 292, 296, 311, 318, 428, 616 | 26.102 | 97.14 | 1.20 | 476 |  | pop 64, 60 gen, 1532 sets |
| North Penn | genetic algorithm, m = 8, seed 16 (rank 20 of 21) | 8: 247, 262, 292, 296, 311, 318, 429, 615 | 26.916 | 96.93 | 1.30 | 469 |  | pop 64, 60 gen, 1770 sets |
| North Penn | genetic algorithm, m = 8, seed 0 (rank 21 of 21) | 8: 245, 266, 280, 292, 296, 313, 317, 616 | 27.556 | 96.97 | 1.34 | 502 |  | pop 64, 60 gen, 1657 sets |
| North Penn | genetic algorithm, m = 8, all 21 seeds (Table S-GA) | — | median 21.225 (18.948-27.556) | median 97.15 (96.93-97.35) | median 1.31 (1.20-1.43) | median 476 |  | 13 of 21 worse than greedy, 8 better; objective vs greedy median +0.9% (-9.9% to +31.0%) |
| North Penn | group lasso, lambda = 0.3 lambda0, refitted | 25 | 31.772 | 97.09 | 0.94 | 61 |  | 14 penalties in one batch |
| North Penn | constant rate on the same stations | 24 | 49.323 | 95.24 | 1.47 | 30 |  | 14 supports in one batch |

North Penn, m = 8: the seed rows are the runs with the highest objective of 21 (seeds 0, 16, 17; rank 1 = lowest objective); the row 'all 21 seeds' summarises every run, each of which is listed in Table S-GA.
