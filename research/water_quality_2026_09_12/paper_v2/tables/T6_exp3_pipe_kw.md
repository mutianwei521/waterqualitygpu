# Exp3: pipe-by-pipe wall-decay identifiability and sensor prescription

## Field systems at theta*: pipe-by-pipe Kw Laplace audit and D-optimal (log-det) sensor prescription (+5 sites)

| dataset | n_obs | sites | pipes | RMSE at theta* | pipes sensitive | identifiable (sd < 0.5 prior) | p_eff | median sd/prior | candidate sites | sites added | identifiable after | p_eff after | Jacobian check | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fairfield | 140 | 15 | 126 | 0.218 | 101 | 0 | 3.86 | 0.996 | 111 | 5 | 3 | 13 | 3.07e-15 | 96.4 |
| Bellingham | 120 | 12 | 168 | 0.132 | 138 | 2 | 3.09 | 1 | 121 | 5 | 8 | 13.2 | 1.67e-16 | 8.69 |
| NorthPenn | 42 | 6 | 399 | 0.182 | 233 | 0 | 3.9 | 1 | 337 | 5 | 14 | 28.2 | 3.77e-15 | 13.2 |


## BWSN2: 12527 nodes / 14822 pipes, 48 h, 193 sampling instants, Kw in ft/d; 20 random sensors + 10 prescribed; groups {'<=6in': 8842, '8in': 3412, '10-12in': 1692, '>=16in': 876}

DLL check 8.8e-15; Jacobian rows vs batched autograd 5.1e-16; one adjoint gradient of 14822 pipe Kw 0.070 s vs projected DLL FD 3e+05 s (4.27e+06x) and batched-engine FD 88.7 s (1.26e+03x); candidate Jacobian 49,408 rows in 52.2 s

| sensor set | sensors | n_obs | predicted identifiable | identifiable (sd < 0.5 prior) | p_eff | sensitive pipes | Kw RMSE identifiable: MAP | grouped prior | median rel err identifiable: MAP | grouped prior | Kw RMSE all: MAP | grouped prior | MAP 2-sd coverage | hold-out node RMSE mg/L: MAP | grouped | sensor fit RMSE (MAP) | noise sd | MAP wall s | audit wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 20 | 3,860 | 9 | 9 | 50 | 3,956 | 0.0548 | 0.187 | 0.0875 | 0.309 | 0.325 | 0.325 | 0.899 | 0.0141 | 0.0143 | 0.0199 | 0.02 | 13.3 | 10.5 |
| random+dopt | 30 | 5,790 | 24 | 27 | 102 | 4,266 | 0.139 | 0.168 | 0.14 | 0.368 | 0.314 | 0.314 | 0.918 | 0.011 | 0.0118 | 0.0198 | 0.02 | 18.7 | 12.4 |
| random+count | 30 | 5,790 | 30 | 27 | 92.4 | 4,554 | 0.125 | 0.162 | 0.0898 | 0.315 | 0.322 | 0.323 | 0.9 | 0.0129 | 0.0134 | 0.0199 | 0.02 | 16.7 | 12.4 |
| random+random | 30 | 5,790 | 10 | 11 | 73 | 4,492 | 0.062 | 0.161 | 0.091 | 0.268 | 0.313 | 0.313 | 0.918 | 0.0125 | 0.0129 | 0.0199 | 0.02 | 15.7 | 12.4 |


timing s: record_s 866.4, build_s 59.3, setup_s 989.0, dll_quality_run_s 20.2, dll_check_s 20.8, jac_check_s 3.9, estimate_random_s 26.0, candidate_jacobian_s 52.2, design_logdet_s 132.0, design_count_s 131.6, estimate_random+dopt_s 32.8, estimate_random+count_s 30.7, estimate_random+random_s 29.7, adjoint_passes_total 574.0, total_s 1464.7


## LTown: 785 nodes / 905 pipes, 48 h, 193 sampling instants, Kw in m/d; 10 random sensors + 5 prescribed; groups {'<=100mm': 710, '110-160mm': 119, '>160mm': 76}

DLL check 5.6e-15; Jacobian rows vs batched autograd 2.2e-15; one adjoint gradient of 905 pipe Kw 0.065 s vs projected DLL FD 77 s (1.19e+03x) and batched-engine FD 0.885 s (13.6x); candidate Jacobian 148,996 rows in 49.9 s

| sensor set | sensors | n_obs | predicted identifiable | identifiable (sd < 0.5 prior) | p_eff | sensitive pipes | Kw RMSE identifiable: MAP | grouped prior | median rel err identifiable: MAP | grouped prior | Kw RMSE all: MAP | grouped prior | MAP 2-sd coverage | hold-out node RMSE mg/L: MAP | grouped | sensor fit RMSE (MAP) | noise sd | MAP wall s | audit wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 10 | 1,930 | 5 | 5 | 27.2 | 470 | 0.0276 | 0.0466 | 0.172 | 0.309 | 0.0622 | 0.0629 | 0.952 | 0.0114 | 0.012 | 0.0202 | 0.02 | 8.54 | 2.4 |
| random+dopt | 15 | 2,895 | 8 | 9 | 52.3 | 712 | 0.0228 | 0.0704 | 0.106 | 0.542 | 0.0615 | 0.0628 | 0.952 | 0.0105 | 0.0118 | 0.0199 | 0.02 | 11.8 | 1.05 |
| random+count | 15 | 2,895 | 15 | 13 | 45.2 | 511 | 0.0213 | 0.0484 | 0.139 | 0.282 | 0.0614 | 0.063 | 0.951 | 0.0116 | 0.012 | 0.0205 | 0.02 | 11.7 | 1.06 |
| random+random | 15 | 2,895 | 5 | 5 | 35.7 | 485 | 0.0281 | 0.0471 | 0.2 | 0.325 | 0.0623 | 0.063 | 0.95 | 0.0114 | 0.012 | 0.0203 | 0.02 | 10.5 | 1.05 |


timing s: record_s 17.9, build_s 19.9, setup_s 39.4, dll_quality_run_s 0.1, dll_check_s 0.2, jac_check_s 2.4, estimate_random_s 12.4, candidate_jacobian_s 49.9, design_logdet_s 1.5, design_count_s 1.4, estimate_random+dopt_s 14.3, estimate_random+count_s 14.2, estimate_random+random_s 13.1, adjoint_passes_total 630.0, total_s 149.5
