# Exp1: booster siting and dosing schedule as one convex program

## Net3: 97 nodes / 119 links, 168 h, 24 slots, 95 candidates, 2280 decision variables, 1514 observations, 1,785,271 fused terms; lo = 0.2 mg/L, hi = 4.0 mg/L

plant only: coverage 0.9534, mean shortfall 0.0046 mg/L, mass 0.000 kg/d


### Greedy forward selection (batched refit over all remaining candidates per stage)

| stage | added | stations | coverage | mean shortfall mg/L | mass kg/d | wall s | batch |
|---|---|---|---|---|---|---|---|
| 1 | 127 | 1 | 0.974 | 0.00192 | 4.2 | 107 | 95 |
| 2 | 211 | 2 | 0.99 | 4.67e-04 | 6.44 | 85.4 | 94 |
| 3 | 131 | 3 | 0.994 | 3.01e-04 | 4.4 | 110 | 93 |
| 4 | 2 | 4 | 0.996 | 2.29e-04 | 2.6 | 85.3 | 92 |
| 5 | 1 | 5 | 0.998 | 9.06e-05 | 2.67 | 110 | 91 |
| 6 | 243 | 6 | 0.998 | 4.52e-05 | 2.58 | 85.6 | 90 |
| 7 | 217 | 7 | 0.998 | 6.00e-05 | 2.09 | 110 | 89 |
| 8 | 231 | 8 | 0.998 | 4.79e-05 | 1.8 | 85.4 | 88 |


### Group-lasso path (lambda fraction), refit on the support, constant-rate baseline on the same support

| lam frac | path stations | coverage | mass kg/d | refit stations | coverage | mass kg/d | const-rate stations | coverage | mass kg/d |
|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 2 | 0.953 | 0.00126 | 2 | 0.958 | 0.0435 | 2 | 0.958 | 0.0665 |
| 0.8 | 2 | 0.953 | 0.00504 | 2 | 0.958 | 0.0435 | 2 | 0.958 | 0.0665 |
| 0.7 | 2 | 0.953 | 0.00902 | 2 | 0.958 | 0.0435 | 2 | 0.958 | 0.0665 |
| 0.6 | 4 | 0.953 | 0.016 | 4 | 0.964 | 0.0878 | 4 | 0.964 | 0.129 |
| 0.5 | 5 | 0.953 | 0.0269 | 5 | 0.966 | 0.0974 | 5 | 0.966 | 0.145 |
| 0.4 | 9 | 0.954 | 0.0553 | 9 | 0.981 | 0.34 | 9 | 0.98 | 0.858 |
| 0.3 | 13 | 0.954 | 0.0955 | 13 | 0.988 | 0.477 | 13 | 0.987 | 1.28 |
| 0.2 | 15 | 0.955 | 0.158 | 15 | 0.989 | 0.429 | 15 | 0.988 | 0.974 |
| 0.1 | 21 | 0.965 | 0.242 | 21 | 0.994 | 0.427 | 20 | 0.993 | 1.1 |
| 0.05 | 25 | 0.983 | 0.307 | 25 | 0.999 | 0.791 | 23 | 0.997 | 2.08 |
| 0.02 | 28 | 0.993 | 0.434 | 28 | 1 | 0.835 | 25 | 0.998 | 2.16 |
| 0.01 | 30 | 0.995 | 0.586 | 30 | 1 | 0.814 | 26 | 0.998 | 2.15 |
| 0.005 | 30 | 0.998 | 0.698 | 30 | 1 | 0.814 | 26 | 0.998 | 2.15 |
| 0 | 36 | 1 | 0.798 | 36 | 1 | 0.798 | 29 | 0.998 | 2.16 |


### Explicit response matrix cross-check: 1514 x 2280, build 0.63 s, dense solve 19.4 s (20623 iters), L_exact 1.06 vs L_power 1.02, D-probe max rel err 7.44e-02, support match True

| lam frac | obj matrix-free | obj dense | rel obj diff | rel u diff | stations free | stations dense |
|---|---|---|---|---|---|---|
| 0.9 | 3.97e-04 | 3.97e-04 | 5.46e-16 | 3.38e-12 | 2 | 2 |
| 0.8 | 3.96e-04 | 3.96e-04 | 3.70e-15 | 8.18e-13 | 2 | 2 |
| 0.7 | 3.93e-04 | 3.93e-04 | 7.03e-15 | 4.86e-13 | 2 | 2 |
| 0.6 | 3.89e-04 | 3.89e-04 | 1.31e-14 | 5.41e-13 | 4 | 4 |
| 0.5 | 3.82e-04 | 3.82e-04 | 2.63e-14 | 5.77e-13 | 5 | 5 |
| 0.4 | 3.66e-04 | 3.66e-04 | 4.99e-14 | 3.74e-13 | 9 | 9 |
| 0.3 | 3.35e-04 | 3.35e-04 | 7.67e-14 | 1.35e-12 | 13 | 13 |
| 0.2 | 2.78e-04 | 2.78e-04 | 8.73e-14 | 8.71e-13 | 15 | 15 |
| 0.1 | 1.86e-04 | 1.86e-04 | 1.36e-13 | 1.91e-12 | 21 | 21 |
| 0.05 | 1.21e-04 | 1.21e-04 | 2.51e-13 | 2.33e-12 | 25 | 25 |
| 0.02 | 6.84e-05 | 6.84e-05 | 4.77e-13 | 4.13e-12 | 28 | 28 |
| 0.01 | 4.26e-05 | 4.26e-05 | 8.49e-13 | 1.05e-11 | 30 | 30 |
| 0.005 | 2.60e-05 | 2.60e-05 | 1.14e-12 | 8.72e-12 | 30 | 30 |
| 0 | 5.54e-06 | 5.54e-06 | 2.22e-05 | 0.151 | 36 | 35 |


DLL re-insertion of every reported design: 15 designs, max rel err 1.08e-13


timing s: record_s 14.8, build_s 14.4, setup_s 30.4, dll_quality_run_s 0.0, group_scale_s 0.2, lipschitz_s 4.3, sweep_s 159.2, refit_s 158.4, const_s 54.4, graph_capture_s 2.4, greedy_s 827.7, mu_sweep_s 148.1, matrix_s 20.1, dll_check_s 12.8, total_s 1417.7

## NorthPenn: 339 nodes / 399 links, 36 h, 36 slots, 328 candidates, 11808 decision variables, 7047 observations, 900,842 fused terms; lo = 0.2 mg/L, hi = 4.0 mg/L

plant only: coverage 0.8692, mean shortfall 0.0195 mg/L, mass 0.000 kg/d


### Greedy forward selection (batched refit over all remaining candidates per stage)

| stage | added | stations | coverage | mean shortfall mg/L | mass kg/d | wall s | batch |
|---|---|---|---|---|---|---|---|
| 1 | 308 | 1 | 0.905 | 0.0118 | 1.19 | 35.6 | 328 |
| 2 | 292 | 2 | 0.927 | 0.00783 | 1.25 | 38.6 | 327 |
| 3 | 266 | 3 | 0.954 | 0.00453 | 1.46 | 35.9 | 326 |
| 4 | 296 | 4 | 0.962 | 0.00296 | 1.38 | 38.6 | 325 |
| 5 | 616 | 5 | 0.965 | 0.00237 | 1.38 | 35.5 | 324 |
| 6 | 615 | 6 | 0.968 | 0.00177 | 1.4 | 38.5 | 323 |
| 7 | 429 | 7 | 0.971 | 0.0014 | 1.26 | 35.7 | 322 |
| 8 | 318 | 8 | 0.972 | 0.00106 | 1.33 | 38.2 | 321 |


### Group-lasso path (lambda fraction), refit on the support, constant-rate baseline on the same support

| lam frac | path stations | coverage | mass kg/d | refit stations | coverage | mass kg/d | const-rate stations | coverage | mass kg/d |
|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 1 | 0.869 | 5.69e-04 | 1 | 0.872 | 0.00936 | 1 | 0.872 | 0.0185 |
| 0.8 | 2 | 0.869 | 0.00128 | 2 | 0.88 | 0.496 | 2 | 0.872 | 0.0297 |
| 0.7 | 6 | 0.869 | 0.00295 | 6 | 0.923 | 0.874 | 6 | 0.885 | 0.194 |
| 0.6 | 10 | 0.869 | 0.00964 | 10 | 0.937 | 0.944 | 10 | 0.917 | 0.972 |
| 0.5 | 17 | 0.869 | 0.0247 | 17 | 0.943 | 0.908 | 17 | 0.926 | 1.09 |
| 0.4 | 20 | 0.87 | 0.0501 | 20 | 0.946 | 0.923 | 19 | 0.93 | 1.16 |
| 0.3 | 25 | 0.872 | 0.0916 | 25 | 0.971 | 0.942 | 24 | 0.952 | 1.47 |
| 0.2 | 37 | 0.874 | 0.172 | 37 | 0.981 | 0.74 | 36 | 0.961 | 1.55 |
| 0.1 | 57 | 0.889 | 0.352 | 57 | 0.989 | 0.748 | 57 | 0.966 | 1.6 |
| 0.05 | 69 | 0.921 | 0.508 | 69 | 0.991 | 0.75 | 68 | 0.966 | 1.65 |
| 0.02 | 82 | 0.963 | 0.684 | 82 | 0.995 | 0.884 | 75 | 0.97 | 1.82 |
| 0.01 | 88 | 0.977 | 0.782 | 88 | 0.995 | 0.886 | 77 | 0.969 | 1.85 |
| 0.005 | 89 | 0.987 | 0.835 | 89 | 0.995 | 0.886 | 77 | 0.969 | 1.85 |
| 0 | 109 | 0.996 | 0.888 | 108 | 0.996 | 0.884 | 78 | 0.971 | 1.92 |


### Explicit response matrix cross-check: 7047 x 11808, build 0.57 s, dense solve 161.7 s (50000 iters), L_exact 6.62 vs L_power 1.74, D-probe max rel err 1.00e+00, support match False

| lam frac | obj matrix-free | obj dense | rel obj diff | rel u diff | stations free | stations dense |
|---|---|---|---|---|---|---|
| 0.9 | 0.00212 | 0.00212 | 0 | 2.17e-13 | 1 | 1 |
| 0.8 | 0.00212 | 0.00212 | 4.09e-16 | 1.16e-13 | 2 | 2 |
| 0.7 | 0.00212 | 0.00212 | 2.05e-16 | 1.32e-13 | 6 | 6 |
| 0.6 | 0.00211 | 0.00211 | 2.06e-15 | 3.01e-13 | 10 | 10 |
| 0.5 | 0.00208 | 0.00208 | 4.59e-15 | 2.02e-13 | 17 | 17 |
| 0.4 | 0.00201 | 0.00201 | 9.28e-15 | 1.57e-13 | 20 | 20 |
| 0.3 | 0.00188 | 0.00188 | 1.65e-14 | 2.46e-13 | 25 | 25 |
| 0.2 | 0.00164 | 0.00164 | 2.83e-14 | 2.52e-13 | 37 | 37 |
| 0.1 | 0.00116 | 0.00116 | 4.65e-14 | 5.99e-13 | 57 | 57 |
| 0.05 | 7.55e-04 | 7.55e-04 | 5.34e-14 | 9.57e-09 | 69 | 69 |
| 0.02 | 4.15e-04 | 4.15e-04 | 6.06e-11 | 3.55e-05 | 82 | 82 |
| 0.01 | 2.69e-04 | 2.69e-04 | 4.17e-04 | 0.0948 | 88 | 87 |
| 0.005 | 1.87e-04 | 1.87e-04 | 7.79e-04 | 0.166 | 89 | 88 |
| 0 | 9.41e-05 | 9.37e-05 | 0.00445 | 0.466 | 109 | 107 |


DLL re-insertion of every reported design: 15 designs, max rel err 2.47e-14


timing s: record_s 9.3, build_s 2.4, setup_s 14.3, dll_quality_run_s 0.0, group_scale_s 0.1, lipschitz_s 1.6, sweep_s 30.4, refit_s 30.1, const_s 29.6, graph_capture_s 1.3, greedy_s 326.5, mu_sweep_s 26.9, matrix_s 162.2, dll_check_s 10.4, total_s 632.7
