# Field calibration and uncertainty (G4 / H2)

## Field calibration (Kb + 4 diameter-class Kw), each method repeated; wall-clock per full fit

| dataset | method | repeats | median RMSE mg/L | min | max | median wall s | median evaluations |
|---|---|---|---|---|---|---|---|
| Bellingham | nominal(INP) | 1 | 0.223 | 0.223 | 0.223 | 0 | - |
| Bellingham | FD-LBFGSB | 3 | 0.132 | 0.132 | 0.132 | 4.61 | 324 |
| Bellingham | ADJ-LBFGSB | 3 | 0.132 | 0.132 | 0.132 | 1.62 | 36 |
| Bellingham | ADJ-LBFGSB-graph | 3 | 0.132 | 0.132 | 0.132 | 0.206 | 28 |
| Bellingham | FD-LBFGSB@ctol0.01 | 1 | 0.132 | 0.132 | 0.132 | 9.01 | 1,020 |
| Bellingham | FD-LBFGSB@ctol0.01,eps1e-3 | 1 | 0.138 | 0.138 | 0.138 | 7.25 | 762 |
| Bellingham (Kb fixed) | nominal(INP) | 1 | 0.223 | 0.223 | 0.223 | 0 | - |
| Bellingham (Kb fixed) | FD-LBFGSB | 3 | 0.142 | 0.142 | 0.142 | 3.62 | 230 |
| Bellingham (Kb fixed) | ADJ-LBFGSB | 3 | 0.142 | 0.142 | 0.142 | 2.1 | 46 |
| Bellingham (Kb fixed) | ADJ-LBFGSB-graph | 3 | 0.142 | 0.142 | 0.142 | 0.336 | 46 |
| Bellingham (Kb fixed) | FD-LBFGSB@ctol0.01 | 1 | 0.142 | 0.142 | 0.142 | 5.81 | 680 |
| Bellingham (Kb fixed) | FD-LBFGSB@ctol0.01,eps1e-3 | 1 | 0.157 | 0.157 | 0.157 | 3.12 | 355 |
| Fairfield | nominal(INP) | 1 | 0.231 | 0.231 | 0.231 | 0 | - |
| Fairfield | FD-LBFGSB | 3 | 0.218 | 0.218 | 0.218 | 2.46 | 210 |
| Fairfield | ADJ-LBFGSB | 3 | 0.218 | 0.218 | 0.218 | 1.21 | 35 |
| Fairfield | ADJ-LBFGSB-graph | 3 | 0.218 | 0.218 | 0.218 | 0.171 | 35 |
| Fairfield | FD-LBFGSB@ctol0.01 | 1 | 0.219 | 0.219 | 0.219 | 3.26 | 660 |
| Fairfield | FD-LBFGSB@ctol0.01,eps1e-3 | 1 | 0.23 | 0.23 | 0.23 | 1.69 | 348 |
| Fairfield (Kb fixed) | nominal(INP) | 1 | 0.231 | 0.231 | 0.231 | 0 | - |
| Fairfield (Kb fixed) | FD-LBFGSB | 3 | 0.228 | 0.228 | 0.228 | 1.1 | 100 |
| Fairfield (Kb fixed) | ADJ-LBFGSB | 3 | 0.228 | 0.228 | 0.228 | 0.668 | 20 |
| Fairfield (Kb fixed) | ADJ-LBFGSB-graph | 3 | 0.228 | 0.228 | 0.228 | 0.0977 | 20 |
| Fairfield (Kb fixed) | FD-LBFGSB@ctol0.01 | 1 | 0.23 | 0.23 | 0.23 | 1.29 | 265 |
| Fairfield (Kb fixed) | FD-LBFGSB@ctol0.01,eps1e-3 | 1 | 0.23 | 0.23 | 0.23 | 0.875 | 180 |
| NorthPenn | nominal(INP) | 1 | 0.261 | 0.261 | 0.261 | 0 | - |
| NorthPenn | FD-LBFGSB | 3 | 0.182 | 0.182 | 0.182 | 13 | 228 |
| NorthPenn | ADJ-LBFGSB | 3 | 0.182 | 0.182 | 0.182 | 1.91 | 37 |
| NorthPenn | ADJ-LBFGSB-graph | 3 | 0.182 | 0.182 | 0.182 | 0.303 | 37 |
| NorthPenn | FD-LBFGSB@ctol0.01 | 1 | 0.183 | 0.183 | 0.183 | 29.8 | 1,248 |
| NorthPenn | FD-LBFGSB@ctol0.01,eps1e-3 | 1 | 0.184 | 0.184 | 0.184 | 9.4 | 414 |
| NorthPenn (Kb fixed) | nominal(INP) | 1 | 0.261 | 0.261 | 0.261 | 0 | - |
| NorthPenn (Kb fixed) | FD-LBFGSB | 3 | 0.186 | 0.186 | 0.186 | 9.53 | 165 |
| NorthPenn (Kb fixed) | ADJ-LBFGSB | 3 | 0.186 | 0.186 | 0.186 | 1.61 | 32 |
| NorthPenn (Kb fixed) | ADJ-LBFGSB-graph | 3 | 0.186 | 0.186 | 0.186 | 0.259 | 32 |
| NorthPenn (Kb fixed) | FD-LBFGSB@ctol0.01 | 1 | 0.187 | 0.187 | 0.187 | 15 | 635 |
| NorthPenn (Kb fixed) | FD-LBFGSB@ctol0.01,eps1e-3 | 1 | 0.204 | 0.204 | 0.204 | 11.8 | 570 |


## One-gradient cost and engine/DLL consistency at the nominal point

| dataset | n_obs | sites | DLL run (Ctol 0) s | DLL run (Ctol 0.01) s | FD gradient s | adjoint gradient s | adjoint gradient, graph s | loss rel diff engine vs DLL | grad rel err vs DLL FD |
|---|---|---|---|---|---|---|---|---|---|
| Bellingham | 120 | 12 | 0.0125 | 0.00697 | 0.0751 | 0.0385 | 0.00716 | 1.09e-15 | 4.75e-10 |
| Bellingham (Kb fixed) | 120 | 12 | 0.0125 | 0.00701 | 0.0627 | 0.0384 | 0.00716 | 1.38e-15 | 1.10e-09 |
| Fairfield | 140 | 15 | 0.00842 | 0.00457 | 0.0505 | 0.0277 | 0.00477 | 1.35e-15 | 6.51e-10 |
| Fairfield (Kb fixed) | 140 | 15 | 0.0084 | 0.00454 | 0.042 | 0.0276 | 0.0047 | 1.90e-15 | 5.07e-10 |
| NorthPenn | 42 | 6 | 0.0515 | 0.0174 | 0.309 | 0.04 | 0.00797 | 1.49e-15 | 7.66e-10 |
| NorthPenn (Kb fixed) | 42 | 6 | 0.0517 | 0.0175 | 0.258 | 0.0404 | 0.00792 | 1.81e-15 | 1.07e-09 |


## H2: leave-one-site-out predictive check (Laplace sampling on the GPU)

| dataset | n_obs | fit RMSE | active bounds | LOSO RMSE | parameter share of predictive variance | LOSO 95% coverage (linearised) | LOSO 95% coverage (sampled) | LOSO coverage (parameter-only band) | samples | sampling wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| Fairfield | 140 | 0.218 | 3 | 0.226 | 0.0213 | 0.936 | 0.936 | 0.279 | 4,096 | 0.771 |
| Bellingham | 120 | 0.132 | 4 | 0.167 | 0.0174 | 0.867 | 0.875 | 0.108 | 4,096 | 1.3 |
| NorthPenn | 42 | 0.182 | 3 | 0.206 | 0.0657 | 0.952 | 0.976 | 0.333 | 4,096 | 0.875 |
