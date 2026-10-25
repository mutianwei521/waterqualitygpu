# Sensitivity of the booster designs to hydraulic conditions and chlorine demand (R2-11)

Coverage: demand-weighted share (%) of node-hours of the evaluation window at or above 0.2 mg/L. *Fixed*: the nominal design (stations and hourly schedule) installed in EPANET as mass-rate sources and run by the library under the scenario. *Refit*: schedule re-optimised under the scenario on the same stations. *Re-solved*: group lasso at the same relative penalty under the scenario, refitted; J is the Jaccard index of its stations with the nominal support. The mass weight is kept at its nominal value; since the refit minimises violation plus mass, under milder scenarios it may trade a little coverage for less chlorine (second table). Every scenario is recorded again (at Ctol = 0 the kinetic ones would not need it).


## Net3

Designs: greedy, 3 stations; group lasso 0.1λ0, 21 stations; constant rate, 20 stations (nominal study, Table S5).

| scenario | plant only | greedy, 3 stations: fixed / refit | group lasso 0.1λ0, 21 stations: fixed / refit | constant rate, 20 stations: fixed / refit | lasso re-solved: stations / coverage / J |
|---|---|---|---|---|---|
| nominal | 95.3 | 99.4 / 99.4 | 99.4 / 99.4 | 99.3 / 99.3 | 21 / 99.4 / 1.00 |
| demand x0.8 | 94.0 | 97.6 / 99.1 | 96.5 / 99.5 | 99.0 / 99.4 | 20 / 99.5 / 0.71 |
| demand x0.9 | 94.4 | 97.5 / 99.2 | 96.8 / 99.4 | 98.6 / 99.3 | 19 / 99.5 / 0.74 |
| demand x1.1 | 96.6 | 98.4 / 99.6 | 98.3 / 99.8 | 99.4 / 99.7 | 15 / 99.6 / 0.71 |
| demand x1.2 | 97.2 | 98.6 / 99.5 | 98.6 / 99.8 | 99.5 / 99.7 | 16 / 99.8 / 0.68 |
| pump 10 two hours later | 95.0 | 99.0 / 99.4 | 97.9 / 99.4 | 98.7 / 99.3 | 22 / 99.5 / 0.79 |
| kb x0.5 | 96.6 | 99.9 / 99.7 | 99.7 / 99.7 | 99.7 / 99.7 | 15 / 99.6 / 0.71 |
| kb x2.0 | 93.1 | 96.1 / 99.0 | 93.9 / 98.9 | 96.7 / 98.8 | 24 / 99.3 / 0.88 |
| kw x0.5 | 96.6 | 99.7 / 99.5 | 99.7 / 99.7 | 99.7 / 99.6 | 13 / 99.6 / 0.62 |
| kw x2.0 | 92.3 | 96.2 / 99.0 | 93.1 / 98.3 | 96.0 / 98.3 | 22 / 99.0 / 0.79 |

Replayed versus library concentrations of every fixed design in every scenario: maximum relative difference 4.9e-14.

| scenario | greedy, 3 stations: fixed max mg/L | group lasso 0.1λ0, 21 stations: fixed max mg/L | constant rate, 20 stations: fixed max mg/L | greedy, 3 stations: refit kg/d | group lasso 0.1λ0, 21 stations: refit kg/d | constant rate, 20 stations: refit kg/d | re-solved kg/d |
|---|---|---|---|---|---|---|---|
| nominal | 4.04 | 3.98 | 1.74 | 4.40 | 0.43 | 1.10 | 0.43 |
| demand x0.8 | 5.60 | 3.29 | 1.06 | 3.65 | 0.39 | 1.24 | 0.33 |
| demand x0.9 | 5.43 | 3.54 | 1.82 | 4.49 | 0.42 | 1.37 | 0.42 |
| demand x1.1 | 4.65 | 1.71 | 1.52 | 2.96 | 0.35 | 1.01 | 0.34 |
| demand x1.2 | 4.15 | 1.53 | 0.88 | 3.19 | 0.29 | 0.99 | 0.36 |
| pump 10 two hours later | 5.50 | 1.09 | 1.27 | 3.80 | 0.43 | 1.25 | 0.43 |
| kb x0.5 | 4.11 | 4.23 | 2.01 | 1.90 | 0.20 | 0.70 | 0.21 |
| kb x2.0 | 3.94 | 3.53 | 1.33 | 6.85 | 0.73 | 1.77 | 0.63 |
| kw x0.5 | 4.15 | 4.42 | 1.97 | 2.94 | 0.24 | 0.83 | 0.25 |
| kw x2.0 | 3.91 | 3.55 | 1.56 | 6.30 | 0.74 | 1.62 | 0.65 |

## North Penn

Designs: greedy, 8 stations; group lasso 0.3λ0, 25 stations; constant rate, 24 stations (nominal study, Table S5).

| scenario | plant only | greedy, 8 stations: fixed / refit | group lasso 0.3λ0, 25 stations: fixed / refit | constant rate, 24 stations: fixed / refit | lasso re-solved: stations / coverage / J |
|---|---|---|---|---|---|
| nominal | 86.9 | 97.2 / 97.2 | 97.1 / 97.1 | 95.2 / 95.2 | 25 / 97.1 / 1.00 |
| demand x0.8 | 87.0 | 96.8 / 97.1 | 95.4 / 96.8 | 95.9 / 95.3 | 27 / 96.8 / 0.79 |
| demand x0.9 | 87.4 | 96.7 / 97.2 | 96.1 / 97.0 | 95.7 / 95.8 | 28 / 97.2 / 0.77 |
| demand x1.1 | 85.9 | 94.9 / 96.9 | 91.5 / 96.5 | 93.5 / 94.4 | 25 / 96.4 / 0.85 |
| demand x1.2 | 86.0 | 93.7 / 97.0 | 89.7 / 96.5 | 92.7 / 94.4 | 27 / 96.6 / 0.79 |
| consumer demand two hours earlier | 88.0 | 96.3 / 96.9 | 95.1 / 96.4 | 95.0 / 95.0 | 20 / 95.7 / 0.73 |
| kb x0.5 | 87.2 | 97.5 / 97.4 | 97.2 / 97.2 | 95.4 / 95.4 | 25 / 97.2 / 1.00 |
| kb x2.0 | 86.6 | 96.2 / 96.9 | 96.7 / 96.8 | 94.9 / 95.0 | 26 / 96.9 / 0.96 |
| kw x0.5 | 88.1 | 99.2 / 98.5 | 98.5 / 98.2 | 97.0 / 96.4 | 27 / 98.3 / 0.93 |
| kw x2.0 | 82.7 | 90.1 / 92.8 | 90.4 / 92.9 | 90.5 / 91.2 | 26 / 92.9 / 0.89 |

Replayed versus library concentrations of every fixed design in every scenario: maximum relative difference 2.3e-14.

| scenario | greedy, 8 stations: fixed max mg/L | group lasso 0.3λ0, 25 stations: fixed max mg/L | constant rate, 24 stations: fixed max mg/L | greedy, 8 stations: refit kg/d | group lasso 0.3λ0, 25 stations: refit kg/d | constant rate, 24 stations: refit kg/d | re-solved kg/d |
|---|---|---|---|---|---|---|---|
| nominal | 2.98 | 4.00 | 4.10 | 1.33 | 0.94 | 1.47 | 0.94 |
| demand x0.8 | 3.65 | 5.00 | 5.12 | 1.11 | 0.76 | 1.23 | 0.74 |
| demand x0.9 | 3.31 | 4.55 | 4.56 | 1.21 | 0.86 | 1.36 | 0.69 |
| demand x1.1 | 2.73 | 3.65 | 3.73 | 1.39 | 1.02 | 1.60 | 1.01 |
| demand x1.2 | 2.48 | 3.35 | 3.42 | 1.47 | 1.08 | 1.74 | 1.04 |
| consumer demand two hours earlier | 2.95 | 3.94 | 4.10 | 1.19 | 0.86 | 1.32 | 0.87 |
| kb x0.5 | 2.98 | 4.03 | 4.10 | 1.31 | 0.93 | 1.46 | 0.93 |
| kb x2.0 | 2.97 | 4.00 | 4.10 | 1.38 | 0.96 | 1.49 | 0.96 |
| kw x0.5 | 3.05 | 4.15 | 4.10 | 1.07 | 0.81 | 1.39 | 0.69 |
| kw x2.0 | 2.92 | 4.00 | 4.10 | 1.63 | 1.05 | 1.56 | 1.05 |

Consumer demand shift: patterns [1, 13] moved (supply patterns [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15] kept as operated).
