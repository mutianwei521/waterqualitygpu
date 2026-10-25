# Independent gradient validation: adjoint of the replay versus central differences of the library (R1-3 / R2-3)

Run r1_G_1723889: NVIDIA GeForce RTX 5090, torch 2.8.0+cu128, EPANET 20306, seed 20261022; library losses summed in long double (eps 1.1e-19). Relative error |FD - g.d| / max(|g.d|, |FD|); steps h = 1e-2 to 1e-6 (BWSN2 per-pipe map: 1e-2, 1e-3, 1e-5). Pass, smooth branch: minimum error <= 1e-6 and central-difference slope between h = 1e-2 and 1e-3 in [1.7, 2.3] (pre-registered: slope between 1e-2 and 1e-4); affine maps: maximum error <= 1e-9.

## S-grad-1. Summary by group and class (errors are the minimum over step sizes of each direction)

| group | class | directions | pass | pass (pre-registered slope) | median | 95th percentile | maximum |
|---|---|---|---|---|---|---|---|
| A | smooth branch | 2698 | 2673 | 2452 | 9.7e-12 | 2.2e-10 | 4.0e-08 |
| A | perturbation from the all-zero reaction point | 123 | - | - | 8.7e-06 | 1.3e-03 | 5.8e-02 |
| A | wall coefficient perturbed from kw = 0 | 59 | - | - | 4.4e-08 | 5.2e-07 | 6.8e-06 |
| A | set-point source near its switch | 63 | - | - | 1.7e-11 | 1.4e-10 | 3.2e-09 |
| BL | smooth branch | 411 | 411 | 378 | 1.3e-10 | 5.5e-09 | 2.0e-07 |
| BL | wall coefficient perturbed from kw = 0 | 131 | - | - | 1.7e-08 | 3.5e-06 | 2.8e-05 |
| BL | parameter on a bound | 13 | - | - | 9.0e-10 | 1.3e-06 | 3.2e-06 |
| BL | affine (slot sources) | 60 | 60 | 60 | 4.4e-15 | 2.8e-13 | 9.0e-13 |
| BW | smooth branch | 30 | 30 | n/a (no 1e-4 step) | 4.3e-10 | 3.5e-08 | 3.2e-07 |
| BW | affine (slot sources) | 15 | 15 | 15 | 2.1e-15 | 1.1e-14 | 1.2e-14 |


## S-grad-2. Smooth and affine directions that did not pass

| group | map | direction | parameter | |g.d| | minimum error | slope 1e-2 to 1e-3 | error by step |
|---|---|---|---|---|---|---|---|
| A | A_Net1_P2 | single | kb#6 | 1.40e-01 | 7.2e-11 | 1.66 | 0.01: 3.3e-09, 0.001: 7.2e-11, 0.0001: 3.0e-09, 1e-05: 1.8e-08, 1e-06: 1.6e-07 |
| A | A_Net2_P1 | single | kw#65 | 4.16e-01 | 2.3e-10 | 2.33 | 0.01: 4.9e-08, 0.001: 2.3e-10, 0.0001: 5.3e-09, 1e-05: 2.5e-09, 1e-06: 8.0e-07 |
| A | A_Net2_P1 | single | kw#53 | 2.41e-01 | 2.0e-10 | 2.54 | 0.01: 7.0e-08, 0.001: 2.0e-10, 0.0001: 6.4e-09, 1e-05: 2.7e-08, 1e-06: 4.6e-07 |
| A | A_Net2_P2 | single | kb#13 | 1.80e+00 | 6.0e-12 | 3.06 | 0.01: 6.9e-09, 0.001: 6.0e-12, 0.0001: 1.0e-09, 1e-05: 1.6e-09, 1e-06: 5.8e-08 |
| A | A_Net1_MIX2_P1 | single | kb#0 | 5.14e-01 | 1.6e-10 | 1.64 | 0.01: 6.9e-09, 0.001: 1.6e-10, 0.0001: 4.7e-10, 1e-05: 8.5e-09, 1e-06: 5.3e-08 |
| A | A_Net1_src_mass_P1 | single | kb#1 | 2.28e-01 | 4.7e-12 | 2.54 | 0.01: 1.6e-09, 0.001: 4.7e-12, 0.0001: 4.7e-10, 1e-05: 1.9e-09, 1e-06: 7.7e-09 |
| A | A_Net1_src_mass_P1 | single | kw#18 | 1.27e-02 | 3.3e-09 | 1.49 | 0.01: 1.0e-07, 0.001: 3.3e-09, 0.0001: 1.9e-08, 1e-05: 9.7e-08, 1e-06: 3.0e-06 |
| A | A_Net1_src_mass_P2 | single | kb#6 | 1.44e-01 | 8.4e-11 | 1.61 | 0.01: 3.4e-09, 0.001: 8.4e-11, 0.0001: 3.6e-09, 1e-05: 1.1e-09, 1e-06: 1.1e-07 |
| A | A_Net1_src_setpoint_P1 | single | kb#1 | 2.76e-01 | 5.3e-11 | 1.44 | 0.01: 1.5e-09, 0.001: 5.3e-11, 0.0001: 1.2e-09, 1e-05: 2.2e-09, 1e-06: 4.2e-08 |
| A | A_Net1_src_setpoint_P1 | single | kw#18 | 1.43e-02 | 1.2e-09 | 1.47 | 0.01: 1.2e-07, 0.001: 4.2e-09, 0.0001: 1.2e-09, 1e-05: 1.5e-07, 1e-06: 1.6e-06 |
| A | A_Net1_src_flowpaced_P1 | single | kw#18 | 1.43e-02 | 1.2e-09 | 1.47 | 0.01: 1.2e-07, 0.001: 4.1e-09, 0.0001: 1.2e-09, 1e-05: 1.5e-07, 1e-06: 8.4e-07 |
| A | A_Net1_src_concen_reservoir_pat_P1 | single | kb#1 | 2.84e-01 | 6.7e-12 | 2.33 | 0.01: 1.4e-09, 0.001: 6.7e-12, 0.0001: 3.4e-10, 1e-05: 3.8e-09, 1e-06: 1.3e-07 |
| A | A_Net1_src_concen_junction_inflow_P1 | single | kb#0 | 4.80e-01 | 1.8e-10 | 1.65 | 0.01: 8.1e-09, 0.001: 1.8e-10, 0.0001: 7.4e-10, 1e-05: 5.8e-09, 1e-06: 8.5e-08 |
| A | A_Net2_MIX2_P1 | single | kb#1 | 1.43e-01 | 1.1e-09 | -0.35 | 0.01: 1.1e-09, 0.001: 2.5e-09, 0.0001: 3.2e-09, 1e-05: 4.4e-08, 1e-06: 1.4e-06 |
| A | A_Net2_MIX2_P2 | single | kb#26 | 1.12e+00 | 4.1e-11 | 1.53 | 0.01: 1.4e-09, 0.001: 4.1e-11, 0.0001: 3.5e-10, 1e-05: 1.6e-09, 1e-06: 3.9e-08 |
| A | A_Net2_MIX2_P2 | single | kb#28 | 7.19e-01 | 1.1e-10 | 1.61 | 0.01: 4.4e-09, 0.001: 1.1e-10, 0.0001: 2.3e-10, 1e-05: 8.3e-09, 1e-06: 4.6e-08 |
| A | A_Net2_MIX2_P2 | single | kb#14 | 1.40e+00 | 5.6e-11 | 1.65 | 0.01: 2.5e-09, 0.001: 5.6e-11, 0.0001: 4.6e-10, 1e-05: 1.9e-09, 1e-06: 1.7e-08 |
| A | A_Net3_CHEM_mixed_tanks_P0 | single | kb#62 | 1.84e+00 | 1.0e-12 | 3.03 | 0.01: 1.1e-09, 0.001: 1.0e-12, 0.0001: 9.3e-11, 1e-05: 1.3e-09, 1e-06: 2.6e-11 |
| A | A_Net3_CHEM_mixed_tanks_P0 | single | kb#102 | 2.04e-01 | 4.6e-11 | 1.34 | 0.01: 1.0e-09, 0.001: 4.6e-11, 0.0001: 4.5e-10, 1e-05: 1.2e-08, 1e-06: 1.3e-07 |
| A | A_Net3_CHEM_mixed_tanks_P1 | single | kb#51 | 4.24e-04 | 1.1e-09 | -1.51 | 0.01: 1.1e-09, 0.001: 3.7e-08, 0.0001: 7.6e-08, 1e-05: 3.5e-06, 1e-06: 2.7e-05 |
| A | A_Net3_CHEM_mixed_tanks_P1 | single | kw#188 | 4.97e-04 | 6.0e-11 | 1.67 | 0.01: 2.8e-09, 0.001: 6.0e-11, 0.0001: 1.6e-10, 1e-05: 3.1e-08, 1e-06: 2.9e-07 |
| A | A_Net2_react_P0 | single | kb#17 | 3.24e-01 | 1.5e-10 | 2.56 | 0.01: 5.3e-08, 0.001: 1.5e-10, 0.0001: 7.4e-10, 1e-05: 1.7e-08, 1e-06: 2.5e-07 |
| A | A_Net2_react_P1 | single | kb#0 | 3.94e-01 | 7.6e-10 | 0.57 | 0.01: 2.9e-09, 0.001: 7.6e-10, 0.0001: 4.0e-09, 1e-05: 2.6e-08, 1e-06: 8.7e-08 |
| A | A_Net2_react_P1 | single | kb#4 | 1.93e-01 | 1.4e-11 | 2.52 | 0.01: 4.5e-09, 0.001: 1.4e-11, 0.0001: 1.6e-10, 1e-05: 8.8e-09, 1e-06: 1.7e-07 |
| A | A_Net2_react_P2 | single | kb#13 | 1.80e+00 | 6.0e-12 | 3.06 | 0.01: 6.9e-09, 0.001: 6.0e-12, 0.0001: 1.0e-09, 1e-05: 1.6e-09, 1e-06: 5.8e-08 |


## S-grad-3. Reaction-factor boundary (Net1, bulk coefficient on all pipes, 300 s step)

| kb (1/d) | minimum factor f | library clamp hits | replay vs library | FD error at h = 1e-3, 1e-5 |
|---|---|---|---|---|
| -10.0 | 0.965 | 0 | 4.7e-15 | 4.4e-07, 4.2e-11 |
| -100.0 | 0.653 | 0 | 1.8e-15 | 4.7e-06, 4.8e-10 |
| -250.0 | 0.132 | 0 | 2.2e-16 | 9.4e-05, 9.3e-09 |
| -280.0 | 0.028 | 0 | 2.2e-16 | 2.5e-03, 1.8e-07 |
| -300.0 | - | 178 | replay refused (f <= 0) | - |
