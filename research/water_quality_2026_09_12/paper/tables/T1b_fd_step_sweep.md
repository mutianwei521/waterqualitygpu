# Finite-difference step sweep against EPANET (cluster libepanet2.so)

Relative error of the EPANET central difference (direction: +1 on every parameter) against the replay gradient.

| case / parameter | mode | Ctol | replay gradient | h=1e-01 | h=1e-02 | h=1e-03 | h=1e-04 | h=1e-05 | h=1e-06 | h=1e-07 |
|---|---|---|---|---|---|---|---|---|---|---|
| Jilin_kw / kb (n=34) | M0_inp_ctol | 0.01 | 425 | 1.2e-03 | 4.3e-04 | 1.7e-09 | 2.3e-11 | 1.2e-10 | 9.8e-11 | 1.3e-08 |
| Jilin_kw / kb (n=34) | M1_ctol0 | 0 | 425 | 1.7e-05 | 1.7e-07 | 1.7e-09 | 1.3e-11 | 1.9e-11 | 9.8e-10 | 6.3e-09 |
| Net1 / kb (n=12) | M0_inp_ctol | 0.01 | 17.1 | 4.3e-03 | 2.6e-02 | 1.2e-08 | 1.4e-10 | 2.4e-10 | 9.9e-10 | 1.1e-08 |
| Net1 / kb (n=12) | M1_ctol0 | 0 | 17.1 | 1.2e-04 | 1.2e-06 | 1.2e-08 | 1.5e-10 | 1.7e-10 | 5.0e-10 | 3.0e-09 |
| Net2_MIX2 / kw (n=40) | M0_inp_ctol | 0.01 | 2.18e+03 | 5.6e-01 | 1.9e-01 | 3.8e-03 | 3.9e-02 | 3.7e-01 | 1.4e+00 | 1.0e+00 |
| Net2_MIX2 / kw (n=40) | M1_ctol0 | 0 | 2.18e+03 | 5.6e-01 | 2.0e-01 | 2.8e-02 | 2.1e-03 | 6.2e-04 | 8.9e-04 | 9.2e-04 |
| Net2 / kb (n=40) | M0_inp_ctol | 0.01 | 410 | 3.0e-03 | 8.8e-03 | 1.5e-02 | 1.7e-01 | 1.7e+00 | 1.1e+00 | 1.0e+00 |
| Net2 / kb (n=40) | M1_ctol0 | 0 | 411 | 2.8e-03 | 1.2e-03 | 1.2e-03 | 1.2e-03 | 1.2e-03 | 1.2e-03 | 1.2e-03 |
| Net2 / kw (n=40) | M0_inp_ctol | 0.01 | 2.22e+03 | 5.6e-01 | 1.9e-01 | 1.7e-02 | 3.3e-02 | 3.1e-01 | 1.5e+00 | 1.0e+00 |
| Net2 / kw (n=40) | M1_ctol0 | 0 | 2.23e+03 | 5.6e-01 | 2.0e-01 | 2.7e-02 | 2.1e-03 | 6.0e-04 | 8.8e-04 | 9.0e-04 |
| Net2_react / c0 (n=36) | M0_inp_ctol | 0.01 | 500 | 3.1e-03 | 9.0e-03 | 2.1e-02 | 2.6e-12 | 1.3e-11 | 1.9e-10 | 9.3e-09 |
| Net2_react / c0 (n=36) | M1_ctol0 | 0 | 500 | 2.3e-15 | 3.3e-14 | 2.2e-13 | 2.3e-12 | 2.6e-11 | 3.5e-10 | 4.6e-10 |
| Net2_react / kb (n=40) | M0_inp_ctol | 0.01 | 200 | 3.2e-04 | 1.6e-02 | 2.3e-02 | 1.2e-09 | 2.3e-11 | 1.1e-09 | 1.9e-08 |
| Net2_react / kb (n=40) | M1_ctol0 | 0 | 201 | 1.2e-03 | 1.2e-05 | 1.2e-07 | 1.2e-09 | 1.1e-10 | 2.2e-10 | 2.1e-08 |
| Net3_CHEM_kw_mixed_tanks / kb (n=117) | M0_inp_ctol | 0.01 | 356 | 1.4e-03 | 6.2e-02 | 7.9e-02 | 1.3e-02 | 1.3e-11 | 1.8e-10 | 1.1e-09 |
| Net3_CHEM_kw_mixed_tanks / kb (n=117) | M1_ctol0 | 0 | 359 | 2.2e-04 | 2.2e-06 | 2.2e-08 | 2.2e-10 | 1.2e-12 | 5.1e-10 | 1.5e-09 |
| Net3_CHEM_kw_mixed_tanks / kw (n=117) | M0_inp_ctol | 0.01 | 757 | 1.4e-02 | 3.3e-02 | 1.5e-01 | 1.4e-02 | 7.6e-11 | 1.0e-10 | 1.5e-09 |
| Net3_CHEM_kw_mixed_tanks / kw (n=117) | M1_ctol0 | 0 | 762 | 6.3e-03 | 6.2e-05 | 6.2e-07 | 6.2e-09 | 6.9e-11 | 1.3e-10 | 3.2e-10 |
| Net3_CHEM_mixed_tanks / kb (n=117) | M0_inp_ctol | 0.01 | 404 | 2.6e-02 | 2.7e-02 | 9.7e-02 | 2.0e-01 | 7.1e-01 | 3.1e-10 | 3.1e-10 |
| Net3_CHEM_mixed_tanks / kb (n=117) | M1_ctol0 | 0 | 413 | 2.4e-04 | 2.4e-06 | 2.4e-08 | 2.4e-10 | 4.2e-12 | 1.6e-10 | 1.6e-10 |
| Net3_CHEM_mixed_tanks / kw (n=117) | M0_inp_ctol | 0.01 | 1.12e+03 | 9.8e-02 | 2.5e-02 | 1.6e-02 | 7.8e-02 | 4.7e-01 | 1.1e-05 | 1.1e-06 |
| Net3_CHEM_mixed_tanks / kw (n=117) | M1_ctol0 | 0 | 1.14e+03 | 1.1e-01 | 4.0e-02 | 8.5e-03 | 1.1e-03 | 1.1e-04 | 1.1e-05 | 1.1e-06 |
