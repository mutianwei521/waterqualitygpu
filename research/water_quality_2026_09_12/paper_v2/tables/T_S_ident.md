# Pipe-level identifiability: dependence on the threshold and on the prior (R1-8)

## S-ident-1. Identifiable pipes for thresholds posterior sd / prior sd < 0.3 / 0.5 / 0.7 / 0.9

| network | setting | pipes | existing sensors | after five prescribed sites |
|---|---|---|---|---|
| Fairfield | field audit, tau = 0.3 ft/d | 126 | 0 / 0 / 0 / 3 | 2 / 3 / 8 / 19 |
| Bellingham | field audit, tau = 0.3 ft/d | 168 | 0 / 2 / 3 / 4 | 5 / 8 / 14 / 17 |
| NorthPenn | field audit, tau = 0.3 ft/d | 399 | 0 / 0 / 1 / 5 | 5 / 14 / 28 / 39 |
| LTown | synthetic, random sensors, tau = 0.5 x class value | 905 | 2 / 5 / 8 / 33 | - |
| BWSN2 | synthetic, random sensors, tau = 0.5 x class value | 14822 | 4 / 9 / 21 / 63 | - |


## S-ident-2. Field audits with prior sd tau = 0.1, 0.3 and 1.0 ft/d (threshold 0.5)

| data set | tau ft/d | identifiable pipes | p_eff | median sd / tau | identifiable after five sites | p_eff after five sites | rank correlation of sd / tau with tau = 0.3 |
|---|---|---|---|---|---|---|---|
| Fairfield | 0.1 | 0 | 0.7 | 0.999 | 0 | 5.3 | 0.995 |
| Fairfield | 0.3 | 0 | 3.9 | 0.996 | 3 | 13.0 | 1.000 |
| Fairfield | 1.0 | 3 | 12.5 | 0.984 | 11 | 26.7 | 0.985 |
| Bellingham | 0.1 | 0 | 1.2 | 1.000 | 3 | 7.2 | 0.983 |
| Bellingham | 0.3 | 2 | 3.1 | 1.000 | 8 | 13.2 | 1.000 |
| Bellingham | 1.0 | 4 | 5.8 | 0.999 | 15 | 23.8 | 0.994 |
| NorthPenn | 0.1 | 0 | 1.2 | 1.000 | 5 | 7.2 | 0.929 |
| NorthPenn | 0.3 | 0 | 3.9 | 1.000 | 14 | 28.2 | 1.000 |
| NorthPenn | 1.0 | 1 | 8.9 | 1.000 | 31 | 47.0 | 0.960 |
