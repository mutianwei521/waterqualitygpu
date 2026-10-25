# Every genetic-algorithm run (R2-11)

Same settings in every run (population 64, 60 generations, at most 1,000 warm-started inner iterations, tournaments of three, four elites, uniform crossover repaired to m stations, mutation probability 0.3); only the random seed differs. Objective relative to greedy selection with the same number of stations. *Collapse*: first generation in which at least half of the population is the best set.


## Net3, m = 3 (greedy: objective 4.887 ×10⁻⁵, coverage 99.37%, 4.40 kg/d)

| seed | stations | objective (×10⁻⁵) | vs greedy | coverage % | mass kg/d | stations shared with greedy | sets evaluated | collapse (generation) | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 127, 131, 211 | 4.887 | +0.0% | 99.37 | 4.40 | 3 | 806 | 16 | 2060 |
| 1 | 131, 211, 3 | 4.871 | -0.3% | 99.34 | 4.41 | 2 | 957 | 21 | 2015 |
| 2 | 131, 211, 3 | 4.871 | -0.3% | 99.34 | 4.41 | 2 | 888 | 11 | 2026 |

3 runs: 0 worse, 2 better and 1 equal (within 0.05%) relative to greedy selection; objective relative to greedy median -0.3%, range -0.3% to +0.0%.

## Net3, m = 8 (greedy: objective 1.456 ×10⁻⁵, coverage 99.83%, 1.80 kg/d)

| seed | stations | objective (×10⁻⁵) | vs greedy | coverage % | mass kg/d | stations shared with greedy | sets evaluated | collapse (generation) | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 127, 131, 139, 153, 213, 243, 255, 1 | 1.206 | -17.2% | 99.83 | 1.57 | 4 | 1560 | 20 | 2093 |

## North Penn, m = 8 (greedy: objective 21.041 ×10⁻⁵, coverage 97.24%, 1.33 kg/d)

| seed | stations | objective (×10⁻⁵) | vs greedy | coverage % | mass kg/d | stations shared with greedy | sets evaluated | collapse (generation) | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 245, 266, 280, 292, 296, 313, 317, 616 | 27.556 | +31.0% | 96.97 | 1.34 | 4 | 1657 | 20 | 502 |
| 1 | 267, 291, 297, 313, 319, 428, 615, 616 | 21.389 | +1.7% | 97.01 | 1.43 | 2 | 1553 | 23 | 470 |
| 2 | 266, 292, 298, 312, 318, 429, 615, 616 | 19.816 | -5.8% | 97.32 | 1.31 | 6 | 1513 | 15 | 474 |
| 3 | 266, 292, 296, 312, 319, 429, 615, 616 | 19.067 | -9.4% | 97.35 | 1.33 | 6 | 1637 | 18 | 576 |
| 4 | 267, 291, 298, 313, 316, 428, 615, 616 | 21.060 | +0.1% | 97.31 | 1.28 | 2 | 1718 | 36 | 579 |
| 5 | 266, 291, 296, 312, 319, 428, 615, 616 | 18.948 | -9.9% | 97.31 | 1.32 | 4 | 1695 | 45 | 582 |
| 6 | 266, 292, 296, 300, 312, 318, 428, 616 | 25.070 | +19.1% | 97.15 | 1.24 | 5 | 1648 | 22 | 576 |
| 7 | 268, 292, 296, 312, 317, 429, 615, 616 | 20.908 | -0.6% | 97.29 | 1.31 | 5 | 1532 | 31 | 476 |
| 8 | 266, 275, 292, 297, 311, 318, 428, 616 | 25.094 | +19.3% | 97.09 | 1.24 | 4 | 1587 | 27 | 576 |
| 9 | 266, 291, 298, 312, 319, 428, 615, 616 | 19.726 | -6.3% | 97.33 | 1.30 | 3 | 1521 | 22 | 583 |
| 10 | 267, 292, 296, 313, 317, 428, 615, 616 | 20.034 | -4.8% | 97.28 | 1.34 | 4 | 1797 | 37 | 576 |
| 11 | 245, 267, 292, 296, 311, 319, 428, 615 | 25.466 | +21.0% | 96.97 | 1.35 | 3 | 1612 | 23 | 477 |
| 12 | 247, 291, 296, 312, 319, 428, 615, 616 | 20.447 | -2.8% | 97.22 | 1.28 | 3 | 1653 | 18 | 470 |
| 13 | 267, 291, 297, 311, 319, 428, 615, 616 | 21.173 | +0.6% | 97.09 | 1.39 | 2 | 1507 | 22 | 476 |
| 14 | 247, 291, 298, 312, 319, 428, 615, 616 | 21.225 | +0.9% | 97.24 | 1.26 | 2 | 1567 | 26 | 469 |
| 15 | 246, 267, 291, 296, 312, 429, 615, 616 | 20.718 | -1.5% | 97.26 | 1.30 | 4 | 1545 | 19 | 475 |
| 16 | 247, 262, 292, 296, 311, 318, 429, 615 | 26.916 | +27.9% | 96.93 | 1.30 | 5 | 1770 | 31 | 469 |
| 17 | 268, 275, 292, 296, 311, 318, 428, 616 | 26.102 | +24.1% | 97.14 | 1.20 | 4 | 1532 | 20 | 476 |
| 18 | 245, 266, 292, 296, 313, 316, 428, 616 | 26.091 | +24.0% | 96.96 | 1.32 | 4 | 1584 | 22 | 472 |
| 19 | 245, 264, 292, 296, 311, 429, 615, 616 | 22.677 | +7.8% | 97.04 | 1.34 | 5 | 1596 | 26 | 469 |
| 20 | 246, 291, 296, 313, 318, 429, 615, 616 | 21.789 | +3.6% | 96.93 | 1.32 | 5 | 1539 | 19 | 468 |

21 runs: 13 worse, 8 better and 0 equal (within 0.05%) relative to greedy selection; objective relative to greedy median +0.9%, range -9.9% to +31.0%.
