# Residual diagnostics of the leave-one-site-out test (R1-6)

Leave-one-site-out residuals of the grouped calibration (Kb + four diameter-class Kw), linearised 95% predictive band.

| data set | observations | sites | variance share of site means (ANOVA p) | lag-1 autocorrelation within sites | same after removing site means | site sd range (Brown-Forsythe p) | Spearman |r| vs prediction | 95% coverage | sites below 0.95 | worst site; coverage without it |
|---|---|---|---|---|---|---|---|---|---|---|
| Fairfield | 140 | 15 | 0.15 (p = 0.1) | 0.18 | 0.03 | 0.050-0.416 (p = 0.10) | -0.15 (p = 0.09) | 0.936 | 5 of 15 | site 178: 0.67 (15 obs); others 0.968 |
| Bellingham | 120 | 12 | 0.58 (p = 8e-16) | 0.63 | 0.08 | 0.066-0.161 (p = 0.31) | -0.30 (p = 0.0007) | 0.867 | 7 of 12 | site 150: 0.10 (10 obs); others 0.936 |
| NorthPenn | 42 | 6 | 0.42 (p = 0.001) | 0.35 | -0.08 | 0.059-0.273 (p = 0.12) | 0.03 (p = 0.8) | 0.952 | 2 of 6 | site 243: 0.86 (7 obs); others 0.971 |


## Coverage of the 95% predictive band by held-out site

| data set | site | observations | linearised | sampled | parameter only | mean residual mg/L |
|---|---|---|---|---|---|---|
| Fairfield | 134 | 10 | 1.00 | 1.00 | 0.80 | +0.013 |
| Fairfield | 106 | 10 | 0.90 | 0.90 | 0.40 | +0.100 |
| Fairfield | 1821 | 8 | 1.00 | 1.00 | 0.25 | +0.005 |
| Fairfield | 178 | 15 | 0.67 | 0.67 | 0.00 | -0.196 |
| Fairfield | 168 | 9 | 0.89 | 0.89 | 0.11 | -0.028 |
| Fairfield | 158 | 9 | 1.00 | 1.00 | 0.11 | +0.142 |
| Fairfield | 144 | 7 | 1.00 | 1.00 | 0.43 | +0.040 |
| Fairfield | 1521 | 9 | 0.89 | 0.89 | 0.33 | +0.047 |
| Fairfield | 153 | 9 | 1.00 | 1.00 | 0.67 | +0.047 |
| Fairfield | 141 | 9 | 1.00 | 1.00 | 0.22 | -0.014 |
| Fairfield | 184 | 9 | 1.00 | 1.00 | 0.33 | -0.020 |
| Fairfield | 172 | 9 | 0.89 | 0.89 | 0.11 | -0.096 |
| Fairfield | 162 | 9 | 1.00 | 1.00 | 0.22 | +0.012 |
| Fairfield | 214 | 9 | 1.00 | 1.00 | 0.22 | +0.057 |
| Fairfield | 219 | 9 | 1.00 | 1.00 | 0.11 | +0.016 |
| Bellingham | 112 | 10 | 0.90 | 0.90 | 0.00 | -0.087 |
| Bellingham | 2 | 10 | 1.00 | 1.00 | 0.00 | +0.038 |
| Bellingham | 4 | 10 | 0.90 | 0.90 | 0.00 | +0.083 |
| Bellingham | 23 | 10 | 1.00 | 1.00 | 0.00 | -0.068 |
| Bellingham | 28 | 10 | 0.90 | 0.90 | 0.10 | -0.148 |
| Bellingham | 34 | 10 | 0.80 | 0.80 | 0.30 | +0.099 |
| Bellingham | 58 | 10 | 0.90 | 0.90 | 0.00 | -0.086 |
| Bellingham | 46 | 10 | 1.00 | 1.00 | 0.10 | -0.009 |
| Bellingham | 64 | 10 | 1.00 | 1.00 | 0.00 | +0.005 |
| Bellingham | 90 | 10 | 1.00 | 1.00 | 0.60 | +0.067 |
| Bellingham | 150 | 10 | 0.10 | 0.20 | 0.00 | +0.348 |
| Bellingham | 160 | 10 | 0.90 | 0.90 | 0.20 | +0.118 |
| NorthPenn | 8 | 7 | 1.00 | 1.00 | 0.71 | -0.065 |
| NorthPenn | 29 | 7 | 1.00 | 1.00 | 0.00 | +0.235 |
| NorthPenn | 243 | 7 | 0.86 | 1.00 | 0.43 | +0.048 |
| NorthPenn | 192 | 7 | 0.86 | 0.86 | 0.29 | -0.207 |
| NorthPenn | 700 | 7 | 1.00 | 1.00 | 0.00 | -0.042 |
| NorthPenn | 197 | 7 | 1.00 | 1.00 | 0.57 | -0.013 |
