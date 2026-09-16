# Figures 2–3: caption changes required after the visual redesign

No manuscript or arXiv caption file was edited. Quantitative values and network coordinates are unchanged. Axes were tightened where the old plot reserved empty space for legends; no measured values were clipped.

## Figure 2

- **a:** Replace “bars” with “symbols with pale stems”. Gold circles: zero-tolerance FD; vermilion triangles: default-tolerance FD; cyan squares: eager adjoint; purple diamonds: CUDA-graph adjoint. Values remain median calibration times. Numbers are median loss-evaluation counts. Three runs support the zero-tolerance FD and adjoint medians; default-tolerance FD has one saved run per system. Solver success flags are retained in the source CSV; visual agreement in RMSE does not imply every solver met its stopping criterion.
- **b:** Open grey circles: input parameters; gold crosses: zero-tolerance FD; purple diamonds: graph adjoint; vermilion up-triangles: default-tolerance FD; magenta down-triangles: default FD with h=0.001. Pale regions span nominal RMSE to the best displayed fit. Symbols retain every method/system summary in the former plot.
- **c:** Replace “bars” with “paired symbols”. Filled teal circles give prediction-interval coverage (parameters plus residual noise); open purple squares give parameter-only interval coverage. Their connecting strokes are not uncertainty intervals. The dashed reference is 95%. Printed percentages denote the parameter share of predictive variance, not coverage. Coverage is observation-count-weighted LOSO coverage.
- **d:** Fairfield is magenta, Bellingham cyan, North Penn teal. Solid curves show existing sites; dashed curves show the local prediction after five D-optimal sites. Curves are independently sorted; equal ranks do not match the same pipe. The sd/prior-sd threshold remains 0.5.
- **e,f:** Prior-dominated pipes are light grey (ratio≥0.9); partly informed pipes cyan (0.5≤ratio<0.9); identifiable pipes purple (ratio<0.5). Existing sites are open squares and proposed sites gold triangles, numbered in selection order. All original geometry and site IDs are retained. These are local linear prescriptions at the fitted parameters, not observations from an installed monitoring expansion.

Suggested concise replacement for the changed graphical clauses: “Calibration-time medians are shown by symbols with pale stems (a). Paired symbols compare predictive and parameter-only LOSO coverage; connecting strokes link two coverage summaries and do not denote uncertainty bounds (c). The maps show local information predictions after adding five D-optimal sites, using the unchanged recorded network geometry (e,f).”

## Figure 3

- Net3 is magenta and North Penn teal throughout. Existing source/reservoir markers remain gold stars; candidate nodes light grey; selected group-lasso support open network-coloured squares; greedy stations filled network-coloured triangles with order labels. Label offsets changed only for readability; node coordinates did not move.
- **a,b:** Maps still show lambda/lambda0=0.4 (9 and 20 stations), together with all eight greedy stations. 
- **c:** Greedy paths are filled circles and connecting lines; refitted supports open squares; constant-rate designs open triangles. The station axis retains its linear section through eight stations and logarithmic section thereafter, including the zero-booster baseline. No design is removed.
- **d:** Every saved refit, constant-rate and greedy design remains plotted. Thin segments pair time-varying and constant-rate doses on the same support. The colours encode networks and the shapes encode methods. Links are neither uncertainty intervals nor global Pareto-optimality certificates.
- **e:** Net3's 9×24 selected dosing schedule is unchanged; the colour scale now runs from near-white through magenta to purple and the colourbar is horizontal. Zero remains zero. Values are kg/h for these one-hour slots; their sum is 0.34048879878 kg/day, agreeing with the stored 0.3404887986333703 kg/day to the saved dosage precision.
- **f:** Grey is plant-only; solid network colour is lambda/lambda0=0.4; dashed is 0.1. The corresponding station counts remain Net3 9/21 and North Penn 20/57. Temporal shortfall is a fraction of nodes, whereas the aggregate coverage follows the experiment's node-hour weighting. No interpolation or smoothing was introduced.

Keep the optimization qualification: the group-lasso path is a continuous relaxation; several saved refits did not meet their stopping condition. The redesigned figure does not turn these designs into certified combinatorial optima.

## Export and verification

- Figure 2: 176×184 mm; Figure 3: 176×190 mm.
- Both: editable SVG/PDF text, 600-dpi RGB TIFF/PNG, minimum extracted PDF glyph size 5.04 pt, no text outside the PDF page.
- Source-data checks cover 36 calibration runs, 36 Figure-2 summaries, 1,386 ranked pipe values, 567 mapped pipe values, 28 monitoring sites; 74 Figure-3 design records, 216 dosing cells, 150 time-series points and 436 mapped nodes.
- Re-run `python fig2_field.py`, `python fig3_exp1.py`, then `python fig23_qa.py` from any working directory with the plotting environment. The scripts resolve their own paths and do not run simulation experiments.
