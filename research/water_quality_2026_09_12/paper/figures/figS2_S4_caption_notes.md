# Caption updates required for redesigned Figures S2--S4

Use the same wording in `paper/arxiv/si_figures.tex` and
`paper/est/supporting_information_est.md`.  The quantitative statements and
table citations are unchanged; only the visual encodings have changed.

## Figure S2

**Figure S2. Finite-difference step sweep for all tested directions.** Each
cell gives the relative error of the library central difference against the
replay gradient for the all-ones direction over one parameter group, at steps
$h=10^{-1}$ to $10^{-7}$, (a) at the input merging tolerance
($C_{\mathrm{tol}}=0.01$ mg/L) and (b) at zero tolerance. Rows are the eleven
network/configuration and parameter directions of Table S2; coloured row
labels identify bulk coefficients $K_b$ (purple), wall coefficients $K_w$
(green) and initial qualities $c_0$ (cyan). Cell colour encodes relative error
on a logarithmic scale from blue (smaller) to orange (larger), and the outlined
circle marks the minimum error within each row. Cells show the reported values
without interpolation or smoothing. Data: Table S2.

## Figure S3

**Figure S3. Engine cost and routing throughput.** (a) Wall-clock time of one
library run, one library finite-difference gradient, one per-operator adjoint
pass and one fused forward-plus-adjoint pass at batch size one, together with
the per-parameter-set cost of the fused replay at a batch of 1,024, for four
configurations. Position is shown on a logarithmic time axis; method colours
and marker shapes follow the legend. (b) Jacobian rows per second on a shared
schedule against batch size $B$, separated into six network facets to avoid
overplotting. Cyan dotted circles denote the gather path, green dashed squares
the sparse (SpMM) path and purple solid open diamonds the automatic
per-operation choice; facet-title colours identify the networks. Data: Table
S3.

## Figure S4

**Figure S4. Field calibration on the 1993 chlorine data sets.** (a, b) Site
RMSE for the nominal model and each calibration method with the bulk
coefficient free (a) or fixed at its input-file value (b). Method colours and
marker shapes follow the shared legend; vertical offsets only separate
coincident values. (c) Wall-clock time per complete fit with the bulk
coefficient free, on a logarithmic axis. Purple brackets and labels give the
speed-up of the execution-graph adjoint over the zero-tolerance
finite-difference fit. Points in (a--c) show the tabulated median where a method
was repeated and the reported value for single-run controls. (d)
Leave-one-site-out (LOSO) coverage of the 95% predictive band (linearised and
sampled) and of the parameter-only band; the vertical gold line marks the
nominal coverage of 0.95. Network labels retain the cross-figure colours.
Data: Table S4.

No other S2--S4 prose changes are required.  Method S5 already describes the
finite-difference sweep without referring to line or marker encodings, Method
S4 refers to the three routing paths generically, and Methods S7--S8 refer only
to the reported calibration and uncertainty quantities.
