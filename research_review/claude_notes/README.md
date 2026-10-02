# Claude project notes

Working notes written by Claude Code during sessions on this project (June–September 2026), copied from the local memory folders on 2026-10-02. They reflect what was true when written; `~/explained_variance` referred to in them has since been removed.

- [EV metric convention](ev-metric-convention.md) — EV = 1 − mean off-diag corr; quantify_variance_corr returns raw corr, callers must apply 1−; which notebooks are correct
- [EV calibration expectations](ev-calibration-expectations.md) — expected high/low EV per dataset, used to validate axis convention
- [HD EV noise vs real](hd-ev-noise-vs-real.md) — place cells' EV_hd≈0.50 at 60 bins is ~82% sampling noise (unsmoothed); modest real residual normEV≈0.4; binning + wording rec; stored pfeiffer -corr.npy are stale
- [Model data blockers](model-data-blockers.md) — missing pred checkpoint + incomplete pitch panorama block regenerating 6 deep-net layers locally
- [Project layout: two copies](project-layout-two-copies.md) — ~/explained_variance is a scrubbed partial copy; run things in ~/variance_analysis with notebooks/run_nb.py
- [HD/MD behavioural coupling](hd-md-behavioural-coupling.md) — HD and MD only weakly coupled in Dusz sessions; MD convention verified
- [Rate-map params and speedups](ratemap-params-and-speedups.md) — floor/vel settings that reproduce the rodent results; Pfeiffer builder vectorised; A5506 lacks tracking
- [Manuscript state](manuscript-state.md) — 2026-09-13 revision of manuscript.tex, what was swapped in, what the author still owes
