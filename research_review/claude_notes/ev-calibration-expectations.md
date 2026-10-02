---
name: ev-calibration-expectations
description: Expected EV patterns per dataset used to validate axis convention and metric correctness
metadata: 
  node_type: memory
  type: project
  originSessionId: dd08380f-7458-493f-a09f-68518a9c7968
---

Maggie's calibration rule for the EV scatters — the result must match the known biology, which also pins down the correct axis labeling (an axis swap is invisible in the symmetric/entangled model data, so calibrate on the asymmetric bio data):

- **Pfeiffer place cells**: high EV(XY), *moderate/noise-inflated* EV(HD). **Recomputed 2026-07-06 from the rate-map .npz (Rat1,Rat4, 20×20×60) — the on-disk `results/rodents/pfeiffer/*-corr.npy` are STALE pre-fix (they show EV_xy≈0.50/EV_hd≈0.04; do NOT use).** Correct notebook pipeline (smooth σ=(1,2)+zoom): EV_xy med ≈0.96, EV_hd med ≈0.50, 100% cells xy>hd. The 0.50 EV_hd is largely sampling noise at fine bins — see [[hd-ev-noise-vs-real]].
- **Dusz HD cells (xy/hd)**: high EV(HD), low EV(XY).
- **Dusz HD cells (hd/md)**: high EV(HD), low EV(MD). `figs/dusz_hdmd_ev_total_best_fit-corr.pdf` already correct (all points high-HD/low-MD, above y=x). ✅
- **Model units (resnet/pred)**: EV(XY) ≈ EV(HD) (entangled, near y=x diagonal).

The stale composite figures embedded the OLD unbounded variance metric (axes to ~1e6). The `-corr` suffixed result/fig files are the correct correlation-metric versions. See [[ev-metric-convention]].
