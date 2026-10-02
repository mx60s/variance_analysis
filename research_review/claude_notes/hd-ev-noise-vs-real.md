---
name: hd-ev-noise-vs-real
description: "How much of place cells' EV_hd is sampling noise vs real HD tuning, across HD bin counts; binning + paper-wording recommendation"
metadata: 
  node_type: memory
  type: project
  originSessionId: e430dad8-c05d-4e89-8eb9-95564391a1c6
---

Analysis 2026-07-06 (Rat1+Rat4 Pfeiffer, 20×20×60 rate-map .npz, notebooks/bio_neurons.ipynb machinery). Question: is place cells' EV_hd≈0.50 (native 60 HD bins) real or noise, to word "place cells largely invariant to HD."

**Exact additive decomposition** (from `quantify_variance_corr` → offdiag, reliability=diag):
EV_hd = 1−offdiag = **(1−reliability)** [noise floor] **+ (reliability−offdiag)** [real, = reliability-normalized EV_hd numerator]. normEV = (reliability−offdiag)/reliability.

Pooled medians across HD bins B=60/30/12/4:
- **Unsmoothed (honest per-bin reliability)**: EV_hd 0.61/0.63/0.60/0.51; reliability 0.50/0.65/0.75/0.83; normEV 0.16/0.42/0.45/0.39. → At native 60 bins EV_hd≈0.61 ≈ **0.50 noise + 0.11 real (~82% noise)**; coarsening removes noise (reliability rises) and reveals a **modest real residual, normEV≈0.4**.
- **Notebook pipeline (smooth σ=(1,2)+zoom, what paper figs use)**: EV_hd 0.50/0.40/0.20/0.07; reliability 0.82/0.85/0.89/0.92; normEV 0.39/0.28/0.09/−0.01. Reproduces prior finding (EV_hd 0.50→~0.07, reliability 0.82→0.92). BUT its collapse-to-0 at coarse bins is partly an **artifact of the fixed σ=2 circular HD smoothing** blurring away tuning (over 4 bins = 90° each); it overstates invariance.

Context: EV_xy≈0.96 ≫ EV_hd (100% cells xy>hd) — cells overwhelmingly spatial. Decoding cross-check (robust to binning): HD decodes weakly above chance — Rat1 60° vs 86°, Rat4 70° vs 80° (Rat2 76°≈75° chance) → real-but-weak HD code, inconsistent with strict invariance.

**Recommendation**: don't quote raw EV_hd at 60 bins (noise-limited, reliability≈0.5); report reliability-normalized EV_hd and/or coarsen to ~12 HD bins (30°, reliability≈0.75–0.8); avoid heavy circular HD smoothing at coarse bins. Wording: keep "predominantly spatial" but soften invariance to "weak, largely-but-not-entirely-noise-driven HD modulation"; anchor on EV_xy≫EV_hd asymmetry + decoding, not a bare EV_hd number. See [[ev-calibration-expectations]] [[ev-metric-convention]].
