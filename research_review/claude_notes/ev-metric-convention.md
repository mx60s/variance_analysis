---
name: ev-metric-convention
description: "How the correlation-based explained-variance (EV) metric is defined, the canonical helpers, and which pipelines use them"
metadata: 
  node_type: memory
  type: project
  originSessionId: dd08380f-7458-493f-a09f-68518a9c7968
---

Metric (paper Eq. 2): **EV(x) = 1 − ⟨C_ij⟩_{i≠j}** (one minus mean off-diagonal cross-fold Pearson correlation). ~[0,1] but can slightly exceed 1.0 when off-diag corr goes negative (seen ~1.04–1.06).

`src/explained_variance.py:quantify_variance_corr(...)` returns the **raw** mean off-diag correlation (NOT 1−). Callers apply `1 −` inline. (Do NOT wrap this in new abstraction functions — Maggie wanted the existing inline call grounded, not a refactor; an over-engineered `explained_variance_xy_hd`/`_hd_md` pair was added then reverted 2026-06-22.)

The grounded call, used inline at every `process_neuron*`:
- `ev_xy = 1 - quantify_variance_corr(fold0, fold1, occ_z, axis=0)[0]`  (location)
- `ev_hd = 1 - quantify_variance_corr(fold0, fold1, occ_z, axis=1)[0]`  (head direction)
Both the `1 −` and the axis assignment matter. Same rate-map layout (axis0=location, axis1=HD) in bio + model.

Smoothing (kept; light): `nan_aware_gaussian_filter(..., circular_axes=(1,))` for location×HD (wrap HD only), `circular_axes=(0, 1)` for HD×MD (both circular). For 3D (x,y,hd) rate-map generation cells, `circular_axes=(0,0,1)` (wrap HD) is correct.

History of bugs fixed 2026-06-20:
- Missing `1−` and swapped axis in `model_neurons.ipynb` and `binning_bias_fig.ipynb`; binning also wrongly normalized to a fraction.
- Inconsistent `circular_axes` across all 6 process_neuron functions (pfeiffer wrapped location; dusz_xyhd wrapped location+HD; dusz_hdmd wrapped only MD; model loop wrapped location; encoder didn't wrap). All now route through the canonical helpers.
- Stale on-disk results: `results/rodents/pfeiffer/*-corr.npy` were from the OLD buggy code (raw, swapped) — recompute from rate maps. dusz `-corr.npy` were already correct-metric.

Both `bio_neurons.ipynb` and `model_neurons.ipynb` are now wired to the canonical helpers (validated locally). See [[ev-calibration-expectations]] and [[model-data-blockers]].
