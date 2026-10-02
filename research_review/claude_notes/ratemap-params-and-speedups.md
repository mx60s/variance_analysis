---
name: ratemap-params-and-speedups
description: "Rate-map build parameters that reproduce the paper's rodent EV results, the Pfeiffer builder speedup, and the one Dusz session with missing tracking data"
metadata: 
  node_type: memory
  type: project
  originSessionId: c9fe392c-1579-4978-80f7-7cbf9a20c0ef
  modified: 2026-09-14T00:31:41.245Z
---

Verified 2026-09-13 in `~/variance_analysis`:
- Dusz location×HD maps must be built with `vel_threshold=2, floor=5` (stored maps have min occupancy 4). `bio_neurons.ipynb` had drifted to `vel_threshold=0, floor=10`, which makes HD cells look location-tuned (A1808: 76% HD-dominant at floor 5 vs 10% at floor 10, because a high floor at 50 ms bins keeps mostly stationary episodes). Notebook cell restored. HD×MD maps use `vel_threshold=1, floor=10` (unchanged, consistent).
- Pfeiffer maps: `tau=0.25 s`, 20×20 location bins, 60 HD bins, `vel 5 cm/s`, `floor=5`.
- `src/pfeiffer_ratemaps.py` lookups were O(samples × time bins) per bin pair; vectorised with searchsorted (bit-identical output, 5 s per rat instead of 5+ min; Rat 3 previously took hours).
- `data/rodents/dusz/Dataset_2/A5506-200914a/Data/Tracking_data.csv` is absent, so that session cannot be rebuilt; its old fold maps are kept and the notebook now skips it with a message.
- Results with these settings: place cells 99% EV_xy>EV_hd (n=551); HD cells 94% EV_hd>EV_xy (n=1489); HD vs MD 100% EV_hd>EV_md (n=1465); trained network layers on the diagonal.

**Why:** Maggie's expected biology only reproduces with the stored-map parameters; the parameter drift silently broke the HD-cell result.

**How to apply:** Don't change floor/vel in the rate-map cells without re-checking the HD-cell scatter. See [[project-layout-two-copies]], [[hd-md-behavioural-coupling]].
