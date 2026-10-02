---
name: model-data-blockers
description: Deep-net EV layer run — RESOLVED; how it was unblocked and run
metadata:
  type: project
---

RESOLVED 2026-07-07. All 10 deep-net EV scatters regenerated with the corrected/grounded
metric on Maggie's M5 (MPS). Blockers that were cleared:
1. Pred-coder checkpoint: dropped in at `data/weights/predictive_coder.ckpt` — verified
   loads strict into PredictiveCoder (157/157 keys).
2. Complete panorama data: extracted from `panorama-90yaw-pitch.tar.gz` (289) and
   `panorama-90yaw.tar.gz` (289) — both dirs now full 17x17 grid.
3. xformers won't build on arm64 → patched `src/ann/predictive_coder.py` to fall back to
   torch `F.scaled_dot_product_attention` (layout [B,M,H,K]->[B,H,M,K]); xformers path
   unchanged where present. Verified on MPS.
4. numpy-2.x break in `classify_neurons` (int() on 1-element array) — coerced in the runner.

Runner: `notebooks/run_model_layers.py` (reuses notebook fns, MPS, lazy model-load,
`EV_MODELS` env). Results: all TRAINED layers entangled (EV_xy≈EV_hd≈0.9+, on diagonal);
`resnet50_untrained_avgpool` is the outlier (0.65/0.72, HD-leaning, ~10 place cells).
Scatters at figs/{model}_{layer}_ev_pitch.pdf; EVs at results/ann/{model}_{layer}/ev_all_neurons_pitch.npz.

Still data-blocked: full 4-rat Pfeiffer EV needs the complete rodent rate maps (only
Rat1/Rat4 local). See [[ev-metric-convention]] and [[ev-calibration-expectations]].
