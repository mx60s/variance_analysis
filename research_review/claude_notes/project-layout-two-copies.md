---
name: project-layout-two-copies
description: "~/explained_variance is an anonymized partial copy; the runnable project with data, weights, and the rodent notebook is ~/variance_analysis"
metadata: 
  node_type: memory
  type: project
  originSessionId: c9fe392c-1579-4978-80f7-7cbf9a20c0ef
  modified: 2026-09-13T21:46:33.132Z
---

`~/explained_variance` (July 2026) is a scrubbed copy of `~/variance_analysis` (names replaced: place->exclusive units, HD->v2, MD->v3, dusz->dataset_1, pfeiffer->dataset_2). It is missing: the rodent EV notebook (`bio_neurons.ipynb`), `src/*_ratemaps.py` and `src/*_decode_solo.py`, all rodent data (`data/rodents`, 17 GB), `data/weights/predictive_coder.ckpt`, `results/rodents`, and two panorama locations. Its `src/explained_variance.py` is identical to the original apart from renames.

**Why:** Maggie thought she was "missing data"; the data was never copied, not lost.

**How to apply:** Run analyses in `~/variance_analysis`. Local runner (no Colab): `~/variance_analysis/notebooks/run_nb.py <notebook>` using the py3.12 venv at `~/variance_analysis/.venv` (rebuilt 2026-09-20 with numba, torch, statsmodels, plotly, cv2, skimage). 2026-09-20: astra's literature/method review was moved to `~/variance_analysis/research_review/` (committed); the 7 `unit_chart.npy` caches were copied over; `~/explained_variance` is slated for deletion once Maggie confirms (only its `.venv` and the anonymized duplicates remain there). Logs go to `~/variance_analysis/logs/`. Previous-session notes (metric convention, calibration expectations, model blockers) live in `~/.claude/projects/-Users-moo-variance-analysis/memory/`. See [[hd-md-behavioural-coupling]].
