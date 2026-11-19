# Invariance and selectivity in neural responses

Code and manuscript for **Invariance and selectivity to spatial variables in deep neural networks and the brain**. The primary analysis measures conditional response variance in CA1 place cells, postsubicular head-direction cells, and vision-model units.

## Paper and reproducibility

- Current LaTeX and figures: [`Spatial_maps_in_CNNs/`](Spatial_maps_in_CNNs/).
- [Public release: paper, Overleaf ZIP, and complete reproduction bundles](https://github.com/mx60s/variance_analysis/releases/tag/manuscript-2026-11-18-luo).
- [Correlation-based alternate](alternatives/correlation/): earlier paper and an unchanged archived reproduction package.

The `manuscript-2026-11-18-luo` release contains the current source and figure-reproduction inputs. The earlier `manuscript-2026-11-18` release remains an immutable pre-Luo snapshot.

For the current manuscript figures, download `variance_reproduction_luo.zip` from the release. It contains the numerical inputs, source snapshots, figure/table manifests, and checksums. It does not contain raw recordings, large activation caches, or model checkpoints.

```bash
unzip variance_reproduction_luo.zip
cd figure_reproduction
python -m pip install -r requirements-figures.txt numba
python reproduce.py --verify-only
python reproduce.py --compile  # Tectonic must be on PATH
```

The source snapshot inside that archive is the authoritative complete figure-reproduction environment. Repository scripts also include raw-data analysis and training tools; some require external datasets, activation caches, or remote storage. Their paths and dependencies are documented in the scripts and archive README.

For Overleaf, upload `overleaf_variance.zip` from the release as a new project and select `manuscript.tex` as the main document. Use pdfLaTeX with BibTeX. The supplement is included in the same source.

## Primary metric

For a joint response map `f(x, y)`, the contribution of x is `E_y Var_x[f(x,y)]`: response variation across x while y is held fixed. We estimate it by multiplying centered maps across folds and averaging over common supported bins. Both axes are normalized by the same cross-fold total signal variance.

The implementation is [`scripts/finalization/variance_estimator.py`](scripts/finalization/variance_estimator.py):

```python
from scripts.finalization.variance_estimator import components
result = components(fold0, fold1)  # matching 2-D maps; missing bins may be NaN
location_ev, heading_ev = result["normalized"]
```

Interactions contribute to both axes; the scores need not sum to one. The unnormalized estimator is unbiased for its target map under fixed common support/weights and unbiased fold maps with independent zero-mean errors. Normalized ratios are not guaranteed unbiased. Binning, smoothing, and missing coverage define the target; correction for noise does not eliminate within-bin sampling bias. Biological unnormalized values have units of Hz squared; normalized values are dimensionless.

The correlation alternate remains available through `src.explained_variance.quantify_variance_corr`; existing correlation workflows are retained and should not be mistaken for the primary variance analysis.

## Current analysis choices

- CA1: published place-cell selection, 545 units; primary joint maps use approximately 10-cm spatial bins.
- HD cells: dataset authors' labels, 1,489 units across 30 square-arena sessions; primary joint maps use 10 x 10 spatial bins and 4-degree headings.
- MD: centered 200-ms local-linear velocity estimate, speed at least 5 cm/s, matched HD/MD decoder time bins. Neighboring windows, thresholds, and resolutions are reported.
- ANN classification: Luo's single spatial-mean threshold, eight-connected 10–144-bin fields (`luo`). Iterative Tanni field detection with and without heading-group stability is supplementary. Main pies combine place and border classifications; exclusive place subsets remain separate.
- Models: pretrained ResNet-50 and VGG-16, an untrained ResNet-50 control, and a retrained predictive model with per-frame normalization preventing future-frame leakage. All are evaluated on the shared Minecraft panorama bank. Pitch folds compare responses across image perturbations rather than repeated stochastic measurements.
- Decoders: held-out contiguous time folds, supported-state masking, and an occupancy-based baseline.

## Installation and checks

Python 3.12 is used for the analysis environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[ann,curated]'  # model and curated-recording dependencies
pip install pytest
python -m pytest tests -q
python scripts/finalization/test_estimator.py
```

`remote/` contains optional Modal workflows; use a configured account only when intentionally starting remote work. `PredictiveCoder(..., causal_normalization=False)` preserves historical checkpoint behavior. The causal-normalization architecture requires a matching retrained checkpoint.

## Data and source layout

Pfeiffer and Foster recordings were supplied by the authors and are not redistributed. The public Duszkiewicz recordings use DANDI 000939, version 0.261016.2310; `scripts/fetch_curated_dusz.py` documents retrieval. The release bundles contain derived numerical inputs sufficient to reproduce the included figures, not the raw recordings. Large panorama banks, checkpoints, and activation caches are not included.

- `src/`: map processing, estimators, decoders, models, and classification.
- `scripts/finalization/`: primary variance estimator, figures, simulations, and supplementary tables.
- `scripts/ca1_selection/`, `scripts/md_revision/`, `scripts/grid_primary/`: selection, movement, coverage, and recovery analyses.
- `scripts/iterative_classification/`: field-classification runner.
- `tests/`: numerical and pipeline correctness tests.
- `notebooks/`, older `figs/`, and `research_review/`: historical/exploratory workflows and audit records; the current manuscript uses `Spatial_maps_in_CNNs/final_figures/` and the release reproduction bundle.

Tanni field-detection source attribution and license are preserved in `src/ann/TANNI_LICENSE.txt` and its source module. Other external datasets and model implementations retain their original terms.
