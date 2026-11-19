# Numerical revalidation, 16 November 2026

This continues the classification/code audit on `review/paper-classification-20261116`. Historical caches and publication figures remain available. New numerical outputs live under `results/review_20261116/revalidation`; preliminary outputs from before the real-map numerical validation are quarantined in `pre_numerical_validation` and must not be used for conclusions.

## Comparisons and meaning

1. `cached_biology`: recompute EV on the original biological fold maps with corrected spatial smoothing and occupancy units. This isolates downstream numerical changes, but retains the historical time-bin assignment issue.
2. `aligned_biology`: regenerate fold maps from raw local data. Each spike-count bin receives the tracking state nearest its center. No extrapolation is allowed across recording boundaries or gaps larger than 1.5 median tracking intervals. Duplicate tracking timestamps are merged with linear XY means and circular heading means; clock reversals fail explicitly. The same aligned targets define cross-validated decoding and occupancy baselines.
3. `aligned_biology_fine`: use 50-ms Pfeiffer and 10-ms Duszkiewicz bins, keeping the minimum occupancy duration fixed (rather than keeping the minimum count fixed). No decoding rerun is needed for this EV sensitivity analysis.
4. `native_xy_coarse_hdmd`: remove interpolation along HD for location×HD EV, and use 36×36 HD×MD bins. The HD×MD smoothing width remains four degrees (0.4 of a ten-degree bin). These are two separate comparisons on different types of map, not an assertion that coarsening and removing interpolation are the same operation.
5. `curated_aligned` and `curated_native_coarse`: repeat the two principal grid choices on the authors' curated NWB data, keeping exactly the same 30 sessions. Kinematic preprocessing is explicit and is therefore a data-release-plus-processing comparison, not a pure change of units.
6. `models`: regenerate pitch folds for the four predictive-coder layers and six CNN configurations. Match the original flattened feature IDs recorded during the fresh clean-panorama classification run. The original predictive checkpoint and ten-repeat static-image evaluation remain unchanged. Static images can be encoded once in eval mode before the exact ten-step temporal computation; this optimization is checked against the original wrappers. Pitch folds are perturbation folds, not biological measurement-noise replicates.

These are exploratory audit sensitivities, not preregistered analyses. All HD×MD runs compare nonwrapping lags of zero, one and three tracking samples. None was chosen because it produced the desired result. Raw scores and reliability are saved, as are normalized scores where reliability is positive. A score that cannot be estimated is missing, not zero. Report paired finite counts separately for raw and normalized summaries. Keep all overlapping ANN classes separate; the iterative field detector is a classification sensitivity analysis.

## Critical implementation checks

The fast correlation implementation uses sufficient statistics for pairwise missing support. A real-map check exposed cancellation when pairwise variance is tiny relative to the slice scale. Numerically ambiguous pairs now use the centered, two-pass reference calculation on the original values. Constant support is explicitly undefined. Synthetic regressions include tiny variance and zero tails; the validated summaries on inspected real maps agree with the reference to better than 1e-12. This numerical check is distinct from proving any statistical property of EV.

The Poisson decoder can compute the same argmax without saving enormous posterior arrays. A regression checks equality against the reference implementation, including unsupported states and partial missing rates. Folds remain contiguous; temporal spike smoothing is restricted to the held-out fold. Neural and occupancy-only errors use the same valid test bins. The review summaries retain low errors rather than discarding values below 0.5.

The released Pfeiffer tracking used here has approximately 33.37-ms intervals. The manuscript's acquisition-rate statement should be distinguished from the sampling rate of these released arrays. The analyzed Pfeiffer population is all 551 supplied excitatory units, not a newly classified place-cell subset.

## Curated data provenance

The authors now recommend the curated NWB release for new analyses:

- https://research.manchester.ac.uk/en/publications/local-origin-of-excitatory-inhibitory-tuning-equivalence-in-a-cor/
- https://dandiarchive.org/dandiset/000939/0.261016.2310

Only behavior and selected excitatory HD-cell spike times were retrieved using HTTP range requests. Raw electrophysiology was not downloaded. Subset files record the DANDI version, asset IDs, source digest, square epoch, cell IDs and subset checksum. An extra available session was downloaded but is excluded from the matched 30-session comparison.

In session A1813-190408, matched curated XY coordinates are approximately 1.6 times the local loader's coordinates, plus translation. The curated file explicitly specifies centimeters. Its HD is clockwise from positive Y, whereas the legacy convention is approximately 90 degrees different; the new headings also include nonconstant corrections. Do not fix all legacy sessions by guessing a global rotation or scale. Use the documented curated coordinates and headings. The MD conversion in the curated runner matches that documented heading reference.

Curated velocity is the norm of a centered XY derivative, Gaussian-smoothed with sigma three tracking samples; gaps are excluded. It is not the old supplied Velocity.mat trace. The legacy-data decoding distances are therefore labeled as legacy coordinate units until calibration is verified; curated XY errors can be labeled centimeters. Comparisons of distance across releases must account for scale and kinematic processing.

## GLM repair

`notebooks/glm.ipynb` no longer replaces conditional ground truth with flat lines; a degree/radian mix-up is also corrected. `scripts/revalidate_glm.py` supplies a separate corrected illustration and held-out comparison. Marginals integrate over the actual independent standard-normal stimulus distribution and match an analytic integral. Conditional slices are labeled separately. An additive log-rate model can match the marginals but miss the interaction; adding x:y recovers it. This is a model-specification example, not evidence that GLMs inherently fail to capture joint tuning.

The existing manuscript GLM figure and its interpretation still require reconciliation with this corrected figure. The publication PDF is not silently updated with a mixture of old examples and new population scores.

## Scope boundary

These analyses do not make Pearson dissimilarity a variance decomposition, establish unbiasedness, establish independent neuron-level replicates, or validate the historical checkpoint's causal training interpretation. No new predictive-model training or natural-history evaluation was performed. A bounded Modal L4 inference run was launched after a pre-launch estimate of under $1 compute with a $5 setup allowance; it evaluates existing weights, not training. Final runtime and completion are recorded separately. Primary grid/time-bin policy and scientific wording should be selected explicitly before rewriting the manuscript's numerical claims and composite figures.

## Completed biological findings and manuscript consequences

All five rebuilt biological configurations are complete: 34 + 34 + 34 + 30 + 30 session/configuration runs, plus 64 rescored historical maps. Corrected decoding covers four Pfeiffer and 30 legacy Duszkiewicz sessions, plus the same 30 Duszkiewicz sessions from the curated release.

The predominant tuning axis remains location in CA1 and HD in the selected HD population. However, CA1 location dominance is 85.2% in the aligned interpolated analysis, 83.3% with finer time bins, and 74.1% on the native grid, with paired-finite coverage 546/551, 539/551 and 529/551 respectively. Curated HD dominance over location is 84.1% on the interpolated grid and 82.5% on the native grid (1489/1489). This supports a graded population contrast, not uniform invariance.

**The manuscript statement that location cannot be decoded from HD cells must be removed.** Curated pooled median XY error is 33.35 cm vs 43.78 cm for the occupancy baseline; 28/30 sessions improve. HD median error is 8.67° vs 81.35°; MD is 61.37° vs 89.78° (30/30 sessions improve for each). These medians must not be silently substituted into sentences describing means. Population decoding need not measure direct conditional selectivity; behavioral correlation may carry information.

The GLM caption's claim that true marginal curves are flat is false for its simulated model. Replace the figure and its interpretation together. State that an additive GLM misses the interaction, while an interaction GLM recovers it.

The path-integration conclusion also requires qualification: coarse MD decoding and weak conditional shape modulation do not establish an impossibility result for downstream integration. Retain the measured contrast, and distinguish it from that mechanistic inference.

## Execution notes

The remote inference request permits L4 or A10 allocation after L4-only requests remained pending without containers. At published 2026-11-16 rates, four CPU cores and 16 GiB RAM plus an A10 total about $1.42/hour (about $0.71 for the 30-minute execution bound); L4 totals about $1.12/hour. Build/load/idle overhead and final billing are separate. This remains within the previously stated under-$1 execution estimate plus $5 setup allowance. No training runs were started.

For the local memory-constrained fallback, scoring reads 128-neuron contiguous blocks rather than repeatedly faulting large strided mmap arrays. A 129-neuron comparison spanning the block boundary produced exactly equal EV, reliability and normalized arrays against the prior scorer. This changes I/O, not the calculation. Legacy Duszkiewicz decoding JSON unit labels were corrected to “legacy coordinate units” after execution; numerical arrays and recorded execution-source hashes were preserved.

The fully rescored ResNet layer1 has identical finite-score masks on Mac and CUDA, maximum raw-EV difference 1.05e-5, and median raw-EV difference 5.32e-8. Normalized-score maximum difference is about 0.029 at tiny reliability, despite median difference 1.07e-7. Keep reliability-normalized results secondary and disclose this instability.

Curated behavioral HD–MD coupling was checked using exactly the EV runner's aligned 50-ms bins, velocity preprocessing and speed ≥1 cm/s threshold. Of 1,039,444 eligible bins, 30.649% have |HD−MD| ≤30°, compared with a 16.698% session-wise empirical-marginal independence reference (calculated by exact circular sorted-range counts). The old manuscript 25% value must change; substantial off-diagonal behavior remains. MD measurement quality and higher-speed neural analyses remain distinct scientific checks, not established by this behavioral calculation.

For the rewrite, distinguish shape selectivity from information available to a population decoder. Pearson-based EV is insensitive to positive gain changes, whereas decoding may exploit gain or behavioral correlations. The audit does not identify which explains the recovered location information. Neither low EV nor relative decoding weakness establishes complete response invariance.

Maintained scripts were reformatted and the model runner gained explicit native-grid, cross-device and resume-identity handling after the initial runs. The code bundle preserves the frozen source used for each remote execution. Biological numerical operations are unchanged by the post-run formatting and unit-label edits; execution hashes are retained.

## Final completion

All 10 model/layer configurations are complete on both grids, with feature IDs, corrected classifications, panorama hashes and the historical predictive checkpoint verified on import. Pre-prediction interpolated medians are location/HD 0.91619/0.91877, and post-prediction 0.91510/0.91203 (8000 units each). Both corrected and iterative place-only subsets retain the qualitative joint-tuning pattern. The broad contrast survives; categorical biological invariance and decoding claims need revision.

All 676 biological score archives and 20 model score archives pass EV/reliability range and shape checks. The test suite passes 24 tests, including a small end-to-end model extraction/scoring run and resume-identity rejection. Remote inference completed on A10 GPUs in two bounded jobs; all apps are stopped. Estimated requested resource time totals approximately $0.83, excluding build/load/idle overhead, storage/egress and credit offsets. This is not an invoice. No new training or simulator collection was performed.
