> Follow-up: the numerical reruns are now complete. See `REVALIDATION_20261116.md` for substantive findings, changed manuscript claims and execution details. Earlier outstanding-run notes below describe the initial classification audit.

# Code review and reclassification, 16 November 2026

Base commit: `e44fee3`. Work is isolated on `review/paper-classification-20261116`. No existing checkpoint or original result cache has been overwritten. Generated review results live in `results/review_20261116`; publication figures have only the requested decoder-label/S4 updates. The manuscript is a review draft, not a numerical revalidation.

## Confirmed and fixed

- Candidate-field stability used `~isnan(field)` although outside-field values were zero. It therefore included other fields/background. It now uses positive finite field bins.
- Field area counted the entire arena; means and variances included zero padding. Statistics now use actual field bins.
- Constant maps produced undefined normalizations/directional ratios. Explicit constant-map behavior added.
- The new field detector ignored `pixel_min_threshold` and `pixel_max_threshold`. It now respects them; notebook/runner calls were changed from the previously ignored `1` to the effective historical threshold `15` so this fix does not silently alter the intended criteria.
- Partial feature subsampling left the old loop length; rectangular grids were reshaped as squares. Both repaired. Classification outputs can be directed to isolated folders.
- Pie plots discarded <1% classes and renormalized the remaining slices. Small classes are now pooled; denominators include every sampled unit, including inactive units. Percentages use one decimal or `<1%`.
- Split generation mutated the input, lost valid values after internal NaNs, and undercounted the larger odd-sized fold. It now preserves inputs and reports true counts; incompatible per-neuron validity masks fail explicitly instead of corrupting shared occupancy.
- Interpolation clipped negative activations implicitly. Signed values are now preserved by default; clipping is opt-in.
- Example smoothing passed `(0,0,1)` as axis indices, wrapping X and Y rather than HD. Corrected to `(2,)`. HD/MD maps wrap both axes.
- EV scoring smoothed already-flattened XY positions, joining row boundaries and omitting one spatial direction. A tested helper smooths a true X/Y/HD grid, then flattens. Model and biological notebook call sites updated.
- Occupancy cutoffs were computed from total counts but compared against half counts. Biological notebook cutoffs are now in per-fold units.
- XY decoding could assign likelihood to states with no finite training rates. These states are now excluded, matching the angular decoder's behavior.

## Classification method versus bug fixes

The default remains the manuscript's adapted single-threshold method: 15-bin minimum, half-map maximum, peak >= half-global-maximum, first/second HD-half stability >=0.3 over >=9 bins above the activity cutoff. It is not the full Tanni procedure. The upstream iterative field hierarchy has been included separately with its source commit and public-domain license. `field_method='tanni_iterative'` is a sensitivity analysis using that hierarchy with the same ANN adaptation. The older thesis code uses a nine-bin minimum, so neither set of current 15-bin counts should be called a literal reproduction of that older thesis.

All overlapping categories remain separate. The paper's highlighted place group is exclusive of border and directional labels. Inclusive place-field-positive units are the union of the four place-containing categories. No regrouping as location-selective has been made.

The original and corrected classifiers run on exactly the same fresh activation arrays in each comparison. Original baseline code is loaded from the specified Git commit, and old/new charts are both retained. Feature subsampling is deterministic on CPU, and original flattened feature IDs are saved. Earlier caches without feature IDs cannot be assumed to contain the same units even when the array shape matches.

## CNN pie-chart recovery

The plotting helper and CNN layer specifications are still present. The current manuscript and earliest tracked manuscript version inspected do not include the CNN pie charts; the available Git history does not establish when or why they disappeared from a prior paper version. This review reconstructs charts from fresh activations for ResNet50 layer1, layer3, layer4 and avgpool, untrained ResNet50 avgpool, and VGG16 features.9. These are regenerated review artifacts, not recovered historical images. The untrained control uses a new recorded seed. Original/fixed/iterative comparisons within each run share exactly the same activations.

## Predictive-coder audit

The historical model's LayerNorm includes time. A checkpoint-loaded counterexample changes future frames while holding the prefix fixed and changes earlier latents (max difference 1.222 in the fixed-seed diagnostic). The upstream Gornet code contains the same temporal normalization. Training BatchNorm also mixes B×L frames. This is not evidence of a transcription mistake unique to this repository, but it prevents assuming strict causal next-frame prediction.

Default legacy behavior was compared to `e44fee3` with the existing checkpoint: maximum latent difference exactly 0. The opt-in `causal_normalization=True` architecture uses per-frame LayerNorm and GroupNorm. It passes future-perturbation tests in train and eval modes. It is a new training architecture, not a silent repair of the old checkpoint. A five-step local training smoke test completed on public upstream trajectories; this is not retraining or evidence of model quality.

Existing evaluation repeats each static frame ten times. Fresh classification intentionally retains that protocol. Natural-history testing remains necessary before attributing behavior to the trained temporal representation. Pitch-based folds are changes in visual input, not independent measurement-noise replicates.

## Data checks

The current checkpoint and both evaluation panorama sets are local. Fresh predictive activation extraction validates image counts and zero pitch, records all input hashes and logs observed-yaw differences. In the fixed-pitch set, 44 frames across 41/289 locations differ from their nominal headings by >0.01 degrees; the largest difference is 5.73 degrees. Historical nominal-bin assignment is retained for the controlled classifier comparison. This is a small fraction of frames but a concrete collection-quality problem to resolve before a clean-data replication.

## Outstanding scientific and regeneration work

- Recompute EV and decoder numerical results in isolated outputs after helper changes. Existing paper values, category highlights and composite panels have not been reconciled to new classifications. Do not interpret updated wording as certification of the old numbers.
- Figure 1's GLM notebook explicitly replaces true conditional slices with flat mean lines. The current paper describes flat marginals. This is a confirmed unresolved illustration error, not a wording-only issue. Replace it with correctly computed marginal/conditional curves and a clearly specified additive-versus-interaction comparison; the current joint Gaussian ridge does not have flat Gaussian-sampling marginals.
- Pearson shape dissimilarity ignores positive gain and additive baseline changes; constant slices are undefined. Equal high scores need not mean equal variance contributions. Broader main-text conclusions still require a methodological revision beyond the requested unbiasedness wording.
- “Noise floor” is a reference, not a universal lower bound or unbiased correction. Heterogeneous reliability and differing support remain relevant.
- Biological joint-bin selection is based on membership of tracking samples in wider spike-count time bins. A time bin may contribute to more than one state if the animal traverses states inside it. Quantify this and compare with aligned state assignment before claiming independent state samples.
- Dusz movement-direction maps use a rolled lag and the CLI/default lag differs; decoder MD uses no identical roll. Resolve the intended alignment and avoid wraparound before regenerating MD conclusions. No arbitrary lag choice was imposed in this review.
- Position units are inferred heuristically in the Dusz loader; verify per-session scale against metadata before broad pooling.
- Existing untrained-CNN caches lack reproducible initialization metadata. New control runs explicitly use seed 413 and are a new controlled initialization, not the original random realization.
- Exact original training-trajectory identity, world/checkpoint linkage and training history remain unverified. See `remote/README.md` for recovered code, worlds, Drive locations, host options and costs.

## Validation and limits

Fourteen regression tests cover field masks, area/constant behavior, category partitioning, pie denominators, rectangular input shape, input-preserving split counts/missingness, signed interpolation, undefined correlations, decoder support, CV coverage, true spatial adjacency and future-frame independence of the new architecture. Core Python/scripts pass compile and selected Ruff fatal/undefined-name checks. Notebook code parses after removing IPython commands (the multiline pip setup cell is intentionally IPython-only). The actual fresh model forward passes and classifiers were exercised on panorama data, and the training loader/optimizer/validation path was smoke-tested locally.

This is not an end-to-end rerun of every biological experiment, a proof of the metric's statistical properties, or a remote deployment test. Paper edits were compiled with Tectonic and relevant pages visually inspected. Credential checks were read-only; no paid remote job has run.
