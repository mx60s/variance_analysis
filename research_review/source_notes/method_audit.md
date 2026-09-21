# Independent method and implementation audit

Audit date: 2026-09-14. Manuscript: `Spatial_maps_in_CNNs_revised_20260914.pdf`, using the page-delimited extraction `tmp/pdfs/manuscript.txt`. No manuscript, original source, model weights, or original results were modified. Notebook code was exported verbatim to `research_review/source_notes/notebook_exports/*.py`; `# CELL` markers preserve the original zero-based notebook cell indices. Line references to those exports below are for navigation, not claims that they were executable end-to-end.

## Overall assessment

**Subsequent repository clarification:** the initial audit concerns the scrubbed `explained_variance` tree. The reviewed PDF is byte-identical to `variance_analysis/Spatial_maps_in_CNNs/manuscript.pdf`. The original `variance_analysis` tree contains the biological notebook, both decoder modules and a predictive-coder checkpoint, so missing-file observations below apply only to the scrubbed copy. Direct comparison confirmed identical `_corr_matrix` and `quantify_variance_corr` definitions, with only terminology changes elsewhere in the estimator file. The checked original model notebook retains the same pitch folds and repeated-static-frame evaluation. The full original biological pipeline has not yet been audited.

The scientific question is worthwhile: when a unit has apparent marginal spatial tuning, how much does its *tuning shape* persist across changes of viewpoint or another behavioral variable? The joint maps can supply compelling descriptive evidence. The present estimator and demonstrations do **not** establish a new method for partitioning response variance or identifying arbitrary stimulus contributions. Several issues are mathematical counterexamples, not merely statistical caveats. The most economical credible revision keeps the joint-tuning comparison but changes the measurement language, repairs the GLM illustration, adds explicit gain/baseline analyses and reliable held-out prediction, and reruns the core comparisons with documented model evaluation and matched sampling. Which biological/model findings survive those changes remains an empirical question; this audit did not regenerate them.

## 1. What is actually estimated?

Source `src/explained_variance.py:117–136` returns the average off-diagonal cross-fold Pearson correlation and the average diagonal correlation. Notebook callers calculate `1 - off_diagonal`, matching manuscript Eq. 2 (p. 20). There is no regression residual, squared prediction error, variance denominator, or variance-component decomposition in this statistic.

Let `h_i(y)` be the true curve over `y` at `x_i`. With complete data and nonconstant curves, define `z_i=(h_i-mean(h_i))/||h_i-mean(h_i)||`. In the noiseless limit the reported statistic is

`D_x = 1 - mean_{i != j}(z_i^T z_j)`.

For `m` slices it also equals

`D_x = m/(m-1) * (1 - ||mean_i z_i||^2)`.

This is a dispersion of centered, unit-normalized slice shapes. That is an interpretable descriptive quantity. It is not a fraction of variance attributable to `x`; calling its two values a “variance partition” or comparing their ratio as fractions is unsupported. On common support, the noiseless range is `[0,m/(m-1)]`, and opposing curves can exceed 1 substantially when there are few slices. General noisy cross-fold or pairwise-missing data do not even retain the common Gram-matrix bound. The appropriate name would be “cross-condition tuning-shape dissimilarity,” accompanied by reliability and the sampled domain.

### Exact counterexamples

These were run through the repository's scoring functions with the Numba decorator removed via AST to avoid optional runtime dependencies. The additive, multiplicative and pure-x cases were also independently confirmed by importing the original module directly using the project's `.venv`, including its original Numba decorator. Script and outputs: `research_review/method_diagnostics.py`, `research_review/source_notes/method_diagnostics.json`. JSON `null` denotes an undefined numeric result.

| True response | Reported noiseless `EV(x), EV(y)` | Implication |
|---|---:|---|
| `h(x,y)=x+y`, positive finite grids | `0, 0`, reliabilities `1,1` | Both variables change firing, yet both “contributions” vanish. |
| `h(x,y)=xy`, positive finite grids | `0, 0`, reliabilities `1,1` | Multiplicative gain interactions are invisible to both axes. |
| `h(x,y)=x` | `NaN, 0` | All curves over `y` are constant, so the supposedly maximally tuned axis has undefined Pearson correlations. |
| Independent noise in the two folds | `0.977, 1.004` in one small check, reliabilities approximately 0 | No reliable signal can produce a near-diagonal, high “EV” cloud. |
| `20+(x-6.5)sin(y)` | `1.091,1.032`, reliabilities `1,1` | The numbers are not bounded explained fractions. |

More generally, `h(x,y)=a(x)g(y)+b(x)` with positive `a(x)` produces `D_x=0`, no matter how strong gain or baseline modulation is. The manuscript mentions scale/offset robustness on p. 20 and gain modulation of place fields on p. 16; that is an intentional property with a different interpretation than complete neuronal invariance. It must be made central rather than treated as a minor technical detail.

For the pure-`x` example, adding arbitrarily small *independent* noise makes `EV(x)` approximately 1 even as noise amplitude tends to zero. The audit obtains `1.0068` at noise amplitude `10^-4`, while reliability for that axis is `-0.020`. Thus recovering “1” for the tuned axis in the current simulations is not a conventional consistent variance estimator: it is a singular zero-denominator case rendered finite by noise. The ground-truth helper in the simulation avoids zero norms by replacing them with 1 and setting constant standardized slices to zero (`binning_bias_fig` export:228–234). That yields dot products of 0 where the actual Pearson estimator returns NaN. Floating-point residuals can further affect near-constant curves. A valid benchmark must specify and test these cases explicitly.

### Scientific implication

The strongest supported reading is “tuning shapes are comparatively stable across head direction in these recorded place cells and vary more across viewpoint in these evaluated ANN units,” conditional on adequate reliability and the validity of preprocessing. The metric cannot justify “the variable explains 95% of firing-rate variance,” “the neuron is insensitive to that variable,” or “these two variables contribute equally.” Equal high values can simply reflect saturation of a shape-dissimilarity score.

## 2. Cross-folding does not by itself remove measurement noise

Manuscript pp. 7 and 20 calls the estimator noise-corrected because folds have independent noise. Independent noise removes an additive cross-product bias in a covariance numerator. Pearson correlation still divides by noise-inflated estimated standard deviations. With usual signal-plus-independent-noise assumptions and many comparison bins,

`C_ij ≈ rho_ij * sqrt(q_i q_j)`

where `rho_ij` is the true correlation and `q_i` is slice reliability. Even for perfectly invariant nonconstant tuning, `EV_raw≈1-q` at homogeneous reliability `q`. Cross-folding therefore produces positive apparent dissimilarity from noise; pp. 31–33 partly acknowledge this, but do not reconcile the stronger main-text claims. The Methods assertion that inflated correlation would lead to *overestimated* `1-correlation` has its sign reversed.

The “floor” `1-mean(diag(C))` is not a universal lower bound or a calibrated test threshold. Finite random estimates can lie on either side, off-diagonal and diagonal entries can involve different valid bins, and heterogeneous reliability introduces bias even with fully occupied maps.

### Heterogeneous reliability counterexample, no sparse bins

Let all 12 true slices have identical shape, with six observation reliabilities `q=1` and six `q=.01`. Then

- Mean diagonal = `.505`.
- Mean off-diagonal = `.28409`.
- Raw dissimilarity = `.71591`, though true dissimilarity is 0.
- The supplementary normalization `1-off/diag` still gives `.43744`.

A 4,000-bin simulation using the actual scorer gives `.43676` for the normalized value. Every bin is present. Thus the positive bias is not solely a low-occupancy artifact. This matters for place fields because off-field slices often have much lower response variation and signal reliability than in-field slices. It also matters when comparing different firing rates, areas, ANN layers, or pitch-sensitive units.

No general uncertainty estimator or hypothesis test is implemented. A single random split is reported on p. 20. Repeating splits assesses partition variability conditional on one recording; it does not supply uncertainty over sessions/animals or independently generated data. Use session/animal resampling or hierarchical analyses for biological population claims, and model-seed/environment resampling for ANN claims.

## 3. The GLM illustration requires replacement

### Proven figure construction error

`notebooks/glm.ipynb`, the `compute_row` code cell (export `glm.py:234–246`), computes real true axis slices, then explicitly replaces each with its constant mean:

`true_fx_flat = np.full_like(x_grid, lam_true_fx.mean())`

`true_gy_flat = np.full_like(y_grid, lam_true_gy.mean())`.

These constants are returned as `true_fx` and `true_gy` and plotted as ground truth. They are neither the true conditional slices nor the true marginals of the simulated response. This makes the claim of flat ground-truth tuning in Fig. 1 (p. 3) invalid for the implemented example. This is a demonstrable plotting/data-definition error; it does not imply any intent.

For the stated model `lambda=A exp(-(cx-sy)^2/(2 sigma^2))`, with independent standard Gaussian `X,Y` (p. 19), the actual marginal is

`E_Y lambda(x,Y) = A sigma/sqrt(sigma^2+s^2) * exp[-c^2 x^2/(2(sigma^2+s^2))]`.

It is non-flat whenever `c != 0`. At 45 degrees, `A=5` and `sigma=1.2`, it is 4.308 at `x=0` and 2.573 at `x=2`. The GLM finding both variables tuned is therefore not, by itself, a false attribution. Both variables truly affect the rate, and both marginals are non-flat under the sampling design.

### The valid model criticism is narrower

The fitted formula is `events ~ x + y + I(x**2) + I(y**2)` (`glm` export:184–189). It omits the cross-term `x*y`. Expanding the true log-rate gives a quadratic with exactly that additional term; the interaction GLM therefore contains the data-generating model exactly. A GLM is linear in fitted coefficients, not required to be additive in raw stimulus variables. Tensor-product bases, categorical joint bins and other interaction bases are legitimate GLMs. The claim on pp. 4, 16–17 that GLMs generally impose multiplicative separability is false. A specific additive log-rate GLM imposes it.

Replace Fig. 1 with (i) a genuine interaction example such as a balanced circular difference tuning or crossover response with flat marginals under a specified reference distribution, and (ii) held-out comparisons of additive vs interaction GLMs/GAMs. State the criticism as “marginal analysis and restricted additive models can conceal context-dependent tuning.” A good interaction model is a complementary estimator, not an adversary.

The pp. 3–4 figure descriptions also reverse the order of the two examples relative to each other; p. 4's sampling paragraph says selectivity was falsely inferred for `x` after describing invariance to `y`. These should be repaired during the larger revision.

## 4. Joint maps help with observed-variable confounding but do not solve identification

The simple marginal confound proof on p. 19 is correct in substance: `E[R|X=x]` changes when `p(Y|X=x)` changes. Uniform marginal sampling of `X` is not needed for this conditional-mean identity. The derivation unexpectedly assumes `h=f*g` after introducing a general `h`; the general identity should precede a clearly labelled separable special case.

A joint map estimates `E[R|X,Y]` on the occupied support. It can distinguish dependence on `X` and `Y` when sufficient combinations are sampled. It cannot identify off-support responses if `Y=X`, precisely the perfect-confounding example in the manuscript. Thus the suggestion on p. 17 that the approach resolves highly correlated location/time/distance variables requires overlapping support or an experimental manipulation. Pairwise deletion does not create missing evidence.

The note `notes_marginals_and_glm.md` says the method “can't be fooled by sampling correlations.” That is too strong. Even without unobserved variables, within-bin variation under a nonuniform conditional distribution changes the target bin mean. Missing pairwise support makes correlations represent different regions of stimulus space. Unobserved speed, posture, reward/task state, egocentric visual content and history can still affect `E[R|X,Y]`; descriptive tuning is not a causal variable contribution.

A counterfactual example makes the support issue precise: on data supported only at `y=x`, the functions `h_1(x,y)=x` and `h_2(x,y)=y` are observationally identical. No estimator using only those observations can determine which variable drives the response.

## 5. Simulation validation does not test the advertised range

**Scope correction:** the author clarified after this audit that the binning-bias notebook is not used in the intended final paper. The implementation observations below are valid for that inspected notebook, but should not be attributed to final-paper simulations without verifying the actual generating code. The local PDF/source still contain related simulation and binning/noise-floor claims; that does not establish notebook-to-figure provenance. If those claims are also removed from the final paper, the corresponding manuscript criticisms are inapplicable. Independent algebraic counterexamples in sections 1–2 are unaffected.

Source: `binning_bias_fig` export:72–107, 111–176, 212–303; Methods p. 21; supplement pp. 31–33.

- “Continuous” simulation draws continuous positions but **snaps to bin centers before evaluating the response** (lines 95–105). It does not test within-bin continuous response variation.
- The “discrete” simulation uses an integrated expected response per bin; its similarity to the center-evaluated version cannot establish that binning introduces no bias for arbitrary continuous tuning or correlated occupancy.
- Response counts are Poisson and rate is `exp(beta0 + amplitude*Gaussian_ridge)` (lines 103–105), rather than the manuscript's Gaussian rate map with additive noise. The distinction may not defeat the descriptive example, but it must match the Methods.
- The loop redraws stimuli **and spike counts** for every value called a “split” (lines 282–293). The manuscript describes 600 partitions of data. The code instead measures variability across synthetic datasets plus partitions.
- The noiseless target is another correlation functional, not known explained variance, and handles constant slices differently from the actual scorer.
- Three rotated-ridge examples do not cover additive responses, multiplicative gain, baseline shifts, null units, heterogeneous reliability, missing support, arbitrary bin widths, smoothing effects, temporal dependence, or partially separable fields.

The current validation supports only limited recovery of this descriptive score in the particular toy distributions. Replace broad “only bias is finite-sampling floor” wording with empirical qualifications until a larger benchmark is completed.

## 6. Preprocessing and comparison protocol

### Proven implementation facts, downstream magnitude not yet quantified

1. **Model location smoothing uses a flattened one-dimensional index.** Model activations have shape `(n_locs,90,n_units)`; the scoring path smooths each `(n_locs,90)` map using `sigma=(1,1)` (`model_units` export:388–404, 420–427; all-layer path 582–592). A `17×17` arena is treated as a single ordered list of 289 locations. This connects the end of one spatial row to the start of the next and misses adjacency on the other spatial axis. It is not isotropic two-dimensional spatial smoothing. Correct by reshaping to `(17,17,90)` before smoothing; flatten only for comparing full spatial maps.

2. **Pitch folds are distinct deterministic conditions.** They are `[-3,+1]` vs `[-1,+3]` degrees (`model_units` export:94–102), with mean pitches -1 and +1. They are not independent identically distributed noise realizations of the same scene. Cross-fold differences can reflect systematic pitch sensitivity; consistent across-fold visual structure can also remain. Treat pitch invariance as another axis or use true repeated realizations under a fixed, specified nuisance distribution. Because a deterministic ANN need not have observation noise, analyzing the exact fixed-pitch response and separately testing pitch robustness is simpler.

3. **Predictive model evaluation repeats a stationary image ten times.** `model_units` export:195–220 repeats each panorama frame via `np.repeat` and reads the final timestep. It does not supply actual recent navigation history. A temporally trained network might use past views to build an invariant representation; stationary repeated-frame tests can suppress that mechanism or create distribution shift. This does not prove the main difference is an artifact, but makes comparison to a claim about temporally learned cognitive maps inconclusive without natural-history tests. Evaluate multiple real trajectories ending at each matched state and quantify both history dependence and viewpoint generalization.

4. **Negative activations are rectified.** `src/explained_variance.py:25–31` clips interpolated arrays at zero for every `masked_zoom` call; `src/ann/classify_units.py:188–190` explicitly rectifies post-prediction features for classification. Rectification changes tuning shape and can create/remove interactions. It is unnecessary for Pearson correlation; report and test signed activations separately.

5. **Interpolation changes the quantity and supplies no new observations.** `zoom_factor=(1,90//17)=(1,5)` upsamples the 90 HD bins to 450 (`model_units` export:404,589). It does not make 289 location bins and 450 HD bins equal, nor is equal axis length needed by the scorer. Duplicating/interpolating slices changes the off-diagonal pair weighting, so binning/upsampling choices alter the estimand. Use original bins and robustness analysis.

6. **Nan-aware smoothing weights bins equally, not by their precision or occupancy.** `src/explained_variance.py:4–22` smooths rate values with a binary validity mask. This differs from smoothing summed spikes and time. It may be a deliberate uniform-bin estimate but should be documented, with sensitivity analysis under uneven occupancy.

7. **Input splitting has small exact bugs.** `src/explained_variance.py:54–55` shuffles a view in place, mutating the caller's visit array. Lines 62–63 report both fold occupancies as `floor(n/2)` even when fold 1 contains one extra observation. It assumes non-NaN observations occupy the initial contiguous prefix; internal NaNs cause valid later observations to be dropped (the diagnostic returns mean 2 instead of 8/3). Arrays built by the present synthetic helper use contiguous prefixes, so the last issue is conditional rather than a demonstrated corruption of those data. `cap=True` takes the first observations before shuffling, so it is not random subsampling from all visits. Unit-dependent missingness also leaves occupancy arrays reflecting the last included unit.

8. **Occupancy conventions are inconsistent.** `quantify_variance_corr` divides supplied occupancy by 2 (line118). The model passes summed fold occupancies; the simulation passes one fold's occupancy (`binning_bias_fig` export:293–299), effectively dividing a half-count again. With the simulation's zero threshold this does not change masking, but any positive threshold would behave differently.

### Model training provenance needs verification

`src/ann/get_activations.py:95–96` loads predictive-coder weights only if the checkpoint exists, silently returning an untrained model otherwise. The cache need not have been generated without weights; that has **not** been demonstrated. But reproducibility requires failing loudly when pretrained weights are requested and recording checkpoint hash, training data/objective, model seed, and exact evaluation inputs. The release omits the actual training runner, so the stated 200-epoch history cannot be verified here.

## 7. Place-cell classification does not implement the stated Methods

Manuscript pp. 22–23 says all applicable Tanni criteria are used, with odd/even stability, threshold 1 Hz, iterative threshold raising, hierarchy selection, minimum 10 bins and `r>.5`. The active “new” code instead does the following (`src/ann/score_utils.py:179–218`): min-max normalizes activations; sums the first vs second half of HD samples (not odd/even); thresholds at the map mean; requires 15 bins, at least half-global-maximum peak and stability `.3`.

Most decisively, `_detect_fields` accepts `threshold_step` but never uses it: it thresholds once and applies filters (`score_utils.py:162–176`). There is no iterative threshold or nested hierarchy logic in this function. Thus claimed replication of the full Tanni procedure is not supported by the released implementation.

Further exact mask bug: extracted candidate fields use zero outside the field (`score_utils.py:156`), but field-stability uses `~np.isnan(field_response_map)` (`score_utils.py:132–134`), which selects the entire map. Consequently the purported field-restricted stability is evaluated over globally above-threshold overlap, not the actual candidate field. Likewise “num pixels” at line223 counts all non-NaN bins, so it reports the whole map size rather than field size. Whether these errors change which units pass depends on the data; they should be repaired before comparing class prevalence.

Notebook “place” labels actually use `exclusive_units_indices` (`model_units` export:384 and564), excluding units also classified as border/direction cells. This is a selected subset of cells with place fields, not all place-field-positive units. Report the selection explicitly and show inclusive results. Testing invariance after selecting HD-stable spatial maps can also build some invariance into selection; use independent classification data and compare the same selection logic across models and recordings.

## 8. Stored result archives and provenance

The diagnostics inspect all six available `results/ann/*/ev_all_units_pitch*.npz` files. They are cached outputs, not newly regenerated scores. The notebook currently writes and reads `ev_all_units_pitch.npz`, whereas four local corrected archives have suffix `_corr.npz`; no matching writer for that suffix was found in the notebook. Corrected caches omit reliability and per-unit IDs, preventing independent noise audit and safe identity matching.

| Archive | Finite pairs | Median stored v1 | Median stored v2 | v1 > v2 |
|---|---:|---:|---:|---:|
| predictive_coder_conv1, corr | 7,630 | .93115 | .86141 | 63.16% |
| predictive_coder_pre_pred, corr | 8,000 | .91618 | .91895 | 47.14% |
| resnet50_layer1, pitch | 7,964 | .04242 | .03554 | 52.75% |
| resnet50_layer1, corr | 7,964 | .96446 | .95758 | 52.75% |
| resnet50_layer3, pitch | 7,987 | .08596 | .06331 | 58.71% |
| resnet50_layer3, corr | 7,987 | .91898 | .90860 | 51.87% |

Both predictive-coder corrected caches have `is_place` entirely false, despite supplement plots labelling 75 place cells. This does not prove the figures are wrong: they may come from another archived run or external output. It proves the current cache set cannot document their selection. Likewise values here must not be substituted into the paper without tracing their origin and axis convention. The corrected ResNet-50 layer1 archive is exactly the complemented, axis-swapped finite subset of the older archive; layer3 is not that same transformation. Exact residual comparisons are saved in the JSON. Establish a single script that recomputes scores, reliabilities, selection labels and paper tables from named inputs with versioned metadata.

## 9. Biological and decoding conclusions: what can and cannot be checked

The repository has no complete biological rate-map runner. `1d_decoding` imports absent modules `src.dataset_2_decode_solo` and `src.dataset_1_decode_solo` (export:70,380). Thus temporal fold construction, occupancy priors, smoothing order, spike counts, correctness of movement direction and training/test leakage cannot be verified here.

Methods p. 25 describes training on **one** of four folds and decoding the remaining three; ordinary four-fold CV usually does the reverse. This may be intended but gives only 25% training data, and the held-out sets overlap across repeats. Confirm against the missing code and avoid treating all decoded windows as independent replicates. Methods p. 26 smooths spikes with a 100-ms kernel before 50-ms decoding. Whether this crosses train/test boundaries depends on the missing implementation; it is a concrete audit question, not a proven leakage finding.

Weak performance of one memoryless independent-Poisson decoder on selected HD cells is a lower bound on information that this decoder can use. It is not an upper bound on information in the population, its temporal code, or its usefulness to a recurrent path integrator. The statements on pp. 15 and17 that HD cells are unlikely to carry a sufficient movement-direction signal require substantial narrowing. Compare an HD-only behavioral predictor of MD with the neural decoder, then ask whether neurons add information about MD conditional on HD, speed and state. Include stronger decoders, population-size curves, real-time alignment, movement-direction uncertainty, and task-relevant path-integration error. Conditional rather than unconditional chance is crucial: MD can be decoded from HD because behavior correlates the two, without dedicated MD coding.

Selecting identified HD cells and then finding strong HD dominance is expected. The interesting claim is the magnitude and form of residual MD dependence *within* HD conditions, with adequate off-diagonal support. Analyze all recorded excitatory cells as a complementary control and report session/animal uncertainty rather than only unit counts.

## 10. Minimal repair and natural expansion

### Minimal credible manuscript

1. Rename the current statistic and restrict its interpretation to stability of tuning shape modulo gain and offset. Define the reference stimulus distribution, support and noise conditions. Present reliability rather than dropping it from caches.
2. Repair/recreate Fig. 1 using true marginals and compare additive and interaction models. Keep the correct pedagogical message about joint vs marginal tuning.
3. Correct spatial smoothing, occupancy metadata, field classification and negative-activation handling; remove unnecessary upsampling. Audit versions/caches and regenerate the decisive comparisons.
4. Test predictive networks with realistic histories, frozen-frame controls and logged checkpoints; show matched-sampling/reliability comparisons between models and recordings.
5. Retain HD–MD results as descriptive shape stability plus decoder performance under the tested protocol, avoiding a claim that the system lacks an adequate path-integration signal.

### More compelling narrative

Ask whether “place-like” responses support **viewpoint-general spatial readout**, then distinguish three mechanisms: stable tuning shape with changing gain; changing baselines; and changes of preferred locations/shape. This turns an estimator weakness into an explicitly chosen scientific distinction, while providing separate measurements for the dimensions that Pearson deliberately removes. Measure decoder generalization across held-out head directions/pitches/histories and compare it with single-unit shape/gain properties. A decoder can generalize despite entangled individual coordinates, so the population result must be measured rather than inferred.

### If genuine variance attribution is desired

Choose a reference product distribution `q(x)q(y)` and decompose the true map into main effects and interaction: `h=mu+f_x+f_y+f_xy`. Main-effect variance fractions then sum with the interaction fraction to 1 for a noiseless map. Total-effect sensitivity for `x` is `E_y Var_x[h(x,y)] / Var[h]`; interactions belong to both total effects, so their sum need not be 1. Under correlated observed sampling, the choice of reference distribution and off-support identification must be explicit.

A useful nonparametric numerator for total-effect sensitivity is an average of cross-fold products of contrasts:

`0.5 * E_{x,x',y}[(h0(x,y)-h0(x',y))*(h1(x,y)-h1(x',y))]`.

With unbiased fold maps, independent fold errors and fixed weights/common support, its expectation is `E_y Var_x[h(x,y)]`. It correctly counts gain and offset changes and works for constant slices. A corresponding cross-fold signal-variance denominator yields a noise-aware ratio; ratio uncertainty and low denominator cases still need explicit treatment, and this observation is not a claim of methodological novelty. Comparison with established variance-component/sensitivity and cross-validated encoding methods is mandatory before presenting it as new.

### Feasible experiment priorities

| Experiment | Question resolved | Relative effort |
|---|---|---|
| Synthetic additive, gain, baseline, crossover, null, heteroskedastic and missing-support suite | Does each named metric estimate what it claims? | Hours to a few days; CPU only. |
| Replot true GLM marginals and fit interaction GLM | Is the Fig. 1 claim valid, and what actually defeats an additive model? | Hours; CPU only. |
| Recompute scores after correct spatial smoothing, no interpolation, signed-activation controls; save reliability | Are model contrasts sensitive to implementation? | A few days if activations exist; inference otherwise. |
| Poissonize/subsample ANN responses under real occupancy and firing rates, with exactly invariant null units | Could brain/model differences arise from sampling and noise? | Several days after biological pipeline recovery. |
| Additive vs multiplicative vs joint smooth encoding models with blocked held-out likelihood | Which tuning changes are necessary and predictive? | Days to 1–2 weeks, dependent on data recovery. |
| Cross-HD and cross-pitch spatial readout with same population and sample sizes | Do the claimed invariances provide useful generalization? | Days with saved activations; moderate compute. |
| Natural-history predictive-coder evaluation against repeated-image controls | Are entangled units a stationary-input artifact? | Moderate to high; needs trajectories/checkpoints, not necessarily retraining. |
| HD-conditioned MD decoder, speed strata, temporal/population scaling, synthetic MD injection | What sensitivity/power does the MD conclusion have? | 1–2 weeks after missing decoding code is recovered. |
| Several environments/seeds and realistic visual histories; optional vision-only invariance training baseline | Is the conclusion about a specific setup or a broader learning objective? | Weeks; new data/inference/training likely. |

These are planning estimates, not measured runtimes. The first two tasks are sufficient to decide the correct estimand and revised claims; the later scientific comparisons require access to the omitted biological analysis and original provenance.
