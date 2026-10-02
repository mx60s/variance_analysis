# 03 — Biological resemblance and usefulness: claim tree

## Rules

- A claim holds if and only if every one of its children holds.
- Numbered top-level claims are independent unless one lists another as a child. A child written "= Claim X" means the whole of Claim X.
- Leaf kinds:
  - **DEF** — a definition.
  - **MATH** — follows from the definitions; the derivation is given.
  - **CITE** — one or more sentences copied from the cited paper. Inline math is retyped in plain text; reference superscripts are removed; nothing else is changed.
  - **EXP** — an experiment. The leaf says that the strongest attack on its parent does not produce the refuting result.
  - **AXIOM** — assumed and not tested. All axioms are listed again in the last section.
- An EXP leaf holds if and only if (a) the attack was run and did not produce the "Refuted if" result, and (b) the same attack produces that result when the effect is planted ("Sensitive if"). A leaf that fails (b) is untested, not survived.
- Status values: `untested`, `survived`, `refuted`, `observed, no margin set`.
- Every δ, ε and N is a free parameter. Each is fixed before the confirmatory run and recorded in the Parameters section. No parameter has been fixed yet.
- No experiment in this document has been run unless its status says otherwise.

## Definitions

- **DEF p, h.** p is position (x, y). h is heading (head direction in the recordings).
- **DEF field.** f_h(p) is a unit's mean response at position p and heading h.
- **DEF S, D.** S is the mean, over all pairs of different headings h ≠ h′, of the Pearson correlation over p between f_h and f_h′. D = 1 − S.
- **DEF D̂, repeat correlation.** D̂ is an estimate of D in which each pair correlation is computed between two disjoint halves of the data. The repeat correlation is the mean correlation between the two halves of the same heading. D̂_v0 is the current code: `1 - quantify_variance_corr(fold0, fold1, occ, axis=1)[0]`; its second return value is the repeat correlation.
- **DEF population-level D̂.** The median of D̂ over all scored units of a population.
- **DEF recordings.** CA1 excitatory cells from four rats in a 2 m × 2 m open arena (`data/rodents/pfeiffer/Rat1…Rat4`; 204, 174, 79 and 94 cells). Maps are 20 × 20 position bins × 60 heading bins, 250-ms samples, speed above 5 cm/s.
  - **CITE** Pfeiffer & Foster (2013), *Nature* 497:74–79: "We recorded from hippocampal neurons while rats performed a spatial memory task, using the statistical power of an open field design in which the goal was one of 36 clearly separated locations within a 2m x 2m arena (Fig 1a)." Methods: "After a rat achieved criterion performance on the task, it was surgically implanted with a microdrive array (25–30 g) containing 40 independently adjustable, gold-plated tetrodes aimed at area CA1 of dorsal hippocampus (20 tetrodes in each hemisphere; 4.00 mm posterior and 2.85 mm lateral to bregma)."
- **DEF model population.** One independently trained model, one fixed layer, a fixed number of units. The replication unit is the trained model.
- **DEF T, W.** Fit a ridge linear decoder from the population to p on one heading sector; regularization is chosen on training-sector data. W is mean position error on held-out blocks of the training sector. T is mean position error on the other sectors, at matched positions. Both are divided by the arena diagonal.
- **DEF P.** The fraction of units that pass a fixed place-cell criterion on heading-averaged maps.
- **DEF native error.** Mean squared error of a model's own decoder output against the true next frame. No probe is involved.

## Claim 0 — CA1 place fields keep their shape across heading, and in trained models that same property predicts whether position can be read out across headings

- 0.1 = Claim 1
- 0.2 = Claim 2

## Claim M — D̂ measures change of spatial field shape across heading in these recordings and models

- **M.1 MATH** If f_h(p) = a_h·g(p) + b_h with a_h > 0 and g not constant, then D = 0.
  - Derivation: Pearson correlation is unchanged by positive scaling and by adding a constant, so every pair correlation is 1.
- **M.2 EXP** The code returns D̂ = 0 for noiseless gain-only and offset-only changes.
  - Attack: score h(x,y) = x + y and h(x,y) = x·y with the repository functions.
  - Refuted if: D̂ ≠ 0 on either axis.
  - Sensitive if: a crossover interaction gives D̂ > 0.
  - Status: survived. Both cases return 0.0 on both axes; 20 + (x − 6.5)·sin(y) returns 1.09 and 1.03 (`source_notes/method_diagnostics.json`).
- **M.3 EXP** The pipeline call compares spatial profiles across headings.
  - Attack: read the call site; check the sign on recordings with a known answer.
  - Refuted if: CA1 place cells score higher on the heading call than on the position call.
  - Sensitive if: head-direction cells show the opposite order.
  - Status: survived. CA1 (Rat1, Rat4): median D̂ across position 0.96, across heading 0.50, position > heading in 100% of cells (notebook pipeline, 2026-07-06). Postsubiculum head-direction cells show the opposite order: the manuscript reports medians of 0.68 across position and 0.92 across heading (1,489 cells, 30 sessions).
- **M.4 EXP** A unit whose spatial profile is constant is reported as not estimable.
  - Attack: score h(x,y) = x with and without small independent noise.
  - Refuted if: the code returns a finite D̂ for the constant-profile axis.
  - Sensitive if: not applicable.
  - Status: refuted for v0. Without noise the code returns NaN. With noise of amplitude 10⁻⁴ it returns D̂ = 1.007 with repeat correlation −0.02 (`method_diagnostics.json`).
- **M.5 EXP** A field shift of δ_shift bins raises D by at least δ_D.
  - Attack: compute noiseless D for synthetic fields shifted by δ_shift across headings on the 20 × 20 grid. Repeat for a width change and an added second field.
  - Refuted if: D < δ_D for any of the three.
  - Sensitive if: not applicable; noiseless computation.
  - Status: untested.
- **M.6 EXP** Identical field shape in every heading does not give D̂ above δ_D.
  - Attack: simulate Poisson spikes from one shape in all headings, on each rat's real trajectory, at each cell's observed rate; score with the frozen estimator.
  - Refuted if: D̂ > δ_D.
  - Sensitive if: a planted shift of δ_shift on the same trajectories gives D̂ larger by δ_D.
  - Status: refuted for v0.
    - Twelve identical slices, six with reliability 1 and six with reliability 0.01, give D̂ = 0.716; the ratio correction 1 − off/diag gives 0.437 (`method_diagnostics.json`).
    - In CA1 (Rat1, Rat4, unsmoothed, 60 heading bins) median D̂ = 0.61 and median repeat correlation = 0.50, so 0.50 of the 0.61 is 1 − repeat correlation (analysis of 2026-07-06; rerun before use).
    - The trajectory simulation has not been run.
- **M.7 MATH** For identical shape in every heading and equal reliability q in every heading, E[D̂_v0] ≈ 1 − q.
  - Derivation: with signal plus independent noise, the cross-half correlation of slices i and j is ρ_ij·√(q_i·q_j). With ρ_ij = 1 and q_i = q this is q. (`source_notes/method_audit.md` §2.)
  - **M.7.1 CITE** Pospisil & Bair (2021), *J Neurosci* 41(26):5638–5651, Results: "The above strategy means that rn is now zero, thus leaving r^s (or r^split) with a bias toward zero that increases with noise as follows:" (r^s is r-hat-sub-s in the source.)
  - **M.7.2 CITE** Pospisil & Bair (2021), bioRxiv 2021.03.29.437413, Abstract: "Yet this metric is biased by trial-to-trial variability: as trial-to-trial variability increases, measured correlation decreases. Major lines of research are confounded by this bias, including the study of invariance of neural tuning across conditions and the similarity of tuning across neurons."
- **M.8 EXP** The two halves share no temporally adjacent samples.
  - Attack: read the fold construction.
  - Refuted if: samples from one pass through a bin can fall in both halves.
  - Sensitive if: not applicable.
  - Status: refuted for v0. `pfeiffer_ratemaps.py:163–164` shuffles the 250-ms samples inside each (x, y, h) bin and cuts the list in half. For models, the halves are pitch groups [−3°, +1°] and [−1°, +3°].
- **M.9 EXP** The score does not depend on heading bin count or smoothing width inside the declared range.
  - Attack: recompute at 60, 30, 12 and 4 heading bins, with and without smoothing.
  - Refuted if: population-level D̂ changes by more than δ_D across settings.
  - Sensitive if: not applicable.
  - Status: refuted for v0. CA1 medians at 60/30/12/4 bins: unsmoothed 0.61/0.63/0.60/0.51; notebook pipeline (σ = (1, 2) plus 5× interpolation) 0.50/0.40/0.20/0.07. (D̂ − (1 − repeat correlation)) / repeat correlation: unsmoothed 0.16/0.42/0.45/0.39; notebook pipeline 0.39/0.28/0.09/−0.01 (2026-07-06).
- **M.10 EXP** Smoothing is applied on the two-dimensional position grid, on original bins, without clipping.
  - Attack: read the scoring path for recordings and for models.
  - Refuted if: smoothing runs along a flattened position index, or the map is interpolated or clipped before scoring.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline (`run_model_layers.py:76–80, 158`: (289, 90) map, sigma (1, 1), 5× heading interpolation, clip at zero in `masked_zoom`). Not checked for the recordings pipeline.
- **M.11 EXP** Every heading pair is scored on the same set of positions, and each cell has at least N_occ such positions.
  - Attack: compute the common valid-bin mask per cell.
  - Refuted if: masks differ between heading pairs, or the common set is smaller than N_occ.
  - Sensitive if: a planted missing bin is detected.
  - Status: untested. v0 uses a pair-specific mask inside `_corr_matrix`.
- **M.12 EXP** Nominal 95% intervals for population-level D̂ contain the true value in at least 90% of simulations, with animals (or trained models) as the resampling unit.
  - Attack: simulate as in M.6 with known D; resample animals.
  - Refuted if: coverage < 90%.
  - Sensitive if: resampling cells in place of animals gives intervals that are flagged as too narrow.
  - Status: untested. No interval procedure is implemented.

## Claim 1 — CA1 place fields keep their shape across heading

- 1.1 = Claim M
- **1.2 EXP** Population-level D̂ across heading is below δ_D in each of the four rats.
  - Attack: score every eligible cell with an estimator that survives Claim M; report per rat.
  - Refuted if: the lower bound of the interval is above δ_D in any rat.
  - Sensitive if: M.6's planted shift is detected on that rat's trajectory.
  - Status: observed, no margin set. v0, Rat1 and Rat4 only: 0.61 unsmoothed at 60 bins; after subtracting 1 − repeat correlation and dividing by repeat correlation, 0.16 at 60 bins and 0.39–0.45 at 4–30 bins (M.9). Rat2 and Rat3 not analysed. The manuscript reports a median of 0.49 over 551 cells from four rats with the notebook pipeline.
- **1.3 EXP** A Poisson model with one spatial shape and a gain and offset per heading predicts held-out spikes as well as a model with a separate spatial shape per heading.
  - Attack: fit both with exposure offsets on blocked training time; compare log-likelihood on held-out blocks; complexity chosen on training data.
  - Refuted if: the flexible model is better by more than δ_ll per spike.
  - Sensitive if: on simulated cells with fields shifted by δ_shift on the real trajectories, the flexible model is better by more than δ_ll.
  - Status: untested.
- **1.4 EXP** The displacement of the field centre across heading sectors is below δ_shift.
  - Attack: estimate the field centre per heading sector on the common positions; take the largest pairwise displacement per cell; report the population median per rat.
  - Refuted if: the lower bound of the interval is above δ_shift in any rat.
  - Sensitive if: as 1.3.
  - Status: untested.
- **1.5 EXP** 1.2–1.4 hold for all eligible excitatory cells and for place cells selected on data not used in 1.2–1.4.
  - Attack: repeat on both sets.
  - Refuted if: any of 1.2–1.4 is refuted in either set.
  - Sensitive if: as 1.3.
  - Status: untested. The v0 place-cell selection in `score_utils.py` uses a stability criterion across heading halves, so selection is not independent of the quantity tested (`source_notes/method_audit.md` §7).

## Claim 2 — Across trained models, lower D̂ predicts lower T, beyond P, W, repeat correlation and population size

- 2.1 = Claim M
- **2.2 EXP** Population-level D̂ differs between the models in the set by at least δ_range.
  - Attack: score every model in the set.
  - Refuted if: the range is below δ_range.
  - Sensitive if: not applicable.
  - Status: observed, no margin set. With v0, every trained layer scored has median D̂ above 0.9 on both axes; untrained ResNet-50 avgpool has 0.65 and 0.72 (`results/ann/*/ev_all_neurons_pitch.npz`, 2026-07-07). The trained models span less than 0.1.
- **2.3 EXP** Position can be decoded within a heading sector in every model.
  - Attack: compute W.
  - Refuted if: W is above ε_within in any model.
  - Sensitive if: a decoder on shuffled position labels is above ε_within.
  - Status: untested.
- **2.4 EXP** D̂ and T are computed on disjoint trajectories.
  - Attack: compare trajectory identifiers.
  - Refuted if: any overlap.
  - Sensitive if: a planted overlap is found.
  - Status: untested.
- **2.5 EXP** Adding D̂ to a regression of T on P, W, repeat correlation and unit count lowers the error of predicting T for held-out trained models by at least δ_pred.
  - Attack: fit on N_models − N_held models; predict T for the rest; repeat over splits.
  - Refuted if: the upper bound of the interval of the error reduction is below δ_pred.
  - Sensitive if: on synthetic populations where T is built to depend on D, the same procedure with N_models populations finds a reduction above δ_pred.
  - Status: untested. Layers and unit subsets of one trained model count as one model.
- **2.6 EXP** The coefficient of D̂ in 2.5 is positive.
  - Attack: read the fitted sign and its interval.
  - Refuted if: the interval includes zero or is negative.
  - Sensitive if: as 2.5.
  - Status: untested.
- **2.7 EXP** 2.5 and 2.6 hold when every population is cut to the same number of units.
  - Attack: subsample to N_units; repeat.
  - Refuted if: either is refuted.
  - Sensitive if: as 2.5.
  - Status: untested.
- **2.8 EXP** 2.5 and 2.6 hold when one architecture family is held out entirely.
  - Attack: leave-one-family-out.
  - Refuted if: either is refuted for any family.
  - Sensitive if: as 2.5.
  - Status: untested.
- **2.9 EXP** A model whose units have lower D̂ than another, at matched W and repeat correlation, does not have higher T.
  - Attack: list all model pairs matched on W within δ_within and on repeat correlation within δ_rel.
  - Refuted if: in more than ε_pairs of such pairs the lower-D̂ model has T higher by δ_T.
  - Sensitive if: as 2.5.
  - Status: untested.

## Claim 3 — Population geometry predicts T beyond D̂

- 3.1 = 2.2, 2.3, 2.4
- **3.2 MATH** If r(p, h) = A·φ(p) + B·ψ(h) and a readout matrix R satisfies R·B = 0, then R·r does not depend on h.
  - Derivation: R·r = R·A·φ(p).
  - Consequence: T can equal W in a population in which every unit responds to heading.
  - **3.2.1 CITE** Kriegeskorte & Wei (2021), "Neural tuning and representational geometry", arXiv:2104.09743v1, Abstract: "The tuning induces the geometry, but different sets of tuned neurons can induce the same geometry." §2.3: "However, the geometry is sufficient for determining how well any given distinction among the stimuli can be decoded (under the assumption of isotropic, stimulus-independent noise)."
- **3.3 DEF** g₁ is the mean length of the heading-induced displacement of the population response at matched positions, after projection through the frozen readout of T. g₂ is the mean cosine between position-discrimination directions fitted at different headings.
- **3.4 EXP** g₁ and g₂ are the only geometric measures that were tried.
  - Attack: read the analysis log.
  - Refuted if: any other geometric measure was computed against T before the confirmatory run.
  - Sensitive if: the log records every measure computed.
  - Status: untested.
- **3.5 EXP** The readout and g₁, g₂ are fitted on one trajectory split, and T is evaluated on another.
  - Attack: compare trajectory identifiers across the three splits.
  - Refuted if: any overlap.
  - Sensitive if: a planted overlap is found.
  - Status: untested.
- **3.6 EXP** Adding g₁ and g₂ to the regression of 2.5 (which already contains D̂) lowers held-out-model prediction error of T by at least δ_geo.
  - Attack: as 2.5.
  - Refuted if: the upper bound of the interval of the error reduction is below δ_geo.
  - Sensitive if: on synthetic populations built as in 3.2 with R·B ≠ 0 of varying size, the procedure finds a reduction above δ_geo.
  - Status: untested.
- **3.7 EXP** Larger g₁ goes with larger T, and larger g₂ with smaller T.
  - Attack: read the fitted signs and intervals.
  - Refuted if: either interval includes zero or has the other sign.
  - Sensitive if: as 3.6.
  - Status: untested.
- **3.8 EXP** 3.6 and 3.7 hold with and without per-unit standardization fitted on training data.
  - Attack: repeat both ways.
  - Refuted if: the verdicts differ.
  - Sensitive if: as 3.6.
  - Status: untested.

## Claim 4 — In at least one trained model, position can be read across headings and heading can still be read

- **4.1 EXP** In that model, T is within δ_T of the error of a decoder trained on the target headings with the same number of samples.
  - Attack: fit the target-heading decoder on separate target-heading training blocks; compare. The position decoder is never given heading or test-set normalization statistics.
  - Refuted if: the lower bound of the interval of the difference is above δ_T in every model.
  - Sensitive if: a synthetic population with R·B ≠ 0 fails.
  - Status: untested.
- **4.2 EXP** In the same model, a decoder of (sin h, cos h), tested on held-out positions, beats the best predictor of heading from position and occupancy alone by at least δ_head.
  - Attack: fit both on training blocks; compare circular error on held-out positions.
  - Refuted if: the difference is below δ_head in every model that passes 4.1.
  - Sensitive if: the same decoder on input pixels beats the baseline.
  - Status: untested.
- **4.3 EXP** 4.1 and 4.2 hold in at least k of N_seed retrainings of that model.
  - Attack: retrain; repeat.
  - Refuted if: fewer than k pass both.
  - Sensitive if: as 4.1 and 4.2.
  - Status: untested.
- **4.4 EXP** 4.2 holds for raw responses and for responses divided by their population norm.
  - Attack: repeat 4.2 after normalization with training-set statistics.
  - Refuted if: the verdicts differ.
  - Sensitive if: as 4.2.
  - Status: untested.

## Claim 5 — Ranking the predictive transformer's components by the D̂ of their units finds a set C that the next-frame prediction depends on

- **5.1 EXP** The checkpoint loads with every key matched, and the prediction at step t does not depend on frames after t.
  - Attack: `strict=True` load; hold frames 1…t fixed, replace later frames, compare the output at t.
  - Refuted if: a key is unmatched, or the output at t changes.
  - Sensitive if: removing the attention mask changes the output at t.
  - Status: load survived on 2026-07-07 (157 of 157 keys). Causality untested; `F.layer_norm(f, f.shape[1:])` on a [B, L, C, H, W] tensor (`predictive_coder.py:480–493`) predicts refutation.
- **5.2 EXP** C is chosen on discovery trajectories only, by ranking components by the population-level D̂ of their units.
  - Attack: read the patching registry.
  - Refuted if: any test trajectory was used, or C was chosen by another rule.
  - Sensitive if: the registry records every selection step.
  - Status: untested.
- **5.3 EXP** Patching C's output from a donor run into a recipient run changes the predicted frame in the direction written down beforehand for that pair.
  - Attack: donor and recipient are real runs that differ in one stated variable. Patch the full output of C. Measure native error against the donor's and the recipient's true next frame.
  - Refuted if: error against the donor's frame falls by less than δ_patch of the clean gap.
  - Sensitive if: patching all components gives the donor's frame.
  - Status: untested.
- **5.4 EXP** Patching a random component set of the same size and activation norm does not reproduce 5.3.
  - Attack: N_rand random sets.
  - Refuted if: more than 5% reach δ_patch.
  - Sensitive if: as 5.3.
  - Status: untested.
- **5.5 EXP** Removing C raises native error, and restoring C's clean output removes the rise.
  - Attack: mean-ablate C at three strengths; then restore.
  - Refuted if: the rise is below δ_abl, or restoration recovers less than δ_rest of it.
  - Sensitive if: removing a whole block raises error by more than δ_abl. δ_abl is set as a fraction of the native-error change caused by a controlled input change with a known effect.
  - Status: untested.
- **5.6 EXP** The ablation in 5.5 changes the operation, not only a probe score.
  - Attack: after ablation, measure the stated variable's effect on the predicted frame in unpatched runs.
  - Refuted if: D̂ or a probe score changes while that effect is unchanged.
  - Sensitive if: ablating a whole block removes the effect.
  - Status: untested.
- **5.7 EXP** 5.3–5.5 hold in at least k of N_seed independently trained models.
  - Attack: repeat per model.
  - Refuted if: fewer than k pass.
  - Sensitive if: as 5.3.
  - Status: untested. No training code is in the repository.
- **5.8 CITE** An intervention that raises a decodability measure need not change behaviour. Xiong, Ji-An, Wilson, Lee & Wei (2026), COLM, arXiv:2605.28854v3, Abstract: "Causal interventions that amplify neural activity along the axis defining the task are insufficient to improve behavioral performance or induce this representational reorganization."

## Claim 6 — Within CA1 sessions, lower D̂ goes with lower T, as in the models

- 6.1 = Claim 2
- **6.2 EXP** Every population used is a set of simultaneously recorded cells from one session.
  - Attack: read the population construction.
  - Refuted if: cells from different sessions are combined.
  - Sensitive if: not applicable.
  - Status: untested.
- **6.3 EXP** Populations are matched in cell count, sample count and the set of positions covered at every heading sector.
  - Attack: compute the three quantities per population.
  - Refuted if: any differs by more than ε_match.
  - Sensitive if: a planted mismatch is flagged.
  - Status: untested.
- **6.4 EXP** Across subpopulations of a session, the slope of T on D̂ is positive, with sessions nested in animals.
  - Attack: draw N_sub subpopulations per session that differ in population-level D̂; fit a mixed model.
  - Refuted if: the interval of the slope includes zero or is negative.
  - Sensitive if: simulated populations on the real trajectories, with T built to depend on D, give a positive slope at the same N.
  - Status: untested. Four animals.
- **6.5 EXP** Decoder splits are contiguous time blocks separated by a gap longer than the smoothing window.
  - Attack: read the split code.
  - Refuted if: adjacent samples fall in training and test sets.
  - Sensitive if: not applicable.
  - Status: untested. The existing decoders (`pfeiffer_decode_solo.py`) were not written for held-out-heading tests.

## Claim 7 — D̂ predicts T better than existing single-unit measures

- 7.1 = Claim 2
- **7.2 CITE** A correlation index of the similarity of tuning-curve shape across conditions exists. Nowak, Dobbins, Gawne, Grzywacz & Amthor (2011), *J Neurophysiol* 105(5):2083–2099, Methods.
  - "This index was similar to that introduced by Rapela et al. (2010) except for two differences: it was based on the median rather than the mean as the measure of central tendency and on the Kendall's tau rank correlation coefficient rather than the Pearson product-moment correlation coefficient as the measure of correlation."
- **7.3 CITE** A split-half single-unit separability index exists, and its authors state that it predicts population linear separability. Rust & DiCarlo (2010), *J Neurosci* 30(39):12978–12995, Materials and Methods.
  - "This metric measures how well the response of a neuron can be explained as resulting from independent tuning for object identity and the identity-preserving image transformation (e.g., retinal position, size)."
  - "We here term this metric single-neuron “linear separability index” (se Fig. 12c) because, although it does not indicate linear separability of object identity in the response of a single neuron, under reasonable assumptions about the distribution of tuning functions in a population, it is a good predictor of linear separability of object identity at the neuronal population level and is a far better predictor than single-unit measures such as receptive field size (Li et al., 2009)."
- **7.4 CITE** Variance of the response over one variable, averaged over the other, is an existing measure for place and head-direction units. Franzius, Sprekeler & Wiskott (2007), *PLoS Comput Biol* 3(8): e166, Methods.
  - "The sensitivity of a function f to spatial position r will be characterized by its mean positional variance ηr, which is the variance of f(r,ϕ) with respect to r averaged over all head directions ϕ: ηr(f) = 〈varr (f(r,ϕ))〉ϕ."
- **7.5 EXP** Each of the indices in 7.2, 7.3 and 7.4, put in place of D̂ in 2.5, lowers held-out prediction error of T by less than D̂ does.
  - Attack: repeat 2.5 once per index on the same models and splits.
  - Refuted if: any index gives a reduction at least as large as D̂'s.
  - Sensitive if: each index detects M.6's planted shift.
  - Status: untested.
- **7.6 EXP** The likelihood gap of 1.3 (flexible minus shared-shape model), put in place of D̂ in 2.5, lowers held-out prediction error of T by less than D̂ does.
  - Attack: as 7.5.
  - Refuted if: its reduction is at least as large as D̂'s.
  - Sensitive if: as 7.5.
  - Status: untested.

## Claim N — No cited prior work reports Claims 1–3

- **N.1 CITE** Single-unit separability was found not to account fully for population readout in visual cortex. Rust & DiCarlo (2010), Results.
  - "Thus, differences in the maintenance of relative selectivity across changes in position, scale, and context between V4 and IT also cannot completely account for the differences in population performance between these two visual areas."
- **N.2 CITE** In mouse V1, a fixed orientation decoder works across luminance while luminance stays encoded, on orthogonal axes. O'Shea, Wei & Priebe (2026), *eNeuro* 13(2):ENEURO.0281-25.2025.
  - Discussion: "These results indicate that the V1 population response carries information about mean luminance, in a manner that does not interfere with using a shared, linear decoder to recover orientation information across scotopic and photopic conditions."
  - Abstract: "Importantly, V1 populations encode changes in stimulus orientation and mean luminance along orthogonal axes in the neural response space, such that a change in one stimulus variable is encoded independently from the other."
  - Results: "dUori and dUSF were not orthogonal, but rather had dot products shifted away from zero, in contrast with the orthogonality of dUori and dUluminance (Fig. 6B)."
- **N.3 CITE** The same group reported the shared decoder first. O'Shea, Nauhaus, Wei & Priebe (2025), *Cell Rep* 44(1):115217, Discussion.
  - "As predicted for a luminance-invariant cortical code, the performance of a shared decoding scheme is close to that of a luminance-dependent decoder on orientation discrimination for the V1 population responses."
- **N.4 CITE** Training a linear decoder on some conditions and testing on others is a published measure, and it is unchanged by rotating the population. Bernardi et al. (2020), *Cell* 183(4):954–967.
  - "As described in reference to Figure 2b–d, CCGP can be computed by training a linear decoder to classify any dichotomy on a subset of conditions, and testing classification performance on conditions not used for training."
  - "Under the assumption that a decoder is linear, CCGP will not change if a linear operation like rotation is performed on the data points."
- **N.5 CITE** That measure has been applied to human hippocampal neurons across agent and gaze, not across heading. Chericoni et al. (2026), arXiv:2603.04747v1.
  - Abstract: "Moreover, their geometry supported generalization across spatial maps, such that a linear rule learned on one agent transfers to another."
  - Results: "Indeed, linear transformability does not imply that a decoder trained in one subspace will function in another, because the decoding axes may not be preserved by the transform."
- **N.6 CITE** Place-like, direction-like and border-like units occur in networks not trained on spatial tasks, and leaving them out does not hurt position decoding more than leaving out random units. Luo, Mok & Love, eLife reviewed preprint 99047, bioRxiv 10.1101/2024.01.10.575026v2.
  - Summary: "We show that deep neural networks (DNNs) for object recognition, which lack spatial grounding, contain numerous units resembling place, border, and head-direction cells. Strikingly, even untrained DNNs with randomized weights contained such units and support decoding of spatial information."
  - Results 2.3: "[...] we randomly excluded the same number of units without regard to the spatial score (Fig. 4B, bottom panel) and observed a similar pattern across four exclusion scenarios, meaning that highly-tuned spatial units contributed no more to spatial cognition than a random selection of units."
- **N.7 CITE** In a trained recurrent network, units tuned to two variables are needed for the computation. Cueva, Wang, Chin & Wei (2020), ICLR, arXiv:1912.10189v2.
  - "We first lesioned connections from CW Shifters to all units, and found that it significantly impaired rotation in the CW direction, and also increased the rotation speed in the CCW direction."
- **N.8 CITE** Nonlinear mixed selectivity helps linear readout. Rigotti et al. (2013), *Nature* 497:585–590.
  - "High-dimensional neural representations have the desirable property of allowing simple readouts such as linear classifiers to implement a large set of input-output relations."
- **N.9 CITE** Place-cell firing in a cylinder was reported to have no directional specificity of its own. Muller, Bostock, Taube & Kubie (1994), *J Neurosci* 14(12):7235–7251, Abstract.
  - "When the distributive hypothesis is put into numeric form, the directional firing profiles that it predicts are extremely similar to the observed directional firing profiles, strongly suggesting that there is no intrinsic directional specificity of place cell firing in the cylinder."
  - "A major conclusion reached from recording in both environments is that the directional firing properties of hippocampal place cells are variable and not fixed; this is true of individual units as well as of the population."
- **N.10 CITE** Heading modulates the firing rate of many hippocampal place cells. These reports concern rate; by M.1, D ignores gain.
  - Rubin, Yartsev & Ulanovsky (2014), *J Neurosci* 34(3):1067–1080, Results: "Among the population of place cells (n = 86/108 cells with significant spatial information, see Materials and Methods), the proportion of neurons with significant head-direction tuning was even higher: 50/86, or 58% of the place cells were significantly tuned to head direction."
  - Jercog et al. (2019), *Nat Commun* 10:2333, Abstract: "Here, utilizing electrophysiological recordings from CA1 pyramidal cells in freely moving mice, we report that a majority of neural responses are modulated by the heading-direction of the animal relative to a point within or outside their enclosure that we call a reference point."
- **N.11 CITE** Gain and shape both vary in real tuning, and a model that assumes one form of variation misses the others. Zhu & Wei (2023), *Nat Commun* 14:2298, Introduction.
  - "Indeed, variability of neural tuning has now been widely reported in neural systems, and furthermore been proposed to exhibit various forms, including multiplicative gain, additive modulation, shift of tuning peaks, and tuning width changes."
- **N.12 CITE** Item-by-item comparison of a model with a brain needs the same stimuli; these models and recordings do not share stimuli. Qian, Geisler & Wei (2025), NeurIPS, arXiv:2506.02164v4, Abstract.
  - "DVC quantifies the image-by-image correlation between the decoded decisions based on the internal neural representations in a classification task."
- **N.13 AXIOM** None of N.1–N.12 tests whether stability of spatial field shape across heading predicts cross-heading position decoding across trained models, or measures that stability in CA1 with gain and sampling noise separated. No closer work exists.

## Parameters (all unset)

δ_D, δ_shift, δ_ll, δ_range, δ_pred, δ_geo, δ_T, δ_within, δ_rel, δ_head, δ_patch, δ_abl, δ_rest, ε_within, ε_pairs, ε_match, N_occ, N_models, N_held, N_units, N_sub, N_seed, N_rand, k, the heading sectors, the place-cell criterion for P, the layer per model, the estimator version v.

The earlier draft gave planning defaults: δ_T = 0.02 arena diagonals; δ_pred = 5% reduction in held-out error; δ_shift = one position bin; eight heading sectors; two model conditions × five trained models each.

## Axioms

- N.13.
- The margins above, once set, are the smallest differences that matter.
