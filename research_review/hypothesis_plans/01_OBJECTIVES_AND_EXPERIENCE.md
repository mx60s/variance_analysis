# 01 — Objectives and experience: claim tree

## Rules

- A claim holds if and only if every one of its children holds.
- Numbered top-level claims are independent unless one lists another as a child. A child written "= Claim X" means the whole of Claim X.
- Leaf kinds:
  - **DEF** — a definition.
  - **MATH** — follows from the definitions; the derivation is given.
  - **CITE** — one or more sentences copied from the cited paper. Inline math is retyped in plain text; nothing else is changed.
  - **EXP** — an experiment. The leaf says that the strongest attack on its parent does not produce the refuting result.
  - **AXIOM** — assumed and not tested. All axioms are listed again in the last section.
- An EXP leaf holds if and only if (a) the attack was run and did not produce the "Refuted if" result, and (b) the same attack produces that result when the effect is planted ("Sensitive if"). A leaf that fails (b) is untested, not survived.
- Status values: `untested`, `survived`, `refuted`, `observed, no margin set`.
- Every δ, ε and N is a free parameter. Each is fixed before the confirmatory run and recorded in the Parameters section. No parameter has been fixed yet.
- No experiment in this document has been run unless its status says otherwise.

## Definitions

- **DEF model.** `PredictiveCoder` in `src/ann/predictive_coder.py`: convolutional encoder, attention block 1, feed-forward block 1, attention block 2, feed-forward block 2, image decoder. Input is a sequence of frames. Output is a predicted next frame for each input frame.
- **DEF p, h, H.** p is position (x, y). h is heading. H is the sequence of frames before the current frame.
- **DEF z.** z_t is the activation of the last input token at one fixed layer. The layer is fixed before the confirmatory run.
- **DEF field.** f_h(p) is a unit's mean response at position p and heading h, averaged over histories that end at (p, h).
- **DEF S, D.** S is the mean, over all pairs of different headings h ≠ h′, of the Pearson correlation over p between f_h and f_h′. D = 1 − S.
- **DEF D̂.** D̂ is an estimate of D from data. Each pair correlation is computed between f_h from one half of the histories and f_h′ from the other, disjoint half. D̂_v0 is the current code: `1 - quantify_variance_corr(fold0, fold1, occ, axis=1)[0]`.
- **DEF model-level D̂.** The median of D̂ over all units of the fixed layer.
- **DEF objective A.** L_pred: mean squared error between predicted and actual next frame.
- **DEF objective B.** L_pred + λ·L_lat. L_lat = SmoothL1( q(z_t, sg(e_{t+1})), sg(z_{t+1}) ). e_{t+1} is the encoder output for the next frame. q is a two-layer MLP. sg is stop-gradient.
- **DEF seed pair.** One A model and one B model with identical initial weights, identical training episodes and identical batch order.
- **DEF G.** Fit a ridge linear decoder from z to p on one heading sector. G = (mean position error on other headings) − (mean position error on held-out episodes in the training sector), at matched positions.
- **DEF native error.** Mean squared error of the model's own decoder output against the true next frame. No probe is involved.
- **DEF memory test set.** Episodes in which the current frame is the same for two histories and the true next frame differs.

## Claim 0 — The latent-transition loss changes the spatial computation of the transformer

- 0.1 = Claim 1
- 0.2 = Claim 2
- 0.3 = Claim 4
- 0.4 = Claim 5
- 0.5 = Claim 6

## Claim S — A and B differ only in whether L_lat sends gradient into the backbone

- **S.1 EXP** A and B in each seed pair start from the same weights and see the same batches.
  - Attack: hash initial weights and the batch index sequence of both runs.
  - Refuted if: any hash differs.
  - Sensitive if: a run with a changed seed gives a different hash.
  - Status: untested. No two-objective trainer exists in the repository.
- **S.2 EXP** In A, L_lat sends no gradient into the backbone.
  - Attack: backpropagate L_lat alone in A; read the gradient norm on every backbone parameter.
  - Refuted if: any norm is nonzero.
  - Sensitive if: the same read in B gives a nonzero norm (S.3).
  - Status: untested.
- **S.3 EXP** In B, L_lat sends gradient into the backbone.
  - Attack: same as S.2, in B.
  - Refuted if: all norms are zero.
  - Sensitive if: S.2 reads zero with the same code.
  - Status: untested.
- **S.4 EXP** A and B have the same architecture and the same auxiliary head q.
  - Attack: diff the two configurations and parameter counts.
  - Refuted if: anything differs other than the stop-gradient on q's inputs.
  - Sensitive if: a deliberately altered configuration is flagged.
  - Status: untested.
- **S.5 EXP** The prediction at step t does not depend on frames after t.
  - Attack: hold frames 1…t fixed, replace frames t+1…L, compare the output at t. Run in train mode and eval mode.
  - Refuted if: the output at t changes.
  - Sensitive if: removing the attention mask makes the output at t change.
  - Status: untested. Reading the code predicts refutation for the unmodified architecture: `F.layer_norm(f, f.shape[1:])` is applied to a tensor of shape [B, L, C, H, W] (`predictive_coder.py:480–493`), so the normalization statistics include all L frames. The encoder's BatchNorm also pools batch × time in train mode.
- **S.6 EXP** Both models predict the next frame better than copying the current frame.
  - Attack: compare native error with the error of the copy-current-frame predictor on held-out episodes.
  - Refuted if: either model's error is not lower by δ_comp.
  - Sensitive if: an untrained model fails this test.
  - Status: untested.
- **S.7 EXP** Neither model's z has collapsed.
  - Attack: compute the variance and effective rank of z over held-out episodes.
  - Refuted if: variance or rank in either model is below ε_rank.
  - Sensitive if: a model trained without stop-gradient on the target collapses and is flagged.
  - Status: untested.
- **S.8 EXP** In B, the head q uses z_t.
  - Attack: on the frozen B backbone, fit a head of the same size that receives only e_{t+1} (z_t set to zero).
  - Refuted if: its L_lat is within δ_head of the full head's L_lat.
  - Sensitive if: on the memory test set, where the same e_{t+1} has different target z_{t+1}, a head without z_t cannot reach the full head's loss.
  - Status: untested.
- **S.9 EXP** No training trajectory appears in the test set.
  - Attack: compare episode identifiers and pose sequences between splits.
  - Refuted if: any overlap.
  - Sensitive if: a planted duplicate is found.
  - Status: untested.
- **S.10 EXP** λ, the stopping rule and the optimizer settings were chosen using validation L_pred only.
  - Attack: read the selection log.
  - Refuted if: any selection step reads D̂, G, or any outcome in Claims 1–7.
  - Sensitive if: the log records every metric read during selection.
  - Status: untested.
- **S.11 EXP** Every training transition is a legal move.
  - Attack: check each consecutive pose pair against the declared pose graph (adjacent position or adjacent heading).
  - Refuted if: any transition is not an edge.
  - Sensitive if: a planted illegal transition is found.
  - Status: untested. No trajectory generator exists in the repository. `data/minecraft/panorama-90yaw/` holds 289 positions × 90 headings of single frames.

## Claim M — D̂ measures change of spatial field shape across heading in this data

- **M.1 MATH** If f_h(p) = a_h·g(p) + b_h with a_h > 0 and g not constant, then D = 0.
  - Derivation: Pearson correlation is unchanged by positive scaling and by adding a constant, so every pair correlation is 1.
- **M.2 EXP** The code returns D̂ = 0 for noiseless gain-only and offset-only changes.
  - Attack: score h(x,y) = x + y and h(x,y) = x·y with the repository functions.
  - Refuted if: D̂ ≠ 0 on either axis.
  - Sensitive if: a crossover interaction gives D̂ > 0.
  - Status: survived. Both cases return 0.0 on both axes; 20 + (x − 6.5)·sin(y) returns 1.09 and 1.03 (`source_notes/method_diagnostics.json`).
- **M.3 EXP** The pipeline call compares spatial profiles across headings.
  - Attack: read `quantify_variance_corr` and its call site; score a synthetic unit with a fixed place field and a synthetic unit with a fixed heading curve.
  - Refuted if: the fixed-place-field unit gets high D̂ from the `axis=1` call.
  - Sensitive if: the two synthetic units get opposite scores.
  - Status: survived by code reading only. `axis` is moved to the slice dimension and the rest is flattened into the profile (`explained_variance.py:123–124`); the pipeline calls `axis=1` on (position, heading) maps (`run_model_layers.py:82`). The synthetic check has not been run.
- **M.4 EXP** A unit whose spatial profile is constant is reported as not estimable.
  - Attack: score h(x,y) = x with and without small independent noise.
  - Refuted if: the code returns a finite D̂ for the constant-profile axis.
  - Sensitive if: not applicable; the refuting result is directly observable.
  - Status: refuted for v0. Without noise the code returns NaN. With noise of amplitude 10⁻⁴ it returns D̂ = 1.007 with repeat correlation −0.02 (`method_diagnostics.json`).
- **M.5 EXP** A field shift of δ_shift bins raises D by at least δ_D.
  - Attack: compute noiseless D for synthetic fields shifted by δ_shift across headings, on the real position grid. Repeat for a width change and for an added second field.
  - Refuted if: D < δ_D for any of the three.
  - Sensitive if: not applicable; noiseless computation.
  - Status: untested.
- **M.6 EXP** Identical field shape in every heading does not give D̂ above δ_D when headings differ in reliability.
  - Attack: simulate one shape in all headings with per-heading noise levels taken from the data; score it.
  - Refuted if: D̂ > δ_D.
  - Sensitive if: a planted shift of δ_shift under the same noise gives D̂ larger by δ_D.
  - Status: refuted for v0. Twelve identical slices, six with reliability 1 and six with reliability 0.01, give D̂ = 0.716. The ratio correction 1 − off/diag gives 0.437 (`method_diagnostics.json`, 4,000-bin simulation with the repository scorer). Untested for any other estimator version.
- **M.7 MATH** For identical shape in every heading and equal reliability q in every heading, E[D̂_v0] ≈ 1 − q.
  - Derivation: with signal plus independent noise, the cross-half correlation of slices i and j is ρ_ij·√(q_i·q_j). With ρ_ij = 1 and q_i = q this is q. (`source_notes/method_audit.md` §2.)
  - Consequence: D̂_v0 fails M.6 whenever q < 1 − δ_D.
  - **M.7.1 CITE** Splitting trials leaves a bias toward zero in the correlation of noisy tuning curves. Pospisil & Bair (2021), *J Neurosci* 41(26):5638–5651, Results, "Corrected estimator of signal correlation": "The above strategy means that rn is now zero, thus leaving r^s (or r^split) with a bias toward zero that increases with noise as follows:" (r^s is r-hat-sub-s in the source.)
  - **M.7.2 CITE** The bias affects studies of tuning invariance across conditions. Pospisil & Bair (2021), "The unbiased estimation of r2 between two sets of noisy neural responses", bioRxiv 2021.03.29.437413, Abstract: "Yet this metric is biased by trial-to-trial variability: as trial-to-trial variability increases, measured correlation decreases. Major lines of research are confounded by this bias, including the study of invariance of neural tuning across conditions and the similarity of tuning across neurons."
- **M.8 EXP** Nominal 95% intervals for D̂ contain the true D in at least 90% of simulations.
  - Attack: simulate known fields on the real (p, h) support at observed noise levels; compute the interval with the frozen procedure.
  - Refuted if: coverage < 90%.
  - Sensitive if: an interval procedure known to be too narrow is flagged.
  - Status: untested. No interval procedure is implemented.
- **M.9 EXP** Smoothing is applied on the two-dimensional position grid.
  - Attack: read the scoring path.
  - Refuted if: smoothing runs along a flattened position index.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline. `process_neuron_loop` smooths a (289, 90) map with sigma = (1, 1); the 17 × 17 grid is one axis of length 289 (`run_model_layers.py:76–80, 158`).
- **M.10 EXP** D̂ is computed on the original bins of signed activations.
  - Attack: read the scoring path.
  - Refuted if: the map is interpolated or clipped at zero before scoring.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline. `masked_zoom` upsamples heading 5× and clips at zero (`explained_variance.py:25–31`; `run_model_layers.py:79, 158`).
- **M.11 EXP** The two halves are independent repeats of the same condition.
  - Attack: list what differs between the halves.
  - Refuted if: the halves differ in a controlled stimulus variable.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline: the halves are pitch groups [−3°, +1°] and [−1°, +3°]. Untested for the new experiments, where the halves are disjoint sets of histories ending at the same (p, h).
- **M.12 EXP** Every heading pair is scored on the same set of positions.
  - Attack: compare the valid-bin mask across heading pairs.
  - Refuted if: masks differ between pairs.
  - Sensitive if: a planted missing bin is detected.
  - Status: untested. v0 uses a pair-specific mask inside `_corr_matrix`.
- **M.13 EXP** Responses come from histories drawn from the training distribution.
  - Attack: read the extraction code.
  - Refuted if: histories are not samples of the training distribution.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline: each frame is repeated ten times and the last token is read (`run_model_layers.py:50–58`). Untested for the new experiments.

## Claim 1 — Objective B lowers D relative to objective A

- 1.1 = Claim S
- 1.2 = Claim M
- **1.3 EXP** Across seed pairs, model-level D̂ is lower in B than in A by at least δ_D.
  - Attack: train N_seed seed pairs to the frozen stopping rule. Compute model-level D̂ on the common test episodes.
  - Refuted if: the upper bound of the 95% interval of the paired difference D̂_B − D̂_A is above −δ_D.
  - Sensitive if: a positive-control model trained with an explicit heading-invariance loss shows a difference below −δ_D with N_seed pairs.
  - Status: untested.
- **1.4 EXP** The difference in 1.3 is not a difference in reliability.
  - Attack: match A and B units on repeat correlation (the diagonal of the correlation matrix); recompute 1.3 on the matched units.
  - Refuted if: the matched difference is above −δ_D.
  - Sensitive if: in simulation, a planted reliability-only difference disappears under the same matching.
  - Status: untested.
- **1.5 EXP** The difference in 1.3 does not depend on which units are scored.
  - Attack: recompute 1.3 on all units, and on units selected as place-like from training-split data only.
  - Refuted if: the difference is above −δ_D in either set.
  - Sensitive if: same as 1.3.
  - Status: untested.
- **1.6 EXP** An independent field model shows the same difference.
  - Attack: for each unit fit (i) one spatial shape with a gain and offset per heading and (ii) a separate spatial shape per heading. Compare held-out squared error. Take the model-level median of error(i) − error(ii).
  - Refuted if: that median is not smaller in B than in A by δ_fit.
  - Sensitive if: planted shifted fields make error(i) − error(ii) rise.
  - Status: untested.
- **1.7 EXP** Marginal place-cell measures do not show the difference.
  - Attack: compute the fraction of place-like units and the mean spatial information per unit, from heading-averaged maps, for A and B.
  - Refuted if: either measure differs between A and B by at least its margin δ_marg.
  - Sensitive if: an untrained model differs from a trained model on these measures.
  - Status: untested.

## Claim 2 — Objective B makes position readable across headings

- 2.1 = Claim S
- **2.2 EXP** Position can be decoded within a heading sector in both models.
  - Attack: within-sector error on held-out episodes.
  - Refuted if: either model's error is above ε_within (in arena units).
  - Sensitive if: a decoder on shuffled position labels is above ε_within.
  - Status: untested.
- **2.3 EXP** G is lower in B than in A by at least δ_G.
  - Attack: N_seed seed pairs; same decoder family, regularization chosen on training-sector validation data; equal training-set sizes; matched test positions.
  - Refuted if: the upper bound of the 95% interval of G_B − G_A is above −δ_G.
  - Sensitive if: the positive-control model of 1.3 shows a difference below −δ_G.
  - Status: untested.
- **2.4 EXP** Within-sector error in B is not higher than in A by more than δ_within.
  - Attack: same runs as 2.3.
  - Refuted if: the lower bound of the interval of the within-sector error difference is above δ_within.
  - Sensitive if: same as 2.2.
  - Status: untested.
- **2.5 EXP** 2.3 holds for every choice of training sector.
  - Attack: repeat 2.3 with each of the N_sector sectors as the training sector.
  - Refuted if: 2.3 is refuted for any sector.
  - Sensitive if: same as 2.3.
  - Status: untested.
- **2.6 EXP** 2.3 holds at convergence and at matched validation L_pred.
  - Attack: evaluate 2.3 at the frozen stopping point and at the pair of checkpoints with equal validation L_pred.
  - Refuted if: 2.3 is refuted at either.
  - Sensitive if: same as 2.3.
  - Status: untested.

## Claim 3 — B keeps heading while making position readable across headings

- 3.1 = Claim 2
- **3.2 EXP** Heading can be decoded from z in B.
  - Attack: ridge decoder of (sin h, cos h) from z, trained on some positions, tested on held-out positions.
  - Refuted if: circular error is above ε_head.
  - Sensitive if: the same decoder on input pixels is below ε_head, and on shuffled labels is above it.
  - Status: untested.
- **3.3 EXP** B's native error on turn steps is not higher than A's by more than δ_turn.
  - Attack: native error on steps where only heading changes.
  - Refuted if: the lower bound of the interval of error_B − error_A is above δ_turn.
  - Sensitive if: a model with heading removed from z by projection fails this test.
  - Status: untested.

## Claim 4 — B moves history into the current token by the end of block 1; A reads it from past tokens in block 2

- 4.1 = Claim S
- **4.2 EXP** On the memory test set, the next frame cannot be predicted from the current frame alone.
  - Attack: train a current-frame-only predictor and a full-history predictor of equal size on the same episodes.
  - Refuted if: the current-frame-only error is within δ_mem of the full-history error.
  - Sensitive if: on a test set without aliasing the two errors are within δ_mem.
  - Status: untested. No memory test set exists.
- **4.3 EXP** Blocking the current token's attention to past tokens in block 2 raises native error less in B than in A, by at least δ_mask.
  - Attack: mask those attention entries at inference on the memory test set; N_seed seed pairs.
  - Refuted if: the upper bound of the interval of (rise in B) − (rise in A) is above −δ_mask.
  - Sensitive if: the rise in A is above δ_mask.
  - Status: untested.
- **4.4 EXP** A mask of the same size on tokens that carry no disambiguating cue does not produce the 4.3 difference.
  - Attack: mask an equal number of block-2 attention entries to tokens before the cue.
  - Refuted if: the difference (rise in B) − (rise in A) is below −δ_mask.
  - Sensitive if: same as 4.3.
  - Status: untested.
- **4.5 EXP** Swapping the post-block-1 current-token state between two histories with the same current frame and different cues moves B's prediction to the donor's next frame.
  - Attack: interchange that state, with block-2 past-token attention masked; measure native error against the donor's and the recipient's true next frame.
  - Refuted if: error against the donor's frame does not fall by at least δ_swap of the clean gap.
  - Sensitive if: swapping the full set of token states produces the donor's frame.
  - Status: untested.
- **4.6 EXP** Swapping that state between two different histories that imply the same next frame leaves B's prediction unchanged.
  - Attack: same interchange, on history pairs with the same true next frame.
  - Refuted if: native error rises by more than δ_equiv.
  - Sensitive if: 4.5 shows the interchange can change the output.
  - Status: untested.
- **4.7 EXP** No route other than attention carries information between tokens.
  - Attack: = S.5, plus: with all cross-token attention masked, perturb past frames and read the current token.
  - Refuted if: the current token changes.
  - Sensitive if: with attention unmasked the current token changes.
  - Status: untested. Code reading predicts refutation for the unmodified architecture (see S.5). A model with per-token normalization is a different model and is named as such.

## Claim 5 — A fixed set of components C in B performs the history integration of Claim 4

- 5.1 = Claim 4
- **5.2 EXP** C is chosen on discovery episodes only.
  - Attack: read the patching registry.
  - Refuted if: any confirmation episode was used to choose C.
  - Sensitive if: the registry records the episode set of every selection step.
  - Status: untested.
- **5.3 EXP** Patching C's output from a donor run into a recipient run moves the prediction toward the donor's next frame on held-out episodes.
  - Attack: donor and recipient share the current frame and differ in the cue. Patch the full output of C from the natural donor run.
  - Refuted if: the fall in error against the donor's frame is below δ_patch of the clean gap.
  - Sensitive if: patching all components gives the donor's frame.
  - Status: untested.
- **5.4 EXP** Patching a random component set of the same size and activation norm does not.
  - Attack: same as 5.3 with N_rand random sets.
  - Refuted if: more than 5% of random sets reach δ_patch.
  - Sensitive if: same as 5.3.
  - Status: untested.
- **5.5 EXP** Removing C raises native error on the memory test set, and putting C's clean output back removes the rise.
  - Attack: replace C's output with its mean over episodes; then restore the clean output with everything else still corrupted upstream of C.
  - Refuted if: the rise is below δ_abl, or restoration recovers less than δ_rest of it.
  - Sensitive if: removing the whole of block 1 raises error.
  - Status: untested.
- **5.6 EXP** 5.3–5.5 hold under two corruption methods and two output metrics.
  - Attack: repeat with (mean ablation, resample ablation) × (full-image error, error in the image region where donor and recipient frames differ).
  - Refuted if: any of the four combinations refutes 5.3, 5.4 or 5.5.
  - Sensitive if: same as 5.3.
  - Status: untested.
  - **5.6.1 CITE** Patching results depend on these choices. Zhang & Nanda (2024), ICLR, arXiv:2309.16042, Abstract: "In several settings of localization and circuit discovery in language models, we find that varying these hyperparameters could lead to disparate interpretability results." §1: "On corruption method, we show that GN and STR can lead to inconsistent localization and circuit discovery outcomes (subsection 3.1)."
- **5.7 EXP** Attention weights are not used as evidence for C.
  - Attack: read the evidence listed for C.
  - Refuted if: any listed evidence is an attention map without an intervention.
  - Sensitive if: not applicable.
  - Status: untested.

## Claim 6 — C is why B predicts better on new trajectories

- 6.1 = Claim 5
- **6.2 EXP** No held-out trajectory composition or cue-gap length appears in training.
  - Attack: enumerate compositions and gap lengths in both splits.
  - Refuted if: any overlap.
  - Sensitive if: a planted overlap is found.
  - Status: untested.
- **6.3 EXP** B has lower native error than A on the held-out compositions, by at least δ_gen.
  - Attack: N_seed seed pairs; also compare with a current-frame-only predictor and a short-history predictor.
  - Refuted if: the upper bound of the interval of error_B − error_A is above −δ_gen.
  - Sensitive if: the full-history predictors beat the current-frame-only predictor on this set.
  - Status: untested.
- **6.4 EXP** 6.3 holds for error in the image region that differs between alternative futures.
  - Attack: repeat 6.3 with the region metric.
  - Refuted if: 6.3 is refuted under the region metric.
  - Sensitive if: same as 6.3.
  - Status: untested.
- **6.5 EXP** Removing C from B removes at least δ_med of B's advantage on held-out compositions, and less than that on familiar compositions.
  - Attack: ablate C as in 5.5; recompute 6.3 on both sets.
  - Refuted if: the advantage on held-out compositions falls by less than δ_med, or falls equally on familiar compositions.
  - Sensitive if: ablating all of block 1 removes the advantage.
  - Status: untested.
- **6.6 EXP** Removing a random component set of the same size does not.
  - Attack: N_rand random sets.
  - Refuted if: more than 5% of random sets remove δ_med of the advantage.
  - Sensitive if: same as 6.5.
  - Status: untested.
- **6.7 EXP** Restoring C's clean output restores the advantage.
  - Attack: as 5.5.
  - Refuted if: less than δ_rest of the lost advantage returns.
  - Sensitive if: same as 6.5.
  - Status: untested.

## Claim 7 — The effect of the objective on D depends on the ratio of rotation to translation in training

- 7.1 = Claim 1
- **7.2 DEF** Experience E1 and E2 are two training sets that differ in the fraction of steps that are rotations.
- **7.3 EXP** E1 and E2 are equal in number of frames, position occupancy, heading coverage and sequence length.
  - Attack: compute the four statistics for both sets.
  - Refuted if: any differs by more than ε_exp.
  - Sensitive if: a planted occupancy difference is flagged.
  - Status: untested.
- **7.4 EXP** The interaction (D̂_B − D̂_A under E2) − (D̂_B − D̂_A under E1) is not zero.
  - Attack: 2 × 2 design, N_seed seed pairs per experience, one common test set.
  - Refuted if: the 95% interval of the interaction lies inside ±δ_int.
  - Sensitive if: a simulated interaction of size δ_int is detected with N_seed pairs.
  - Status: untested.
- **7.5 EXP** The field model of 1.6 shows the same interaction.
  - Attack: compute the 1.6 quantity in the 2 × 2 design.
  - Refuted if: its interaction interval lies inside ±δ_fit.
  - Sensitive if: same as 1.6.
  - Status: untested.

## Claim 8 — D̂ separates A from B better than existing shape measures

- 8.1 = Claim 1
- **8.2 CITE** A correlation index of the similarity of tuning-curve shape across conditions of other stimulus variables exists, in a Pearson form and a rank form. Nowak, Dobbins, Gawne, Grzywacz & Amthor (2011), *J Neurophysiol* 105(5):2083–2099, Methods, "Similarity Index".
  - "Within each tuning category, the similarity of the shapes of those tuning curves was quantified off-line using a similarity index."
  - "This index was similar to that introduced by Rapela et al. (2010) except for two differences: it was based on the median rather than the mean as the measure of central tendency and on the Kendall's tau rank correlation coefficient rather than the Pearson product-moment correlation coefficient as the measure of correlation."
  - "The index was computed in three steps: first, the tuning curves were normalized with respect to their peaks, then the median normalized tuning curve for the cell was determined, and finally the Kendall's tau rank correlation coefficient between the data constituting that median curve and the data corresponding to all the normalized tuning curves was computed yielding the index value."
- **8.3 CITE** A split-half separability index for tuning across a second variable exists. Rust & DiCarlo (2010), *J Neurosci* 30(39):12978–12995, Materials and Methods, "Single-neuron measurement of linear separability".
  - "This metric measures how well the response of a neuron can be explained as resulting from independent tuning for object identity and the identity-preserving image transformation (e.g., retinal position, size)."
  - "For these neurons, we computed the independent tuning prediction by computing the singular value decomposition of one matrix (M1 = USV′) and then used the first principal component to determine the expected response to each object across the six transformations (Mpred was the product of the first columns of U and V′). Finally, we computed the element-wise Pearson's correlation of Mpred and the separately measured M2."
- **8.4 EXP** The similarity index of 8.2 does not separate A from B as well as D̂.
  - Attack: compute the index on the same units and test episodes; compare the standardized paired difference between A and B with that of D̂.
  - Refuted if: the index's standardized difference is at least that of D̂.
  - Sensitive if: the index separates the positive-control model of 1.3 from A.
  - Status: untested.
- **8.5 EXP** The separability index of 8.3 does not separate A from B as well as D̂.
  - Attack: compute the index on the same units and test episodes; compare the standardized paired difference between A and B with that of D̂.
  - Refuted if: the index's standardized difference is at least that of D̂.
  - Sensitive if: the index separates the positive-control model of 1.3 from A.
  - Status: untested.
- **8.6 EXP** The field-model quantity of 1.6 does not separate A from B as well as D̂.
  - Attack: as 8.4 with error(i) − error(ii).
  - Refuted if: its standardized difference is at least that of D̂.
  - Sensitive if: as 8.4.
  - Status: untested.
- **8.7 EXP** A noise-corrected correlation does not separate A from B as well as D̂.
  - Attack: as 8.4 with the r̂²_ER estimator of M.7.2's source applied to each heading pair.
  - Refuted if: its standardized difference is at least that of D̂.
  - Sensitive if: as 8.4.
  - Status: untested.

## Claim N — No cited prior work reports Claims 1–6

- **N.1 CITE** Next-image prediction in this model family gives latents from which position can be decoded, with spatially localized units. Gornet & Thomson (2024), *Nature Machine Intelligence* 6, 820–833.
  - Abstract: "While learning a next-image prediction task, the agent automatically constructs an internal representation of the environment that quantitatively reflects spatial distances."
  - Results: "To measure the positional information in the predictive coder, we train a neural network to predict the agent’s position from the predictive coder’s latent units (Fig. 1a)."
  - Results: "We demonstrate that each unit in the neural network’s latent space activates at distinct, localized regions—akin to place fields in the mammalian brain—in the environment’s physical space (Fig. 5a)."
  - Methods: "The decoder architecture consists of several layers designed to extract this positional information: a convolutional layer transforms the input to a higher dimension (256), followed by a ReLU activation for non-linearity."
- **N.2 CITE** The next-latent objective is published. Its authors did not analyze the learned representations. Teoh et al., "Next-Latent Prediction Transformers Learn Compact World Models", arXiv:2511.05963v4.
  - §3.2: "NextLat additionally enforces transition consistency (Equation 2) of the hidden states by introducing a latent dynamics model p_ψ that predicts the next hidden state h_{t+1} directly from (h_t, X_{t+1})."
  - Abstract: "Empirically, across benchmarks in world modeling, reasoning, planning, and language modeling, NextLat demonstrates significant gains over standard next-token prediction and other baselines in downstream accuracy, representation compression, and lookahead planning."
  - §6: "On the analysis side, we do not study the structure of the learned representations under NextLat, leaving open questions about how the method shapes latent spaces."
  - Their loss also has a KL term (their Eq. 5). Objective B in this plan does not.
- **N.3 CITE** The guarantee for the next-latent objective assumes exact optimization. Teoh et al., §3.1.
  - "Here we analyze an idealized next-latent prediction transformer which successfully optimizes both next-token prediction and next-latent prediction with respect to an underlying data distribution."
- **N.4 CITE** Next-token prediction alone already gives linearly decodable belief states in transformers trained on hidden-Markov data. Shai et al. (2024), NeurIPS 37, arXiv:2405.15943.
  - Abstract: "Leveraging the theory of optimal prediction, we anticipate and then find that belief states are linearly represented in the residual stream of transformers, even in cases where the predicted belief state geometry has highly nontrivial fractal structure."
  - §4.4: "In the experiments we ran, we generated training data using HMMs."
- **N.5 CITE** The closest reinforcement-learning study varied task and action space, not the loss. Vijayabaskaran & Cheng (2022), *PLoS Comput Biol* 18(10): e1010320.
  - Results: "Hence, our simulations revealed that neither navigation strategy nor task demands alone determine the evolution of spatial representations, but rather a combination of the two (Fig 6B)."
  - Methods: "Because the action space defines the reference frame in which navigation takes place, in our model, we view the strategy as being identical to the action space used."
- **N.6 CITE** The closest transformer study varied the training data of GPT-2 models on grid paths, with the same next-token loss. Baumgartner, Spens, Burgess & Manescu (2025), arXiv:2511.13371v1.
  - §1: "Here, we conduct a controlled experiment comparing models trained on exploratory random walks (Foraging Model) versus shortest-path finding (SP Models)."
  - Abstract: "Using behavioural, representational and mechanistic analyses, we uncover two fundamentally different learned algorithms."
- **N.7 CITE** Training input statistics change tuning in a recurrent head-direction network. Cueva, Wang, Chin & Wei (2020), ICLR, arXiv:1912.10189v2, §4.
  - "When networks are trained with small angular velocities, we observe the presence of more units with strong head direction tuning but minimal angular velocity tuning."
  - "Conversely, when networks are trained with large AV inputs, fewer Compass units emerge and more units become Shifter-like and exhibit both HD and AV tuning (Fig. 5c,f,i)."
- **N.8 AXIOM** None of N.1–N.7 reports the effect of a latent-transition loss on spatial field shape, on cross-heading position decoding, or on where history is integrated in a visual prediction transformer. No closer work exists.

## Parameters (all unset)

δ_D, δ_shift, δ_fit, δ_marg, δ_comp, δ_head, δ_G, δ_within, ε_within, ε_head, δ_turn, δ_mem, δ_mask, δ_swap, δ_equiv, δ_patch, δ_abl, δ_rest, δ_gen, δ_med, δ_int, ε_rank, ε_exp, N_seed, N_sector, N_rand, λ, the layer of z, the estimator version v.

## Axioms

- N.8.
- The margins above, once set, are the smallest differences that matter.
