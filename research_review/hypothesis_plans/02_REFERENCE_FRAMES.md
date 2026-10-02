# 02 — Reference frames: claim tree

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

- **DEF model.** `PredictiveCoder` in `src/ann/predictive_coder.py` with weights `data/weights/predictive_coder.ckpt`: convolutional encoder, attention block 1, feed-forward block 1, attention block 2, feed-forward block 2, image decoder. Input is a sequence of frames. Output is a predicted next frame for each input frame.
- **DEF p, h, l, a, o.** p is observer position (x, y). h is heading. l is the position of a named landmark. a is a fixed direction in the environment. o is a fixed point in the environment. R(θ) is rotation by θ.
- **DEF frame family F.** Five coordinate systems, fixed before any test:
  - W: (p, h).
  - A: l − p.
  - E: R(−h)(l − p).
  - C: R(−a)(p − o).
  - B: distance and observer-relative bearing to a named boundary segment.
- **DEF field in frame f.** f_h(q) is a unit's mean response at frame coordinate q and heading h, averaged over histories that end there.
- **DEF S, D.** S is the mean, over all pairs of different headings h ≠ h′, of the Pearson correlation over q between f_h and f_h′. D = 1 − S. D_f is D computed with q taken from frame f. D_W is D over position.
- **DEF D̂.** An estimate of D in which each pair correlation is computed between two disjoint halves of the histories. D̂_v0 is the current code: `1 - quantify_variance_corr(fold0, fold1, occ, axis=1)[0]`.
- **DEF model-level D̂.** The median of D̂ over all scored units of the fixed layer.
- **DEF scene intervention.** A rendered scene that differs from a base scene by exactly one stated change.
- **DEF discovery scenes, test scenes.** Two disjoint sets of scenes and landmark positions. Everything that is chosen is chosen on discovery scenes.
- **DEF native error.** Mean squared error of the model's own decoder output against the true next frame. No probe is involved.
- **DEF encoding model of frame f.** Ridge regression from a fixed spline basis of frame f's coordinates to a unit's response. All frames get the same number of basis functions.

## Claim 0 — Units whose spatial fields change shape across heading encode a landmark relation, and the model uses it to predict the next frame

- 0.1 = Claim 1
- 0.2 = Claim 2
- 0.3 = Claim 4
- **0.4 EXP** The f\* of Claim 2 is a landmark or boundary frame.
  - Attack: read which frame won 2.5–2.10.
  - Refuted if: f\* is W or C.
  - Sensitive if: 3.3 shows that a population built in frame C is identified as C.
  - Status: untested.

## Claim G — The model under study is the trained checkpoint, run as it was trained

- **G.1 EXP** The checkpoint loads with every key matched.
  - Attack: `load_state_dict(..., strict=True)`; record the file hash.
  - Refuted if: any key is missing or unexpected.
  - Sensitive if: a checkpoint with one renamed key fails.
  - Status: survived on 2026-07-07 (157 of 157 keys). Hash not recorded. Rerun.
- **G.2 EXP** The loader fails when the checkpoint file is missing.
  - Attack: call the loader with a path that does not exist.
  - Refuted if: it returns a model.
  - Sensitive if: not applicable.
  - Status: refuted by code reading. `get_activations.py:95–96` loads weights only `if weights_path and weights_path.exists()` and otherwise returns the randomly initialized network.
- **G.3 EXP** Activations read through hooks reproduce the output of `forward`.
  - Attack: rebuild the decoder output from hooked activations; compare with `forward`.
  - Refuted if: the maximum absolute difference is above ε_num.
  - Sensitive if: a hook on the wrong module gives a difference above ε_num.
  - Status: untested.
- **G.4 EXP** The prediction at step t does not depend on frames after t.
  - Attack: hold frames 1…t fixed, replace frames t+1…L, compare the output at t.
  - Refuted if: the output at t changes.
  - Sensitive if: removing the attention mask makes the output at t change.
  - Status: untested. Reading the code predicts refutation: `F.layer_norm(f, f.shape[1:])` is applied to a tensor of shape [B, L, C, H, W] (`predictive_coder.py:480–493`), so the normalization statistics include all L frames.
- **G.5 EXP** On histories from the training distribution, the model predicts the next frame better than copying the current frame.
  - Attack: compare native error with the copy-current-frame error.
  - Refuted if: native error is not lower by δ_comp.
  - Sensitive if: the randomly initialized model fails.
  - Status: untested. No trajectories from the training distribution are in the repository.
- **G.6 EXP** A renderer produces scene pairs that differ only in the stated change.
  - Attack: render base and edited scenes; difference the images outside the edited object and its shadow.
  - Refuted if: pixels differ outside that region.
  - Sensitive if: a planted second change is detected.
  - Status: untested. No renderer, landmark coordinates or scene manifest are in the repository. `data/minecraft/panorama-90yaw/` holds one fixed scene, 289 positions × 90 headings.
- **G.7 EXP** The model takes no action input.
  - Attack: read `PredictiveCoder.forward` and `get_latents`.
  - Refuted if: an action tensor enters the computation.
  - Sensitive if: not applicable.
  - Status: survived by code reading. `get_latents(x, actions=None, which=-1)` never uses `actions` (`predictive_coder.py:502–530`). `forward` takes images only.

## Claim M — D̂ measures change of field shape across heading in this data

- **M.1 MATH** If f_h(q) = a_h·g(q) + b_h with a_h > 0 and g not constant, then D = 0.
  - Derivation: Pearson correlation is unchanged by positive scaling and by adding a constant, so every pair correlation is 1.
- **M.2 EXP** The code returns D̂ = 0 for noiseless gain-only and offset-only changes.
  - Attack: score h(x,y) = x + y and h(x,y) = x·y with the repository functions.
  - Refuted if: D̂ ≠ 0 on either axis.
  - Sensitive if: a crossover interaction gives D̂ > 0.
  - Status: survived. Both cases return 0.0 on both axes; 20 + (x − 6.5)·sin(y) returns 1.09 and 1.03 (`source_notes/method_diagnostics.json`).
- **M.3 EXP** The pipeline call compares profiles across headings.
  - Attack: read `quantify_variance_corr` and its call site; score a synthetic unit with a fixed place field and one with a fixed heading curve.
  - Refuted if: the fixed-place-field unit gets high D̂ from the `axis=1` call.
  - Sensitive if: the two synthetic units get opposite scores.
  - Status: survived by code reading only (`explained_variance.py:123–124`; `run_model_layers.py:82`). The synthetic check has not been run.
- **M.4 EXP** A unit whose profile is constant in the chosen frame is reported as not estimable.
  - Attack: score h(x,y) = x with and without small independent noise.
  - Refuted if: the code returns a finite D̂ for the constant-profile axis.
  - Sensitive if: not applicable.
  - Status: refuted for v0. Without noise the code returns NaN. With noise of amplitude 10⁻⁴ it returns D̂ = 1.007 with repeat correlation −0.02 (`method_diagnostics.json`).
- **M.5 EXP** A field shift of δ_shift bins raises D by at least δ_D.
  - Attack: compute noiseless D for synthetic fields shifted by δ_shift across headings, on the real grid. Repeat for a width change and an added second field.
  - Refuted if: D < δ_D for any of the three.
  - Sensitive if: not applicable; noiseless computation.
  - Status: untested.
- **M.6 EXP** Identical field shape in every heading does not give D̂ above δ_D when headings differ in reliability.
  - Attack: simulate one shape in all headings with per-heading noise levels taken from the data; score it.
  - Refuted if: D̂ > δ_D.
  - Sensitive if: a planted shift of δ_shift under the same noise gives D̂ larger by δ_D.
  - Status: refuted for v0. Twelve identical slices, six with reliability 1 and six with reliability 0.01, give D̂ = 0.716. The ratio correction 1 − off/diag gives 0.437 (`method_diagnostics.json`). Untested for any other estimator version.
- **M.7 MATH** For identical shape in every heading and equal reliability q in every heading, E[D̂_v0] ≈ 1 − q.
  - Derivation: with signal plus independent noise, the cross-half correlation of slices i and j is ρ_ij·√(q_i·q_j). With ρ_ij = 1 and q_i = q this is q. (`source_notes/method_audit.md` §2.)
  - **M.7.1 CITE** Pospisil & Bair (2021), *J Neurosci* 41(26):5638–5651, Results: "The above strategy means that rn is now zero, thus leaving r^s (or r^split) with a bias toward zero that increases with noise as follows:" (r^s is r-hat-sub-s in the source.)
- **M.8 EXP** Smoothing is applied on the two-dimensional grid of the frame coordinate.
  - Attack: read the scoring path.
  - Refuted if: smoothing runs along a flattened index.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline. `process_neuron_loop` smooths a (289, 90) map with sigma = (1, 1); the 17 × 17 grid is one axis of length 289 (`run_model_layers.py:76–80, 158`).
- **M.9 EXP** D̂ is computed on the original bins of signed activations.
  - Attack: read the scoring path.
  - Refuted if: the map is interpolated or clipped at zero before scoring.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline. `masked_zoom` upsamples heading 5× and clips at zero (`explained_variance.py:25–31`; `run_model_layers.py:79, 158`).
- **M.10 EXP** The two halves are independent repeats of the same condition.
  - Attack: list what differs between the halves.
  - Refuted if: the halves differ in a controlled stimulus variable.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline: the halves are pitch groups [−3°, +1°] and [−1°, +3°].
- **M.11 EXP** Every heading pair, in every frame, is scored on the same set of coordinate bins.
  - Attack: compare the valid-bin mask across heading pairs and across frames.
  - Refuted if: masks differ.
  - Sensitive if: a planted missing bin is detected.
  - Status: untested. v0 uses a pair-specific mask inside `_corr_matrix`. In frame E the covered coordinates rotate with h, so the common set is smaller than in frame W.
- **M.12 EXP** Responses come from histories drawn from the training distribution.
  - Attack: read the extraction code.
  - Refuted if: histories are not samples of the training distribution.
  - Sensitive if: not applicable.
  - Status: refuted for the v0 model pipeline: each frame is repeated ten times and the last token is read (`run_model_layers.py:50–58`).

## Claim 1 — Spatial fields of units in the fixed layer change shape across heading

- 1.1 = Claim G
- 1.2 = Claim M
- **1.3 EXP** Model-level D̂_W is above δ_D.
  - Attack: score all units chosen in 1.6 with an estimator that survives Claim M.
  - Refuted if: the lower bound of the 95% interval of model-level D̂_W is below δ_D.
  - Sensitive if: a synthetic population with fields shifted by δ_shift across headings scores above δ_D.
  - Status: observed under refuted preconditions. With the v0 pipeline the median D̂_W is above 0.9 in every trained layer scored and 0.72 in untrained ResNet-50 avgpool (`results/ann/*/ev_all_neurons_pitch.npz`, run 2026-07-07). That run fails M.8, M.9, M.10 and M.12.
- **1.4 EXP** D̂_W is above the D̂ of a population with identical shape in every heading and the same repeat correlations.
  - Attack: simulate that population on the real support; compare.
  - Refuted if: the observed model-level D̂_W is within δ_D of the simulated value.
  - Sensitive if: as 1.3.
  - Status: untested.
- **1.5 EXP** A model with one spatial shape and a gain and offset per heading predicts held-out responses worse than a model with a separate spatial shape per heading.
  - Attack: fit r(p,h) = a(h)·f(p) + b(h) and an interaction model with the same number of parameters; compare held-out squared error.
  - Refuted if: the error difference is below δ_fit.
  - Sensitive if: planted shifted fields give a difference above δ_fit; planted gain-only fields give a difference below it.
  - Status: untested.
- **1.6 EXP** Units are chosen without using test data.
  - Attack: read the selection code.
  - Refuted if: selection reads any response used in 1.3–1.5.
  - Sensitive if: not applicable.
  - Status: untested.
- **1.7 EXP** 1.3–1.5 hold for histories from the training distribution, not only for a repeated static frame.
  - Attack: repeat 1.3–1.5 on histories from the training distribution ending at matched (p, h).
  - Refuted if: any of 1.3–1.5 is refuted on those histories.
  - Sensitive if: as 1.3.
  - Status: untested.

## Claim 2 — One frame in F predicts unit responses under scene interventions better than world coordinates and better than image features

- 2.1 = Claim G
- 2.2 = Claim 1
- **2.3 MATH** With one landmark at a fixed location, frames W and A fit any data equally well.
  - Derivation: l − p is an invertible function of p when l is constant.
  - Consequence: fixed-scene fits cannot choose between W and A. 2.5–2.8 use scenes in which l changes.
- **2.4 EXP** The frame f\* and its encoding model are fixed on discovery scenes.
  - Attack: read the analysis log.
  - Refuted if: any test scene was used to choose f\*, its basis, its regularization or the landmark.
  - Sensitive if: the log records the scene set of every fitting step.
  - Status: untested.
- **2.5 EXP** When the landmark moves by Δl, responses follow f\*'s prediction.
  - Attack: predict test-scene responses, without refitting, from the encoding models of f\*, of W, and of image features (pixels and encoder output). For frame E the predicted coordinate is q′ = q + R(−h)Δl.
  - Refuted if: f\*'s held-out error is not lower than the better of W and image features by δ_enc; or responses move opposite to f\*'s predicted direction.
  - Sensitive if: a synthetic population built in frame f\* and passed through the same analysis is identified as f\*.
  - Status: untested.
- **2.6 EXP** When the observer turns by Δh with p and l fixed, responses follow f\*'s prediction.
  - Attack: as 2.5. For frame E the predicted coordinate is q′ = R(−Δh)q.
  - Refuted if: as 2.5.
  - Sensitive if: as 2.5.
  - Status: untested.
- **2.7 EXP** When observer and landmark are translated together, responses do not change by more than δ_same.
  - Attack: render the translated pair; compare responses.
  - Refuted if: the change is above δ_same.
  - Sensitive if: translating the observer alone changes responses by more than δ_same.
  - Status: untested. Applies only if f\* is A, E or B.
- **2.8 EXP** When the landmark's appearance changes and its geometry does not, responses do not change by more than δ_same.
  - Attack: re-texture the landmark; compare responses.
  - Refuted if: the change is above δ_same.
  - Sensitive if: moving the landmark changes responses by more than δ_same.
  - Status: untested.
- **2.9 EXP** 2.5 and 2.6 hold at every step of a fixed ladder of encoding-model sizes.
  - Attack: repeat with each basis size on the ladder, then with a nonlinear model of matched size.
  - Refuted if: f\* loses at any step.
  - Sensitive if: as 2.5.
  - Status: untested.
- **2.10 EXP** 2.5 and 2.6 give the same verdict for Cartesian and polar encodings of f\*.
  - Attack: repeat with both encodings.
  - Refuted if: the verdicts differ.
  - Sensitive if: as 2.5.
  - Status: untested.
- **2.11 EXP** The model still predicts the next frame in edited scenes.
  - Attack: native error on test scenes.
  - Refuted if: native error is above ε_native.
  - Sensitive if: native error on scrambled scenes is above ε_native.
  - Status: untested.

## Claim 3 — The frame with the lowest D̂ on discovery scenes is f\*

- 3.1 = Claim M
- 3.2 = Claim 2
- **3.3 EXP** On synthetic populations built in a known frame and sampled on the real support, the lowest-D̂ frame is the known frame.
  - Attack: build one population per frame in F; add noise at observed levels; rank frames by model-level D̂_f.
  - Refuted if: the known frame is ranked first in fewer than 1 − ε_sel of runs.
  - Sensitive if: a population built in frame W is not ranked first under frame E.
  - Status: untested.
- **3.4 EXP** On discovery scenes, the lowest-D̂ frame is the f\* of Claim 2.
  - Attack: rank the five frames by model-level D̂_f on discovery scenes. Determine f\* from 2.5–2.10 without reading the ranking.
  - Refuted if: they differ.
  - Sensitive if: 3.3.
  - Status: untested.
- **3.5 EXP** The ranking does not change with bin size, smoothing width or support threshold inside the declared ranges.
  - Attack: recompute the ranking at every setting on the declared grid.
  - Refuted if: the first-ranked frame changes.
  - Sensitive if: not applicable.
  - Status: untested.
- **3.6 EXP** The first-ranked frame is not first because its profiles are nearly constant.
  - Attack: compare the across-bin variance of profiles in the first-ranked frame with that in frame W.
  - Refuted if: the variance ratio is below ε_var.
  - Sensitive if: a synthetic frame that makes profiles constant is flagged.
  - Status: untested. See M.4.
- **3.7 EXP** Held-out fit of the encoding models on discovery scenes does not pick f\* from fewer scenes than D̂ does.
  - Attack: subsample discovery scenes; at each size record whether lowest D̂ and highest held-out R² each pick f\*.
  - Refuted if: the encoding-model rule reaches 1 − ε_sel accuracy at a smaller or equal number of scenes.
  - Sensitive if: both rules reach 1 − ε_sel at the full number of scenes.
  - Status: untested.
- **3.8 EXP** Heading-averaged place-field measures do not pick f\*.
  - Attack: rank frames by mean spatial information of heading-averaged maps in each frame.
  - Refuted if: that ranking puts f\* first.
  - Sensitive if: not applicable.
  - Status: untested.

## Claim 4 — A fixed set of components C carries the f\* coordinate, and the next-frame prediction depends on it

- 4.1 = Claim G
- 4.2 = Claim 2
- **4.3 EXP** C and the patch rule are chosen on discovery scenes only.
  - Attack: read the patching registry.
  - Refuted if: any test scene was used.
  - Sensitive if: the registry records the scene set of every selection step.
  - Status: untested.
- **4.4 EXP** Patching C's output from a donor run (landmark at l′) into a recipient run (landmark at l) moves the predicted frame toward the rendered frame for l′.
  - Attack: donor and recipient are real runs on rendered scenes. Patch the full output of C. Measure native error against the rendered frame for l′ and for l.
  - Refuted if: error against the l′ frame falls by less than δ_patch of the clean gap.
  - Sensitive if: patching all components gives the l′ frame.
  - Status: untested.
- **4.5 EXP** The reverse patch moves the prediction the other way.
  - Attack: swap donor and recipient.
  - Refuted if: as 4.4.
  - Sensitive if: as 4.4.
  - Status: untested.
- **4.6 EXP** Patching from a donor with a different scene and the same f\* coordinate leaves the prediction unchanged.
  - Attack: choose donors matched on q and different in p, h and l.
  - Refuted if: native error against the recipient's own frame rises by more than δ_equiv.
  - Sensitive if: 4.4 shows the patch can change the output.
  - Status: untested.
- **4.7 EXP** Patching a random component set of the same size and activation norm does not reproduce 4.4.
  - Attack: N_rand random sets.
  - Refuted if: more than 5% reach δ_patch.
  - Sensitive if: as 4.4.
  - Status: untested.
- **4.8 EXP** Removing C raises native error, and putting C's clean output back removes the rise.
  - Attack: replace C's output with its mean; then restore the clean output.
  - Refuted if: the rise is below δ_abl, or restoration recovers less than δ_rest of it.
  - Sensitive if: removing a whole block raises error.
  - Status: untested.
- **4.9 EXP** The patch in 4.4 changes the landmark region of the predicted frame and leaves the rest of the frame within δ_off.
  - Attack: split native error into the image region where the l and l′ frames differ and the rest.
  - Refuted if: error outside the region rises by more than δ_off.
  - Sensitive if: patching all components changes the landmark region.
  - Status: untested.
- **4.10 EXP** 4.4–4.9 hold under two corruption methods and two output metrics.
  - Attack: (mean ablation, resample ablation) × (full-image error, region error).
  - Refuted if: any combination refutes any of 4.4–4.9.
  - Sensitive if: as 4.4.
  - Status: untested.
  - **4.10.1 CITE** Zhang & Nanda (2024), ICLR, arXiv:2309.16042, Abstract: "In several settings of localization and circuit discovery in language models, we find that varying these hyperparameters could lead to disparate interpretability results."
- **4.11 EXP** Every patch is the full output of a component taken from a real run, not a learned subspace.
  - Attack: read the patching code.
  - Refuted if: any patch is a projection onto a fitted direction or subspace.
  - Sensitive if: not applicable.
  - Status: untested.
  - **4.11.1 CITE** Makelov, Lange & Nanda (2023), arXiv:2311.17030, Abstract: "Counterintuitively, even if a subspace intervention makes the model’s output behave as if the value of a feature was changed, this effect may be achieved by activating a dormant parallel pathway leveraging another subspace that is causally disconnected from model outputs."

## Claim 5 — While the landmark is hidden, the model keeps the f\* coordinate from earlier frames

- 5.1 = Claim 2
- 5.2 = G.4
- **5.3 EXP** Earlier frames carry information about the next frame that the current frame lacks.
  - Attack: on episodes where the landmark is hidden in the current frame, train a current-frame-only predictor and a full-history predictor of equal size with the same inputs the model gets.
  - Refuted if: the current-frame-only error is within δ_mem of the full-history error.
  - Sensitive if: on episodes with the landmark visible the two errors are within δ_mem.
  - Status: untested.
- **5.4 EXP** The f\* coordinate can be decoded from z while the landmark is hidden.
  - Attack: ridge decoder fit on discovery histories, tested on held-out histories.
  - Refuted if: its error is not below the single-frame decoder's and the shuffled-history decoder's by δ_dec.
  - Sensitive if: with the landmark visible the decoder is below ε_dec.
  - Status: untested.
- **5.5 EXP** Two histories with the same current frame and different earlier landmark relation give different predicted frames, each closer to its own true next frame.
  - Attack: native error of each prediction against both true frames.
  - Refuted if: own-frame error is not lower than other-frame error by δ_mem.
  - Sensitive if: 5.3.
  - Status: untested.
- **5.6 EXP** 5.4 and 5.5 hold after the landmark's appearance is changed.
  - Attack: repeat with re-textured landmarks.
  - Refuted if: either is refuted.
  - Sensitive if: as 5.4.
  - Status: untested.

## Claim 6 — C is why the model predicts well on new combinations of heading and landmark position

- 6.1 = Claim 4
- **6.2 EXP** No test combination of heading and landmark position appears in the scenes used for training or discovery.
  - Attack: enumerate combinations in each set.
  - Refuted if: any overlap.
  - Sensitive if: a planted overlap is found.
  - Status: untested.
- **6.3 EXP** Removing C raises native error more on new combinations than on familiar ones, by at least δ_med.
  - Attack: ablate C as in 4.8 at three strengths; compare the rise on both sets.
  - Refuted if: the difference in rise is below δ_med.
  - Sensitive if: removing a whole block raises error on both sets.
  - Status: untested.
- **6.4 EXP** Removing a random component set of the same size does not reproduce 6.3.
  - Attack: N_rand random sets.
  - Refuted if: more than 5% reach δ_med.
  - Sensitive if: as 6.3.
  - Status: untested.
- **6.5 EXP** Restoring C's clean output removes the 6.3 difference.
  - Attack: as 4.8.
  - Refuted if: less than δ_rest of the difference is removed.
  - Sensitive if: as 6.3.
  - Status: untested.

## Claim 7 — Claims 2 and 4 hold in independently trained models and in scenes not used so far

- **7.1 EXP** A training procedure for this architecture exists and reproduces the checkpoint's native error.
  - Attack: train one model; compare native error with the checkpoint's.
  - Refuted if: the difference is above δ_comp.
  - Sensitive if: a model trained for one epoch differs by more than δ_comp.
  - Status: untested. No training code is in the repository.
- **7.2 EXP** With the frame rule, the patch rule and all parameters frozen, Claim 2 holds in at least k of N_seed models.
  - Attack: repeat Claim 2 per model on held-out scenes. Unit identity may be re-aligned on discovery scenes only.
  - Refuted if: fewer than k models pass.
  - Sensitive if: 3.3 passes for rotated and distributed synthetic codes.
  - Status: untested.
- **7.3 EXP** Claim 4 holds in at least k of N_seed models.
  - Attack: as 7.2 for Claim 4.
  - Refuted if: fewer than k models pass.
  - Sensitive if: as 4.4.
  - Status: untested.

## Claim N — No cited prior work reports Claims 2–4

- **N.1 CITE** A sequence model's board state is linear in a player-relative frame and not in an absolute frame, and interventions in that frame change outputs. Nanda, Lee & Wattenberg (2023), BlackboxNLP, pp. 16–30.
  - §8: "Previously unbeknownst, we demonstrated that the board state in OthelloGPT is linearly represented by encoding the colour of each tile relative to the player at each timestep (Mine, Yours, Empty) as opposed to absolute colour (Black, White, Empty)."
  - §1: "Put differently, by pushing the model’s activations in the directions of Mine, Yours, or Empty, we can alter the model’s belief state of the board, and change its predictions accordingly."
- **N.2 CITE** Cells that code a world-oriented vector to an object exist (frame A). Høydal, Skytøen, Andersson, Moser & Moser (2019), *Nature* 568, 400–404.
  - Abstract: "Here we show that a large fraction of medial entorhinal cortex neurons fire specifically when mice are at given distances and directions from spatially confined objects."
  - bioRxiv v1 preprint (doi 10.1101/286286), Results: "In the majority of cells that had displaced firing fields (offsets of more than 4 cm), the vector relationship (distance and direction from the object) was maintained when the object was moved to a new location on a subsequent trial (Fig. 1d), suggesting that the cell’s activity was determined by the location of the object and not the distal cues."
  - bioRxiv v1 preprint, Discussion: "The vector representation is allocentric, i.e. the firing fields are fixed in room coordinates, irrespective of the animal’s direction of movement, distinguishing them from the egocentrically tuned goal-vector cells of the hippocampus."
- **N.3 CITE** Cells that code observer-relative bearing to items exist (frame E). Wang, Chen, Lee, Deshmukh, Yoganarasimha, Savelli & Knierim (2018), *Science* 362(6417):945–949.
  - Abstract: "Many LEC neurons showed tuning for egocentric bearing of external items, whereas MEC cells tended to represent allocentric bearing."
- **N.4 CITE** Task and action space change which frame emerges in a visual reinforcement-learning agent. Vijayabaskaran & Cheng (2022), *PLoS Comput Biol* 18(10): e1010320.
  - Abstract: "Furthermore, we find that place-cell-like allocentric representations emerge preferentially in guidance when using an allocentric strategy, whereas egocentric vector representations emerge when using an egocentric strategy in aiming."
- **N.5 CITE** A transformer with recurrent position encodings is related to a hippocampal model; it has no visual input. Whittington, Warren & Behrens, arXiv:2112.04035.
  - §7: "We have shown that TEM, a current model of the hippocampal formation, is closely related to a transformer with recurrent position encodings."
- **N.6 CITE** A model of reusable object-related maps reproduces object-vector fields; it reports no new recordings and no network intervention. Piray & Daw (2025), *Nat Commun* 16, 7444.
  - Abstract: "The model treats each object as an alteration to a baseline map linked to open space, creating predictive maps by combining object-related representations compositionally, providing insights into brain processes supporting efficient, flexible planning."
- **N.7 CITE** Interventions on internal features change behaviour in a transformer trained on text mazes. Spies et al., arXiv:2412.11867.
  - §1: "We show that transformers trained to solve mazes acquire causally relevant WMs - intervening upon these in the latent space of SAEs predictably affects model behaviour."
- **N.8 CITE** The interchange intervention used in Claim 4 is a published procedure. Geiger et al. (2025), *JMLR* 26(83):1–64, arXiv:2301.04709, §2.5.
  - "Specifically, the causal model is provided a ‘base’ input and an intervention is performed that fixes some variables to be the values they would have if different ‘source’ inputs were provided."
- **N.9 CITE** Landmarks anchor the head-direction population. Ajabi, Keinath, Wei et al. (2023), *Nature* 615, 892–899.
  - Results: "Tracking this offset, we observed a rotational response after cue reappearance that matched the cue shift. We refer to this phenomenon as a reset (Fig. 2a)."
- **N.10 CITE** Self-motion can act on a position code as a group of transformations. Gao, Xie, Wei, Zhu & Wu (2021), NeurIPS, arXiv:2006.10259, Abstract.
  - "In this paper, we conduct theoretical analysis of a general representation model of path integration by grid cells, where the 2D self-position is encoded as a higher dimensional vector, and the 2D self-motion is represented by a general transformation of the vector."
- **N.11 AXIOM** None of N.1–N.10 tests a physical reference frame with scene interventions in a visual prediction transformer, or uses shape correlation across heading to select the frame. No closer work exists.

## Parameters (all unset)

δ_D, δ_shift, δ_fit, δ_comp, δ_enc, δ_same, δ_patch, δ_equiv, δ_abl, δ_rest, δ_off, δ_mem, δ_dec, δ_med, ε_num, ε_native, ε_sel, ε_var, ε_dec, N_rand, N_seed, k, the named landmark(s) (at most two), a, o, the boundary segment, the layer of z, the basis-size ladder, the declared grid of bin size / smoothing / support, the estimator version v.

The earlier draft gave planning defaults: δ_enc = 5% relative error reduction over the best baseline; δ_patch = 10% of the clean gap; N_seed = 5.

## Axioms

- N.11.
- The margins above, once set, are the smallest differences that matter.
