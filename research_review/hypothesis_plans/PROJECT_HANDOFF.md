# Shared project handoff and evidence standards

This companion supplies repository provenance, verified reusable assets, implementation gates, and a common experimental record for the three hypothesis documents. It is intended to travel with those documents and the original repository. It records inspection on September 14, 2026; it is not a report of newly executed scientific experiments.

## Documents and intended use

**Current method-centered contract:** Claim M in each plan file is the shared entry point (the former METHOD_CENTRAL_CLAIMS.md was folded into it). The correlation score is the proposed principal measure of field-shape stability across heading, subject to Claim M and to the comparison claims (01 Claim 8; 02 leaves 3.7–3.8; 03 Claim 7).

- [Objectives and experience](01_OBJECTIVES_AND_EXPERIENCE.md): intervene on training pressure and test changes in representation and computation.
- [Reference frames](02_REFERENCE_FRAMES.md): test whether a specified physical coordinate system explains apparent entanglement and is used by the model.
- [Biological resemblance and usefulness](03_BIOLOGICAL_RESEMBLANCE_AND_USEFULNESS.md): test which properties predict access, transfer, and native computational use.

Each can support a separate paper. Combining them requires evidence for each connecting claim; completing one does not establish the others. Start by critically reviewing the claim tree of the selected document, rather than treating its proposed headline as a result to obtain. Negative and dissociative outcomes are explicitly allowed.

The original [research review](../RESEARCH_REVIEW.md), [Wei-program review](../WEI_RESEARCH_PROGRAM.md), and [transformer-mechanism review](../TRANSFORMER_MECHANISMS_DIRECTION.md) provide wider context. They are useful reading, but essential assumptions and implementation requirements are repeated in this packet.

## Repository identity and provenance

**REPO_ROOT = `/Users/moo/variance_analysis`** is the original scientific project. At inspection its Git HEAD was `c04cac25e7f3faafbe7113f8e97dd5745a6a33ef` and `git status --short` was empty. **REVIEW_ROOT = `/Users/moo/variance_analysis/research_review`** contains these documents. `/Users/moo/explained_variance` is a scrubbed copy, not a Git repository at inspection. Its terminology is anonymized and it omits important original assets. Do not silently substitute it for the original biological pipeline.

The reviewed paper is [manuscript.pdf](/Users/moo/variance_analysis/Spatial_maps_in_CNNs/manuscript.pdf), with source [manuscript.tex](/Users/moo/variance_analysis/Spatial_maps_in_CNNs/manuscript.tex) and bibliography [map.bib](/Users/moo/variance_analysis/Spatial_maps_in_CNNs/map.bib). The PDF was previously verified byte-identical to `Spatial_maps_in_CNNs_revised_20260914.pdf` in the scrubbed copy, SHA-256 `7e8f5d056e6ec626238ecd3fab7f46e2a5e944a9616efc9ea86ca17e5c3febf7`. This establishes which manuscript was reviewed, not provenance of every figure.

The author clarified that `notebooks/binning_bias_fig.ipynb` is not part of the intended final paper. Its notebook-specific defects are not evidence against final-paper experiments. The estimator’s algebraic interpretation is a separate issue. If simulation/noise-floor claims remain in the manuscript, identify the actual generating implementation before evaluating them.

[ASSET_MANIFEST.json](ASSET_MANIFEST.json) records 34 selected original paths, existence, sizes, small-file hashes, and selected NumPy archive headers. Inspection did not load checkpoint weights or execute the model, and array-header inspection did not validate array values. Relative paths in the manifest are relative to REPO_ROOT. Update this root on another machine; preserve the manifest as the baseline snapshot.

## What is actually reusable

| Asset under REPO_ROOT | Verified contents or interface | Appropriate reuse and limit |
|---|---|---|
| `notebooks/model_neurons.ipynb` | Model loading, activation extraction, conditional maps, layer comparisons and figures | Recover the original analysis and its settings. Extract a small executable pipeline; do not execute all cells blindly. It includes Colab mounts, package installation and system font commands. |
| `src/ann/get_activations.py` | `load_model`, predictive-model wrappers, ResNet/VGG/ViT paths, cached activations | Reuse preprocessing and layer access after verifying against the checkpoint. Model-library weights may require downloads; the project metadata does not pin their versions. |
| `src/ann/predictive_coder.py` | `PredictiveCoder`, encoder/decoder, two attention blocks and feed-forward blocks; additional classes also present | Candidate small mechanism study. The loader selects `PredictiveCoder`; the existence of other classes does not establish that their weights were trained or used. |
| `data/weights/predictive_coder.ckpt` | File present, 6,128,789 bytes; hash recorded | Starting checkpoint, subject to strict loading, output sanity checks and training-provenance recovery. File presence is not a successful reproduction. |
| `data/minecraft/panorama-90yaw/` | Per-location archives with RGB, yaw, pitch and position | Frozen-scene heading tests and a possible explicitly constructed discrete observation environment. These are views, not recovered natural trajectories. |
| `data/minecraft/panorama-90yaw-pitch/` | Per-location archives with 360 images and yaw/pitch arrays | Test pitch sensitivity and reconstruct view ordering. Do not interpret 360 images as 360 distinct yaw angles. |
| `data/ann_activations/predictive_coder_{pre_pred,post_pred}_panorama_clean/acts.npz` | Inspected clean archives: activations `(289, 90, 8000)`, locations `(289, 2)` | Screen conditional tuning and candidate readouts. Reconstruct heading order, unit selection and history protocol before reuse. No interventions can be established from these frozen arrays alone. |
| `data/ann_activations/predictive_coder_pre_pred_panorama_pitch/acts.npz` | Activations `(289, 360, 8000)`, locations `(289, 2)` | Screen nuisance sensitivity; inspect source to reconstruct pitch groups. Raw activation array is approximately 3.33 GB before copies. |
| `results/ann/predictive_coder_pre_pred/ev_all_neurons_pitch{,_corr}.npz` | `ev_xy`, `ev_hd`, `is_place`; 8,000-element vectors; one variant also has `n_neurons` | Historical output reconciliation. The `ev` names do not establish a variance interpretation; saved flags do not establish correct cell selection. |
| `src/explained_variance.py` | `quantify_variance_corr`, `_corr_matrix`, masked smoothing/zoom and splitting helpers | Recover and calibrate a conditional shape-correlation statistic. Do not make it the sole invariance, interaction, information, or usefulness metric. |
| `src/ann/classify_neurons.py`, `src/ann/score_utils.py` | Classification and scoring infrastructure | Candidate definitions for descriptive biological resemblance. Reconcile selection rules and independently choose evaluation samples before comparing performance. |
| `notebooks/bio_neurons.ipynb` | Pfeiffer and Duszkiewicz pipeline calls, cached plots and summaries | Starting point for biological validation. Hard-coded paths and notebook execution history are not a reproducibility manifest. |
| `src/pfeiffer_ratemaps.py`, `src/dusz_ratemaps.py` | Rate-map builders and loaders | Recover sampling, speed, selection and time-window conventions. Check program defaults against explicit notebook arguments and actual directories. |
| `src/pfeiffer_decode_solo.py`, `src/dusz_decode_solo.py` | Existing decoding implementations | Baselines and input handling, not demonstrated transfer tests or causal analyses. Audit splits, priors and time alignment. |
| `data/rodents/pfeiffer/Rat1` … `Rat4` | Original data directories present; Rat1 includes `Open1` and `Open2` | Potential within-session population analysis. Confirm simultaneity, identifiers and timing in loaded data before combining cells. |
| `data/rodents/dusz/Dataset_1`, `Dataset_2` | Original inputs present, with `Dataset_metadata.xlsx` in parent directory | Resolve region, session, animal and environment identity from metadata and source papers. Do not infer these solely from folder names or the older README. |
| `results/rodents/pfeiffer/Rat1/fold0_xyhd_20x20x60.npz` | `rate_map (20,20,60,204)`, `occupancy (20,20,60)` and coordinate edges | A concrete cached pilot input. Its companion metadata records tau 0.25, speed threshold 5.0, speed filtering, 20 position bins per axis and 60 heading bins; units and full fold provenance still need source verification. |

For a representative clean panorama archive, `rgb` is `(90,64,64,3)` uint8; `yaw` and `pitch` are each `(90,)`; `position` is `(3,)`. The pitch archive counterpart has 360 rows. Read these arrays explicitly instead of inferring degrees, sign, handedness, Cartesian axes or pitch groups from filenames.

Large compressed `.npz` files do not provide ordinary memory-mapped access to their contained arrays. A second copy, float64 conversion, or several workers may multiply memory usage. Inspect headers first, then select a bounded session/view/unit subset or create a documented chunked derived store. Preserve source files.

### Assets that have not been established

The inspected analysis paths do not establish a reproducible training command, optimizer schedule, checkpoint ancestry, a live scene renderer, landmark coordinates, collision geometry, or natural trajectory/action logs paired with every checkpoint. This is a recovery list, not a claim that the entire original project or upstream project lacks them. Search and verify before rebuilding.

A useful fallback is a **new, explicitly declared discrete pose-graph environment** using saved images as observations. Legal transitions must be specified and checked; adjacent image locations alone do not prove a navigable edge. Synthetic paths from this graph are new training/evaluation data, not authentic trajectories from the original model. Moving landmarks or testing new geometry requires additional rendering or a different controlled environment. A new environment can test a hypothesis but cannot retroactively validate the original scene-specific claim.

## Model and measurement gates before hypothesis tests

### G1. Verify the actual trained model

`_make_predictive_coder` loads weights only if the supplied path exists; otherwise it returns a randomly initialized model without raising an error. Require an existing checkpoint, record its hash, use strict key/shape matching, and explicitly label untrained controls. `load_model(..., output_layer='pre_pred'/'post_pred')` returns a wrapper for latents. Mechanistic evaluation needs access to the core model’s native prediction and the correct input/target alignment, not just wrapper output.

The selected `PredictiveCoder.get_latents` has an `actions` argument that is not used in its body. The loaded `PredictiveCoder.forward` takes images. Do not silently add action conditioning and attribute a resulting change only to loss. Some latent paths are under `torch.no_grad`; activation replacement may still be possible, but gradient-based methods require an intentional interface change and revalidation.

### G2. Verify temporal semantics

The generic activation extractor repeats each static image ten times. This supplies one specific history, not a test of normal navigation. Reconstruct genuine or explicitly synthetic histories and vary incoming paths at matched endpoints.

An additional code-level concern is `F.layer_norm(f, f.shape[1:])` in `PredictiveCoder`, where `f` has shape `[B,L,C,H,W]`. Its normalization spans sequence positions. Earlier-position activations can therefore depend on later supplied frames despite masked attention; training-mode encoder normalization also needs checking. **This has not been measured here and is not a demonstrated defect in a historical experiment.** Last-token evaluation has no later supplied frame, but interpreting an intermediate temporal circuit or per-token training target requires a stronger check.

Before temporal-mechanism claims, perturb suffix frames while holding a prefix fixed and test earlier outputs; inspect target alignment, train/eval modes, normalization and mask dimensions. Distinguish the complete-model behavior from an assumed causal-transformer abstraction. Recover upstream semantics before altering normalization; changing it can invalidate checkpoint comparability. A newly trained corrected model is a new experimental condition, not a silent repair of historical evidence.

### G3. Specify the estimand

The historical plotted dissimilarity is based on one minus mean cross-fold Pearson correlation of conditional curves. The function itself returns mean off-diagonal correlation first and mean diagonal repeat correlation second. It is not a variance partition. Positive gain and additive baseline invariance is deliberate for the conditional-shape target; constant slices can make Pearson undefined, and independent folds do not automatically remove denominator noise attenuation. Cross-variable scores near one are not evidence of equal explained-variance contributions. Select the heading axis to compare spatial profiles across headings; selecting position answers the reciprocal question.

Use separate quantities for response modulation, field-shape stability, context-general readout, and native computational use. Validate every chosen score on positive controls and plausible adversarial controls: pure one-variable tuning, gain-only modulation, additive mixing, a genuine interaction, sparse support, heteroscedastic noise, and a population basis change where appropriate. Do not reuse the excluded binning notebook as final-paper validation without an independent reason and new provenance.

### G4. Verify geometry, support and selection

Preserve the true two-dimensional position grid and circular heading. Smoothing a flattened 289-position axis does not implement spatial smoothing. Pitch groups are deterministic nuisance changes rather than interchangeable independent noise repeats. Match support, report excluded conditions, and select units independently of the response differences or transfer scores being tested. Different biological and artificial environments permit a within-system comparison of defined properties; they do not supply matched stimulus-by-stimulus neural/model similarity.

## How to attack a hypothesis fairly

Use five distinct outcome labels for each claim: **supported at stated scope**, **contradicted at stated scope**, **equivalent within the meaningful margin**, **inconclusive**, and **not tested**. Evidence for a necessary prerequisite is not evidence for the downstream claim. A branch-specific failure does not disprove all possible coordinate systems, all objectives, or all meanings of cognitive map.

Before confirmatory analysis, specify the tested population/conditions, primary endpoint, expected directional contrast if any, smallest scientifically meaningful difference, interval/uncertainty method, family of alternatives, and exclusions. Set tolerances using resolution, repeatability, control separation and the intended computation. Mark exploratory pilot choices as exploratory; freeze them before testing fresh seeds, sessions or conditions. Proposed numeric margins in a narrative document are design starting points, not validated biological constants.

For a claim of meaningful improvement, a precise interval within the prespecified negligible-effect range can defeat the claim. A non-significant p-value alone cannot. For a universal necessity claim, one valid counterexample may suffice; for a probabilistic predictive claim, estimate out-of-sample predictive value across the appropriate independent units. Replication units are usually models/seeds, animals, sessions or environments—not thousands of neurons sharing one outcome.

Use blocked trajectory/time splits with a gap appropriate to smoothing and history length. Fit preprocessing, coordinates, probes, selection rules and hyperparameters on training data. Preserve a final test set that did not choose the narrative. In neural analysis, include occupancy/behavior baselines and sufficient common support. In intervention analysis, test specific output changes, controlled donor/recipient pairs, appropriate negative controls and recovery where feasible; undifferentiated damage does not identify a mechanism.

The mechanism claims are scoped to the chosen model, input distribution and intervention. Multiple pathways or compensation may make a single ablation inconclusive. A successful edit can also create a new computation; check whether the candidate is active and functionally connected during unmodified execution. These distinctions should appear in the claim record, not be introduced only after an unwelcome result.

## Implementation and result contract

No experiment runner described by the plans has been implemented as part of this document task. Existing module names above are real; proposed pipeline stages and output names below are requirements for the next implementation, not commands that can already be run.

1. **Inventory and smoke test:** resolve paths and versions; hash checkpoint/code; load a bounded input; verify axes, normalization, prediction target and model output; record failures before evaluating hypotheses.
2. **Dataset manifest:** for every example record `sample_id`, `environment_id`, `trajectory_id`, time/index, position, heading convention, pitch, available landmarks, context length/history identity, source file and split. Record model/checkpoint/seed/objective separately. Biological records additionally need animal/session/unit IDs and time-bin conventions. Distinguish unknown from absent values.
3. **Frozen analysis configuration:** save method version, population target, context/profile axes, support and weighting, `S`/`D` convention, handling of non-estimable slices and calibration. Include close assay comparators and the specific practical advantage under test. Save coordinate candidates, controls, thresholds, selection rules, primary endpoints, uncertainty and multiple-comparison handling. For training, save objective equations, loss weights, schedules, matched inputs and compute/exposure accounting.
4. **Pilot diagnostics:** use inexpensive subsets to test identifiability, dynamic range and runtime. Preserve unsuccessful pilots. Do not use their sample count as confirmatory replication.
5. **Confirmatory results:** save per-example predictions/errors, aggregate contrasts, independent-unit uncertainty, split hashes and claim statuses. All figures should be generated from these outputs without manual data edits.
6. **Critical interpretation:** for each claim record which alternatives remain, what would reverse the assessment, and whether the narrative should narrow, change or stop.

A minimal result record can use fields `claim_id`, `scope`, `primary_endpoint`, `effect_estimate`, `uncertainty_interval`, `meaningful_margin`, `control_results`, `independent_n`, `status`, `alternative_explanations`, and `next_decision`. A missing margin or unavailable control should be explicit, not encoded as zero. Save `config.json`, `data_manifest`, `results`, `claim_status.json`, and figure provenance within a new experiment output directory. Exact file formats may follow the implementation’s needs.

The strongest handoff to a subsequent agent is this packet plus the original repo, with an instruction to **audit the selected claim tree and first attempt its cheapest decisive falsifier**. Do not assume all three narratives must survive. A well-supported reason to abandon a branch is a successful scientific outcome of this process.
