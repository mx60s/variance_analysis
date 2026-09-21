# Reference frames: does apparent spatial entanglement reveal a useful computation?

Research hypothesis and implementation brief, 14 September 2026. **These are proposed experiments, not new findings.** Read with [PROJECT_HANDOFF.md](/Users/moo/variance_analysis/research_review/hypothesis_plans/PROJECT_HANDOFF.md); the essential source paths and scientific requirements are repeated here so this document also works with the original repository alone. Claim identifiers F1–F6 are stable labels for preregistration, results, and critique.

## Broad narrative and scope

**Method-centered revision:** conditional shape correlation is the proposed discovery assay. It should nominate a physical-frame hypothesis whose identifying predictions can then be tested independently. The shared [M1–M4 claims](METHOD_CENTRAL_CLAIMS.md) require a defined shape target, adequate estimation, comparative value and independent scientific validation. This is stronger than merely including the old score in a figure and weaker than assuming it can discover any reference frame. F0 states the method's application-specific role.

The original paper asks whether units that resemble place or head-direction cells retain their tuning when another spatial variable changes. A promising alternative interpretation is that some conditional changes are expected consequences of a different physical coordinate system. A response to an object at a particular observer-relative bearing can look complicated in laboratory position × heading coordinates while implementing a simple spatial relation.

The proposed question is **whether a predictive visual transformer's apparent entanglement contains a compact relational computation that predicts unseen transformations and contributes to its own predictions**. The result must remain open: fixed visual templates, distributed world-centered codes, genuine interactions, or history-dependent uncertainty could explain the observations better. We should not search indefinitely until some coordinate system works.

This connects the coauthors' mechanistic-interpretability and computational-neuroscience interests through Wei's work on learned navigation circuits and transformations of population states. It can stand alone without changing the training objective. A successful objective comparison would be an extension linking this document to the first narrative; a biological resemblance–utility comparison belongs to the third. Neither is required to establish one convincing mechanism.

The smallest compelling paper would show: a reproducible conditional response pattern; an independently testable physical interpretation; and a selective intervention connecting that interpretation to native model behavior. The method-centered version also requires useful assay-guided selection under F0 and a defensible comparative-value claim under M3. A stronger paper would explain how history updates that state and why it supports transfer. Simply obtaining a better coordinate probe is insufficient for this narrative.

## Relevant prior work and the novelty boundary

- **Nanda, Lee & Wattenberg (2023), “Emergent Linear Representations in World Models of Self-Supervised Sequence Models.”** OthelloGPT admits a simpler mine/yours than absolute-color description, with vector interventions affecting model behavior. This already establishes the value of choosing semantic coordinates carefully. Our possible contribution is a physical transformation and its learned implementation in visual prediction. [Published paper](https://aclanthology.org/2023.blackboxnlp-1.2/).
- **Vijayabaskaran & Cheng (2022), “Navigation task and action space drive the emergence of egocentric and allocentric spatial representations.”** Task and action space shape spatial responses and generalization in visual RL agents. Task-dependent frames are established; identifying an internal transformation in this transformer would be more specific. [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010320).
- **Whittington, Warren & Behrens (2022), “Relating transformers to models and neural representations of the hippocampal formation.”** The transformer–TEM connection uses structured position and sensory pathways. It is a strong comparator for relational memory, but not evidence that our less constrained checkpoint implements its factorization. [ICLR paper](https://arxiv.org/html/2112.04035).
- **Høydal et al. (2019), “Object-vector coding in the medial entorhinal cortex.”** Object displacement and object/environment changes provide an operational test of relative spatial tuning. Origin and directional axis must both be specified: object-centered does not mean egocentric. The accessible abstract and figures support this comparison. [Nature](https://www.nature.com/articles/s41586-019-1077-7).
- **Piray & Daw (2025), “Reconciling flexibility and efficiency: medial entorhinal cortex represents a compositional cognitive map.”** Their model links reusable object-related representations to flexible planning and neural response types. A generic claim that relative maps support transfer would overlap substantially; a discovered transformer operation would add a different kind of evidence. [Nature Communications](https://www.nature.com/articles/s41467-025-62733-7).
- **Geiger et al. (2025), “Causal Abstraction.”** This supplies a vocabulary for comparing an hypothesized computation and neural-network interventions. It need not become a new methodological contribution. Patching requires controls: successful subspace edits alone can admit misleading interpretations, as the interpretability-illusion debate illustrates. [JMLR](https://www.jmlr.org/papers/v26/23-0058.html), [Makelov et al.](https://arxiv.org/abs/2311.17030), [Wu et al. reply](https://arxiv.org/abs/2401.12631).

Wei's [learned HD circuit study](https://arxiv.org/html/1912.10189), [group-representation grid model](https://arxiv.org/abs/2006.10259), and [landmark-driven HD dynamics study](https://www.nature.com/articles/s41586-023-05813-2) motivate testing transformations and computational roles rather than ranking units by purity. This is thematic fit, not knowledge of his present preferences.

## Existing work: reusable material and limits

Use `/Users/moo/variance_analysis` as the original project. `/Users/moo/explained_variance` is its anonymized analysis copy and the location of these planning documents.

| Asset | Reuse | What it does not establish |
|---|---|---|
| [manuscript.tex](/Users/moo/variance_analysis/Spatial_maps_in_CNNs/manuscript.tex), [map.bib](/Users/moo/variance_analysis/Spatial_maps_in_CNNs/map.bib) | Original scientific motivation, conditional-tuning examples, literature context | Final support for a new frame or circuit claim |
| [model_neurons.ipynb](/Users/moo/variance_analysis/notebooks/model_neurons.ipynb), [run_model_layers.py](/Users/moo/variance_analysis/notebooks/run_model_layers.py) | Recover labels, plots, and baseline extraction; notebook documents 17 × 17 positions, 90 headings, four pitches | Existing predictive extraction repeats each static image ten times and reads the last token; it is not natural-history evaluation |
| [panorama-90yaw](/Users/moo/variance_analysis/data/minecraft/panorama-90yaw), [panorama-90yaw-pitch](/Users/moo/variance_analysis/data/minecraft/panorama-90yaw-pitch) | Fixed-scene screening and current-view controls; location filenames permit metadata reconstruction | No verified landmark-motion counterfactuals or independent repeated trials; pitch changes are nuisance interventions |
| [cached predictive activations](/Users/moo/variance_analysis/data/ann_activations/predictive_coder_post_pred_panorama_clean/acts.npz), corresponding `pre_pred`, `conv1`, `down1`, and pitch directories | Fast candidate discovery and cache reproduction | Full component states, native outputs, causal evidence, or arbitrary missing unit identities |
| [get_activations.py](/Users/moo/variance_analysis/src/ann/get_activations.py), [predictive_coder.py](/Users/moo/variance_analysis/src/ann/predictive_coder.py), [checkpoint](/Users/moo/variance_analysis/data/weights/predictive_coder.ckpt) | Preprocessing, model definition, two attention modules, two FFNs, encoder and native image decoder | Successful checkpoint recovery, training-target alignment, and renderer recovery still require explicit tests |
| [explained_variance.py](/Users/moo/variance_analysis/src/explained_variance.py), [classify_neurons.py](/Users/moo/variance_analysis/src/ann/classify_neurons.py) | Legacy baseline and unit selection for comparison | Pearson shape dispersion is not a general variance fraction; gain and offsets can disappear from it |
| [bio_neurons.ipynb](/Users/moo/variance_analysis/notebooks/bio_neurons.ipynb), biological ratemap/decoder scripts in `src` | Optional comparison of response properties | No causal identification of the same mechanism in rodents; existing session metadata alone do not imply manipulated landmarks |

The prior audit is [RESEARCH_REVIEW.md](/Users/moo/variance_analysis/research_review/RESEARCH_REVIEW.md). Its legacy binning-notebook defects are **not assumed to affect the intended final paper**. The important current issues are distinct: correct XY geometry rather than smoothing a flattened location index; distinguish gain from shape; treat pitch folds as different conditions; verify cache provenance and unit selection. Reproduce or replace only the measures required here.

Files for old image collections also exist under `data/minecraft/small_world_r17` and `small-world-r360`; ordering and trajectory provenance are unverified. Do not treat numbered images as a verified trajectory. No renderer, movable-landmark scene manifest, or faithful training loop was located in the inspected source tree. Recover upstream material or build a clearly labeled new controlled environment. A new model is a successor experiment, not a replication of the original checkpoint.

## Coordinate hypotheses and identifying interventions

Let position be $p_t\in\mathbb R^2$, heading $h_t$, landmark $j$'s location $l_j$, environmental directional anchor $a$, and planar rotation $R(\theta)$. Prespecify this finite family before evaluation:

1. **World coordinates:** $p_t$, with circular heading $(\cos h_t,\sin h_t)$.
2. **Landmark-centered, world-oriented vector:** $q^A_{j,t}=l_j-p_t$.
3. **Observer-relative landmark vector:** $q^E_{j,t}=R(-h_t)(l_j-p_t)$, or its distance and sine/cosine bearing. These Cartesian/polar encodings are robustness checks for one physical hypothesis, not independent discoveries.
4. **Environment-anchored vector:** $q^C_t=R(-a)(p_t-o)$, with a prespecified origin $o$ and anchor direction $a$.
5. **Boundary relation:** distance and observer-relative bearing to a specified boundary segment; nearest-boundary identity must be included when it switches.

Compare with additive position/heading models, gain-modulated place fields, equally regularized interaction models, current-image features, and visual-template controls. Use at most two named landmarks initially. Do not optimize landmark location on the test set or introduce a new frame after observing its failures without declaring an exploratory follow-up.

**These are physical hypotheses, not arbitrary latent rotations.** A fixed invertible activation change preserves the existence of linear readouts, but says nothing about landmark anchoring. Freely fitting a different alignment at every heading can absorb interactions without explaining them. A physical hypothesis specifies responses to scene interventions:

| Intervention | Observer-relative prediction | Discriminating purpose |
|---|---|---|
| Turn by $\Delta h$ at fixed $p,l$ | $q^{E\prime}=R(-\Delta h)q^E$ | Separates observer- from world-oriented quantities |
| Move landmark by $\Delta l$, fixed observer | $q^{E\prime}=q^E+R(-h)\Delta l$ | Separates landmark anchoring from fixed-world location |
| Translate observer and landmark together | Same $q^E$, subject to changed background | Distinguishes relative relation from unrelated scene templates |
| Change appearance while preserving geometry | Same proposed spatial coordinate | Tests identity/texture dependence |
| Hide landmark after different informative histories | Coordinate prediction depends on remembered state | Tests persistent state rather than visible-template detection |

At a fixed landmark location, $l-p$ is an invertible reparameterization of $p$; observational fits cannot identify the anchor. Rotating all visible cues and the observer together may preserve the entire input. A visual-only model cannot infer an unobservable absolute north. Design cue conflicts or local edits whose inputs and competing predictions actually differ. Even a true egocentric bearing detector may be purely visual; persistent spatial state is a stronger, separate claim.

## Shared experimental specification

**Primary discovery assay:** define each candidate physical profile variable and the residual context variable before scoring. Compute expected-profile similarity `S`, estimated by cross-fold correlations, and optionally report `D=1−S`. For spatial fields across heading, the compared-slice axis is heading. Freeze common physical samples, domain, resolution, smoothing and pair weighting; reject raw rankings created by unequal overlap or by a coordinate transform that makes profiles constant. A new frame can change the estimand even when the same correlation formula is used. Use matched-support simulations to establish that candidate rankings are interpretable, and allow abstention when comparable domains cannot be defined. Fitted response models below are independent checks and fair competing discovery methods.

Preserve each observation's scene, physical pose, cue configuration, appearance, history, temporal index, split, and valid counterfactual pair. Train/validation/test separation must occur at trajectory and scene/cue-configuration level before constructing overlapping windows. Reserve novel angle–position combinations and transform compositions. Freeze preprocessing, frame family, feature dictionary, regularization selection, intervention strength, and candidate unit/subspace selection on training/validation data.

Use a capacity ladder for encoding models—matched-size spline/Fourier dictionaries with ridge regularization followed by a matched nonlinear model. Report error versus complexity rather than a single favorable probe size. For activation prediction, use held-out standardized MSE or $R^2$ against training-set centering/scaling; report reliable variance separately if justified. Deterministic network repeats do not create a biological noise ceiling. For outputs, use verified native prediction loss plus prespecified geometric attributes where independently measurable. A frozen evaluation network is an auxiliary metric, not ground truth.

**Pilot decision margins:** choose practically meaningful margins on development data, then lock them before confirmation. A useful initial target is a 5% relative error reduction over the strongest prespecified baseline for representation/transfer comparisons. For targeted output edits, use an improvement of 10% of the clean-recipient–valid-counterfactual output gap, with off-target degradation below a separately set tolerance. These are planning defaults, not established scientific thresholds; replace them if pilot measurement precision or task relevance makes them unsuitable, documenting the change before held-out testing.

Use paired, cluster-aware intervals over independent trajectories and scenes. One checkpoint supports a checkpoint-specific claim. For a learned-mechanism claim across training runs, aim for five independent seeds after a one-seed feasibility pilot; determine scene/trajectory counts by simulation showing at least 80% power at the locked smallest effect and adequate equivalence precision. Do not count units or video frames as independent model replications. A confidence interval spanning both negligible and meaningful effects is unresolved, not a killed hypothesis.

## Structured claims and attacks

### F0. The correlation assay usefully selects a testable physical-frame hypothesis

**Claim:** under a fixed finite candidate family, a calibrated change in conditional shape agreement can guide frame selection that succeeds on identifying physical interventions, with a demonstrated practical or explanatory benefit over prespecified alternatives.

**Current evidence:** the correlation implementation and conditional fields exist; no frame-selection validity has been demonstrated. Fixed-arena observations are discovery data only.

**Test/support:** calibrate candidate scores using known physical-frame synthetic populations sampled on matched support. On discovery scenes, freeze a candidate-selection rule using the score and compare it with marginal field descriptions, a close shape/separability method and matched-complexity encoding-model selection. Evaluate selected candidates on new landmark/observer transformations, using response predictions and native outputs rather than simply rescoring transformed data. Record false selections, abstentions and precision against a declared practical margin. Successful independent selection and a defensible M3 advantage support the claim.

**Best killing result:** the assay confidently prefers incorrect frames in calibrated synthetic or new physical interventions, or rankings reverse under innocuous resolution/support choices while identifying predictions favor another account. If equal-quality established methods select the same hypotheses with no demonstrated additional benefit, methodological distinctiveness fails even if F2 succeeds.

**Confounds/branch:** lower dispersion is not intrinsically a better coordinate system, and different transformed domains may not admit a common score comparison. An arbitrary rebinning cannot supply a physical explanation. If F0 fails, a frame mechanism discovered by another approach can remain valuable, but it is not validation of this method as the discovery contribution. F1–F6 remain separately testable.

### F1. The motivating conditional variation is reproducible response structure

**Claim.** At a declared layer and input protocol, a nontrivial, held-out set of responses changes with heading at fixed location beyond offset/gain changes and measurement artifacts.

**Current evidence:** legacy plots and caches motivate this; static-history extraction, metric meaning, and provenance limit interpretation. No new replication has been run.

**Test and supporting result:** re-extract matched endpoints under natural histories and static repeats; quantify raw dependence, gain/offset, and residual shape separately. Fit $r(p,h)=a(h)f(p)+b(h)$, then compare a matched-capacity interaction alternative on held-out samples. Visualize physical XY grids. A reproducible residual effect exceeding the registered margin supports the motivation.

**Best killing result:** the apparent shape changes disappear under correct geometry, valid extraction, and gain/offset controls, with an interval excluding meaningful residual variation. A static-only effect that disappears under representative histories kills the claim about normal temporal operation, though not the static observation.

**Confounds/power:** select units on training data; include all selected units in test summaries. Sparse support, unsuitable smoothing, and low firing/activation variance can mimic effects. Synthetic additive, multiplicative, and interaction codes must distinguish those cases at the observed support.

**If killed:** revise toward stable fields with modulation, or abandon the “apparent entanglement” premise. Do not manufacture a frame rescue for an artifact.

### F2. A prespecified physical frame explains and predicts that variation

**Claim.** At least one finite-family relational coordinate gives a simpler, transferable account than fixed-world and image-template alternatives, including environmental interventions that identify its anchor.

**Current evidence:** no identifying landmark manipulation exists in the reviewed project. Fixed-scene correlations are screening only.

**Test and supporting result:** fit response functions on several cue configurations; predict responses to unseen landmark positions, headings, appearance variants, and combined transformations without refitting. Require advantage at matched complexity and consistent physical predictions. A low-dimensional state may follow a rotation/translation rule even if individual units do not become “pure.”

**Best killing result:** the selected frame systematically predicts the wrong direction of response change when the landmark moves, while a fixed-world or appearance model succeeds. Alternatively, sufficiently precise held-out comparisons show no meaningful benefit beyond the strongest baseline across the registered family.

**Confounds/power:** novel edits may cause generic distribution failure. Verify native outputs remain informative; include small supported edits and oracle-coordinate synthetic positives. Match landmark visibility and coordinate coverage. Composition tests use actual scene transforms, not merely fitted matrix products that satisfy identities by construction.

**If killed:** reject this finite family and scope; retain generic interaction or visual-feature explanations. Another unconstrained coordinate search is a new exploratory study, not confirmation.

### F3. The candidate state carries a remembered relation beyond the current view

**Claim.** Where prediction requires information absent from the current image, the model retains the proposed relation from history. This claim is optional for a visible relational code, but mandatory for calling it a persistent spatial state.

**Current evidence:** static repeated-image caches cannot support this claim.

**Test and supporting result:** construct matched terminal images with histories that differ in a previously visible landmark relation; choose a continuation distribution with distinct valid target predictions. Test decodability and native output dependence during occlusion, across held-out histories. Compare single-frame and shuffled-history baselines.

**Best killing result:** validated histories contain useful disambiguating information, a capable history baseline with the same supplied inputs, context length and target distribution exploits it, yet the model's state and outputs are insensitive to it with tight equivalence bounds. A privileged pose/landmark/action oracle is only a task-validity ceiling and cannot establish observability from the checkpoint's inputs. If the apparent relation tracks only current pixels and vanishes under appearance controls, the persistent-state interpretation fails.

**Confounds/power:** do not require an observation-only predictor to know an unprovided future action. Recover training horizon and conditional target distribution. The model's temporal normalization can expose earlier tokens to later supplied frames; test prefixes separately before making causal online claims. Last-token evaluation alone does not automatically suffer future-input leakage.

**If killed:** a visible relational-feature mechanism may still justify F2/F4. Drop memory, latent-map maintenance, and path-integration language.

### F4. A specific component-level operation uses the coordinate in native prediction

**Claim.** An identifiable pathway writes, transforms, or reads the candidate state, with counterfactual effects consistent with the proposed operation.

**Current evidence:** accessible modules make intervention possible; no causal evidence exists yet. `get_latents(..., actions=None)` in the loaded `PredictiveCoder` does not use actions; another action-related class in the file does not establish action input to this checkpoint.

**Test and supporting result:** choose real donor/recipient histories with simulator-grounded counterfactual targets. Hook encoder outputs, attention outputs, FFNs, residual additions and post-normalization states. Patch a selected state, ablate its proposed pathway, and test restoration. Measure the original decoder's output toward the predicted counterfactual, preserving unrelated attributes. Require same-state donors, reverse swaps, matched-norm random/irrelevant subspaces and complementary-path controls. Confirm held-out examples with a frozen patch rule.

**Best killing result:** targeted edits demonstrably alter the proposed state or operation, with independently validated specificity and sensitivity, yet leave native outputs unchanged; or all output changes are generic damage matched by controls. Changing only an auxiliary probe score is insufficient to establish that the operation was disrupted. A rival pathway provides stronger contrary evidence when valid interventions hold the proposed state fixed while varying the rival and reproduce the predicted effect. Statistical conditioning alone is competing explanatory evidence, not causal control. Failure of one patch method alone is not enough to reject all distributed implementations.

**Confounds/power:** hybrid states can be off-manifold; restoration can exploit decoder artifacts. Positive controls must show native outputs can change appropriately, and interventions must genuinely change the target state without replacing the whole computation. An attention visualization is not this test.

**If killed:** retain an encoding description if F2 survives; remove the mechanistic claim. A modest descriptive paper needs another substantive result.

### F5. The identified operation contributes to a specified generalization advantage

**Claim.** The same operation helps native prediction transfer across a declared transformation, beyond generic accuracy or access by an external decoder.

**Current evidence:** none. This is an extension beyond coordinate identification.

**Test and supporting result:** choose one primary transfer axis, such as novel heading–landmark-position combinations. Compare intact, targeted-intervention and matched-damage control models. Predict in advance which transfer condition should worsen and which matched familiar condition should remain less affected; seek restoration of that selective loss. Separately report external position readout if useful, labeling it an access test.

**Best killing result:** the operation has a clear native role under familiar conditions but selective removal leaves transfer equivalent after controls, or its supposed benefit disappears when baseline prediction quality is matched. A code can be used without causing the proposed transfer advantage.

**Confounds/power:** equal aggregate damage does not guarantee identical removed information. Test multiple intervention magnitudes and valid natural counterfactuals; avoid asserting unique mediation from one lesion. Match support and quantify clean transfer performance before perturbing it.

**If killed:** publish the operation, if F2/F4 are strong, without the usefulness explanation. Pass the dissociation to the biological-resemblance narrative as a constraint.

### F6. The account replicates beyond a selected scene or checkpoint

**Claim.** The physical rule and operation recur within a stated model/training family, with tolerable variation in unit identities and latent basis.

**Current evidence:** one original checkpoint and one main fixed arena do not establish this.

**Test and supporting result:** repeat frozen selection and testing across independent seeds and held-out scenes. Allow train-only alignment of feature identity while keeping physical predictions and intervention logic fixed. Report prevalence and failed runs, not only exemplar cells.

**Best killing result:** the rule works only in the discovery scene/seed and fails precise replication despite comparable native performance and supported stimulus variation. Flexible scene-specific retraining of the explanatory model cannot rescue this claim.

**Confounds/power:** a seed may implement a different mechanism; that heterogeneity is a result. A method incapable of finding rotated/distributed versions of known synthetic states lacks the sensitivity needed for a strong negative.

**If killed:** restrict the paper to a transparently selected mechanistic case study, or study solution multiplicity. Do not claim a universal transformer reference frame.

## Execution gates, deliverables, and paper decisions

**Gate 0: recover a trustworthy executable.** Assert checkpoint existence and strict loading; the current loader otherwise silently returns a random model. Log file hash, repository revision, dependency versions, preprocessing, full tensor shapes, and native output target alignment. Compare `forward` and latent-hook reconstruction numerically. `F.layer_norm(f, f.shape[1:])` spans sequence and feature dimensions: test suffix sensitivity of earlier outputs and padding/prefix behavior before any temporal-mechanism inference. Do not silently fix it and describe the resulting model as unchanged. Training-mode encoder BatchNorm also requires review if retraining.

**Gate 1: inexpensive screening.** Establish M1/M2, reproduce selected cached examples, then run F1 using the calibrated correlation assay and independent field checks. Test F0's candidate-ranking behavior on known synthetic frames before applying it to existing images as exploratory screening. Deliver an asset manifest, method/version specification, comparator results, candidate registry and calibration curves. This can proceed before simulator recovery; it cannot confirm landmark anchoring.

**Gate 2: identifying scene manipulations.** Recover an editable renderer and history generator, or explicitly choose a new small controlled model. Produce scene manifests, valid paired histories, train/validation/test IDs, and a transform-prediction table before activation analysis. Run F2, adding F3 if persistent memory is intended. A failed observability or counterfactual-validity check stops the relevant test rather than becoming negative model evidence.

**Gate 3: mechanism and confirmation.** Freeze a candidate, execute F4, then prioritize F5 or F6 according to intended scope. Store interventions and native outputs for independent review. Proposed new deliverables are `frame_candidates.yaml`, `scene_manifest.jsonl`, `splits.json`, `encoding_results.parquet`, `counterfactual_pairs.jsonl`, `patch_results.parquet`, and `claim_decisions.md`; use a new experiment directory, leaving original assets intact.

Three figures could support a focused method-centered paper: assay characterization and fair candidate-selection comparisons; held-out physical transformations validating the selected frame; selective native-output intervention and restoration. A fourth transfer/replication figure expands scope. A successful F2/F4 mechanism without F0/M3 may support a mechanism paper, but not the proposed method-centered contribution. Figure captions must distinguish old observations, newly reproduced results, and new model experiments. See the shared method claims for close shape-assay citations as well as the physical-frame literature above.

Feasibility is presently highest for static screening, conditional for checkpoint/hooks, and uncertain for renderer/history recovery. No GPU/runtime benchmark was performed. Benchmark one extraction batch and one training run before giving compute estimates. If recovery dominates the project, prefer a clearly scoped successor experiment over weeks spent forcing unverifiable original provenance.

The decisive stopping rule is scientific: **if a frame neither predicts identifying interventions nor explains native behavior, this mechanism narrative has not earned a paper merely because the tuning looks simpler.** Conversely, a valid coordinate transformation with a selective computational role can be worthwhile at workshop scope even if it is not universal, history-based, or caused by a new objective.
