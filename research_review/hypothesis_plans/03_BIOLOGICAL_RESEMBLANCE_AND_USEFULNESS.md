# Plan 03 — Which biological resemblance predicts computational usefulness?

**Research proposal, September 14, 2026. No proposed result below has been demonstrated.** Read alongside [PROJECT_HANDOFF.md](/Users/moo/variance_analysis/research_review/hypothesis_plans/PROJECT_HANDOFF.md), the [original manuscript](/Users/moo/variance_analysis/Spatial_maps_in_CNNs/manuscript.tex), and the [claim audit](/Users/moo/variance_analysis/research_review/RESEARCH_REVIEW.md). This document specifies hypotheses to attack, rather than a conclusion to preserve. “Kill” means contradict the stated, bounded claim with a valid, sensitive experiment; an underpowered null or broken pipeline does not count.

## Broad narrative and contribution

**Method-centered revision:** the principal candidate biological property is conditional tuning-shape stability measured by the project's correlation assay. The question is whether that assay reliably isolates a useful distinction and whether it adds predictive or explanatory value beyond conventional cell labels and close existing shape measures. [M1–M4](METHOD_CENTRAL_CLAIMS.md) and U0 make those commitments falsifiable. Gain/baseline invariance is intentional; finite-data noise bias is not. Other field fits and population geometry are comparators and independent validators, not automatic replacements for the central assay.

Spatial models are often assessed by whether units resemble place or head-direction cells. Such resemblance is informative only relative to a specified question. Stable place-like fields might facilitate a location readout that transfers across heading; a mixed population might achieve the same result through a different organization; and heading-dependent modulation might support prediction rather than obstruct localization. We propose asking **which measurable biological properties predict a particular computational advantage, and whether those properties identify operations that the model actually uses.**

The original project already contrasts conventional spatial selectivity with conditional tuning in visual models and rodent recordings. Its missing bridge is from those observations to function. This direction retains that empirical starting point while making its interpretation falsifiable. It also gives transformer mechanistic interpretability a central role: separate an external experimenter's ability to decode location from the transformer's use of a spatial state to generate outputs.

The broad principle is established. Kriegeskorte and Wei explain why distinct neuronal tuning can induce equivalent population geometry. O’Shea and colleagues' 2025 and 2026 visual-cortex studies connect orientation stability across luminance to retained luminance coding and population organization. They are closely connected studies, not independent replications. A spatial paper must contribute a new, controlled relationship or dissociation in the tested systems, rather than rediscover that tuning and population function differ. [Kriegeskorte & Wei, 2021](https://arxiv.org/abs/2104.09743), [O’Shea et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850277/), [O’Shea, Wei & Priebe, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12893793/).

This question fits Wei's published program and the coauthor's transformer interests. His recent LLM study distinguishes linear separability from in-context learnability and finds that amplifying a task-aligned axis is insufficient to improve behavior. Its relevant lesson is to test the algorithmic use of a representation, not presume that improving an access metric improves the model. This is intellectual alignment inferred from publications, not a prediction about his preferences. [Xiong et al., 2026, inspected preprint v3](https://arxiv.org/html/2605.28854).

A bounded paper could establish one reproducible mismatch between single-unit resemblance and transfer, then explain it geometrically. A stronger paper would identify and intervene on the computation behind that mismatch. Broad brain–transformer mechanistic equivalence is outside the proposed evidence.

## Definitions to freeze before testing

**Primary task:** estimate allocentric position in one environment from a frozen representation, using a fixed readout trained on some headings and evaluated on disjoint headings at common positions. Position units must be explicit; report Euclidean error divided by arena diagonal. This is “location access across heading,” not unrestricted navigation competence.

**Secondary task:** the transformer's own next-view or other verified training-task output on held-out natural histories. Define the native loss and an endpoint-matched spatial contrast before selecting components. A newly trained position probe is never the native output. A heading decoder conditioned on position diagnoses retained context; it is not evidence against location invariance.

**Resemblance is a vector, not a universal scalar:**

| Property | Operational measure | What it does not establish |
|---|---|---|
| Conventional place-like selectivity | Independently selected spatial information/field criteria; prevalence with uncertainty | Conditional stability or native use |
| Conditional field shape | Principal assay: mean cross-fold correlation `S` of spatial profiles across headings, or `D=1−S`; independent validation: held-out shared-shape versus flexible interaction fit | Equality of raw response amplitude, a variance fraction, absence of heading information, or useful transfer |
| Gain/baseline modulation | Separate change in fitted amplitude/baseline with heading | Necessarily harmful interference |
| Population organization | Held-out stability of position-discrimination directions; context variation projected onto a frozen location readout | Global invariance from one local angle |

For spikes, compare nested Poisson count models with exposure offsets: position-only; shared position shape with heading-dependent gain/baseline; flexible position × heading tuning. Enforce identifiable constraints and positivity. For signed model features, use appropriate continuous-response models, not Poisson by default. Fit complexity with training-only validation; compare predictive scores on unseen trajectories. Shared-shape models are descriptive approximations, not unique causal decompositions. Wei's tuning-variability work is a relevant comparator, not something the correlation score replaces. [Zhu & Wei, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10121715/).

**Statistical contract:** choose a smallest meaningful effect before confirmatory analysis. A practical provisional transfer margin is 0.02 arena diagonals in mean error; revise it using application requirements and an independent pilot, then freeze it. For incremental prediction, provisionally require at least a 5% reduction in held-out error of predicting model-level transfer versus a predeclared baseline. These are design choices, not literature constants. Simulate power at those margins with realistic trajectory/animal/model clustering. An interval wholly inside the equivalence region supports a practically null effect; an interval spanning material positive and negative effects is inconclusive. Use animal/model-seed as top-level resampling units, with nested sessions/trajectories. Repeated subsets and layers of the same model are dependent observations.

**Assay contract:** `quantify_variance_corr` returns mean correlation, not its complement. Select heading as the slice axis when comparing spatial profiles, preserve the position grid, and freeze the score version, support, reliability handling and aggregation across units. High `S` means stable shape; high `D` means unstable shape. Gains and offsets can change real responses while leaving noiseless shape scores unchanged. U2's primary predictor is this calibrated assay summary; a fitted-model stability measure is a prespecified comparator. Report raw and revised versions separately if calibration requires a new estimator.

## Structured claims and adversarial tests

### U0 — The correlation assay earns its role as a measure of a biologically meaningful property

**Claim:** the assay provides an adequate and practically useful measurement of conditional field-shape stability in the selected recordings/models, with scientific value beyond simply assigning another number to familiar fields.

**Current evidence:** biological fields, score caches and mathematical invariances motivate the target. Measurement adequacy and comparative utility remain unestablished; raw historical `ev` values cannot establish either.

**Test/support:** pass M1/M2 with trajectory-matched stable/gain/shift/noise controls. Compare the score with independent field-displacement/shape fits, conventional spatial selectivity, close shape/separability correlations and justified noise-aware correlation estimators. Freeze one useful advantage before confirmation—such as interpretable discrimination, measurement efficiency, or candidate selection for U2/U3. Require held-out biological/model findings and a calibrated effect exceeding the declared margin. U2 supplies the specific positive transfer-prediction test; it is not assumed by U0.

**Best killer:** plausible reliability or gain-related sampling changes dominate the score's biological/model ordering; or fair comparisons reveal no claimed measurement/practical advantage and no independently informative finding attributable to conditional shape analysis. Matching familiar methods numerically can preserve an empirical application, but defeats a claim of a distinctive new measurement principle.

**Confounds/branch:** different measures may intentionally target different properties, so disagreement with gain-sensitive metrics is not automatically failure. Conversely, naming gain invariance does not excuse noise-induced gain sensitivity. If U0 fails, assess whether the functional science survives with another assay and explicitly change the contribution. A precise U2 dissociation can be a useful limitation finding for an adequate assay; it does not demonstrate positive predictive validity.

### U1 — The proposed biological benchmark is a reproducible conditional response property

**Claim:** within the selected CA1 recordings, stable spatial field shape across heading survives explicit separation of gain and sampling effects. The analogous HD-across-location result in the manuscript's PoSub population is a secondary replication target, not required for the first paper.

**Current evidence:** manuscript figures and cached maps suggest this pattern. The reported CA1 total is 551 cells from four rats. Existing scores are not explained-variance fractions, and the new analysis has not reproduced the biological finding. Verify region/session identifiers from source metadata; do not infer them from the older README.

**Test:** regenerate raw aligned samples for each animal/session; choose inclusion independently of held-out tuning; report the calibrated correlation-assay summary as the primary shape measure and fit the nested models above as independent checks. Estimate conditional field displacement and gain on common support. Simulate known stable and shifting fields on actual trajectories at observed spike rates. Include all eligible excitatory cells plus independently selected place cells; show animal-level results and exclusions. A conclusion supported only by the independent fits requires reporting the assay's failure separately under U0.

**Best killer:** calibrated models recover experimentally meaningful field shifts in trajectory-matched positive controls, yet biological field shifts reliably exceed the preregistered meaningful-shift margin on matched support, or apparent stability disappears with independent selection and adequate reliability. This kills a claimed biological stability benchmark, not all biological resemblance.

**Support required:** stable-shape models predict held-out biological responses within the predeclared practical tolerance of the flexible alternative, while gain estimates can remain nonzero. Define that tolerance from detectable displacement, e.g. one preregistered spatial bin, and report likelihood sensitivity rather than treating nonsignificance as equivalence.

**Confounds and branch:** sparse joint occupancy can make shape unidentifiable; this is an inconclusive benchmark. If killed, remove the new brain–model ranking. Continue model-only with a biologically motivated criterion, or redefine the measured biological property as structured mixed tuning. Do not select a new favorable region after seeing results without labeling exploration.

### U2 — Conditional field stability predicts cross-heading location access beyond conventional cell resemblance

**Claim:** among the tested trained model populations, calibrated field stability adds predictive value for fixed-readout transfer beyond place-like prevalence, within-heading location access, reliability and representation size.

**Current evidence:** the original discussion proposes this connection; it has not tested held-out-heading transfer. Existing numerical scores are only candidate-feature leads. This is the most direct attack on the old narrative's functional implication.

**Test:** for each independent model seed, measure the calibrated correlation-assay population summary on a metric-estimation split and transfer on separate trajectories. Fit a small preregistered prediction model: baseline controls first, then the assay summary. Compare a separate matched model using fitted-field stability or a close existing shape metric; do not count success of a replacement as success of the original assay. Evaluate incremental prediction on held-out seeds; with several architectures, leave an entire architecture family out as a harder secondary test. Freeze the primary layer or nest layer selection. Use neuron-count and sample-count curves; analyze equal dimensions as well as full representations. Match within-heading accuracy using a prespecified tolerance or report adjusted comparisons, without silently dropping inferior systems.

**Best killer:** stable-field-rich populations repeatedly transfer worse than less stable populations with comparable within-heading performance and reliability, and adding stability has practically zero or negative held-out predictive benefit with a sufficiently narrow interval. This is stronger than a nonsignificant correlation.

**Support required:** a positive, practically relevant incremental relationship replicated across seeds and not explained by one architecture, layer, or reliability threshold. A two-model contrast is a case study, not an established predictive law.

**Confounds and branch:** hundreds of units do not provide hundreds of independent model-performance observations. Narrow score range also makes a null uninformative. If killed, a reproducible counterexample can motivate U3: field stability is an incomplete functional criterion. Do not claim biological resemblance generally lacks value; only the tested property/task relationship failed.

### U3 — Population organization explains transfer failures or successes missed by single-unit fields

**Claim:** cross-heading alignment of location-relevant directions predicts transfer differences left unexplained by conditional field stability.

**Current evidence:** established theory motivates this, but there are no project measurements yet. In the calibration model `r(p,h)=A φ(p)+B ψ(h)`, a readout satisfying `WB=0` can retain location despite individually mixed units. Orthogonality is sufficient under suitable assumptions but unnecessary; appropriate complementary subspaces suffice.

**Test:** fit position-pair discrimination directions and the location readout using training data. On independent trajectories, estimate heading-induced displacements at matched positions, their projection through the frozen readout, and direction agreement across multiple locations/headings. Predict errors on a third split. Compare the baseline from U2 against its augmentation with two predefined geometric measures; avoid a large post hoc metric search. Test a nonlinear probe as a diagnostic of access beyond the linear class.

**Best killer:** reliable geometry differences have no incremental predictive value within a calibrated equivalence margin, or the proposed geometric explanation predicts which transfer fails in the wrong direction on new headings. A valid counterexample with aligned local directions but poor global transfer specifically kills an overbroad local-angle explanation.

**Support required:** held-out transfer errors or a U2 dissociation are predicted across several position contrasts and headings. Geometry measured using the same decoder test errors would be circular; independent splits are essential. A same-sample algebraic decomposition of decoder error is bookkeeping, not out-of-sample explanatory evidence.

**Confounds and branch:** feature scaling and trial noise influence geometry. Report training-only standardization, unstandardized sensitivity, and noise-aware estimates. If killed, test gain normalization or a bounded reference-frame hypothesis as a new exploratory explanation, with fresh confirmation data. If nothing predicts transfer, publish the descriptive dissociation only if it is reliable and informative.

### U4 — Useful location access can coexist with retained heading information

**Claim:** in at least one replicated model condition, fixed location readout transfers across heading while an independently fitted readout accesses heading beyond position/occupancy associations. This is a conjunction; both parts must pass.

**Current evidence:** the sensory precedent establishes an analogous possibility, not this spatial result. The original manuscript's inference from weak nuisance decoding to absent information is unsupported. Mixed responses and gain also have computational roles in Wei's navigation work. [Cueva et al., 2020](https://arxiv.org/html/1912.10189), [Ajabi et al., 2023](https://www.nature.com/articles/s41586-023-05813-2).

**Test:** use U2's location transfer and a separate heading prediction of sine/cosine, evaluated across held-out positions with circular error. Compare with training-derived conditional behavioral baselines and shuffled controls preserving relevant temporal structure. The location decoder receives neither true heading nor test-condition normalization statistics. Any heading-conditioned correction is a separately named readout with privileged information.

**Best killer:** no tested condition satisfies both criteria with adequate precision: transfer consistently fails whenever heading remains accessible, or heading access falls below the predeclared meaningful advantage whenever transfer succeeds. This kills coexistence in the tested systems, not its general mathematical possibility. Decoder failure alone cannot establish literal information absence.

**Support required:** transfer is within the preregistered margin of a target-heading readout trained on separate target-heading training trajectories, and conditional heading decoding beats the best prespecified behavioral baseline materially, replicated across seeds.

**Confounds and branch:** matched location support can select atypical trajectories; report retained coverage. Global gain can carry heading without changing shape; test raw and training-defined normalized representations. If killed, focus on the measured tradeoff. Do not force an “orthogonal multiplexing” interpretation if covariance or downstream readout does not support it.

### U5 — A representation property identifies a computation used by the transformer

**Claim:** an identified internal operation contributes selectively to native spatial prediction; intervention effects follow the proposed computational account. This is the optional mechanistic core, not implied by U2–U4.

**Current evidence:** the local predictive transformer exposes internal modules, but no circuit is established. Readout geometry and high biological resemblance are candidate discovery tools only.

**Test:** select an operation using discovery trajectories—for example, integrating landmark evidence with heading. Make a signed, example-specific prediction of its effect on next-view output. Patch natural matched donor activations into recipient histories, with the claimed variable changed and other observable factors controlled. Test held-out pairs, matched-norm/size random or unrelated patches, complementary components, several intervention strengths, and restoration where meaningful. Record native loss, the specific output contrast, field/gain measures, and transfer separately. Validate checkpoint and temporal semantics before interventions.

**Best killer:** an intervention independently verified to disrupt the proposed operation or state relation leaves native spatial behavior equivalent within a sensitive margin, after accounting for the specified redundant pathways; or its native-output effects are fully reproduced by matched generic damage and fail predicted directional/restoration tests. Changing only a resemblance/probe score does not establish disruption of the operation: in that case the metric-to-use inference fails, while the circuit test remains inconclusive. With validated disruption and sensitivity, the negative rejects the stated native-role claim within its intervention scope, even if decoder access improves.

**Support required:** predicted selective output changes, appropriate controls and restoration, across held-out examples and independent models. Define the native margin using a controlled input change with known output consequences, then freeze a fraction of that reference effect before confirmation. Redundant mechanisms can conceal necessity; report the tested intervention scope.

**Confounds and branch:** successful steering can exploit an unnatural direction. Use on-distribution donors and check collateral degradation. Joint changes in resemblance and function do not prove resemblance is the sole mediator. A larger mediation claim would require stronger interventions distinguishing competing pathways. If killed, retain a functional-access paper and state that the native mechanism was not established. [Activation-patching methods](https://proceedings.iclr.cc/paper_files/paper/2024/file/06a52a54c8ee03cd86771136bc91eb1f-Paper-Conference.pdf), [interpretability-illusion analysis](https://arxiv.org/abs/2311.17030).

### U6 — The relevant property–function relationship extends to recorded populations

**Claim:** within neural sessions, conditional field stability or the successful U3 geometry predicts transfer in the same qualitative direction as in models. This optional bridge concerns a relationship, not matched representational identity.

**Current evidence:** raw recordings, maps and decoder code exist. The old decoding figures do not test this relationship. The model and rodent datasets lack matched visual experiences.

**Test:** perform the same within-system transfer assay using simultaneously recorded neurons; match population sizes, sample counts, temporal windows and support. Analyze session effects nested within animal. Start with CA1; treat the reverse HD-across-position assay as separately specified. Plot within-system effects and uncertainty before considering cross-system similarity. Do not concatenate cells from different sessions as if simultaneously recorded.

**Best killer:** adequate biological sampling and positive controls reveal a reproducible opposite or practically null within-system property–function relationship, despite a robust model relationship. Reversing an absolute brain–model performance ordering after reliability matching does not itself kill this claim: both systems could still have the same within-system relationship. Any absolute-ranking claim would require a separately specified hypothesis.

**Support required:** comparable qualitative relationships survive sampling sensitivity; exact numerical equality is unnecessary and would require a separate equivalence hypothesis. Decision-variable correlation is a relevant task-specific comparison method, but direct per-item DVC cannot be assumed without corresponding stimuli/decisions. [Qian, Geisler & Wei, NeurIPS 2025](https://arxiv.org/abs/2506.02164v4).

**Confounds and branch:** animal behavior couples position, heading and history; four CA1 animals constrain precision. No spike intervention here establishes a brain mechanism. If killed or underidentified, remove cross-system generalization and retain the model result with a qualified biological motivation.

## Reusable project material and required changes

Read-only inspection confirms the following assets. Existence and readable metadata are not reproduced output provenance.

| Asset | Reuse and boundary |
|---|---|
| [bio_neurons.ipynb](/Users/moo/variance_analysis/notebooks/bio_neurons.ipynb) | Zero-based cell 7 defines Pfeiffer paths, Rat1–4 and counts `[204,174,79,94]`; cells 8–10 build/load 20×20×60 maps. Cells 15–19 use Dusz Dataset1/2 square sessions, exclude `B0703-211208`, and use 20×20×90 maps. Extract configuration and inclusion logic; rebuild confirmatory splits. |
| [Pfeiffer data](/Users/moo/variance_analysis/data/rodents/pfeiffer) | `Rat1/Open1/Position_Data.mat` and `Spike_Data.mat`, with equivalent Rat2–4 directories. Loader expects position columns time/x/y/HD and excitatory IDs. Verify units, timestamps and selection. Open2 exists but is not automatically an independent animal replication. |
| [Dusz data](/Users/moo/variance_analysis/data/rodents/dusz) | `Dataset_metadata.xlsx` and `Dataset_1/A3701-191119/Data` exemplify real source inputs. Loader reads behavioral epochs, tracking, cell types, spikes and velocity. Verify biological region/session assignments against source metadata. |
| [pfeiffer_ratemaps.py](/Users/moo/variance_analysis/src/pfeiffer_ratemaps.py) and [dusz_ratemaps.py](/Users/moo/variance_analysis/src/dusz_ratemaps.py) | Reuse loaders/count binning after alignment tests. Existing map builders shuffle observations within joint bins: they are not the required blocked transfer splits. Spatial smoothing must respect the 2D grid before flattening. |
| [Pfeiffer caches](/Users/moo/variance_analysis/results/rodents/pfeiffer/Rat1) | Actual `fold0_xyhd_20x20x60.npz`, `fold1_xyhd_20x20x60.npz`, and metadata: tau 0.25 s, speed threshold 5, 20 position bins/axis, 60 heading bins. Useful for shape examples and recovery checks; insufficient for new trial-level transfer. |
| [Dusz example cache](/Users/moo/variance_analysis/results/rodents/dusz/Dataset_1/A3701-191119) | Actual `fold_rate_xyhd_20x20x90.npz` metadata: tau 0.05 s, speed threshold 2, 20 bins/axis, 90 heading bins. Multiple older cache variants coexist; record hashes and producer configuration. |
| [pfeiffer_decode_solo.py](/Users/moo/variance_analysis/src/pfeiffer_decode_solo.py) and [dusz_decode_solo.py](/Users/moo/variance_analysis/src/dusz_decode_solo.py) | Reuse circular-error/alignment and Bayesian-decoder baselines after verification. Existing CV/chance outputs are not held-out-heading evidence. Replace splitting and add strong regularized probes/behavioral baselines. |
| [model_neurons.ipynb](/Users/moo/variance_analysis/notebooks/model_neurons.ipynb), [get_activations.py](/Users/moo/variance_analysis/src/ann/get_activations.py), [predictive_coder.py](/Users/moo/variance_analysis/src/ann/predictive_coder.py) | Reuse panorama extraction and hooks after the shared handoff's loading/history gates. The checkpoint exists at `data/weights/predictive_coder.ckpt`; existence does not establish successful loading. Static repeated-frame evaluation cannot substitute for natural-history evaluation. |
| [estimator](/Users/moo/variance_analysis/src/explained_variance.py) and [diagnostics](/Users/moo/variance_analysis/research_review/method_diagnostics.py) | Principal candidate shape-correlation assay, subject to M1/M2/U0. Synthetic additive, multiplicative, constant-slice and noise cases are calibration seeds. Preserve estimator-version provenance if revised. The unused binning notebook is not required and is not evidence about final-paper validity. |

## Implementation sequence and deliverables

1. **Recovery gate:** manifest source/checkpoint hashes, environment, source units and session identities. Export aligned samples with `system_id, animal_id, session_id, seed, trajectory_id, time, position, heading, speed, layer, representation`; store spikes/count exposure separately. Retain lineage to raw records and explicit masks. Do not run historical notebooks wholesale into old results directories.
2. **Calibration gate:** synthetic stable-shape/gain/shift/noise populations sampled on real trajectories. Verify known shape effects and decoder transfer/dissociations; prevent train/test smoothing leakage. Include an orthogonal change of population basis with a correspondingly transformed linear readout: it can change individual-unit fields while preserving outputs. This is an analytical calibration, not an intervention proving that the network naturally uses rotated neurons. Metric failure stops confirmatory inference.
3. **Small pilot:** one recoverable model and one well-covered CA1 session, eight heading sectors as an initial design, sector-pair transfer matrix on common spatial bins. Use contiguous blocks with a gap exceeding feature/smoothing history. Reduce bin resolution based on occupancy before inspecting claim outcomes; retain the same rule across systems.
4. **Confirmatory functional study:** pass M1/M2 and test U0's declared comparative-value claim; preregister the calibrated assay as U2's primary predictor, U3 as the independent population explanation, U1 as the biological anchor and U4 as secondary. Use two model conditions with an initial target of five independent training seeds each; estimate whether that sample can distinguish the chosen effect. If not, increase independent replication or narrow the claim. Ten models support an initial paired condition contrast, not a credible many-covariate predictive law: keep U2/U3 regressions exploratory until independent model count and metric variation support the prespecified control set and held-out validation. Frozen original checkpoints remain case-study anchors. Additional objectives from Plan 01 provide controlled model variation, but layers/subsets cannot replace seeds.
5. **Optional mechanism/brain bridge:** proceed to U5 only after a stable phenomenon and tractable intervention appear; U6 requires sufficient raw common support. Neither is mandatory for a useful focused paper.

For transfer, fit standardization and regularization on source-heading training blocks. Tune on separate source blocks; evaluate at disjoint headings/held-out blocks. Fit the target-heading comparison decoder on separate target training blocks with equal sample count. Compare occupancy-only predictions, pixels/simple visual features, random and trained models, and a calibrated positive-control population. Report learning curves and raw errors as well as the transfer gap; equal gaps can conceal equally poor decoding. Do not tune heading partitions on successful outcomes.

Write machine-readable `manifest.json`, `splits.parquet`, `metrics.parquet`, `predictions.parquet`, and, for U5, `interventions.parquet` including donors, recipients and controls. A `claim_verdicts.md` must report each U-ID using the shared outcomes: supported at stated scope, contradicted at stated scope, equivalent within the meaningful margin, inconclusive, or not tested, with effect intervals and links to figures. Archive failed controls and exclusions.

**Feasibility:** calibration and a frozen-feature pilot plausibly require several focused days after recovery; a replicated analysis roughly two to four weeks. New model seeds and mechanistic work can require several additional weeks or months. These are planning estimates, not benchmarked compute budgets. Time-aligned data availability, common support and faithful checkpoint execution dominate uncertainty.

**Paper endpoints:** a method-centered workshop paper can characterize the correlation assay against close alternatives and establish one informative U2/U3 relationship or dissociation with independent controls. U0/M3 determine what methodological value can be claimed; a negative U2 result supports a limitation only if the assay itself is adequate. A stronger paper adds U5 with selective causal evidence. If another measure carries the result, change the contribution explicitly. If no property survives calibration or intervals remain wide, do not market a general resemblance–usefulness law. See the shared method claims for assay-specific prior work and novelty limits.
