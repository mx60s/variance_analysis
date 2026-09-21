# Transformer spatial mechanisms: targeted source notes

September 14, 2026. Targeted review for a joint research direction combining the manuscript, Wei’s program, and the coauthor’s interest in mechanistic interpretability. This is a focused novelty check, not an exhaustive review of transformer interpretability. All sources below are primary papers or official proceedings. Paper findings and proposed experiments are separated.

## Closest primary works

### 1. Li et al., ICLR 2023: Emergent World Representations

[ICLR paper](https://openreview.net/pdf?id=DeG07_TcZvT); [October 24, 2022 preprint](https://arxiv.org/abs/2210.13382).

**Source summary:** An Othello next-move transformer develops an internal board representation without board-state supervision. Nonlinear probes recover board state, and interventions on internal activations alter subsequent predictions according to counterfactual board configurations. This provides intervention evidence beyond decoding; the work does not establish a complete circuit implementing the game rules.

**Implication:** Merely finding spatial state inside a sequence predictor is established. A new study should explain which representation the computation uses, under which conditions, and what changes when the objective or experience changes.

### 2. Nanda, Lee & Wattenberg, BlackboxNLP, December 2023: Emergent Linear Representations in World Models of Self-Supervised Sequence Models

[Official proceedings and paper](https://aclanthology.org/2023.blackboxnlp-1.2/).

**Source summary:** In Othello-GPT, switching from absolute piece colors to current-player-relative labels reveals a linear board representation. Vector interventions then control predictions, and the authors investigate how the representation is computed. The current-player coordinate system changes what a simple probe can reveal.

**Implication:** This is almost exactly the methodological intuition behind “apparent entanglement reveals another reference frame.” The conceptual point is therefore not new. A spatial extension earns novelty by identifying a particular frame or frame-conversion mechanism, demonstrating its causal use, and explaining why the task selects it. Failure of a location/heading factorization is a reason to test alternative coordinates, not evidence that one has been found.

### 3. Spies et al., 2025: Transformers Use Causal World Models in Maze-Solving Tasks

[Inspected full preprint, March 5, 2025 version](https://arxiv.org/html/2412.11867); [ICLR 2025 World Models workshop version](https://openreview.net/pdf/c78cc4604feabf2ce55771cbb3d3c9a7ef8b8188.pdf). Original preprint December 16, 2024.

**Source summary:** Maze transformers encode connectivity through attention-head outputs and SAE features. The authors connect these analyses using head patching, then change latent edge features to alter generated paths. Some latent interventions support mazes whose input serialization causes failure. Models with different positional encodings have different feature organization, but also differ in head counts; this is not a perfectly isolated positional-encoding comparison.

**Implication:** Probes, interpretable features, feature construction, and downstream intervention should be distinct evidence levels. A particularly relevant distinction is failure to construct a useful code from inputs versus failure to use an available code. The existing manuscript’s decoder or static-history failures do not distinguish these. SAE use is optional: add it only if simpler, hypothesis-driven subspaces cannot isolate the proposed mechanism.

### 4. Baumgartner, Spens, Burgess & Manescu, November 17, 2025: Cognitive Maps in Language Models: A Mechanistic Analysis of Spatial Planning

[Full preprint](https://arxiv.org/html/2511.13371v1). The inspected manuscript bears a CogInterp workshop label; do not infer a separate archival publication status from that label.

**Source summary:** GPT-2 models trained on random-walk prediction versus shortest-path tasks show different spatial representations and generalization. Direction-token ablations suggest differences in where history information is consolidated. Cross-context patching supports transferable direction information in early attention outputs. The intermediate integration mechanism remains unresolved. Task, trajectory structure, context length, loss masking, optimization schedule, and fine-tuning history differ across conditions, limiting an objective-only interpretation.

**Implication:** This is a major novelty constraint on “training regimes determine cognitive maps.” A sharper experiment could separate objective from experience and test the mechanism of frame conversion or context-invariant access. It should evaluate common downstream tasks, not equate different training-task scores. Their directional tokens are world-cardinal directions; that does not settle transformations between camera heading and world coordinates.

**Evidence qualification:** Zeroing old direction-token states demonstrates dependence on information at those positions under that intervention. It does not by itself prove that the retained state is a unique Cartesian map. Similarly, orthogonal coordinate probe weights do not establish statistically independent coordinates or explain the computation. These are reasons to build more discriminating tests, not dismiss the reported phenomenon.

### 5. Kniazev & Fijalkow, May 13, 2026: Transformers Linearly Represent Highly Structured World Models

[Inspected full preprint](https://arxiv.org/html/2605.18847v1).

**Source summary:** A Sudoku transformer represents constraints through row, column, and box features. The authors patch substructure directions, mean-ablate specialized heads with control regions, and ablate final-MLP naked-single neurons using another simultaneous naked single as a within-puzzle control. These support specific functional roles. Limitations include one model/task, incomplete attention-selection explanation, and one protocol for world-model direction interventions.

**Implication:** Useful internal ontology can differ from the surface decomposition chosen by an analyst. A proposed spatial paper must go beyond displaying a cleaner embedding in a newly chosen frame. The within-example selective intervention design is an excellent template: alter the hypothesized coordinate while preserving an independent control computation, and predict the resulting action or prediction change.

### 6. Piotrowski, Riechers, Filan & Shai, ICML, July 2025: Constrained Belief Updates Explain Geometric Structures in Transformer Representations

[Official proceedings](https://proceedings.mlr.press/v267/piotrowski25a.html). Proceedings abstract inspected; do not attribute unverified detailed ablation results to this note.

**Source summary:** In controlled hidden-Markov-model tasks, the authors connect transformer attention and representation geometry to constrained Bayesian belief updating. Their analysis predicts attention, embedding, and OV structure from optimal prediction modified by architectural constraints. Subsequent layers refine the representation.

**Implication:** A compelling bridge to Wei is a computational hypothesis that predicts internal geometry. With partial visual observations, the appropriate state may be a belief over location and heading, rather than one perfectly disentangled point estimate. Testing this is valuable only if ambiguity and history are explicitly controlled; a general “Bayesian world model” expansion would exceed the immediate project.

## Essential non-transformer predecessor

**Vijayabaskaran & Cheng, PLOS Computational Biology, October 31, 2022: Navigation task and action space drive the emergence of egocentric and allocentric spatial representations.** [Full paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010320).

**Source summary:** A visually driven deep-Q network learns navigation toward fixed unmarked goals or visible moving cues. Task and available action frames influence both strategy and learned spatial responses; agents can solve either task with either constrained action frame. This dissociates task, action frame, and representational labels more than a simple allocentric-good/egocentric-bad account allows.

**Implication:** The user’s first and second narratives already intersect in a direct navigation predecessor. A new result should identify the computation selected by objectives, and establish its consequences under controlled transfer or intervention. Swapping an output vocabulary between world and body directions alone would be a weak discovery: the labels already prescribe a coordinate transformation.

## Proposed tractable pilot and scope

**Question:** Does an objective change select a different internal spatial coordinate computation, and does that explain transfer better than single-unit biological resemblance?

Start with a single recoverable model and an evaluation pilot before committing to retraining. Use matched states varying heading or landmark configuration. Fit a small prespecified set of alternatives: world location; egocentric landmark bearing/distance; goal-relative vector if goals exist; and appropriate interactions. Compare held-out prediction of activations and downstream access at matched model complexity. An egocentric model needs actual landmark geometry; merely relabeling axes is insufficient. If no candidate explains a reliable pattern, avoid launching a mechanistic claim.

For a compact transformer experiment, use a small egocentric visual or symbolic navigation environment with identifiable pose from history. Hold the backbone, input trajectories, tokenization, compute budget, and evaluation constant; compare two prediction objectives, such as immediate versus a specified longer-horizon prediction. Treat this as changing temporal prediction requirements, not automatically a pure test of invariance. Avoid changing target representation, action labels, architecture, and trajectory statistics simultaneously. A matched common decoder can assess access, but causal usefulness requires testing the model's own computation or a explicitly trained common task head.

The decisive addition would be one counterfactual test. Find donor and recipient histories that differ in a controlled spatial relation, patch a candidate representation or its producing component, and test whether the output changes exactly as the proposed coordinate computation predicts. Include unrelated-feature or matched-norm controls, multiple donor pairs, and a restoration test where feasible. Whole-state patching may transfer too many variables; selective subspace patching needs validation that it preserves the controls. General output damage is not evidence for the intended coordinate.

Only after the pilot motivates it, replicate the objective contrast across a small number of seeds and held-out headings, landmark arrangements, or trajectory families. If varying experience is a competing explanation, a small objective-by-experience factorial is stronger than expanding the architecture catalogue. It is extra work, not a prerequisite to a carefully scoped objective comparison with fixed data.

**Feasibility:** Frozen-feature frame tests could be a short pilot if activations and simulator metadata are ready. A small transformer with controlled data is feasible in principle but requires a new training and intervention pipeline; the current CNN paper does not become a transformer mechanism paper through a text revision. Existing pretrained maze models can test analysis ideas cheaply, but do not answer questions about egocentric visual invariance without redesign. Do not promise runtimes before checking hardware, environment, and training scale.

## How the three narratives fit together

The coherent causal sequence is **objective and experience → internal coordinate computation → transfer or navigation performance**. Biological resemblance supplies competing candidate explanatory features along that sequence. A useful result might show that a heading-modulated representation implements a valid coordinate conversion while a more place-like representation fails a particular transfer task. It might instead show that stable location access mediates improved transfer. Both outcomes are informative.

Comparing two models cannot establish a general law that one biological property predicts computational usefulness. To make that claim, define resemblance properties separately, assess held-out prediction of useful behavior over enough independently trained models/conditions, and ideally intervene on the hypothesized property. The practical first paper can remain a mechanistic case study, with biology motivating the candidate hypotheses rather than requiring a sweeping brain–model ranking.

The joint intellectual contribution need not be “make transformers look like place cells.” Wei’s normative and representational-geometry interests can motivate what an effective code should preserve; the coauthor’s mechanistic-interpretability interests can establish how a learned network constructs and uses that code. That is an inference about compatibility, not a prediction of his interest or approval.
