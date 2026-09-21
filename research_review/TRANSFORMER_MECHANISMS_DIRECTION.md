# Objectives, reference frames, and useful spatial representations in transformers

For the subsequent hypothesis-by-hypothesis implementation briefs, see the [three-plan packet](hypothesis_plans/README.md). It includes explicit falsifiers, exact reusable assets, experimental gates, and decisions for narrowing or abandoning each narrative.

**Current framing update:** the packet now explicitly tests the correlation assay as a central contribution through [shared method claims](hypothesis_plans/METHOD_CENTRAL_CLAIMS.md) and O0/F0/U0 application claims. The broader mechanism discussion below supplies scientific context; its earlier permissive treatment of the assay as optional is superseded for this method-centered pass. Centrality remains conditional on calibration and demonstrated value.

**The strongest joint direction is to use training objectives as an intervention, reference-frame computation as the candidate mechanism, and generalization as the functional test.** Biological resemblance then becomes something to explain and evaluate. This accommodates an interest in transformer mechanisms while remaining close to Xue-Xin Wei’s work on learned navigation circuits and representational geometry.

The proposed question is: **How does the training objective change the spatial computation a transformer learns, and which aspects of that computation explain robust spatial access?** The answer may involve invariance, equivariance, memory retrieval, or an explicit transformation between coordinate systems. None should be assumed before testing.

This recommendation prioritizes scientific interest over the smallest possible revision. It could become a substantially new paper built from the original project. The existing data and conditional-tuning observations are useful starting points; preserving the original estimator or every biological section is not a requirement for the new question.

## Fit to the two research interests

| Direction | Connection to Wei’s published work | Mechanistic-interpretability opportunity | Recommended role |
|---|---|---|---|
| Learning objectives and experience determine spatial organization | Learned navigation circuits, normative explanations, and relationships between task demands and representations | Hold the backbone and experience fixed, change training pressure, then identify which internal computation changes | **Lead experimental axis**; initially change objective without simultaneously changing experience |
| Apparent entanglement reflects another reference frame | Spatial transformations, population geometry, and useful conjunctive codes | Identify a coordinate system that predicts both response transformations and causal effects on outputs | **Candidate mechanism**, not a promised discovery |
| Biological resemblance predicts computational usefulness | Tuning versus geometry, invariant readout, and task-relevant representation comparison | Test whether resemblance identifies components actually used for a specified computation | **Evaluation question**, with either agreement or dissociation informative |

These are judgments of fit, not predictions of Wei’s personal preferences. The connection to transformers is already present in his publication record. Xiong et al. study how LLM geometry changes during in-context learning; an intervention that amplifies a task-aligned axis does not improve learning, and behavioral modeling distinguishes access by an external decoder from the algorithm the model appears to use. This is close to the distinction the proposed spatial study needs. The inspected source is the 2026 author preprint.[^1] Wei’s learned HD model also connects conjunctive responses to circuit roles, rather than treating mixed tuning as intrinsically defective.[^2]

## What the interpretability literature changes

**Changing coordinates can reveal a simpler mechanism, but that observation already has a strong precedent.** Nanda, Lee and Wattenberg show that Othello state is more simply represented relative to the current player than as absolute black/white occupancy, and use vector interventions to change behavior.[^3] A spatial analogue therefore needs more than a better probe. It should predict how the representation transforms and demonstrate a role in the model’s own computation.

**Task-dependent reference frames are also established.** Vijayabaskaran and Cheng study how navigation tasks and action spaces affect egocentric and allocentric representations.[^4] The claim “objectives affect spatial representations” is too broad to be the novelty. A new contribution could identify a specific transformer computation, the conditions under which it emerges, and the generalization it enables.

Transformer–TEM already connects a deliberately structured transformer with hippocampal computation, while Piray and Daw connect compositional relative maps with planning and biological response types.[^12][^13] This makes the joint theme well motivated but also means that “relational codes support generalization and resemble the brain” is not sufficient novelty. A learned mechanism in the tested predictive transformer would be a more specific contribution.

**There is a particularly close transformer study.** Baumgartner et al.’s 2025 preprint compares foraging and shortest-path training, with layerwise ablations and output patching. It reports different spatial solutions. Its conditions also differ in trajectory structure, training protocol, and supervision, so it does not isolate an objective-only effect.[^5] A controlled comparison is one opportunity, but it must reveal something beyond confirming that training regimes matter. The intermediate integration mechanism is a more promising target than another map visualization.

**Objective changes that encourage compact state are an existing research line.** NextLat adds a latent-prediction objective and evaluates spatial world-model quality and generalization, including Manhattan trajectories. Its current inspected version is a June 2026 preprint.[^6] The Belief State Transformer is an ICLR 2025 architecture-and-objective proposal.[^7] These are useful interventions or baselines, not ideas to reintroduce as novel. Neither makes the causal organization of the original visual model, its reference frame, or its relationship to biological field stability a settled question.

**Prediction does not imply failure to learn state.** Shai et al. find belief-state geometry in next-token transformers on controlled stochastic processes.[^8] Thus the useful hypothesis is conditional: under a specified task, capacity, and experience distribution, a particular objective favors a different solution. It is not that ordinary prediction inherently cannot form a world model.

The selected papers establish a crowded but relevant intersection. This targeted review is sufficient to refine the proposal; it is not an exhaustive survey of all transformer circuits, self-supervised objectives, or spatial neuroscience. Detailed companion notes cover [spatial transformer mechanisms](source_notes/mech_spatial.md) and [reference frames](source_notes/reference_frames_mechanisms.md).

## A concrete first project

Start with a small visual spatial transformer whose input histories and internal activations can be controlled. The local predictive-model implementation already exposes two attention blocks and intervening feed-forward modules, providing a possible entry point; successful checkpoint recovery and faithful temporal evaluation still need verification. A smaller controlled transformer could be preferable if that recovery is costly, but results from it must not be described as a reproduction of the original checkpoint.

Use **the same architecture and the same trajectories under two objectives**. One candidate is the existing prediction objective versus prediction augmented with a published latent-dynamics objective. A simpler alternative is one-step versus appropriately controlled multi-step prediction. Neither guarantees location invariance: heading may remain necessary for prediction in both conditions. The experiment asks what changes, including the possibility that only history compression changes.

Keep the observation encoder, input information, temporal context, data exposure, and main model capacity comparable. Account for auxiliary-head capacity, extra gradients, optimization effort, and training compute. Evaluate both at matched exposure and across learning curves; a weak or undertrained baseline is not evidence about the preferred computation. If actions or pose labels are supplied, give comparable information to both conditions and identify the extra supervision explicitly. Same-place/different-view contrastive training is a useful positive control, but directly rewarding heading invariance makes its emergence unsurprising.

The first experiment should change objective alone. Add a second experience distribution only after finding a clear effect: for example, turn-rich versus translation-rich trajectories evaluated on the same test distribution. A full objective × experience design is stronger for separating their effects, but not necessary before a pilot demonstrates feasibility.

For each trained condition, ask four linked questions:

1. **What is represented?** Measure position, heading, and landmark-relative variables, along with calibrated field shape and gain. Use held-out combinations and matched-capacity probes.
2. **In what coordinates?** Test egocentric landmark bearing/distance, allocentric position, and environment- or landmark-anchored coordinates against controlled transformations.
3. **What does the model use?** Intervene on candidate states and pathways and measure the model’s own prediction or action output.
4. **What generalizes?** Test unseen heading–location combinations, new incoming histories, cue transformations, and one clearly specified downstream spatial task.

An external decoder answers an access question. It does not establish that the original transformer performs that decoding. A mechanistic contribution must reach the third question.

## Making the reference-frame hypothesis falsifiable

Suppose a unit responds when a landmark is ahead and to the left. Its position × heading map will look mixed in laboratory coordinates, yet it may have simple egocentric tuning. Predictable consequences distinguish this from an unconstrained reinterpretation: moving the landmark should shift its preferred locations; rotating the observer should shift preferred headings; moving observer and landmark together should preserve the relative relation where the scene permits it.

Use transformations that dissociate candidates. Rotating a cue, rotating the arena, rotating the observer, and renaming coordinate axes are different manipulations. An arbitrary invertible change of latent basis is not evidence for a physical reference frame. A more successful visual-feature model is also not automatically a coordinate system.

A fixed landmark’s relative position can simply reparameterize laboratory position, so fixed-arena observations alone cannot identify landmark anchoring. Rotating the observer and the entire scene together can leave the visual input identical; a visual model cannot recover an unobservable absolute north. Choose manipulations whose candidate frames make observably different predictions, and distinguish a visible image-template response from a remembered spatial relation using appearance changes or temporary occlusion.

Fit any proposed transformation on training conditions and test new rotations, translations, environments, and compositions. If a single constrained rule predicts how activity and behavior change, that is stronger evidence than finding a nonlinear decoder that recovers coordinates. Egocentric quantities can be equivariant under some interventions while a downstream output remains invariant; state explicitly which transformation and level of representation each claim concerns.

A useful mechanism hypothesis might be: **one pathway retrieves landmark-relative information from history; another combines it with heading to update a location-relevant state.** Candidate tests include swapping histories that end at matched poses, patching a heading-related contribution while keeping landmark evidence fixed, and checking whether a predicted transformation restores the appropriate output. These are proposed tests, not claims that the current model implements this circuit.

## What would count as mechanistic evidence?

There are three increasingly strong outcomes: a variable is decodable; a controlled intervention affects behavior as predicted; a specified computation explains that effect through identifiable components. A good small mechanism paper can localize and test one transformation without reverse-engineering the entire network.

Prefer matched natural donor/recipient examples, test interventions on held-out conditions, and compare targeted patches with matched-size controls and complementary components. Use both disruptive and restorative interventions where meaningful, and check that broad model degradation is not driving the effect. Subspace steering alone can misidentify a feature’s native role; the interpretability-illusion literature and subsequent debate make fidelity an empirical requirement, not grounds to dismiss all patching.[^9] Patching metrics and corruption choices also materially affect conclusions.[^10]

A sparse autoencoder may help discover candidate features, but it is not required. Attention maps alone do not establish a mechanism. Start from explicit spatial variables and counterfactual predictions, then use tools that test those predictions. Causal abstraction provides a formal vocabulary if needed, without requiring the project to invent a new interpretability method.[^11]

## Biological resemblance as a scientific test

Define usefulness before ranking resemblance. Position transfer across heading, landmark anchoring, trajectory composition, and next-view prediction are different tasks. A feature useful for one may be dispensable for another. Select a small set of biological properties—conditional field stability, gain modulation, or a population-geometric property—and ask which predict performance beyond baseline task accuracy and representation size.

Use multiple seeds, population subsets, or model instances to assess a relationship. Hundreds of units sharing one model-level score are not hundreds of independent tests. For stronger causal evidence, intervene on a candidate operation and test whether both the biological property and the specific functional advantage change. Such an intervention may affect multiple properties, so it does not by itself prove that resemblance is the sole mediator.

This permits several useful outcomes. Stable place-like fields might support data-efficient transfer. A population might transfer despite lacking those fields because it implements a different coordinate transformation. Or a biologically suggestive property might be incidental to the tested computation. These would answer the original paper’s missing bridge with different conclusions.

Existing rodent data can anchor the biological property, but cannot establish that transformers and brains implement the same causal mechanism. If no neural reanalysis is feasible, frame the work as testing a biologically motivated criterion in models rather than claiming new biological mechanism.

## Scope and decision points

The recommended first commitment is a pilot, not the entire causal chain. Once model training and data generation work, one to two focused weeks could establish an objective-dependent effect, test competing coordinate descriptions, and attempt one intervention. A replicated causal study is plausibly several additional weeks to months; simulator recovery, optimization, and circuit identifiability dominate uncertainty. These are planning judgments, not measured compute estimates.

A workshop-sized paper could center on one decisive result: a reference-frame interpretation that predicts a causal transformation, or a controlled objective change that alters how a spatial state is computed. The broader paper would connect that mechanism to transfer and biological criteria. It need not also introduce a new learning algorithm.

If the new objective only improves generic accuracy, or an alternative coordinate system only improves a probe, the strongest narrative remains unsupported. If the same mechanism appears under both objectives, that may favor the reference-frame story while weakening the objective story. If no tractable mechanism emerges, a careful functional comparison remains possible, but it should be recognized as a different endpoint.

The most useful discussion proposal is: **“Can we explain how training pressure selects the spatial computation inside a transformer, then test when that computation produces the biological properties that matter for transfer?”** This supplies a computational-neuroscience question and a transformer-mechanism question within one experiment, while leaving the result open.

## Sources

[^1]: Xiong et al. (2026), [Large language models reorganize representational geometry during in-context learning](https://arxiv.org/html/2605.28854), inspected author preprint.
[^2]: Cueva et al. (2020), [Emergence of functional and structural properties of the head direction system by optimization of recurrent neural networks](https://arxiv.org/html/1912.10189), ICLR.
[^3]: Nanda, Lee & Wattenberg (2023), [Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://aclanthology.org/2023.blackboxnlp-1.2/), BlackboxNLP.
[^4]: Vijayabaskaran & Cheng (2022), [Navigation task and action space drive the emergence of egocentric and allocentric spatial representations](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010320), PLOS Computational Biology.
[^5]: Baumgartner et al. (2025), [Cognitive Maps in Language Models: A Mechanistic Analysis of Spatial Planning](https://arxiv.org/html/2511.13371v1), inspected preprint.
[^6]: Teoh et al., [Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/html/2511.05963v4), inspected preprint revised June 15, 2026.
[^7]: Hu et al. (2025), [The Belief State Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/01b3dea1871f7cea1e0e6be1f2f085bc-Abstract-Conference.html), ICLR.
[^8]: Shai et al. (2024), [Transformers Represent Belief State Geometry in their Residual Stream](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html), NeurIPS.
[^9]: Makelov, Lange & Nanda, [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030); [reply by Wu et al.](https://arxiv.org/abs/2401.12631). The methodological debate should not be reduced to a blanket claim that successful interventions are invalid.
[^10]: Zhang & Nanda (2024), [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://proceedings.iclr.cc/paper_files/paper/2024/file/06a52a54c8ee03cd86771136bc91eb1f-Paper-Conference.pdf), ICLR.
[^11]: Geiger et al. (2025), [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://www.jmlr.org/papers/v26/23-0058.html), JMLR.
[^12]: Whittington, Warren & Behrens (2022), [Relating transformers to models and neural representations of the hippocampal formation](https://arxiv.org/html/2112.04035), ICLR.
[^13]: Piray & Daw (2025), [Reconciling flexibility and efficiency: medial entorhinal cortex represents a compositional cognitive map](https://www.nature.com/articles/s41467-025-62733-7), Nature Communications.
