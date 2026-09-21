# Spatial neuroscience and vision-model literature audit

Searched 14 September 2026. Primary papers, publisher records, author code, and author-institution copies were prioritized. This is a targeted literature review, not an exhaustive systematic search. Judgments below distinguish what the literature establishes from proposed experiments. No biological result in the local manuscript was independently reproduced for this subtask.

## Consequential changes to the narrative

The best opportunity is a **controlled benchmark distinguishing marginal spatial tuning, conditional response structure, and population generalization**. The present broad story—vision models resemble spatial cells only marginally, whereas real cells are invariant, suggesting a missing ingredient for cognitive maps—is too categorical. It conflates different levels of representation, overlooks direct older invariance models, and treats context-dependent biology as an ideal dichotomy.

Three separations are essential:

1. Similar-looking single-unit rate maps do not establish matching joint tuning. This remains a useful empirical comparison, but is not a newly discovered principle.
2. Different joint single-unit tuning does not establish deficient population information or failed cognitive mapping. A linear mixing of invariant population features can change unit tuning without destroying linear readout. This is a mathematical observation, independent of any particular paper.
3. A discrepancy in two vision architectures does not establish that vision cannot supply invariant spatial representations, or that path integration is necessary. Compare learning objectives, exploration statistics, temporal context, and readout before inferring a missing sensory modality.

## Direct model comparators

### Gornet & Thomson (2024), Automated construction of cognitive maps with visual predictive coding

[Nature Machine Intelligence, DOI 10.1038/s42256-024-00863-1](https://www.nature.com/articles/s42256-024-00863-1). Published 18 July 2024.

The paper's evidence goes beyond marginal single-cell maps: it evaluates latent spatial distances, localization relative to landmarks, and vector navigation. Thus the draft's repeated description that these studies “only examined” individual marginal tuning is inaccurate as a characterization of the whole paper. A narrower claim—that its cell-like comparisons did not establish the particular location-by-heading invariance being tested—is defensible after checking the relevant analyses. Showing entangled units would constrain biological resemblance but would not overturn the reported population geometry. A fair extension should reproduce geometry/localization on original or closely matched conditions, then test heading-held-out generalization. Changes in environment, trajectory, architecture, and prediction performance must be documented before calling the new setup a replication.

### Luo, Mok & Love (2025 revision), The inevitability and superfluousness of cell types in spatial cognition

[Reviewed Preprint v2, DOI 10.7554/eLife.99047.2](https://elifesciences.org/reviewed-preprints/99047v2/figures), revised 10 November 2025; v1 was 28 August 2024. [Methods](https://elifesciences.org/reviewed-preprints/99047), [author responses](https://elifesciences.org/reviewed-preprints/99047/reviews), [original source code](https://github.com/don-tpanic/Space).

This is a null-model argument, not simply a proposal that CNNs instantiate the hippocampus. It studies pretrained and random networks, spatial decoding, cell-like unit classification, and unit exclusion. Their central functional claim concerns availability of spatial information without a privileged role for conventionally labeled units. The draft's joint tuning comparison addresses classification validity and biological similarity, but does not itself rebut decoding or redundancy. Criticism of their HD criterion needs a minimal reproducible example, exact version/commit, corrected criterion, and rerun classifications; it should not imply that all their findings depend on that classifier. Publisher listings identify v2 as a **reviewed preprint**, assessed “Important” with “Incomplete” evidence; avoid presenting that assessment as an independent disproof.

### Franzius, Sprekeler & Wiskott (2007), Slowness and Sparseness Lead to Place, Head-Direction, and Spatial-View Cells

[PLOS Computational Biology, DOI 10.1371/journal.pcbi.0030166](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030166), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC1963505/).

This is a crucial direct predecessor: feedforward processing of visual input, trained with slow feature analysis plus sparse coding, generates invariant spatial responses depending on exploration statistics. It explicitly studies position/head-direction joint space and contrasts localized, directional, and invariant responses. The relative speeds of translation and rotation alter what the model learns; its linear-track results also include conjunctive cells. It supplies a counterexample to a broad claim that visual input cannot yield invariant place/HD-like representations. Include it as a positive control and an interpretation of training-distribution effects, not just a distant citation. The sharper unanswered question is why some visual objectives/exploration regimes produce conditional stability while others do not.

### Weber & Sprekeler (2018), Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity

[eLife, DOI 10.7554/eLife.34560](https://elifesciences.org/articles/34560), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5927772/).

This model explicitly aims to explain selectivity and invariance together. Excitatory/inhibitory input statistics determine localized, periodic, and invariant response patterns along spatial dimensions. It is not a raw-pixel deep-network comparator, but it substantially narrows novelty claims about introducing joint selectivity/invariance as a modeling target. It motivates an extension comparing input correlation structure with learned output invariance, including mechanisms other than path integration. A model that generates both invariant and conjunctive units would also be a useful calibration family for the proposed statistic.

## Biological grounding and precedents for analysis

### Muller et al. (1994), On the directional firing properties of hippocampal place cells

[Journal of Neuroscience, DOI 10.1523/JNEUROSCI.14-12-07235.1994](https://pubmed.ncbi.nlm.nih.gov/7996172/).

The distributive hypothesis already formalized how occupancy coupling creates apparent directional tuning from a place-only response. It explained cylinder-foraging data well, but failed to explain genuine directional firing on an eight-arm maze. The same paper therefore supports both the draft's sampling-confound motivation and a strong qualification: directional properties change with environment/task. Cite this where motivating correlated sampling, and phrase any new biological result as a quantified comparison in the sampled context, rather than a discovery that rodent place cells are categorically HD invariant.

### Burgess et al. (2005), Characterizing multiple independent behavioral correlates of cell firing in freely moving animals

[Hippocampus, full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC2678277/).

This develops multivariable treatment of correlated behavioral variables and discusses the earlier distributive hypothesis. It is relevant methodological background and an appropriate conventional-model comparator. Its existence means the manuscript should distinguish limitations of a specified multiplicatively separable model from limitations of all joint or GLM analysis. A model with interaction regressors is not limited to separable tuning simply because it uses a GLM likelihood/link.

### Rubin, Yartsev & Ulanovsky (2014), Encoding of Head Direction by Hippocampal Place Cells in Bats

[Journal of Neuroscience, DOI 10.1523/JNEUROSCI.5393-12.2014](https://pubmed.ncbi.nlm.nih.gov/24431464/), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6608343/).

In crawling bats, spatially selective cells also carry directional information after behavioral-coupling controls. The paper includes per-spatial-bin directional estimation and simulations where preferred directions differ between locations, demonstrating failure of a multiplicatively separable estimate in those cases. This is directly relevant to both joint-map motivation and separability criticism, although bats are not evidence against a specifically rodent dataset result. Distinguish stable tuning shape from gain modulation: a gain-modulated conjunctive neuron may look invariant to Pearson correlation of conditioned curves.

### Jercog et al. (2019), Heading direction with respect to a reference point modulates place-cell activity

[Nature Communications, DOI 10.1038/s41467-019-10139-7](https://www.nature.com/articles/s41467-019-10139-7), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6536526/).

Freely moving mouse CA1 responses can depend on heading relative to a reference point, combining position and direction. Such a code can appear complicated in Cartesian position-by-allocentric-heading coordinates while having a simple interpretation in a different reference frame. It directly warns against interpreting entanglement as intrinsically unstructured or biologically implausible. A natural extension is to ask whether model interactions are captured by egocentric boundary/reference-point variables versus remaining unstructured after this transformation.

### Gerlei et al. (2020), Grid cells are modulated by local head direction

[Nature Communications, DOI 10.1038/s41467-020-17500-1](https://www.nature.com/articles/s41467-020-17500-1), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7445272/), [author code](https://github.com/MattNolanLab/grid_cell_analysis).

Cells classified as pure grid cells can show multiple preferred directions within fields, with directional preferences differing across fields. The study uses local analyses and controls for the distributive hypothesis. This is especially close to the conceptual claim that marginal properties miss local position-by-direction dependence. It also illustrates that interactions may be computationally useful, potentially distinguishing views at one location. It does not directly contradict a CA1/place-cell or postsubicular/HD comparison, but blocks a general brain-disentangled/model-entangled dichotomy. Its data/code provide a promising external benchmark for detecting interactions that a valid framework should recover.

### Raudies et al. (2015), Head direction is coded more strongly than movement direction in a population of entorhinal neurons

[Brain Research, DOI 10.1016/j.brainres.2014.10.053](https://pubmed.ncbi.nlm.nih.gov/25451111/), [author PDF](https://www.bu.edu/hasselmo/RaudiesHasselmo2015.pdf).

The direct predecessor analyzes epochs when HD and movement direction disagree, finds predominantly HD coding, and shows difficulties caused by substituting HD for translational direction in grid models. Therefore the manuscript's HD-versus-MD result should be framed as quantitative extension to its recorded region/population and analysis, not a new discovery of that distinction. Stronger evidence would test the population readout of movement direction after controlling HD and speed, particularly during backward/lateral movement. Marginal HD dominance or Pearson shape stability alone cannot establish absence of movement information in population activity or small rate changes.

### Preston-Ferrer et al. (2016), Anatomical organization of presubicular head-direction circuits

[eLife, DOI 10.7554/eLife.14592](https://elifesciences.org/articles/14592).

Presubicular HD responses during passive rotation were associated especially with layer-3 pyramidal neurons, whereas other layers/cell classes had different tuning. This supports reporting anatomical and recording-condition specificity. It also helps resolve terminology: dorsal presubiculum is classically called postsubiculum. The draft should make its region naming consistent without silently treating MEC, presubiculum, and postsubiculum as interchangeable populations.

### Winter et al. (2015), Passive Transport Disrupts Grid Signals in the Parahippocampal Cortex

[Current Biology, DOI 10.1016/j.cub.2015.08.034](https://www.sciencedirect.com/science/article/pii/S0960982215010052).

Passive transport disrupted grid properties while preserving broad HD properties, supporting a distinction among spatial signals and movement-related mechanisms. It does not establish that every spatial representation requires path integration. If the draft retains strong mechanistic discussion, separate grid-generation evidence from place/HD invariance and from what a visual ANN can represent.

### Liu et al. (2026), Directional dynamics in the entorhinal cortex of male mice driven by behavioral constraints

[Nature Communications, DOI 10.1038/s41467-026-70289-3](https://www.nature.com/articles/s41467-026-70289-3), [author institution abstract](https://mgv.pku.edu.cn/english/research/rp/5cf2064f3b9842ca9041085bcf8418c1.htm), [Crossmark record](https://crossmark.crossref.org/dialog/?doi=10.1038%2Fs41467-026-70289-3).

This recent study compares freely exploring male mice with head-fixed assisted transport. Approximately half of free-foraging HD cells lost tuning reversibly; other neurons gained directional tuning, especially with richer sensory cues. The new tuning involved experience and a spatial component. This is a direct reason to qualify functional types by behavior and environment. Crossmark lists first online 9 March 2026 and version-of-record license from 22 April 2026; cite year 2026 to avoid inconsistent aggregator dates. It is a biological context extension, not an exact replication of the manuscript's postsubicular recordings.

### Peng et al. (2025), Grid cells accurately track movement during path integration-based navigation despite switching reference frames

[Nature Neuroscience, DOI 10.1038/s41593-025-02054-6](https://www.nature.com/articles/s41593-025-02054-6). Published 10 September 2025.

During a self-motion navigation task, grid-population representations track movement across changes of reference frame; their orientation drift predicts homing direction. The results challenge a rigid global-map description and show useful movement information at population level despite unstable conventional spatial fields. This sharpens the distinction between failing a rate-map criterion and failing navigation. It also requires narrowing statements that a movement-direction representation remains unidentified: the unresolved source/computation of inputs to a particular circuit is different from movement information being decodable from grid populations. The paper explicitly leaves the site of path integration—within grid circuitry or upstream—open.

## Claims/evidence decision map

| Draft-level claim | Literature-informed judgment | Necessary change or evidence |
|---|---|---|
| Marginals can hide interactions and sampling confounds | Sound but well established | Cite Muller, Burgess, Rubin, Gerlei; make a precise estimand or benchmark the contribution |
| Joint tuning is the neglected missing analysis in vision-map papers | Plausible narrowly for selected unit comparisons; overstated globally | Acknowledge existing population analyses and direct older joint/invariance models |
| The studied ANN units differ from recorded biological units | Potentially useful empirical claim | Match sampling/noise/smoothing; report effect sizes across animals, sessions, model seeds, environments |
| Biological place/HD cells have invariant responses | Must be restricted to response property and context | Separate absolute-rate invariance, shape invariance, gain, reference frames, and region/task |
| ANN entanglement argues against cognitive maps | Unsupported at present | Measure population geometry and heading-held-out readout/generalization |
| Vision-only training cannot account for invariant spatial cells | Contradicted as a general claim by older sensory learning work | Compare objectives/exploration; do not infer necessity of path integration |
| HD cells do not provide movement direction for path integration | Prior support exists; manuscript can extend it | Test conditional population decoding and rate/gain sensitivity; qualify circuit/region |
| Luo's classifier invalidates their main result | Not established by current draft description | Reproduce exact bug/classifier effect and distinguish cell labels from decoding/exclusion conclusions |

## Experiments that naturally sharpen the paper

These are proposed analyses, not reported outcomes. Feasibility is conditional on obtaining raw activity, behavioral trajectories, model checkpoints and/or image inputs; score-only NPZ files cannot answer most of them.

1. **Matched conditional-tuning benchmark (highest priority).** Fit identical estimators to neural and model joint maps after downsampling model observations, injecting controlled count noise, and matching occupancy/coverage. Report shape similarity separately from gain/offset and full-rate predictability. Include pure, gain-modulated, locally directional, and shifted-field simulations. Low compute; raw data and a validated estimator are the main requirements. This determines whether the original empirical claim survives measurement differences.
2. **Held-out heading population readout (best limited expansion).** Train position decoders on disjoint HD sectors and test on held-out sectors; reverse the roles for HD across spatial regions. Compare within-condition interpolation, nonlinear decoding, and random feature/pixel baselines with matched unit counts and regularization. Use blocked splits to avoid neighboring-frame leakage. Modest compute with stored activations. This connects joint structure to useful generalization and fairly engages Gornet/Luo.
3. **Representation-basis control.** Apply random orthogonal rotations or invertible linear transformations to latent populations and recompute unit statistics and decoding. A changed unit score with unchanged readout exposes what is genuinely single-unit rather than population-level. Cheap with activations; interpret neural axes as biologically meaningful but avoid claiming code-level defects from axis-sensitive measurements.
4. **Exploration/objective intervention (best mechanistic expansion).** Train prediction, reconstruction, temporal-slowness, and optionally contrastive objectives with matched architectures/data. Cross translation:rotation ratio and coupled:independent motion. Include a slow-feature positive control. Measure prediction performance as well as invariance/generalization. Moderate to high compute; choose a small factorial design after identifying the cheapest successful setup. This tests a credible mechanism: invariance depends on objective and exploration statistics.
5. **Reference-frame explanation.** Test whether interactions are explained by bearing/distance to walls or salient reference points, controlling model capacity and held-out likelihood. Useful for both ANN units and biological examples. Modest analysis compute but additional environment metadata needed. This could shift the narrative from “entanglement is failure” to identifying what coordinates the system actually represents.
6. **External biological stress test.** Apply the validated method to a locally directional grid-cell dataset and to a separate hippocampal context, ensuring it recovers known conditional structure. Moderate engineering, little training compute; public-data format/coverage is the risk. Strongly improves credibility beyond a convenient invariant sample.
7. **HD-versus-MD generalization.** Analyze large HD–MD disagreement epochs with speed thresholds and tracking-uncertainty controls, fit gain-sensitive encoding models, and test residual population MD decoding conditioned on HD. Moderate data-analysis effort. A successful result extends Raudies regionally; a null should be stated as a power/decoding bound, not absence of all movement information.

The smallest convincing revision is a careful estimator/measurement repair plus a restricted model–brain comparison. The most coherent broader paper would add experiments 2–4 and explain **when spatial information becomes usable across changes of viewpoint, and how that relates to individual-unit invariance**. Neither a universal defeat of visual models nor a new path-integration mechanism is justified by the current comparison alone.

## Strongest skeptical objection to the proposed broader story

A skeptical reviewer could reasonably say: “That marginal tuning is insufficient is old; single-unit selectivity is basis-dependent; population generalization is already a standard representation-learning test. Why does combining these produce new science?” Merely adding a decoder would not answer this. The paper needs at least one discriminating result: (a) existing marginal criteria rank models differently from cross-condition biological/functional tests; (b) a specific objective or sampling intervention causally changes that ranking while controlling task performance; or (c) an estimand demonstrably recovers a meaningful biological distinction that standard methods miss at realistic sample sizes. If the empirical outcome is only that random CNNs have weaker heading generalization than selected place cells, the value is likely a careful benchmark/resource or methods note, not a major mechanistic account.

Further, heading is not automatically a nuisance: image prediction requires preserving it, and navigation may require both position and orientation. Define which downstream task should generalize over which variable. A prediction model should not be penalized for retaining necessary information; test whether invariant position is also accessible, its sample efficiency, and whether that information generalizes under the intended intervention. For the same reason, a positive result for invariant single units need not imply optimal behavior, and a negative one need not imply a bad model.
