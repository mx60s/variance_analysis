# Xue-Xin Wei: tuning, geometry, independent coding, and model comparison

Prepared September 14, 2026. This is a targeted primary-source review for positioning *Invariance and selectivity to spatial variables in deep neural networks and the brain*. The assessment uses the existing project review and the corrected manuscript provenance recorded there. It does not reproduce the manuscript's neural results. Published claims, our deductions, and proposed interests are separated below. No contact with the author was made.

## Main assessment

The most compelling connection is a progression from **conditional tuning → population geometry → context-general readout**. This connects the spatial paper to several of Wei's publications more naturally than arguing that a good spatial representation should eliminate information about heading. A putative nuisance for one readout can remain useful for another. The manuscript could test whether a location code supports stable access across viewing directions while preserving information needed for orientation or visual prediction.

This is a judgment about fit to the published work, not knowledge of Wei's current preferences. Its central concepts are already present in his papers, so the contribution must be a new spatial result, a verified distinction between neural resemblance and functional access, or an explanatory intervention. Relabeling the current correlation scores as a new theory of invariant coding would not accomplish that.

## Six closest sources

### 1. Independent encoding with retained contextual information

**Citation:** O'Shea, R. T., Wei, X.-X., & Priebe, N. J. (2026). *Independent Encoding of Orientation and Mean Luminance by Mouse Visual Cortex*. **eNeuro, 13**(2), ENEURO.0281-25.2025. Published January 13, 2026. [Article and author responses](https://www.eneuro.org/content/13/2/ENEURO.0281-25.2025.full); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12893793/). DOI: 10.1523/ENEURO.0281-25.2025.

**Published finding, 64 words:** Mouse V1 populations retain information about mean luminance while maintaining orientation encoding across luminance conditions. The study analyzes response changes for pairs of orientations and lighting conditions, finding approximately orthogonal population encoding axes. Individual neurons respond heterogeneously to luminance; the population organization is compared with random and shared encoding models. Other visual feature pairs provide comparisons showing that independent encoding is not a universal property of every pair.

**Implication:** This is the closest conceptual precedent. Our paper should distinguish stable field shape, absence of nuisance information, and independent population access. A spatial application could be complementary, but cannot claim the general principle as new. Do not equate orthogonality of particular finite-difference vectors with complete statistical independence or global generalization across an entire continuous manifold. Measure those separately.

### 2. A direct precedent for cross-context fixed readouts

**Citation:** O'Shea, R. T., Nauhaus, I., Wei, X.-X., & Priebe, N. J. (2025). *Luminance invariant encoding in mouse primary visual cortex*. **Cell Reports, 44**(1), 115217. DOI: 10.1016/j.celrep.2024.115217. [Publisher full text](https://www.sciencedirect.com/science/article/pii/S2211124724015687); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850277/).

**Published finding, 53 words:** The study compares visual responses across scotopic and photopic conditions. Luminance-dependent changes inherited from the retina remain evident in LGN, whereas mouse V1 selectivity is comparatively stable. Population analyses test whether orientation discrimination and a fixed decoding projection generalize across lighting conditions. Thus the work supplies an empirical precedent for evaluating invariance through readout transfer.

**Implication:** A location decoder trained at one set of headings and tested at another would extend a familiar evaluation strategy into navigation. It would not introduce cross-context decoding itself. The useful contrast could be where invariance appears in visual versus spatial processing, or whether spatially suggestive model cells deliver that functional benefit. The 2025 and 2026 papers are distinct studies but overlap in their experimental material; count them as a connected line, not two independent replications.

### 3. Single-cell tuning does not uniquely determine computational conclusions

**Citation:** Kriegeskorte, N., & Wei, X.-X. (2021). *Neural tuning and representational geometry*. **Nature Reviews Neuroscience, 22**, 703–718. DOI: 10.1038/s41583-021-00502-3. [Publisher](https://www.nature.com/articles/s41583-021-00502-3); [author manuscript](https://arxiv.org/abs/2104.09743).

**Published argument, 49 words:** This perspective relates neuron-level tuning to population geometry and information. Tuning induces representational geometry, but multiple sets of tuning functions can generate the same geometry. The authors connect geometry to discrimination and ideal-observer performance under the relevant assumptions, and recommend studying both individual tuning and population organization.

**Implication:** The current inference from individual-unit entanglement to deficient cognitive maps would sit uneasily beside this argument. A stronger contribution tests that bridge empirically: do conditional single-cell differences predict differences in cross-heading location access? A model could fail the biological cell-level comparison and still succeed at a specified readout. Conversely, good marginal place fields might coexist with poor transfer. Those are informative outcomes, not threats to the project.

### 4. Decompose modulation before interpreting invariance

**Citation:** Zhu, R. J. B., & Wei, X.-X. (2023). *Unsupervised approach to decomposing neural tuning variability*. **Nature Communications, 14**. DOI: 10.1038/s41467-023-37982-z. [Publisher](https://www.nature.com/articles/s41467-023-37982-z); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10121715/).

**Published finding, 70 words:** Poisson functional PCA models smooth tuning variability through latent components in log-rate space. It encompasses gain changes and can capture tuning shifts and sharpening. Applied to macaque V1, it identifies structured modulation and examines implications for Fisher information and representational geometry. The empirical implementation assumes tuning is stable within stimulus blocks, examining fluctuations at roughly ten-second timescales; the presented framework concerns smooth one-dimensional tuning.

**Implication:** The existing score's blindness to gain is especially consequential in this context. Treating gain, field movement, and tuning-shape changes separately would connect naturally with this work. Pf-PCA is a relevant comparison or starting point, not automatically a suitable plug-in for irregularly sampled two-dimensional foraging data. A justified adaptation would have to address occupancy, temporal sampling, and the latent-timescale assumption. A full new latent-variable method is unnecessary for the first workshop analysis.

### 5. Compare task-relevant access, not only overall similarity

**Citation:** Qian, Y. E., Geisler, W. S., & Wei, X.-X. *Quantifying task-relevant representational similarity using decision variable correlation*. Accepted at **NeurIPS 2025**; inspected author version **arXiv:2506.02164v4**, revised March 15, 2026. [Version and acceptance statement](https://arxiv.org/abs/2506.02164v4); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803327/). Some earlier listings use a different title.

**Published finding, 67 words:** Decision variable correlation compares stimulus-by-stimulus outputs of linear readouts from two representations for a classification task. It targets task-relevant similarity rather than unrestricted representational alignment. In the evaluated macaque V4/IT and image-classification models, brain–model agreement is lower than within-kind agreement, and greater model classification accuracy does not necessarily improve agreement. The analysis includes dimensionality control and an explicit correction for attenuation under specified noise assumptions.

**Implication:** This supports testing whether similar-looking spatial codes make similar functional distinctions. However, the present model and rodent data do not contain the same images in a common arena: naively calculating stimulus-by-stimulus brain–model DVC would be unjustified. Use transfer performance first; use DVC for matched model conditions or a redesigned comparison with defensible correspondences. Its split correction is not a blanket endorsement of any cross-fold correlation estimator.

### 6. Noise geometry matters for population capacity

**Citation:** Le, D., & Wei, X.-X. (2026). *Split-trial analysis reveals the information capacity of neural population codes*. **eLife Reviewed Preprint**, v1, August 3, 2026, marked not revised at inspection. DOI: 10.7554/eLife.111658.1. [Reviewed-preprint record and reviews](https://elifesciences.org/reviewed-preprints/111658/reviews); [bioRxiv version posted January 29, 2026](https://www.biorxiv.org/content/10.1101/2025.11.27.690979v2), DOI: 10.1101/2025.11.27.690979.

**Reported result, 65 words:** The authors estimate information-limiting noise by comparing decoded estimates from disjoint neural subpopulations. Simulations and applications include the mouse head-direction system, mouse V1, and macaque prefrontal cortex. This is a reviewed preprint, not a verified final version of record. Public reviews discuss finite-population and finite-trial behavior, including whether common decoder-training samples can induce correlated errors that complicate interpretation.

**Implication:** A capacity expansion would fit an active line but is a larger undertaking than readout transfer. It requires simultaneous populations and explicit noise assumptions. A pseudo-population cannot establish the recorded population's correlated-noise limit. The shared vocabulary of splitting and correlation should not obscure the different estimand or make our shape score a capacity measure. Treat this method as a candidate requiring validation in the present sampling regime.

## Two supporting connections

**Kay et al. (2024), including Wei.** *Emergent neural dynamics and geometry for generalization in a transitive inference task*. **PLOS Computational Biology, 20**(4), e1011954. DOI: 10.1371/journal.pcbi.1011954. [Article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011954). The work studies trained networks with different task solutions and compares their behavioral predictions with human working-memory transitive inference. It supplies a useful style of question: identify competing computations through held-out combinations and discriminating predictions. For our project, manipulate trajectory history or learning statistics only after showing which functional generalization needs explanation; merely plotting a low-dimensional manifold would add little.

**Wei & Stocker (2015).** *A Bayesian observer model constrained by efficient coding can explain 'anti-Bayesian' percepts*. **Nature Neuroscience, 18**, 1509–1517. DOI: 10.1038/nn.4105. [Primary paper record](https://pubmed.ncbi.nlm.nih.gov/26343249/); [closely related primary theoretical paper](https://papers.neurips.cc/paper/4489-efficient-coding-provides-a-direct-link-between-prior-and-likelihood-in-perceptual-bayesian-inference.pdf). This line connects environmental statistics, encoding constraints, and behavioral inference. Its relevance is a hypothesis about why a particular experience distribution makes location easier to access across headings, not a license to describe the present score as coding efficiency. An efficiency claim would require an explicit objective and resource/noise constraints.

## Proposed narratives and decisive experiments

The following are proposed research questions, not findings from Wei's papers or results already established in this project.

### A. A modest paper: field stability and independent readout are different tests

**Question:** Do place-like model units preserve spatial field shape across heading, and does their population support heading-general location readout?

This allows a compact result even if the answers differ. Three analysis levels would be sufficient: calibrated per-cell gain/shape measurements; conditional population structure; a fixed decoder tested on unseen heading sectors. A single carefully evaluated visual model and one biological population can support a bounded case study, provided neural and model sampling/reliability are treated fairly. The same questions can be asked in the reciprocal direction for a head-direction population if that does not dilute the paper.

**Experiments:** Use true histories for temporal models; measure field displacement, gain/baseline changes, and held-out interaction-model improvement; train regularized location readouts on designated heading sectors and test on held-out sectors. Include same-heading oracle or within-condition performance so inability to encode location at all is distinguished from inability to transfer. Report performance versus cell count and training sample count. A known invariant synthetic population and gain-modulated controls establish what each test measures.

**Feasibility:** Approximately one to three focused weeks once usable activations, neural data, and analysis provenance are restored; original-repository recovery may reduce the earlier missing-code burden. These are planning judgments. A 2–4 day pilot on one session and a model subset should determine whether a full comparison is informative.

**Contribution beyond the closest precedents:** A reproducible spatial model benchmark linking conditional fields to functional generalization. Cross-context decoding alone is familiar; the new content must be a reliable model–neural contrast or a consequential dissociation between the two criteria.

### B. A stronger paper: geometry predicts which modulation interferes with readout

**Question:** Which forms of heading dependence change the usable location signal, and which merely carry additional information?

**Experiments:** Estimate location-discrimination vectors at several matched headings and heading-change vectors at matched positions. Analyze whether target directions remain aligned across context, and whether nuisance changes project onto the actual fixed decoder. Compare these measurements with transfer errors. Use independent data for fitting axes and evaluation, common occupancy support, and noise-aware estimation. Test several location pairs; two-dimensional position plus circular heading does not reduce to one universal pair of axes. A low dot product at one condition does not prove global invariance.

**Synthetic discrimination:** Construct populations with preserved field shape and varying gains whose gain directions are either harmless or disruptive to a fixed location decoder; also include a rotated population basis with unchanged linear-access properties. This separates cell identity, encoding geometry, and readout. Fit both linear and modest nonlinear decoders to locate the level of access at which a gap occurs. Report geometry as an explanation of measured transfer, not as a substitute for it.

**Feasibility:** Roughly an additional one to two analysis weeks beyond a validated transfer pipeline. This is the strongest incremental option because it uses existing activity and generates a sharper inference without requiring network training. A result that the same single-cell score conceals materially different transfer behavior would make the metric distinction scientifically useful.

### C. An ambitious paper: experience shapes context-general spatial access

**Question:** Does navigation experience organize a stable location readout while preserving heading information required for other computations?

**Experiments:** Hold architecture and data exposure approximately fixed while manipulating one factor: genuine versus shuffled trajectory order, temporal history length, or rotational versus translational experience. Compare single-cell modulation, target/nuisance geometry, and transfer together. A model trained to predict images may need heading; demanding it disappear from the entire representation would confound the task with the benchmark. A causal result could show that history changes how information is organized without eliminating useful sensory variables.

**Feasibility:** Two to six additional weeks, depending on reproducibility and training cost, with multiple seeds and a held-out environment or trajectory regime. Begin only if the analysis-only pilot exposes a robust phenomenon and the intervention distinguishes specific explanations. Retraining several unrelated architectures is less informative than one controlled intervention.

### D. A distinct branch: preserve information under structured fluctuations

An alternative alignment with the tuning-variability and noise-capacity papers is to ask how heading-dependent gain, field drift, and shared fluctuations separately limit spatial precision. It would require simultaneous recordings, sufficient joint support, and careful differentiation between observable heading effects and latent fluctuations. This is potentially valuable but is not the shortest route to publishing the current work. It should become a separate central question only if the raw data make it particularly tractable.

## What to emphasize in a proposal to Wei

Lead with the unresolved spatial question and a concrete pilot, not a claim to extend all of his research at once. A useful framing is: **the original conditional tuning result may distinguish preservation of single-cell fields from independent population access; we can now test which distinction actually matters for spatial generalization.** Show one figure that separates gain from shape, one reliable brain/model or model-history contrast, and the planned held-out-heading test.

Offer a stopping rule: finish a modest, well-calibrated case study if it yields a clear result; broaden into geometry or an experience intervention only when the pilot exposes something explanatory. Avoid assuming publication ambitions or personal preferences from his publication record. The fit is strongest when the project produces a new answer about spatial representations using the conceptual discipline already visible in his work.
