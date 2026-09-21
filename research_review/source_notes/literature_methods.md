# Methods literature audit

Prepared 2026-09-14. Scope: the cross-fold slice-correlation estimator, joint tuning, separability, dependent sampling, variance attribution, and population generalization. Read the supplied manuscript extraction (`tmp/pdfs/manuscript.txt`). Sources below are original research, author-hosted research papers, or official software documentation. This is a targeted broad review, not a systematic review or guarantee of exhaustive coverage. Dates use article metadata rather than search-engine crawl dates. Source summaries are deliberately short; recommendations and mathematical examples are our analysis.

## Main finding

The paper has a potentially useful empirical comparison, but the proposed statistic is not a new general solution to variance attribution or an unbiased measure of response invariance. Its closest intellectual relatives are established tuning-shape separability metrics in sensory neuroscience, noise-corrected tuning correlation, and crossvalidated representational distances. Joint tuning itself is longstanding. A defensible contribution would be a carefully controlled comparison of **conditional spatial tuning and its functional consequences**, supported by established estimators and a precise separation of response sensitivity, gain modulation, tuning-shape stability, and population generalization.

## Closest precedents and what each changes

### Essential spatial precedent: Franzius, Sprekeler & Wiskott (2007)

**Slowness and Sparseness Lead to Place, Head-Direction, and Spatial-View Cells**, *PLOS Computational Biology*, 3:e166, published August 31, 2007. [Primary article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030166).

This visual-learning model already quantifies location/heading selectivity through conditional variance: `eta_r = E_phi Var_r f(r,phi)` and `eta_phi = E_r Var_phi f(r,phi)`, evaluated over uniform joint sampling. It explicitly links the resulting place, HD, and view responses to movement statistics.

**Implication:** unusually close precedent for the spatial question and a variance-based replacement. Under product weighting these are unnormalized total-effect sensitivity measures. A repaired variance metric is not itself new; the claim must rest on controlled new evidence and functional or mechanistic insight.

### 1. Grunewald & Skoumbourdis (2004): joint tuning, separability, and information

**The integration of multiple stimulus features by V1 neurons**, *Journal of Neuroscience*, 24:9185–9194. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6730054/); [DOI](https://doi.org/10.1523/JNEUROSCI.1884-04.2004).

They measured full direction×disparity response matrices, quantified joint tuning and separability, compared alternative combinations of single-feature functions, and related separability to information. They explicitly distinguished single-feature tuning, separable joint tuning, and nonseparable joint tuning. A multiplicative interaction can be statistically significant while remaining separable.

**Implication:** the opening conceptual argument and cartoon taxonomy are established. Cite this near the first claim about marginal tuning insufficiency. The new setting and controlled brain–model result, rather than discovering the need for joint tuning, must carry novelty. Specify the scale on which an interaction is defined.

### 2. Rust & DiCarlo (2010): selectivity, tolerance, and split-half separability

**Selectivity and Tolerance (“Invariance”) Both Increase as Visual Information Propagates from Cortical Area V4 to IT**, *Journal of Neuroscience*. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2975390/); [DOI](https://doi.org/10.1523/JNEUROSCI.0179-10.2010); [author course PDF](https://www.cns.nyu.edu/~david/courses/sm12/Readings/rust-dicarlo-2010.pdf).

They distinguish transformation tolerance from mere receptive-field extent and use object×transformation response matrices. Their single-neuron separability analysis splits repeated trials, constructs a separable prediction from one half, and correlates it with the independently measured half. Population tolerance is also evaluated.

**Implication:** exceptionally close precedent for both the vocabulary and split-half joint-tuning logic. The proposed score is not identical, but novelty cannot be claimed at the level of comparing conditional tuning curves using held-out correlations. Strong template for connecting single-unit structure with population readout.

### 3. Nowak et al. (2011): correlation-based conditional shape similarity

**Separability of stimulus parameter encoding by on-off directionally selective rabbit retinal ganglion cells**, *Journal of Neurophysiology*, 105:2083–2099. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3094178/); [DOI](https://doi.org/10.1152/jn.00941.2010).

They compare direction tuning across speed and luminance, asking whether conditional tuning differs only by gain. Their shape-similarity index normalizes curves, forms a median curve, and computes a Kendall correlation. The paper identifies a Pearson/mean-based antecedent in Rapela et al. (2010).

**Implication:** slice-shape correlation already answers a separability question. Calling the manuscript score “conditional tuning-shape dissimilarity” would align with this literature; calling it the amount a variable drives responses would not. The biological gain changes are signal, even when shape correlation is one.

### 4. Rapela et al. (2010): antecedent of tuning-curve similarity

**ePPR: a new strategy for the characterization of sensory cells from input/output data**, *Network*, 21:35–90. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/20735338/); [DOI](https://doi.org/10.3109/0954898X.2010.488714).

Metadata verified. The precise claim that it introduces the mean/Pearson version of the shape-similarity index is verified from Nowak et al.'s Methods, not independently from the full ePPR text. Treat it as an antecedent worth checking, rather than claiming exact equivalence to the manuscript estimator.

### 5. Pospisil & Bair (2021): attenuation persists after splitting

**Accounting for Biases in the Estimation of Neuronal Signal Correlation**, *Journal of Neuroscience*, 41:5638–5651. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8244973/); [DOI](https://doi.org/10.1523/JNEUROSCI.2775-20.2021).

Signal correlation has distinct biases from response variability and correlated noise. The paper develops corrected expected-response correlation estimators and a split-trial form. Separating trials addresses shared-noise contamination; the correction for noisy tuning curves is a separate step.

**Implication:** the manuscript's argument “independent folds imply noise-corrected Pearson correlation” is insufficient. Benchmark against this family, with heteroskedastic counts, sparse occupancy, and low-signal slices. Pairwise correction is not automatically achieved by dividing an average off-diagonal correlation by the average diagonal reliability.

### 6. Pospisil & Bair (2021): what explained variance means under neural noise

**The unbiased estimation of the fraction of variance explained by a model**, *PLOS Computational Biology*, 17:e1009212. [Primary article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009212).

They analyze noise bias in squared prediction correlation and develop an estimator targeting correlation of expected responses, with confidence intervals and signal-to-noise diagnostics. Low signal can leave very broad uncertainty even with a corrected estimator.

**Implication:** rename the raw `1−r` quantity; it is neither squared prediction correlation nor a variance fraction. Report uncertainty and identifiability rather than treating reliable-looking point estimates as universal evidence. Noise correction and low variance are different goals.

### 7. Walther et al. (2016): an unbiased distance alternative

**Reliability of dissimilarity measures for multi-voxel pattern analysis**, *NeuroImage*. [Author PDF](https://diedrichsenlab.org/pubs/Walther_Neuroimage_2016.pdf).

They compare neural pattern dissimilarities and show that crossvalidated squared-distance estimates have a meaningful zero under identical true responses, unlike conventional positively biased distances. Noise normalization improves reliability; the recommended crossvalidated Mahalanobis form is a population metric.

**Implication:** adapt the independent-fold inner-product principle to slice differences. This preserves gain and offset if vectors are left uncentered. It needs a stated weighting metric and independent errors; heteroskedasticity and shared preprocessing still matter. It is established machinery, not a novel estimator solely because applied to spatial slices.

### 8. Diedrichsen & Kriegeskorte (2017); Diedrichsen et al. (2020): geometry and covariance

**Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis**, *PLOS Computational Biology*, 13:e1005508. [Primary article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005508).

This work connects encoding models, pattern components, and RSA through the second moment of neural activity, and gives a formal model-comparison framework. It provides established context for comparing joint spatial geometry across neural data and networks.

**Comparing representational geometries using whitened unbiased-distance-matrix similarity**. [Author preprint](https://arxiv.org/abs/2007.02789).

This develops covariance-aware comparisons of crossvalidated dissimilarities. Pairwise dissimilarities are statistically dependent; noise biases can induce spurious structure. **Implication:** do not count all slice pairs as independent samples, or treat split repetitions as independent biological replicates.

### 9. Schütt et al. (2023): statistical inference beyond the current sample

**Statistical inference on representational geometries**, *eLife*, e82566. [Primary article](https://doi.org/10.7554/eLife.82566).

The paper develops methods for inference over sampled subjects and conditions, including two-factor bootstrap and crossvalidation, and discusses biases in distance estimation.

**Implication:** brain–model comparisons should state whether the inference concerns these cells, these animals, these environments, or a broader population. Animal/session/environment and training-seed variability cannot be replaced by repeated random fold splits.

### 10. Kobak et al. (2016): dPCA already decomposes main and interaction variance

**Demixed principal component analysis of neural population data**, *eLife*, 5:e10989. [Primary article](https://elifesciences.org/articles/10989).

Its marginalization procedure decomposes each neuron's activity into task-parameter effects and interactions before extracting population readouts. For the specified factorial average, components are uncorrelated; the paper also corrects estimated marginalization variances for residual noise. dPCA's readouts operate on neural activity rather than requiring the marginalization at test time.

**Implication:** a very relevant baseline. It does not solve missing combinations or arbitrary dependent sampling by itself. Separate “single-neuron joint decomposition” from “population demixing”; analyze both. Existing dPCA methods make a general new variance-decomposition pitch difficult.

### 11. Wood (2006) and mgcv: GLMs do not require separability

**Low-rank scale-invariant tensor product smooths for generalized additive mixed models**, *Biometrics*, 62:1025–1036. [Research record](https://pubmed.ncbi.nlm.nih.gov/17156276/); [DOI](https://doi.org/10.1111/j.1541-0420.2006.00574.x).

Wood constructs efficient smooth multivariate terms from marginal bases, supporting models with interacting variables and separately interpretable smoothness penalties. The current [official tensor smooth documentation](https://www.stat.ethz.ch/R-manual/R-devel/library/mgcv/html/te.html) explicitly supplies full multivariate smooths (`te`) and interaction-only smooths (`ti`) alongside main effects.

**Implication:** replace “GLMs assume a product of marginal tuning curves” with “a log-link model containing only separate functions of each variable imposes multiplicative separability.” A model linear in tensor-basis coefficients can model nonlinear interactions in raw behavioral variables. Compare main-effects-only log-rate, additive-rate, gain/separable, and joint-interaction models using held-out data.

### 12. Balzani et al. (2020): scalable nonlinear encoding during navigation

**Efficient estimation of neural tuning during naturalistic behavior**, NeurIPS 33. [Primary conference paper](https://papers.nips.cc/paper_files/paper/2020/hash/94d2a3c6dd19337f2511cdf8b4bf907e-Abstract.html); [PDF](https://papers.nips.cc/paper/2020/file/94d2a3c6dd19337f2511cdf8b4bf907e-Paper.pdf).

The Poisson GAM estimates spline nonlinearities, feature contributions, and uncertainty during naturalistic navigation, and is applied to monkey recordings. It targets efficient estimation under many candidate variables.

**Implication:** realistic encoding-model comparator and implementation reference. An additive GAM without multivariate terms still needs augmentation to detect arbitrary joint interactions; do not overstate this particular implementation as already solving all interactions. The manuscript should engage this applied literature rather than only a 2008 retinal GLM reference.

### 13. Hooker (2007): dependence and unsupported regions

**Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables**, *Journal of Computational and Graphical Statistics*, 16:709–732. [Author bibliography](https://www.gileshooker.com/papers.html); [research paper](https://doi.org/10.1198/106186007X237892); [paper text mirror](https://citeseerx.ist.psu.edu/document?doi=22f631355f1fc74f3f08a7241dd089ed39fb5147&repid=rep1&type=pdf).

Weighted functional ANOVA diagnoses functions with dependent inputs while limiting emphasis on regions of very low probability. The motivation includes poor extrapolation by learned models outside the data's support.

**Implication:** evaluating a balanced Cartesian spatial grid and evaluating naturally occupied spatial states are different estimands. A joint map does not eliminate dependence problems if many combinations are absent and smoothing invents estimates there.

### 14. Chastaing, Gamboa & Prieur (2012/2015): dependent-input sensitivity

**Generalized Hoeffding-Sobol decomposition for dependent variables—application to sensitivity analysis**, *Electronic Journal of Statistics* (2012). [Author preprint](https://arxiv.org/abs/1112.1788); [DOI](https://doi.org/10.1214/12-EJS749).

They give a generalized decomposition and sensitivity indices under conditions on the joint input distribution. The independent-input decomposition is a special case. The [numerical follow-up](https://arxiv.org/abs/1303.4372) develops estimation based on hierarchical orthogonality.

**Implication:** there is extensive prior theory for the manuscript's stated variance-attribution goal. Uniform/product-distribution ANOVA is straightforward for fully measured model grids, but observational occupancy needs explicit assumptions; dependent-input methods do not magically identify off-support neural responses.

### 15. Owen & Prieur (2017): alternative attribution conventions

**On Shapley Value for Measuring Importance of Dependent Inputs**, *SIAM/ASA Journal on Uncertainty Quantification*. [Author paper](https://artowen.su.domains/reports/shapleydependent.pdf); [DOI](https://doi.org/10.1137/16M1097717).

They motivate Shapley attribution when dependent inputs make ANOVA-style importance problematic, using explicit examples to clarify the allocation of importance.

**Implication:** there is no attribution-free meaning of “the contribution of x.” A predictive allocation can credit a correlated proxy; a controlled response-invariance question asks something else. Shapley need not be added as another metric, but the distinction should inform the chosen estimand.

### 16. Bernardi et al. (2020): abstraction need not live in pure neurons

**The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex**, *Cell*, 183:954–967.e21. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8451959/); [DOI](https://doi.org/10.1016/j.cell.2020.09.031).

Cross-condition generalization performance (CCGP) tests whether a decoder transfers to combinations excluded from decoder training. Neural and artificial populations can jointly support abstraction and high-dimensional mixed codes. Their discussion explicitly explains why inspecting individual neurons misses population organization.

**Implication:** the strongest natural expansion is to test whether differences in slice stability predict out-of-condition spatial decoding. Correlations across nonlinear components of different neurons matter, so single-cell “entanglement” does not establish absence of a useful population map.

### 17. Chericoni et al. (2026): a current spatial generalization precedent

**Neural geometry in the human hippocampus enables generalization across spatial position and gaze**. *Neuron*, available online September 9, 2026, in-press corrected proof. [Publisher article](https://www.sciencedirect.com/science/article/pii/S0896627326006136); [DOI](https://doi.org/10.1016/j.neuron.2026.08.004); [earlier March 5 arXiv version](https://arxiv.org/abs/2603.04747).

Human hippocampal recordings during virtual pursuit reveal mixed spatial/agent/gaze responses, related subspaces, and decoder generalization across spatial maps. Its functional tests distinguish linear subspace alignment from direct decoder transfer.

**Implication:** current, close context for a geometry/abstraction expansion, but not the same experiment as testing location decoding across held-out head directions. Do not substitute human gaze/agent results for rodent head-direction evidence.

**Verified status:** the arXiv record had no journal reference, but the publisher record independently identifies the Neuron version as available online September 9, 2026. Use the journal DOI for the updated manuscript.

## Mathematical audit: consequences for the manuscript's estimand

These are direct deductions from the stated statistic, not borrowed empirical findings.

Let `D_A = 1 − mean_{i != j} corr(f(a_i,·), f(a_j,·))` for noiseless slices where correlation exists.

1. **Additive sensitivity is invisible.** For `f(a,b)=u(a)+v(b)` with nonconstant `u,v`, all within-axis slice correlations equal one, so both scores equal zero. Both variables nevertheless contribute positive response variance.
2. **Positive multiplicative gain is invisible.** For `f(a,b)=u(a)v(b)` with positive, nonconstant factors, both scores equal zero. This is mixed response sensitivity and separable shape, not invariance of the response.
3. **Pure selectivity creates undefined correlations on the relevant axis.** If `f(a,b)=u(a)`, the slices used to estimate `D_A` are constant vectors over `b`. Their Pearson correlations are undefined. A numerical convention mapping them to zero creates `D_A=1` without a population-correlation estimand there. The noisy-data limit does not make the noiseless statistic defined.
4. **The range is not a variance fraction.** Individual `1−r` lies in `[0,2]`; cross-fold pairwise averages have no generic conservation law across variables. Even an ordinary squared prediction correlation would require a different definition and interpretation.
5. **Splitting does not remove attenuation.** With independent zero-mean measurement error, cross-fold covariance removes shared noise in its numerator, but each ordinary Pearson denominator still includes noise variance. Thus invariant true shapes can produce positive dissimilarity. Heterogeneous slice reliability makes a single global correction especially questionable.
6. **A uniform map and a behavioral map answer different questions.** An observational marginal uses `p(b|a)`; a standardized marginal uses a chosen common `q(b)`. A joint map can support the latter only where it is measured adequately. Correlated sampling is not solved simply by slicing estimated cells.

These issues are fundamental, not minor phrasing. They do not imply that the visual observations or brain–model contrast are false; they imply that the current numerical summaries do not establish their stated interpretation.

## A feasible repair and expansion path

### Minimum defensible repair

- Define the biological target first: **strict response invariance**, **shape invariance allowing gain/offset**, **predictive importance**, or **cross-condition generalization**. Report separate quantities when the question needs more than one.
- Keep conditional map plots and optionally retain slice correlation as a descriptive shape statistic. Report slope/gain and offset changes separately, with weak/constant slices identified as non-estimable rather than silently assigned a number.
- Use common support, occupancy-matched analyses, and simulated sampling of model units along recorded animal trajectories. Use contiguous/session folds where needed, with smoothing fitted/applied separately. Matching only bin counts is inadequate.
- Fit nested hypotheses: pure location, pure HD, additive response, multiplicative/log-additive response, and flexible joint interaction. Compare held-out likelihood or response prediction; penalize model flexibility. Poorly identified bins should yield uncertainty rather than interpolated certainty.
- Validate under additive, gain-only, moving-field, conjunction, flat, and pure-axis functions; include nonuniform/near-collinear trajectories, Poisson and overdispersed errors, temporal dependence, and varying slice signal strength. These are diagnostic benchmarks, not cosmetic tests.

### A true variance decomposition, if desired

On a fully covered factorial grid with a declared product weighting, decompose each fold's expected map into grand mean, A main effect, B main effect, and interaction using functional ANOVA. Estimate each component's signal energy with independent-fold inner products. For example, `Vhat_AB = <f_AB^(0), f_AB^(1)>_q`. Under unbiased map estimates and independent fold errors, this targets the corresponding squared component norm. Total-effect sensitivity to A includes A's main-effect and interaction energy; shape change is a different statistic. Ratios of unbiased estimates need not be unbiased, and may be unstable at low signal; report component estimates, intervals, and denominator reliability. Under missing support, prefer supported-domain prediction comparisons or explicitly justified dependent-input decomposition rather than filling the full grid silently.

Alternatively, average cross-fold slice-difference inner products: `<f_i^(0)−f_j^(0), f_i^(1)−f_j^(1)>_q`. This estimates real changes in the mean response profile, preserving amplitude and offset, and has a genuine zero under invariance. It measures conditional sensitivity; it is not automatically an orthogonal allocation of total variance. Both options draw on established methods, so novelty would lie in the scientific comparison and validation protocol.

### Most valuable expansion: conditional code structure predicts functional transfer

Train a spatial decoder on a subset of HD sectors and test on held-out HD sectors; train an HD decoder on a subset of spatial regions and test on held-out regions. Compare within-condition decoding, direct transfer, and transfer after a learned low-complexity alignment. Match cell numbers, occupancy, signal strength, and training data. Evaluate simple linear readouts and a more flexible readout separately. For circular HD, use sine/cosine targets or appropriate circular loss; for XY, distance error is more meaningful than arbitrary accuracy alone.

**Interpretation branches:** if networks retain ordinary decoding but fail transfer where the brain succeeds, the paper supports a functional distinction between spatial information and a reusable spatial code. If networks transfer despite single-unit shape changes, the result instead demonstrates that cell-type similarity is a poor proxy for population abstraction. If both systems have comparable conditional structure after controls, the current contrast should be reduced, and the useful contribution may be a negative validation/reanalysis rather than a broad mechanistic claim.

**Feasibility:** the core estimator repairs and synthetic stress tests are low-compute work; data plumbing and unit-selection auditing are likely the main cost. Encoding-model fits and cross-HD transfer on existing dense model grids are moderate effort. Biological cross-condition transfer is constrained by support and recording duration, not merely GPU availability. New network training with matched environments and objectives is a larger, optional step. A new recording experiment is warranted only if existing data cannot separate heading from movement or cannot populate enough held-out combinations.

## Suggested framing

Potential title direction: **Conditional spatial tuning distinguishes local sensory codes from reusable spatial representations** — but only after the transfer experiment establishes reusability differences. With current evidence alone, use the narrower **Joint spatial tuning reveals differences between visual network units and rodent spatial cells**.

The useful narrative is: marginal spatial tuning is a weak constraint; quantify multiple well-defined levels of joint structure; use matched biological/model measurements; test which differences matter to readout. Avoid a blanket rejection of GLMs, a general claim that vision cannot form maps, or an assumption that nonlinear mixed selectivity is intrinsically inferior. The field already knows interactions, separability, and mixed population codes; what can still be useful is a rigorous spatial benchmark that establishes where existing model comparisons succeed or fail.

## Skeptical-reviewer test: when this would still be redundant

Adding shape, gain, and CCGP panels is not sufficient by itself. A reviewer could fairly say that separability is old, visual spatial coding is old, and population abstraction is old. A publishable expanded narrative needs a result that changes an inference: for example, established place-cell classifications rank models as similar while matched out-of-heading transfer reverses that ranking; or controlled changes to temporal learning objective or experience selectively improve cross-heading transfer without improving marginal place tuning. That would tell readers which model validation target matters and why.

Conversely, the case would be weak if the observed difference disappears under matched field of view, occupancy, unit selection, or reliability; if it is already predicted by a simple feedforward visual baseline and no functional consequence is established; if the network cannot reasonably be expected to perform the tested generalization because its objective and experience never require it; or if an existing visual SFA baseline already matches the biology and the paper still frames vision versus path integration as the decisive distinction. In those cases, a narrower correction/reanalysis could remain useful, but a broad methods or mechanism paper would be hard to defend.

A positive result for visual networks is not a failed expansion: transfer despite unstable single cells would support a different, useful warning against inferring map quality from cell types. However, that finding must be demonstrated under meaningful held-out combinations, not inferred from ordinary within-distribution decoding. New-method development is warranted only if existing variance, separability, and transfer tools cannot answer the scientific question under the data's support constraints.
