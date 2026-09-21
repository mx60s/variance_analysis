# The correlation method as a central, testable contribution

**Current framing:** the project investigates conditional tuning-shape stability using its correlation method, then tests when that measurement reveals scientifically useful structure. The method is a candidate central contribution. Its interpretation, estimation and added value must each survive independent attacks. Centrality does not require a claim that Pearson correlation itself is new, or that the implementation cannot change.

This document supplies four shared claim cards, M1–M4, for the three narrative plans. Each plan also has a method-specific claim: O0 for objective effects, F0 for reference-frame discovery, and U0 for functional relevance. A scientific result can survive failure of the method-centered framing; that outcome should be labeled a change of contribution rather than counted as validation of the method.

## Measurement definition and reusable implementation

Let `s` denote the variable along a tuning profile, such as position, and `c` the context, such as heading. Let `f_c(s)` be the expected response on a declared common set of profile bins. The intended population quantity is

`S = mean_(c != c') corr_s(f_c(s), f_c'(s)); D = 1 − S`.

High `S`, or low `D`, means similar centered, normalized profile shape across contexts. It allows positive context-specific gain and additive baseline changes. It is not equality of responses, explained variance, absence of context information, or fixed-decoder generalization. For finite valid correlations, `D` can extend to 2; do not assume a variance-fraction range of 0–1.

The original [implementation](/Users/moo/variance_analysis/src/explained_variance.py) contains `_corr_matrix` and `quantify_variance_corr`. The latter returns **mean off-diagonal correlation and mean diagonal correlation**, in that order. The first return is called `variation` locally, but it is `S_hat`, not `D_hat`; downstream code may take one minus it. Audit this conversion explicitly. The diagonal is a repeat-agreement diagnostic, not an automatic noise correction.

The function moves its selected `axis` to the slice/context dimension and flattens all remaining dimensions into the profile. To measure spatial shape stability across heading, heading must be the compared-slice axis and position the profile domain. This is different from comparing heading curves across positions. Unit and history axes must not accidentally enter the profile. The code uses pooled occupancy divided by two and pair-specific valid-bin masks; neither establishes genuine common support or balanced occupancy in the two folds.

For `f_c(s)=a_c g(s)+b_c`, `a_c>0`, every valid noiseless pair has correlation one. This is the intended invariance, not a bug. Negative gain reverses the profile and is not in that equivalence class. Constant or insufficiently varying profiles have no identifiable Pearson shape and should be marked non-estimable, not assigned maximal stability or instability. A signed network feature requires interpretation on its actual response scale; arbitrary feature-basis rotations can change single-unit shape scores while preserving population readout.

Reuse the original conditional-map code, [diagnostic examples](/Users/moo/variance_analysis/research_review/method_diagnostics.py), saved fields and raw-data access. The unused binning notebook is not a prerequisite or evidence about final-paper validation. Model-based fits and response-sensitive measures below are independent checks and comparators; they need not replace the method as the principal assay.

## Shared claims and their strongest attacks

### M1 — The method isolates a well-defined conditional shape property

**Claim:** on declared common support with nonconstant expected profiles, the statistic quantifies pairwise shape agreement while deliberately ignoring positive gain and constant baseline shifts.

**Current evidence:** the code and Pearson algebra support the narrow affine-invariance statement. They do not establish that every biologically relevant field change produces a large score, or that low average dissimilarity establishes global separability.

**Test/support:** derive the noiseless equivalence class and verify the code-to-symbol mapping. Use pure location tuning across headings, gain/baseline modulation, shifted fields, changed width, added fields, and interactions. Report individual pair correlations and profile plots as well as their aggregate. Strong support consists of exact expected identities in valid cases and declared sensitivity in the field-change family of interest.

**Best killer:** a valid, noiseless, nonconstant, common-support gain/baseline-only case changes the implemented score, or materially different target shapes repeatedly collapse to the same score within the intended effect range. The first defeats implementation fidelity; the second defeats the proposed sensitivity for that phenomenon, not Pearson's algebra. An aggregate hiding a scientifically important subgroup defeats a claim about that subgroup unless its distribution is also measured.

**Controls/branch:** distinguish numerical error, axis reversal, masking and estimator-version differences from a false mathematical claim. If only a narrower shape property is measurable, narrow the claim. Do not repair its meaning by renaming it variance or information.

Equal aggregate scores for different profiles do not by themselves refute this definition: it is not an injective description of every shape. A failed sensitivity test must concern a prespecified meaningful change relative to a reference or candidate decision, rather than demand that one scalar distinguish all possible functions.

### M2 — The property can be estimated reliably in the intended data regime

**Claim:** a specified version of the estimator identifies meaningful shape differences with calibrated uncertainty and controlled nuisance sensitivity in the actual sampling/noise regime.

**Current evidence:** the existing cross-fold procedure is available; previously identified attenuation and heterogeneous reliability prevent claiming this is already established.

**Test/support:** simulate known conditional fields on actual trajectory/support patterns; vary rate, gain, baseline, dispersion, noise dependence, occupancy, binning and smoothing. Include stable controls with unequal slice reliability and context-dependent gain. In noiseless common-support data, gain leaves correlation unchanged; in finite noisy data it changes signal-to-noise ratio and can change the estimate. Benchmark error, interval coverage and detection of a preregistered meaningful field change, including an abstention/non-estimable category.

Compare raw cross-fold correlation with an appropriately justified expected-response correlation estimator and held-out shared-shape versus flexible field models. Fit calibration on development simulations and test new parameter regimes. Pairwise/slice uncertainty does not substitute for animal/model replication.

**Best killer:** plausible matched stable-field controls produce changes as large as the claimed empirical contrast, ordering follows reliability rather than true shape, or uncertainty cannot separate stable and changing fields at usable sample sizes despite adequate controls. Failure to calibrate a meaningful regime defeats use as the central quantitative assay there.

**Controls/branch:** missing support makes some comparisons unidentifiable; a wide interval is inconclusive, not evidence of invariance. A simple mean off-diagonal/mean diagonal ratio is not presumed valid. A corrected estimator may retain the same population target; record it as a new estimator version. Replacing the target with variance or prediction error is a different method, even if it uses the same maps.

### M3 — The assay earns a practical role beyond reasonable existing analyses

**Claim:** for a declared scientific decision, the method has at least one demonstrable advantage: interpretable distinction, reliable discrimination, sample efficiency, computational practicality, or useful candidate selection.

**Current evidence:** the project supplies motivating examples and an implementation. Comparative advantage is unestablished. Existing conditional-shape and separability methods are close precedents.

**Test/support:** compare against conventional marginal selectivity/field criteria, direct shared-shape-plus-gain fits, a close existing shape/separability measure, and noise-aware profile correlation where applicable. Add raw response-sensitive measures to show what the deliberate gain invariance excludes; do not demand they solve a different task. Match support, information, regularization budget and evaluation splits. Before comparison, select which advantage is claimed and its meaningful margin. A simple transparent assay can be useful without beating every alternative on every metric.

**Best killer:** the assay's supposed distinctive finding is equally clear and reliable under an established close comparator, with no demonstrated practical or interpretive benefit; or its apparent advantage disappears under fair calibration and matched data. This defeats the stated added-value claim. It does not logically make every empirical result unpublishable.

**Controls/branch:** an underfit additive-only GLM is not a fair stand-in for all encoding models. If the method is essentially an established assay adapted to this setting, credit that lineage and let a substantive spatial result carry the paper. Do not promise methodological novelty unsupported by comparison.

### M4 — The method reveals a distinction independently confirmed to matter scientifically

**Claim:** the assay's output informs a preregistered decision whose validity is tested with evidence not defined by that output: an objective contrast, a physical-frame prediction, or functional transfer/native use.

**Current evidence:** old fields and scores motivate these applications; none of the proposed new validations has been established.

**Test/support:** separate discovery and validation. Use O0, F0 or U0 to specify the decision. Hold out model seeds, trajectories or scene manipulations as appropriate. Validate with independent response fits, physical counterfactuals, downstream transfer or native-output interventions. A useful result can delimit a failure of the assay—for example, a rigorous dissociation from transfer—but that supports a limitations finding, not positive predictive validity.

**Best killer:** the assay consistently selects wrong candidates or predicts effects that do not occur in sensitive held-out tests, or only succeeds when the validation reuses its own transformed scores. A probe directed by the score and evaluated by the same score is not independent confirmation.

**Controls/branch:** changing a score alone is not intervening on a computation. If independent validation supports the scientific phenomenon while the assay adds no value, retain the science and explicitly reconsider method centrality. If both fail, stop that narrative rather than move thresholds until it succeeds.

## Comparators and citation obligations

Conditional shape correlation has direct precedent in [Nowak et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3094178/), while split-half separability and population tolerance are linked in [Rust & DiCarlo (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2975390/). These limit claims of conceptual novelty. [Pospisil & Bair (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8244973/) distinguish signal-correlation biases and provide relevant correction machinery; their assumptions need checking here. [Franzius et al. (2007)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030166) already use conditional variance for spatial selectivity, supplying a comparator for a different response-sensitive target. [Zhu & Wei (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10121715/) motivate explicitly modeling tuning variability, including gain and shape.

The potential contribution is a characterized assay plus a consequential application or limitation. The packet does not establish that the method is a new estimator, superior to these alternatives, or publishable by itself.

## Versioning and decision record

For every result record `method_version`, `population_target`, `context_axis`, `profile_axes`, common support/weighting, pair exclusions, fold construction, gain/offset policy, uncertainty and `S` versus `D` convention. Freeze the aggregation over units and contexts before confirmation. A common-support or bias-correction change may preserve the target but changes the estimator. Weighted Pearson, a new pair weighting, or a different coordinate-domain measure can change the target too; document and revalidate instead of silently treating scores as interchangeable.

An implementing agent should first attack M1/M2 on a bounded subset, then compare the method's utility through the selected narrative. The earlier scientific claim IDs remain available; passing them with a replacement method does not automatically pass M1–M4. Use the shared handoff's five outcome statuses and preserve a clear decision: retain unchanged, retain with calibrated revision, narrow its interpretation, use as an established secondary assay, or replace it and change the paper's contribution.
