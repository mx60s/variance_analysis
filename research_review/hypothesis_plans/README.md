# Three research narratives to test and potentially reject

This packet contains three research proposals with **25 explicit claim cards**: four shared method claims and seven claims per narrative. Each proposal gives its broad narrative, prior work, reusable project evidence, operational hypotheses, discriminating experiments, strongest falsifiers, interpretation limits, and implementation sequence. No proposed experiment has been executed as part of preparing the packet.

**Current priority: the correlation method is a candidate central contribution.** Start with [METHOD_CENTRAL_CLAIMS.md](METHOD_CENTRAL_CLAIMS.md): define its conditional-shape target, test estimation adequacy, establish value against close alternatives, and independently validate its scientific use. The three applications below must earn that role; a scientific result obtained with another measure does not automatically validate the method. Keeping the same target with a calibrated estimator revision is allowed and must be documented.

| Document | Central question | What would most decisively undermine the proposed story? |
|---|---|---|
| [01 — Objectives and experience](01_OBJECTIVES_AND_EXPERIENCE.md) | Does the assay reveal a meaningful objective-dependent shape distinction, and how does it relate to access and computation? | Calibrated shape is equivalent, or score differences reflect nuisance rather than shape. Separate access/mechanism claims can fail independently. |
| [02 — Reference frames](02_REFERENCE_FRAMES.md) | Does the assay help identify a physical frame that predicts transformations and participates in computation? | Score-selected frames fail identifying interventions or have no useful selection advantage. A frame found by another approach does not validate assay centrality. |
| [03 — Biological resemblance and usefulness](03_BIOLOGICAL_RESEMBLANCE_AND_USEFULNESS.md) | Does correlation-measured shape stability explain useful access beyond conventional summaries and close alternative assays? | The score reflects noise or adds no claimed held-out value. A reliable functional dissociation can delimit its usefulness without demonstrating positive predictive validity. |

The [shared handoff](PROJECT_HANDOFF.md) records repository identity, exact reusable assets, model/measurement gates, evidence standards, and expected experiment records. The [machine-readable asset manifest](ASSET_MANIFEST.json) records the inspected source snapshot and selected array schemas. Give the next agent this folder together with `/Users/moo/variance_analysis`. The original scientific repo, not the anonymized copy, contains the manuscript source, bibliography, biological pipelines and predictive checkpoint.

## Claim map

| Plan | Core claims | Optional expansion | Scope if the expansion fails |
|---|---|---|---|
| Shared method | M1: target; M2: estimation; M3: comparative value; M4: independent scientific validation | Corrected estimator with unchanged declared target; broader applications | If the assay adds no value, change its role and the contribution explicitly. |
| Objectives | O0: assay reveals objective-dependent shape structure | O1: location access; O1a: heading retention; O2/O2a: integration/mechanism; O3: native transfer; O4: experience interaction | Shape/access dissociations are possible. An access result with O0 failure does not establish the method-centered story. |
| Frames | F0: useful assay-guided selection; F1: real variation; F2: physical account; F4: native operation | F3: remembered relation; F5: transfer advantage; F6: broader replication | A mechanism can survive F0 failure, but its discovery no longer supports the assay's claimed role. |
| Usefulness | U0: useful adequate assay; U1: biological anchor; U2: score predicts transfer; U3: population explanation | U4: coexistence; U5: native mechanism; U6: model–recording relationship | A valid U2 dissociation can characterize the assay's limits; an unreliable score cannot establish that scientific dissociation. |

“Core” does not mean that all claims must be positive. A well-controlled failure of U2, for example, can be the central finding if U3 explains a reliable dissociation. Each document specifies the interpretation after a claim is killed. The stronger combined chain—objective changes a mechanism, which changes transfer, whose biological signatures are informative—requires separate evidence for every link.

## Instructions for a critical implementing agent

1. Read the shared method claims, shared handoff and one selected plan. First audit the score's target, axis convention, calibration and proposed advantage; then audit the application. Treat all claims as proposals to criticize, not desired conclusions.
2. Recheck code/data provenance and model interfaces before running historical notebooks. The existing checkpoint loader can silently produce an untrained model when weights are absent; temporal normalization requires checking before causal claims; the legacy correlation score is not explained variance.
3. Start with the cheapest test that could meaningfully defeat the selected hypothesis. For objective work, validate the task and loss gradients before training a sweep. For reference frames, distinguish observational screening from identifying scene interventions. For usefulness, establish support and independent replication before a many-covariate predictive analysis.
4. Use a pilot to set scientifically justified margins, controls and sample sizes, then freeze them before confirmation. Do not turn a wide null interval into falsification or add favorable hypotheses after inspecting the final test set.
5. Return a claim-verdict table with effect estimates, intervals, passed/failed controls, remaining alternatives and a clear continue/narrow/stop decision. Preserve negative results and test failures separately.

The plans exclude trivial final-state sufficiency tests, require same-input baselines for observability, distinguish changing a probe from disrupting an operation, and separate within-system relationships from absolute brain–model rankings. Experimental feasibility and scientific results remain to be established.

The documents use absolute links for currently available project files and relative links within this packet. On another machine, map `/Users/moo/variance_analysis` to the local scientific repo and `/Users/moo/variance_analysis/research_review` to the local review directory. The text and manifest preserve repo-relative filenames so missing links do not obscure the intended inputs.
