# Claim trees

Three research plans, each written as a tree of claims. A claim holds if and only if all of its children hold. Leaves are definitions, one-line derivations, sentences quoted from papers, or experiments stated as "the strongest attack does not produce the refuting result". The rules are at the top of each file.

| File | Headline claim (Claim 0) |
|---|---|
| [01_OBJECTIVES_AND_EXPERIENCE.md](01_OBJECTIVES_AND_EXPERIENCE.md) | The latent-transition loss changes the spatial computation of the transformer. |
| [02_REFERENCE_FRAMES.md](02_REFERENCE_FRAMES.md) | Units whose spatial fields change shape across heading encode a landmark relation, and the model uses it to predict the next frame. |
| [03_BIOLOGICAL_RESEMBLANCE_AND_USEFULNESS.md](03_BIOLOGICAL_RESEMBLANCE_AND_USEFULNESS.md) | CA1 place fields keep their shape across heading, and in trained models that same property predicts whether position can be read out across headings. |

Each file contains its own copy of Claim M (the correlation score D̂ measures change of field shape across heading). The former `METHOD_CENTRAL_CLAIMS.md` was folded into those copies.

Old claim IDs map as follows.

| Old | New |
|---|---|
| M1, M2 | Claim M in each file |
| M3 | 01 Claim 8; 02 leaves 3.7–3.8; 03 Claim 7 |
| M4 | the claims that list Claim M as a child |
| O0, O1, O1a, O2, O2a, O3, O4 | 01 Claims 1, 2, 3, 4, 5, 6, 7 |
| F0, F1, F2, F3, F4, F5, F6 | 02 Claims 3, 1, 2, 5, 4, 6, 7 |
| U0, U1, U2, U3, U4, U5, U6 | 03 Claims M, 1, 2, 3, 4, 5, 6 |

[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) and [ASSET_MANIFEST.json](ASSET_MANIFEST.json) record file provenance. They are unchanged.
