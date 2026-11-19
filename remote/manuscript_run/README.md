# Manuscript remote-run source snapshots

These are the exact orchestration sources used for Minecraft collection and the paper-recipe training/analysis run. Core portable training and trajectory code is in `scripts/retraining/` at the repository root; the manuscript run uses `train_recipe.py`, not the earlier AdamW pilot `train.py`.

The wrappers retain original local paths, Modal volume/image identifiers, and relative references for provenance. They require configuration before use on another machine/account. They contain no account credentials. Running them may incur compute charges; simply cloning the repository does not start a job.

The Minecraft world source is the public `environments/small-world` directory in `mx60s/masters-thesis` at commit `e1956875f6fd1cff2dd02519b40c96973630e8de`. The legacy renderer uses Minecraft 1.11.2 and Malmo 0.37.0. Minecraft runtime binaries, raw trajectory renders, checkpoints, and credentials are not redistributed here.

The paper uses the two-head, SGD/OneCycle recipe with per-frame causal normalization. Earlier pilot wrappers elsewhere in `remote/` are retained as historical workflows, not the final manuscript configuration.
