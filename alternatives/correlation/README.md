# Correlation-based alternate

The primary manuscript uses cross-fold conditional response variance. This directory preserves the alternate correlation-based manuscript and figures as an earlier analysis version; it does not include the subsequent variance-version revisions.

The callable alternate remains `src.explained_variance.quantify_variance_corr`. It returns mean off-diagonal correlation; callers apply `1 - correlation`. It compares tuning shape and is insensitive to positive gain and additive baseline changes. It is not an unbiased variance estimator.

The complete frozen source, numerical inputs, figures, and verification manifest are available as `correlation_reproduction.zip` in the [2026-11-18 release](https://github.com/mx60s/variance_analysis/releases/tag/manuscript-2026-11-18). Extract it and follow its README. The archive is preserved unchanged.

`paper/manuscript.tex`, `paper/map.bib`, and `paper/final_figures/` form a self-contained LaTeX project.
