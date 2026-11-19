"""Run time-bin, grid and data-release sensitivity comparisons."""

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from regenerate_biology import run, sessions

root = Path("results/review_20261116/revalidation")
configs = [
    ("aligned_biology", {}),
    ("aligned_biology_fine", dict(no_decoding=True, fine=True)),
    ("native_xy_coarse_hdmd", dict(no_decoding=True, native=True, hdmd_bins=36)),
    ("curated_aligned", dict(curated_dir=root / "curated_inputs")),
    (
        "curated_native_coarse",
        dict(
            no_decoding=True,
            native=True,
            hdmd_bins=36,
            curated_dir=root / "curated_inputs",
        ),
    ),
]
for name, kw in configs:
    for kind, d, r in sessions():
        if "curated_dir" in kw and kind != "dusz":
            continue
        run(kind, d, r, root / name, **kw)
    print("CONFIG COMPLETE", name, flush=True)
print("SUITE COMPLETE", flush=True)
