"""Summary numbers for the rodent analyses reported in the manuscript.

Reads the saved rate maps and EVs from results/rodents/ (written by bio_neurons.ipynb) and the
decoding errors (written by 1d_decoding.ipynb), and prints:
  - fraction of cells with EV_xy > EV_hd etc. and median EVs
  - median finite-sampling floor 1 - <C_ii> and reliability-normalized contribution
  - HD-MD coupling in the pooled occupancy
  - mean absolute decoding errors (mean over rats/sessions of the per-session mean)

Usage (from the repo root):  python notebooks/rodent_summary.py
"""
import os, sys, glob, warnings
import numpy as np
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
from src.explained_variance import nan_aware_gaussian_filter, masked_zoom, quantify_variance_corr

warnings.filterwarnings("ignore", category=RuntimeWarning)
P = ROOT / "results/rodents/pfeiffer"
D = ROOT / "results/rodents/dusz"


def ev_and_reliability(f0, f1, occ_z, axis, min_occ):
    off, rel = quantify_variance_corr(f0, f1, occ_z, axis=axis, min_occ=min_occ)
    return 1 - off, rel


def report(name, ev_a, ev_b, label_a, label_b, rel_a, rel_b):
    ev_a, ev_b, rel_a, rel_b = map(np.asarray, (ev_a, ev_b, rel_a, rel_b))
    ok = np.isfinite(ev_a) & np.isfinite(ev_b) & np.isfinite(rel_a) & np.isfinite(rel_b)
    a, b, ra, rb = ev_a[ok], ev_b[ok], rel_a[ok], rel_b[ok]
    norm_a = (ra - (1 - a)) / ra
    norm_b = (rb - (1 - b)) / rb
    print(f"{name}: n={ok.sum()}  {label_a}>{label_b}: {100 * np.mean(a > b):.1f}%")
    print(f"  median {label_a}={np.median(a):.3f} (floor {np.median(1 - ra):.3f}, normalized {np.median(norm_a):.3f})")
    print(f"  median {label_b}={np.median(b):.3f} (floor {np.median(1 - rb):.3f}, normalized {np.median(norm_b):.3f})")


# Pfeiffer place cells, location x HD (20 x 20 x 60), as in bio_neurons.ipynb
n_xy, n_hd = 20, 60
max_bins = max(n_hd, n_xy ** 2)
zoom = (max_bins / n_xy ** 2, max_bins / n_hd)
rows = []
for r in range(1, 5):
    f0 = np.load(P / f"Rat{r}/fold0_xyhd_{n_xy}x{n_xy}x{n_hd}.npz")["rate_map"]
    f1 = np.load(P / f"Rat{r}/fold1_xyhd_{n_xy}x{n_xy}x{n_hd}.npz")["rate_map"]
    occ = np.load(P / f"Rat{r}/ratemap_xyhd_{n_xy}x{n_xy}x{n_hd}.npz")["occupancy"]
    x, y, h, n = f0.shape
    f0, f1 = f0.reshape(x * y, h, n), f1.reshape(x * y, h, n)
    occ_z = masked_zoom(occ.reshape(x * y, h), zoom)
    min_occ = np.percentile(occ_z[occ_z > 0], 10)
    for k in range(n):
        a = masked_zoom(nan_aware_gaussian_filter(f0[:, :, k], (1, 2), circular_axes=(1,)), zoom)
        b = masked_zoom(nan_aware_gaussian_filter(f1[:, :, k], (1, 2), circular_axes=(1,)), zoom)
        rows.append(ev_and_reliability(a, b, occ_z, 0, min_occ) + ev_and_reliability(a, b, occ_z, 1, min_occ))
rows = np.array(rows)
report("Pfeiffer place cells", rows[:, 0], rows[:, 2], "EV_xy", "EV_hd", rows[:, 1], rows[:, 3])

# Duszkiewicz HD cells, location x HD (20 x 20 x 90)
n_loc, n_hd = 20, 90
max_bins = max(n_loc ** 2, n_hd)
zoom = (max_bins / n_loc ** 2, max_bins / n_hd)
rows = []
for p in sorted(D.glob(f"Dataset_*/*/fold_rate_xyhd_{n_loc}x{n_loc}x{n_hd}.npz")):
    m = np.load(p)
    f0, f1, occ = m["fold0_map"], m["fold1_map"], m["occupancy"]
    y, x, h, n = f0.shape
    f0, f1 = f0.reshape(y * x, h, n), f1.reshape(y * x, h, n)
    occ_z = masked_zoom(occ.reshape(y * x, h), zoom)
    min_occ = np.percentile(occ_z[occ_z > 0], 10)
    for k in range(n):
        a = masked_zoom(nan_aware_gaussian_filter(f0[:, :, k], sigma=(1, 1), circular_axes=(1,)), zoom)
        b = masked_zoom(nan_aware_gaussian_filter(f1[:, :, k], sigma=(1, 1), circular_axes=(1,)), zoom)
        rows.append(ev_and_reliability(a, b, occ_z, 1, min_occ) + ev_and_reliability(a, b, occ_z, 0, min_occ))
rows = np.array(rows)
report("Duszkiewicz HD cells (location x HD)", rows[:, 0], rows[:, 2], "EV_hd", "EV_xy", rows[:, 1], rows[:, 3])

# Duszkiewicz HD cells, HD x MD (90 x 90); axis 0 is HD, axis 1 is MD
rows, occ_total = [], None
paths = sorted(D.glob("Dataset_*/*/fold_rate_hdmd_90x90.npz"))
for p in paths:
    m = np.load(p)
    f0, f1, occ = m["fold0_map"], m["fold1_map"], m["occupancy"]
    occ_total = occ.astype(float) if occ_total is None else occ_total + occ
    min_occ = np.percentile(occ[occ > 0], 10)
    for k in range(f0.shape[2]):
        a = nan_aware_gaussian_filter(f0[:, :, k], sigma=(1, 1), circular_axes=(0, 1))
        b = nan_aware_gaussian_filter(f1[:, :, k], sigma=(1, 1), circular_axes=(0, 1))
        rows.append(ev_and_reliability(a, b, occ, 0, min_occ) + ev_and_reliability(a, b, occ, 1, min_occ))
rows = np.array(rows)
report("Duszkiewicz HD cells (HD x MD)", rows[:, 0], rows[:, 2], "EV_hd", "EV_md", rows[:, 1], rows[:, 3])

centers = (np.arange(90) + 0.5) * 4.0
dist = np.abs((centers[:, None] - centers[None, :] + 180) % 360 - 180)
print(f"HD-MD coupling over {len(paths)} sessions: P(|MD-HD| < 30 deg) = {occ_total[dist < 30].sum() / occ_total.sum():.3f} "
      f"(independent: {np.mean(dist < 30):.3f})")


# Decoding: mean absolute error, averaged over rats or sessions
def mean_error(dirs, pattern, key):
    per_session = []
    for d in dirs:
        files = sorted(glob.glob(str(d / pattern)))
        if not files:
            continue
        e = np.concatenate([np.abs(np.asarray(np.load(f, allow_pickle=True)[key]).ravel()) for f in files])
        per_session.append(np.mean(e[np.isfinite(e)]))
    return np.mean(per_session), len(per_session)


rats = sorted(P.glob("Rat[0-9]"))
sessions = sorted(p for p in D.glob("Dataset_*/*") if p.is_dir())
for label, dirs, pattern, key, unit in [
    ("Pfeiffer location", rats, "errors_xy_cv*_100x100.npz", "errors_cm", "cm"),
    ("Pfeiffer location, chance", rats, "errors_xy_chance_cv*_100x100.npz", "errors_cm", "cm"),
    ("Pfeiffer HD", rats, "errors_hd_cv*_360.npz", "errors_deg", "deg"),
    ("Pfeiffer HD, chance", rats, "errors_hd_chance_cv*_360.npz", "errors_deg", "deg"),
    ("Duszkiewicz location", sessions, "errors_xy_cv*_all.npz", "errors", "cm"),
    ("Duszkiewicz location, chance", sessions, "errors_xy_chance_cv*_all.npz", "errors", "cm"),
    ("Duszkiewicz HD", sessions, "errors_hd_cv*_all.npz", "errors_deg", "deg"),
    ("Duszkiewicz HD, chance", sessions, "errors_hd_chance_cv*_all.npz", "errors_deg", "deg"),
    ("Duszkiewicz MD", sessions, "errors_md_cv*_all.npz", "errors_deg", "deg"),
    ("Duszkiewicz MD, chance", sessions, "errors_md_chance_cv*_all.npz", "errors_deg", "deg"),
]:
    err, n = mean_error(dirs, pattern, key)
    print(f"{label}: {err:.1f} {unit} (n={n})")
