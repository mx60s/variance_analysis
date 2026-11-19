"""Review-only scatter and session-level decoding figures from validated outputs."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/review_20261116/revalidation"


def decoding_figure(out):
    specs = [
        ("aligned_biology", "pfeiffer", "xy", "CA1 · location", "cm"),
        ("aligned_biology", "pfeiffer", "hd", "CA1 · heading", "degrees"),
        ("curated_aligned", "dusz", "xy", "HD population · location", "cm"),
        ("curated_aligned", "dusz", "hd", "HD population · heading", "degrees"),
        ("curated_aligned", "dusz", "md", "HD population · movement", "degrees"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.5), layout="constrained")
    for ax, (variant, pop, axis, title, unit) in zip(axes, specs):
        pairs = []
        for p in sorted((R / variant).glob(pop + "*/decoding.npz")):
            with np.load(p) as z:
                keys = [
                    k
                    for k in z.files
                    if k.startswith(axis + "_") and k.endswith("_errors")
                ]
                if not keys:
                    continue
                e = np.concatenate([z[k] for k in keys])
                b = np.concatenate([z[k.replace("_errors", "_baseline")] for k in keys])
                v = np.isfinite(e) & np.isfinite(b)
                if v.any():
                    pairs.append([np.median(e[v]), np.median(b[v])])
        a = np.asarray(pairs)
        for e, b in a:
            ax.plot([0, 1], [e, b], color=".65", alpha=0.55, lw=0.8, zorder=1)
        ax.scatter(np.zeros(len(a)), a[:, 0], color="#276ca5", s=25, zorder=2)
        ax.scatter(np.ones(len(a)), a[:, 1], color="#d58a32", s=25, zorder=2)
        ax.set(
            xticks=[0, 1],
            xticklabels=["Neural", "Occ-based"],
            ylabel=f"Session median error ({unit})",
            title=title,
            xlim=(-0.3, 1.3),
            ylim=(0, None),
        )
        ax.text(
            0.03,
            0.97,
            f"{int((a[:, 0] < a[:, 1]).sum())}/{len(a)} sessions improve",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Corrected decoding: each line connects the same held-out bins within one session",
        fontsize=13,
    )
    fig.savefig(out / "decoding_sessions.png", dpi=170)
    fig.savefig(out / "decoding_sessions.pdf")
    plt.close(fig)


def ev_figure(out):
    data = []
    for pop, variant, title in [
        ("pfeiffer", "native_xy_coarse_hdmd", "CA1 · 4 rats"),
        ("dusz", "curated_native_coarse", "HD cells · curated, 30 sessions"),
    ]:
        arrays = [np.load(p)["ev"] for p in (R / variant).glob(pop + "*/xyhd.npz")]
        data.append((title, np.concatenate(arrays), None))
    for p in sorted((R / "models").glob("*/ev_native.npz")):
        with np.load(p) as z:
            data.append(
                (
                    p.parent.name.replace("predictive_coder", "PC"),
                    z["ev"].copy(),
                    z["is_place"].copy(),
                )
            )
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), layout="constrained")
    for ax, (name, a, mask) in zip(axes.flat, data):
        valid = np.isfinite(a).all(1)
        ax.scatter(
            a[valid, 0], a[valid, 1], s=3, alpha=0.12, color="#465d72", rasterized=True
        )
        if mask is not None:
            v = valid & mask
            ax.scatter(
                a[v, 0], a[v, 1], s=5, alpha=0.35, color="#c73b40", rasterized=True
            )
        ax.plot([0, 2], [0, 2], "--", lw=0.7, color=".5")
        ax.set(
            title=name,
            xlim=(-0.03, 2.03),
            ylim=(-0.03, 2.03),
            xlabel="Location EV",
            ylabel="HD EV",
            aspect="equal",
        )
        ax.text(
            0.03,
            0.97,
            f"finite {valid.sum():,}/{len(a):,}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(data) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Native-grid EV • red: corrected place-only model units • all finite biological units shown\nEV is dimensionless correlation dissimilarity and can exceed 1",
        fontsize=12,
    )
    fig.savefig(out / "ev_populations.png", dpi=170)
    fig.savefig(out / "ev_populations.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    decoding_figure(a.output)
    ev_figure(a.output)
