"""Reproduce final paper plots from versioned analysis archives (no model inference)."""

import sys, json, argparse, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.ann.fig import plot_neuron_2d_3d

plt.rcParams.update(
    {
        "font.size": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
DETAIL = [
    "exclusive_place_cells",
    "exclusive_border_cells",
    "place_and_border_not_direction_cells",
    "exclusive_direction_cells",
    "place_and_direction_not_border_cells",
    "border_and_direction_not_place_cells",
    "place_border_direction_cells",
    "active_no_type",
    "dead_neurons",
]
DLABEL = [
    "Place only",
    "Border only",
    "Place + border",
    "HD only",
    "Place + HD",
    "Border + HD",
    "Place + border + HD",
    "Other active",
    "Inactive",
]
COLORS = [
    "#e69f00",
    "#d55e00",
    "#f0c674",
    "#0072b2",
    "#cc79a7",
    "#56b4e9",
    "#6a51a3",
    "#cccccc",
    "#f3f3f3",
]


def save(fig, out, name):
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def scatter(ax, ev, title, x="Location", y="HD", place=None):
    good = np.isfinite(ev).all(1)
    a = ev[good]
    ax.scatter(a[:, 0], a[:, 1], s=3, c="#446e91", alpha=0.22, rasterized=True)
    if place is not None:
        b = ev[place]
        ax.scatter(
            b[:, 0],
            b[:, 1],
            s=5,
            c="#c43a35",
            alpha=0.45,
            rasterized=True,
            label="Place only",
        )
    lo = min(-0.05, float(np.floor(a.min()*10)/10)-0.05)
    hi = max(1.25, float(np.ceil(a.max()*10)/10)+0.05)
    assert ((a >= lo) & (a <= hi)).all()
    ax.plot([lo, hi], [lo, hi], "--", c=".35", lw=0.7)
    ax.set(
        xlim=(max(lo,-.15), min(hi,1.25)),
        ylim=(max(lo,-.15), min(hi,1.25)),
        xlabel=f"Normalized variance ({x})",
        ylabel=f"Normalized variance ({y})",
        title=title,
    )
    ax.set_aspect("equal")
    outside = ((a < -.15) | (a > 1.25)).any(1)
    if outside.any():
        inset = ax.inset_axes([.08,.50,.35,.35])
        inset.scatter(a[:,0],a[:,1],s=2,c="#446e91",alpha=.4,rasterized=True)
        inset.plot([lo,hi],[lo,hi],"--",c=".4",lw=.5)
        inset.set(xlim=(lo,hi),ylim=(lo,hi),title="Full range")
        inset.tick_params(labelsize=6);inset.title.set_fontsize(7)
        ax.text(.03,.04,f"{outside.sum()} outside main view; all in inset",transform=ax.transAxes,fontsize=6)
    ax.text(
        0.03,
        0.98,
        f"n = {len(a):,} finite / {len(ev):,}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )


def categories(path):
    d = json.loads((path / "classifications.json").read_text())
    v = np.array([len(d.get(k + "_indices", [])) for k in DETAIL])
    ids = np.load(path / "unit_ids.npy")
    members = [i for k in DETAIL for i in d[k + "_indices"]]
    assert sorted(members) == list(range(len(ids))), (
        "Classes must partition all unit indices"
    )
    return d, v


def models(base, out):
    paths = [
        ("Predictive " + l, base / "paper_recipe/analysis" / l)
        for l in ["conv1", "down1", "pre_pred", "post_pred"]
    ]
    cnn = base / "finalization/cnns"
    paths += [
        ("ResNet " + l, cnn / "resnet50" / l)
        for l in ["layer1", "layer3", "layer4", "avgpool"]
    ]
    paths += [
        ("Untrained ResNet", cnn / "resnet50_untrained/avgpool"),
        ("VGG features.9", cnn / "vgg16/features.9"),
    ]
    assert all((p / "ev.npz").exists() for n, p in paths), "Missing model input archive"
    summary = {}
    for title, p in paths:
        z = np.load(p / "ev.npz")
        ev = z["ev"]
        d, v = categories(p)
        place = d["exclusive_place_cells_indices"]
        name = (
            "pred_" + p.name
            if "paper_recipe" in str(p)
            else p.parent.name + "_" + p.name
        )
        fig, ax = plt.subplots(figsize=(3.5, 3.4), layout="constrained")
        scatter(ax, ev, title, place=place)
        save(fig, out, name + "_ev")
        broad = np.array([v[0] + v[1] + v[2], v[4] + v[5] + v[6], v[3], v[7], v[8]])
        fig, ax = plt.subplots(figsize=(4.5, 3.4), layout="constrained")
        wedges, _, _ = ax.pie(
            broad,
            colors=["#e69f00", "#cc79a7", "#0072b2", "#cccccc", "#f3f3f3"],
            autopct=lambda x: f"{x:.1f}%" if x >= 5 else "",
            pctdistance=0.73,
            textprops={"fontsize": 9},
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title(title + f" (n={v.sum():,})")
        ax.legend(
            wedges,
            ["Location only", "Location + HD", "HD only", "Other active", "Inactive"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=2,
            fontsize=8,
            frameon=False,
        )
        save(fig, out, name + "_pie")
        good = np.isfinite(ev).all(1)
        norm = z["normalized"]
        r = z["reliability"]
        summary[title] = {
            "n": len(ev),
            "n_finite": int(good.sum()),
            "median": np.median(ev[good], axis=0).tolist(),
            "location_greater": float(np.mean(ev[good, 0] > ev[good, 1])),
            "place_only": len(place),
            "detailed_counts": v.tolist(),
            "broad_counts": broad.tolist(),
            "normalized_median": np.nanmedian(norm, axis=0).tolist(),
            "reliable_both_n": int(((r > 0.5).all(1) & good).sum()),
        }
        if title in [
            "Predictive pre_pred",
            "Predictive post_pred",
            "ResNet layer3",
            "ResNet layer4",
        ]:
            a = np.load(p / "gallery_maps.npz")["maps"]
            selections = json.loads((p / "gallery_selection.json").read_text())
            j = next(
                (
                    i
                    for i, s in enumerate(selections)
                    if s.get("paper_example", False) or ("paper_example" not in s and s["classification"] == "exclusive_place_cells")
                ),
                0,
            )
            s = selections[j]
            render3d(
                a[j], out, name + "_example", title + f" unit {s['unit_id']}", False
            )
            summary[title]["example"] = s
    # Detailed supplement retains all nine disjoint classes.
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    bottom = np.zeros(len(paths))
    for k, (label, color) in enumerate(zip(DLABEL, COLORS)):
        vals = np.array(
            [summary[n]["detailed_counts"][k] / summary[n]["n"] * 100 for n, p in paths]
        )
        ax.barh(np.arange(len(paths)), vals, left=bottom, label=label, color=color)
        bottom += vals
    ax.set(
        yticks=np.arange(len(paths)),
        yticklabels=[n for n, p in paths],
        xlabel="Percentage of all analyzed units",
        xlim=(0, 100),
    )
    ax.invert_yaxis()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    save(fig, out, "detailed_classes")
    (out / "model_statistics.json").write_text(json.dumps(summary, indent=2))
    # Compose at final publication proportions so labels remain legible.
    for name, chosen in [("cnn_main", paths[4:8]), ("pred_summary", paths[2:4])]:
        fig, axs = (
            plt.subplots(1, 4, figsize=(10, 3.4), layout="constrained")
            if name == "pred_summary"
            else plt.subplots(
                2, len(chosen), figsize=(10, 6.4), layout="constrained", squeeze=False
            )
        )
        if name == "pred_summary":
            axs = np.array([axs[:2], axs[2:]])
        for col, (title, p) in enumerate(chosen):
            z = np.load(p / "ev.npz")
            d, v = categories(p)
            scatter(
                axs[0, col], z["ev"], title, place=d["exclusive_place_cells_indices"]
            )
            broad = [v[0] + v[1] + v[2], v[4] + v[5] + v[6], v[3], v[7], v[8]]
            wedges, _, _ = axs[1, col].pie(
                broad,
                colors=["#e69f00", "#cc79a7", "#0072b2", "#cccccc", "#f3f3f3"],
                autopct=lambda x: f"{x:.1f}%" if x >= 5 else "",
                pctdistance=0.72,
                textprops={"fontsize": 11},
                wedgeprops={"edgecolor": "white"},
            )
            axs[1, col].set_title(f"All {sum(broad):,} units", fontsize=10)
        fig.legend(
            wedges,
            ["Location only", "Location + HD", "HD only", "Other active", "Inactive"],
            loc="outside lower center",
            ncol=5,
            frameon=False,
            fontsize=10,
        )
        save(fig, out, name)
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 7), layout="constrained")
    for ax, (title, p) in zip(axs.flat, paths[:4]):
        d, v = categories(p)
        scatter(
            ax,
            np.load(p / "ev.npz")["ev"],
            title,
            place=d["exclusive_place_cells_indices"],
        )
    save(fig, out, "pred_all_layers")


def render3d(a, out, name, title, biological):
    f = plot_neuron_2d_3d(
        a[None],
        0,
        axes_names=("X", "Y" if biological else "Z", "HD"),
        n_slices=4,
        cmap="Jet",
        z_spacing=1.0,
        grid_color="white",
        fig_size=(1500, 620),
        overall_title=title,
        firing_rate=biological,
    )
    if biological:
        # Keep physical units and zero baseline; make the mean's spatial pattern legible.
        mean = np.nanmean(a, axis=2)
        peak = float(np.nanmax(mean))
        f.data[1].update(coloraxis="coloraxis2", zmin=0, zmax=peak)
        f.update_layout(
            coloraxis2=dict(
                colorscale="Jet",
                cmin=0,
                cmax=peak,
                colorbar=dict(
                    title="Mean firing rate (Hz)",
                    orientation="h",
                    x=0.50,
                    y=-0.16,
                    len=0.26,
                    thickness=12,
                ),
            ),
            margin=dict(b=110),
        )
        f.layout.annotations[1].text = "Mean over HD"
    f.update_scenes(zaxis=dict(showticklabels=False, title=""))
    f.add_annotation(
        xref="paper",
        yref="paper",
        x=-0.025,
        y=0.5,
        text="HD (0–360°)",
        textangle=-90,
        showarrow=False,
        font=dict(size=12),
    )
    f.update_layout(font=dict(size=20), title=dict(font=dict(size=24)))
    for annotation in f.layout.annotations:
        annotation.font.size = 18
    f.update_layout(
        coloraxis_colorbar=dict(tickfont=dict(size=18), title=dict(font=dict(size=18)))
    )
    if biological:
        f.update_layout(
            coloraxis2_colorbar=dict(
                tickfont=dict(size=18), title=dict(font=dict(size=18))
            )
        )
    f.write_image(out / (name + ".pdf"))
    f.write_image(out / (name + ".png"))


def biology(base, out):
    review = ROOT / "results/review_20261116/revalidation"
    stats = {}
    for kind, variant, prefix in [
        ("pfeiffer", "ca1_place_primary", "ca1"),
        ("dusz", "curated_aligned", "hd"),
    ]:
        dirs = sorted((review / variant).glob(kind + "_*"))
        score_dirs = sorted((review / "grid_10").glob("dusz_*")) if kind == "dusz" else dirs
        data = [np.load(p / "xyhd.npz") for p in score_dirs]
        ev = np.concatenate([z["ev"] for z in data])
        rel = np.concatenate([z["reliability"] for z in data])
        norm = np.concatenate([z["normalized"] for z in data])
        good = np.isfinite(ev).all(1)
        stats[prefix] = {
            "n": len(ev),
            "n_finite": int(good.sum()),
            "median": np.median(ev[good], axis=0).tolist(),
            "location_greater": float(np.mean(ev[good, 0] > ev[good, 1])),
            "reference": np.nanmedian(1 - rel, axis=0).tolist(),
            "normalized": np.median(norm[np.isfinite(norm).all(1)], axis=0).tolist(),
            "n_normalized": int(np.isfinite(norm).all(1).sum()),
        }
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
        scatter(
            axs[0], ev, "a  CA1 place cells" if prefix == "ca1" else "a  HD population"
        )
        for ax, axis, letter in zip(
            axs[1:], ["xy", "hd"] if prefix == "ca1" else ["hd", "xy"], ["b", "c"]
        ):
            errors = []
            bases = []
            meds = []
            for p in dirs:
                z = np.load(p / "decoding.npz")
                e = np.concatenate([z[f"{axis}_fold{k}_errors"] for k in range(4)])
                b = np.concatenate([z[f"{axis}_fold{k}_baseline"] for k in range(4)])
                errors.append(e)
                bases.append(b)
                meds.append([np.nanmedian(e), np.nanmedian(b)])
            pooled = [
                np.nanmedian(np.concatenate(errors)),
                np.nanmedian(np.concatenate(bases)),
            ]
            ax.bar([0, 1], pooled, color=["#0072b2", "#e69f00"], width=0.6)
            for v in meds:
                ax.plot([0, 1], v, "o-", c=".3", alpha=0.3, lw=0.6, ms=2)
            ax.set(
                xticks=[0, 1],
                xticklabels=["Neural", "Occ-based"],
                ylabel="Median error (cm)"
                if axis == "xy"
                else "Median error (degrees)",
                title=letter
                + "  "
                + ("Location" if axis == "xy" else "Head direction"),
            )
            stats[prefix][axis + "_decoding"] = {
                "pooled_medians": pooled,
                "session_medians": meds,
            }
        save(fig, out, prefix + "_summary")
    for p in sorted((base / "finalization/biology_examples").glob("*.npz")):
        z = np.load(p)
        render3d(
            z["map"],
            out,
            p.stem,
            (
                "CA1 rat " + p.stem.split("_")[1]
                if p.stem.startswith("pfeiffer")
                else "HD cell " + p.stem.split("_")[1]
            )
            + f" | unit {int(z['unit_id'])}",
            True,
        )
    from md_figures import build
    stats["md"] = build(base.parent, out, scatter)
    (out / "biological_statistics.json").write_text(json.dumps(stats, indent=2))


def marginal(out):
    # Analytic bivariate Gaussian sampling: h depends on X alone; integrate X|Y.
    g = np.linspace(-3, 3, 250)
    xx, yy = np.meshgrid(g, g)
    sig = 0.7
    rate = np.exp(-(xx**2) / (2 * sig**2))
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for col, rho in enumerate([0, 0.6, 0.95]):
        density = np.exp(
            -(xx**2 - 2 * rho * xx * yy + yy**2) / (2 * (1 - rho * rho))
        ) / (2 * np.pi * np.sqrt(1 - rho * rho))
        axs[0, col].imshow(density, origin="lower", extent=[-3, 3, -3, 3], cmap="jet")
        axs[0, col].set(title=f"Sampling correlation ρ = {rho}", xlabel="x", ylabel="y")
        marginal = (
            sig
            / np.sqrt(sig * sig + 1 - rho * rho)
            * np.exp(-((rho * g) ** 2) / (2 * (sig * sig + 1 - rho * rho)))
        )
        axs[1, col].plot(g, marginal, c="#0072b2")
        axs[1, col].set(
            xlabel="y",
            ylabel="E[h(X,Y) | Y=y]",
            ylim=(0, 1),
            title="Observed marginal tuning",
        )
    save(fig, out, "marginal_confound")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("outputs", type=Path)
    p.add_argument(
        "--part", choices=["all", "models", "biology", "marginal"], default="all"
    )
    a = p.parse_args()
    out = a.outputs / "finalization/figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("models", models),
        ("biology", biology),
        ("marginal", lambda b, o: marginal(o)),
    ]:
        if a.part in ["all", name]:
            fn(a.outputs, out)
