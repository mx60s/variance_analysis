"""Build a descriptive, coverage-aware revalidation report from completed outputs."""

import sys, json, csv, html, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = ROOT / "results/review_20261116/revalidation"
LABELS = {
    "cached_biology": "Existing maps + corrected EV",
    "aligned_biology": "Aligned legacy data",
    "aligned_biology_fine": "Aligned, finer time bins",
    "native_xy_coarse_hdmd": "Native XY grid / 36×36 HD–MD",
    "curated_aligned": "Curated NWB, aligned",
    "curated_native_coarse": "Curated NWB, native / 36×36",
}


def stats(a, norm, primary=0):
    v = np.isfinite(a).all(1)
    vn = np.isfinite(norm).all(1)
    return dict(
        n_total=len(a),
        n_finite=int(v.sum()),
        median_primary=float(np.median(a[v, primary])) if v.any() else None,
        median_other=float(np.median(a[v, 1 - primary])) if v.any() else None,
        primary_greater_percent=float(100 * np.mean(a[v, primary] > a[v, 1 - primary]))
        if v.any()
        else None,
        n_normalized=int(vn.sum()),
        normalized_primary=float(np.median(norm[vn, primary])) if vn.any() else None,
        normalized_other=float(np.median(norm[vn, 1 - primary])) if vn.any() else None,
    )


def collect():
    rows = []
    for variant, label in LABELS.items():
        for group, mode, primary in [
            ("pfeiffer", "xyhd", 0),
            ("dusz", "xyhd", 1),
            ("dusz", "hdmd", 0),
        ]:
            lags = (
                [0, 1, 3] if mode == "hdmd" and variant != "cached_biology" else [None]
            )
            for lag in lags:
                if variant == "cached_biology":
                    paths = list(
                        (R / variant).glob(
                            ("*pfeiffer*" if group == "pfeiffer" else "Dataset_*")
                            + mode
                            + "*.npz"
                        )
                    )
                else:
                    paths = list(
                        (R / variant).glob(
                            f"{group}_*/{mode if lag is None else 'hdmd_lag' + str(lag)}.npz"
                        )
                    )
                if not paths:
                    continue
                a = np.concatenate([np.load(p)["ev"] for p in paths])
                norm = np.concatenate([np.load(p)["normalized"] for p in paths])
                rows.append(
                    dict(
                        variant=variant,
                        population=group,
                        comparison=mode,
                        lag=lag,
                        sessions=len(paths),
                        **stats(a, norm, primary),
                    )
                )
    return rows


def session_stats():
    rows = []
    for variant in LABELS:
        if variant == "cached_biology":
            continue
        for p in sorted((R / variant).glob("*/*.npz")):
            if p.stem != "xyhd" and not p.stem.startswith("hdmd_lag"):
                continue
            group = p.parent.name.split("_")[0]
            z = np.load(p)
            primary = 1 if group == "dusz" and p.stem == "xyhd" else 0
            rows.append(
                dict(
                    variant=variant,
                    session=p.parent.name,
                    comparison=p.stem,
                    **stats(z["ev"], z["normalized"], primary),
                )
            )
    return rows


def models():
    rows = []
    for p in sorted((R / "models").glob("*/ev*.npz")):
        z = np.load(p)
        meta = json.loads((p.parent / "summary.json").read_text())
        classification = Path(meta["classification"])
        masks = {
            "all": np.ones(len(z["ev"]), bool),
            "corrected_place_only": z["is_place"],
        }
        t = classification / "tanni_iterative/classifications.json"
        if t.exists():
            masks["iterative_place_only"] = np.isin(
                np.arange(len(z["ev"])),
                json.loads(t.read_text())["exclusive_place_cells_indices"],
            )
        for group, mask in masks.items():
            if mask.any():
                rows.append(
                    dict(
                        model=p.parent.name,
                        grid="native" if p.stem == "ev_native" else "interpolated",
                        group=group,
                        **stats(z["ev"][mask], z["normalized"][mask]),
                    )
                )
    return rows


def decode():
    rows = []
    for variant in ["aligned_biology", "curated_aligned"]:
        for group in ["pfeiffer", "dusz"]:
            paths = sorted((R / variant).glob(group + "*/decoding.npz"))
            for axis in ["xy", "hd", "md"]:
                errors = []
                base = []
                sessions_better = []
                for p in paths:
                    z = np.load(p)
                    ek = [
                        k
                        for k in z.files
                        if k.startswith(axis + "_") and k.endswith("_errors")
                    ]
                    if not ek:
                        continue
                    e = np.concatenate([z[k] for k in ek])
                    b = np.concatenate(
                        [z[k.replace("_errors", "_baseline")] for k in ek]
                    )
                    v = np.isfinite(e) & np.isfinite(b)
                    errors.append(e[v])
                    base.append(b[v])
                    sessions_better.append(float(np.median(e[v]) < np.median(b[v])))
                if errors:
                    e, b = np.concatenate(errors), np.concatenate(base)
                    rows.append(
                        dict(
                            variant=variant,
                            population=group,
                            axis=axis,
                            sessions=len(errors),
                            valid_test_bins=len(e),
                            median_error=float(np.median(e)),
                            median_occupancy_baseline=float(np.median(b)),
                            sessions_better=sum(sessions_better),
                            unit="degrees"
                            if axis != "xy"
                            else (
                                "cm"
                                if group == "pfeiffer" or variant == "curated_aligned"
                                else "legacy coordinate units"
                            ),
                        )
                    )
    return rows


def csvfile(out, name, rows):
    if not rows:
        return
    with open(out / name, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(rows, columns):
    def fmt(v):
        if v is None:
            return "—"
        return f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))

    return (
        "<table><thead><tr>"
        + "".join("<th>" + html.escape(title) + "</th>" for k, title in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + fmt(row.get(k)) + "</td>" for k, title in columns)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    bio = collect()
    ann = models()
    dec = decode()
    csvfile(out, "biological_ev_by_session.csv", session_stats())
    csvfile(out, "biological_ev_comparison.csv", bio)
    csvfile(out, "model_ev_comparison.csv", ann)
    csvfile(out, "decoding_comparison.csv", dec)
    status = {
        "cached_maps": len(list((R / "cached_biology").glob("*.npz"))),
        "models": len(list((R / "models").glob("*/ev.npz"))),
        "native_models": len(list((R / "models").glob("*/ev_native.npz"))),
        "biological_configs": {
            v: len(list((R / v).glob("*/complete.json")))
            for v in LABELS
            if v != "cached_biology"
        },
    }
    (out / "completion_status.json").write_text(json.dumps(status, indent=2))
    selected = [x for x in bio if x["lag"] in [None, 0]]
    bc = [
        ("variant", "Analysis"),
        ("population", "Population"),
        ("comparison", "Axes"),
        ("sessions", "Sessions"),
        ("n_finite", "Paired finite"),
        ("n_total", "All cells"),
        ("median_primary", "Primary EV"),
        ("median_other", "Other EV"),
        ("primary_greater_percent", "Primary > other %"),
    ]
    mc = [
        ("model", "Model / layer"),
        ("grid", "Grid"),
        ("group", "Group"),
        ("n_finite", "Paired finite"),
        ("median_primary", "Location EV"),
        ("median_other", "HD EV"),
        ("primary_greater_percent", "Location > HD %"),
    ]
    dc = [
        ("variant", "Analysis"),
        ("population", "Population"),
        ("axis", "Variable"),
        ("sessions", "Sessions"),
        ("median_error", "Decoder median"),
        ("median_occupancy_baseline", "Occ-based median"),
        ("unit", "Unit"),
        ("sessions_better", "Sessions improved"),
    ]
    # A compact summary uses descriptive medians, never treats units as independent replicates.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    bplot = [
        x
        for x in selected
        if x["comparison"] == "xyhd"
        and (
            (
                x["population"] == "pfeiffer"
                and x["variant"] in ["aligned_biology", "native_xy_coarse_hdmd"]
            )
            or (
                x["population"] == "dusz"
                and x["variant"] in ["curated_aligned", "curated_native_coarse"]
            )
        )
    ]
    mplot = [
        x for x in ann if x["group"] == "corrected_place_only" and x["grid"] == "native"
    ]
    for ax, rows, title in [
        (axes[0], bplot, "Biological populations: alignment / grid sensitivity"),
        (axes[1], mplot, "Model place-only units • native grids"),
    ]:
        for i, row in enumerate(rows):
            if row["n_finite"] == 0:
                continue
            x, y = row["median_primary"], row["median_other"]
            if row.get("population") == "dusz":
                x, y = y, x
            ax.scatter(
                x,
                y,
                s=70,
                label=(
                    ("CA1" if row["population"] == "pfeiffer" else "HD cells")
                    + " · "
                    + (
                        "native grid"
                        if "native" in row["variant"]
                        else "aligned, interpolated"
                    )
                )
                if "population" in row
                else row["model"].replace("predictive_coder", "PC"),
            )
        ax.plot([0, 1.1], [0, 1.1], "--", color=".6", lw=1)
        ax.set(
            xlim=(0, 1.1),
            ylim=(0, 1.1),
            xlabel="Median location EV",
            ylabel="Median HD EV",
            title=title,
        )
        ax.legend(fontsize=7, loc="lower left")
        ax.set_aspect("equal")
    fig.suptitle(
        "Descriptive tuning-shape contrasts • EV is dimensionless, not a variance fraction",
        fontsize=13,
    )
    fig.savefig(out / "ev_summary.png", dpi=160)
    plt.close(fig)
    intro = (
        """<h1>Expected variance: numerical revalidation</h1><p>16 November 2026. All reruns are isolated from historical caches and publication figures.</p>
<p class="notice"><strong>Interpretation review, not a submission-ready manuscript.</strong> Use the completion table below to check coverage. Percentages and medians are descriptive; neurons within a session are not independent biological replicates. Undefined scores are reported as missing, never zero.</p>
<h2>What changes scientifically</h2><p><strong>The broad model–brain tuning contrast survives the completed reruns.</strong> All ten model/layer configurations were evaluated on both grids with matched feature IDs. In the pre-prediction latent layer, interpolated median location/HD EV is 0.916/0.919; post-prediction is 0.915/0.912 (8,000 units each). Corrected and iterative place-only subsets retain substantial dependence on both variables. This supports the joint-tuning comparison while requiring more qualified biological and GLM claims.</p><p>The biological populations retain their predominant tuning axis, but the strong categorical claims need revision. Corrected CA1 location dominance is approximately 74–85% across the checked grid/time-bin choices, rather than 99%. In the curated HD-cell population, HD exceeds location EV in approximately 82–84% of cells. On the coarser HD×MD grid, HD dominates almost every cell with full score coverage; the fine grid has substantial missing support.</p><p><strong>The claim that location cannot be decoded from the HD population is contradicted by the corrected decoding.</strong> Curated location error has a pooled median of about 33.4 cm versus 43.8 cm for the occupancy baseline, and improves in 28 of 30 sessions. HD remains much more accurate (about 8.7° versus 81.3°); MD is coarse but above baseline (about 61.4° versus 89.8°). These decoding results describe relative accuracy, not absence of information, and do not by themselves prove a signal cannot support path integration. EV measures changes in tuning shape and is insensitive to positive gain changes; decoding can exploit gain and behavioral correlations. Those possible explanations have not been separated here, so lower EV is not evidence of zero decodable information.</p><h2>What was checked</h2><ul><li>Existing biological maps with corrected EV processing, followed by maps rebuilt with one state per spike-count time bin.</li><li>Finer time bins with fixed minimum occupancy duration; native location×HD grids without interpolation; 36×36 HD×MD grids with the same 4° smoothing width; MD lags 0, 1 and 3 tracking samples without wraparound.</li><li>All four predictive-coder layers and six CNN configurations on fresh pitch-perturbed panoramas, matched to recorded feature IDs and the corrected classifications, on both interpolated and native grids.</li><li>Four-fold contiguous held-out decoding and training-occupancy baselines; the same aligned time-bin targets for both.</li><li>The newer curated NWB release for the same 30 Duszkiewicz sessions, including actual units and angle conventions.</li></ul>
<p><a href="biological_ev_comparison.csv">Biological EV counts and scores</a> · <a href="biological_ev_by_session.csv">Session-level EV summaries</a> · <a href="model_ev_comparison.csv">Model EV and classification sensitivity</a> · <a href="decoding_comparison.csv">Decoding comparison</a> · <a href="glm_corrected.pdf">Corrected GLM illustration</a></p>
<h2>Completion</h2><pre>"""
        + html.escape(json.dumps(status, indent=2))
        + """</pre>
<h2>Biological EV</h2><p>Primary means location for Pfeiffer CA1 excitatory units, and HD for the Duszkiewicz HD-cell population. “Other” is HD, location, or MD as specified by the axes. Medians and dominance percentages use cells with both raw scores finite. These percentages describe score orderings, not noise-free estimates of the fraction of truly selective neurons. Normalized scores and their separate valid counts are in the CSV; normalization is not an unbiased correction. The Pfeiffer analysis includes all 551 supplied excitatory units, not a newly selected place-cell subset. Full lag comparisons are in the CSV.</p>"""
    )
    body = (
        intro
        + table(selected, bc)
        + """<h2>Models</h2><p>Pitch folds compare visual perturbations, not independent measurement-noise samples. Repeated static frames preserve the existing evaluation protocol. The historical predictive checkpoint is unchanged; its temporal-normalization issue remains relevant to causal interpretation. “Place-only” excludes border/direction overlap; no regrouping was imposed. Iterative classification remains a sensitivity analysis.</p>"""
        + table(ann, mc)
        + """<img src="ev_summary.png" alt="Median location and heading EV comparisons">
<p><a href="ev_populations.pdf">Full native-grid neuron scatter plots</a> · <a href="decoding_sessions.pdf">Session-level decoding figure</a></p><h2>Decoding</h2><p>Medians pool the finite held-out time bins shared by neural and occupancy-based predictions. “Sessions improved” counts sessions whose median neural error is lower. Low errors are retained. Legacy Duszkiewicz XY values are deliberately not labeled centimeters: calibration was not verified in that release. Curated NWB positions explicitly use centimeters. A change in raw distance between releases is not by itself a change in decoding quality.</p>"""
        + table(dec, dc)
        + """
<img src="decoding_sessions.png" alt="Session medians comparing neural decoding with the occupancy baseline"><p>Each line joins paired medians from the same recording session; this is descriptive, not an independent-session significance test.</p><h2>Why the data release matters</h2><p>The <a href="https://research.manchester.ac.uk/en/publications/local-origin-of-excitatory-inhibitory-tuning-equivalence-in-a-cor/">authors recommend the curated DANDI release for new analyses</a>. We retrieved only behavior and selected spike-time subsets from <a href="https://dandiarchive.org/dandiset/000939/0.261016.2310">release 0.261016.2310</a>, not raw ephys. In matched session A1813-190408, curated positions are approximately 1.6× the local loader's coordinates; headings also have an approximately 90° convention change plus nonconstant corrections. The newer heading reference is clockwise from +Y. MD is converted to that same reference. Curated speed is explicitly estimated from the XY derivative, smoothed with a three-sample Gaussian; this changes both tracking and kinematic processing, so it is not a pure unit-rescaling experiment.</p>
<h2>Behavioral HD–MD coupling</h2><p>With the curated coordinates and the same aligned 50-ms bins and speed ≥1 cm/s policy, heading and movement are within 30° in 30.65% of eligible bins. The session-wise empirical-marginal independence reference is 16.70% (uniform reference 16.67%). Thus the old 25% number changes, while substantial off-diagonal behavior remains. <a href="behavioral_coupling.csv">Session summaries</a> · <a href="behavioral_coupling.json">Pooled result and method</a>. This behavioral check does not validate MD measurement accuracy at low speeds or establish path-integration sufficiency.</p><h2>Corrected GLM demonstration</h2><p>The old notebook replaced true conditional curves with flat lines and mixed degree/radian arguments in another cell. Both are corrected. The new illustration integrates true marginals over the actual standard-normal stimulus distribution and labels conditional slices separately. An additive log-rate GLM can reproduce the marginals while missing the joint interaction; adding x:y recovers the known interaction. Held-out log likelihood improves by approximately 0.105 and 0.127 nats per observation in the two examples. Analytic and quadrature marginals agree to floating-point precision. This supports analysis of joint tuning, not a general claim that GLMs fail.</p><img src="glm_corrected.png" alt="True joint maps, additive and interaction GLMs, genuine marginals and conditional slices">
<h2>Numerical validation and boundaries</h2><p>A real-map check caught cancellation in the accelerated pairwise correlation calculation near constant support. Numerically ambiguous pairs now use the original two-pass calculation, with explicit constant-slice handling. The validated implementation agrees with reference summaries on inspected real maps to better than 10⁻¹². Preliminary outputs are quarantined under pre_numerical_validation and excluded here. Tests cover missing data, tiny variance, constant tails, alignment gaps, duplicate timestamps, nonwrapping lag, fold partitioning and equivalence of the fast decoder.</p><p>Remaining decisions include the primary grid/time-bin policy, how to report low reliability and missing scores, and the revised scientific wording. None of these reruns proves that correlation dissimilarity is a variance decomposition or unbiased estimator. A full ResNet-layer-1 Mac/CUDA comparison gives a maximum raw-EV difference of about 1.05×10⁻⁵ with identical finite-score masks. Reliability-normalized values differ by up to 0.029 at very low reliability; they are a sensitivity summary, not a guaranteed correction. <a href="crossdevice_score_validation.json">Cross-device score check</a>. New causal predictive-model training and natural-history evaluation have not been performed. Two bounded Modal A10 inference jobs completed, and all apps are stopped. Requested resource time is estimated at about $0.83, excluding build/load/idle overhead and credit offsets; this is not a verified invoice. Pre-launch estimates and the $5 allowance were provided before execution. <a href="remote_execution.json">Execution and cost record</a>. No training was started.</p>"""
    )
    css = "body{font:16px/1.6 system-ui;color:#1d2935;max-width:1250px;margin:35px auto;padding:0 24px;background:#fafbfc}h1{font-size:32px}h2{margin-top:35px}a{color:#17618b}table{font-size:13px;border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border-bottom:1px solid #dce2e8;text-align:left}th{background:#e9f0f6}img{max-width:100%}.notice{background:#fff0d4;border-left:4px solid #b5841b;padding:16px}pre{background:#edf2f6;padding:15px;overflow:auto}"
    (out / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Numerical revalidation</title><style>'
        + css
        + "</style><body>"
        + body
        + "</body></html>"
    )
    print(status)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    main(args.output)
