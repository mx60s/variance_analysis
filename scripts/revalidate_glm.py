"""Corrected demonstration: additive log-rate GLM misses an interaction.

Both marginals integrate over the actual independent standard-normal stimulus
distribution. Conditional slices are labeled separately. An interaction GLM is
included to avoid implying a general limitation of GLMs.
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import gammaln
from numpy.polynomial.hermite import hermgauss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rate(x, y, theta):
    u = x * np.cos(np.deg2rad(theta)) - y * np.sin(np.deg2rad(theta))
    return 5 * np.exp(-u * u / (2 * 1.2**2))


def predict(model, x, y):
    x, y = np.broadcast_arrays(x, y)
    return np.asarray(
        model.predict(pd.DataFrame({"x": x.ravel(), "y": y.ravel()}))
    ).reshape(x.shape)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 11})
    fig, axs = plt.subplots(2, 5, figsize=(11.5, 6.2), layout="constrained")
    summaries = []
    grid = np.linspace(-2.5, 2.5, 120)
    xx, yy = np.meshgrid(grid, grid)
    nodes, weights = hermgauss(50)
    nodes *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    for row, theta in enumerate([60, 45]):
        rng = np.random.default_rng(row)
        x, y = rng.normal(size=(2, 20000))
        counts = rng.poisson(rate(x, y, theta))
        df = pd.DataFrame(dict(x=x, y=y, spikes=counts))
        additive = smf.glm(
            "spikes ~ x + y + I(x**2) + I(y**2)", df, family=sm.families.Poisson()
        ).fit()
        interaction = smf.glm(
            "spikes ~ x + y + I(x**2) + I(y**2) + x:y", df, family=sm.families.Poisson()
        ).fit()
        for ax, z, title in zip(
            axs[row, :3],
            [
                rate(xx, yy, theta),
                predict(additive, xx, yy),
                predict(interaction, xx, yy),
            ],
            ["True joint tuning", "Additive log-rate\nGLM", "GLM with x:y\ninteraction"],
        ):
            im = ax.imshow(
                z,
                origin="lower",
                extent=[-2.5, 2.5, -2.5, 2.5],
                cmap="viridis",
                vmin=0,
                vmax=5,
            )
            ax.set(title=title, xlabel="x", ylabel="y")
            fig.colorbar(im, ax=ax, fraction=0.046, label="Expected count")
        marginal = rate(grid[:, None], nodes[None, :], theta) @ weights
        c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
        analytic = (
            5
            * 1.2
            / np.sqrt(1.2**2 + s * s)
            * np.exp(-c * c * grid * grid / (2 * (1.2**2 + s * s)))
        )
        np.testing.assert_allclose(marginal, analytic, atol=1e-10)
        for ax, true, ap, ip, title in [
            (
                axs[row, 3],
                marginal,
                predict(additive, grid[:, None], nodes[None, :]) @ weights,
                predict(interaction, grid[:, None], nodes[None, :]) @ weights,
                "Marginal\nE[h(x,Y)]",
            ),
            (
                axs[row, 4],
                rate(grid, 1, theta),
                predict(additive, grid, 1),
                predict(interaction, grid, 1),
                "Conditional slice\nh(x, 1)",
            ),
        ]:
            ax.plot(grid, true, color="black", label="True", lw=2)
            ax.plot(grid, ap, "--", color="#d55e00", label="Additive")
            ax.plot(grid, ip, ":", color="#0072b2", label="Interaction", lw=2)
            ax.set(title=title, xlabel="x", ylim=(0, 5.3))
            ax.legend(frameon=False, fontsize=10)
        x, y = rng.normal(size=(2, 20000))
        truth = rate(x, y, theta)
        k = rng.poisson(truth)
        ll = {
            name: float(np.mean(k * np.log(mu) - mu - gammaln(k + 1)))
            for name, mu in [
                ("oracle", truth),
                ("additive", predict(additive, x, y)),
                ("interaction", predict(interaction, x, y)),
            ]
        }
        summaries.append(
            dict(
                theta=theta,
                heldout_mean_loglik=ll,
                interaction_gain_nats_per_observation=ll["interaction"]
                - ll["additive"],
                marginal_quadrature_max_error=float(np.max(abs(marginal - analytic))),
            )
        )
        axs[row, 0].text(
            -0.25, 1.1, f"{theta}°", transform=axs[row, 0].transAxes, fontsize=14
        )
    fig.savefig(out / "glm_corrected.png", dpi=160)
    fig.savefig(out / "glm_corrected.pdf")
    (out / "glm_validation.json").write_text(json.dumps(summaries, indent=2))
    print(summaries)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    main(args.output)
