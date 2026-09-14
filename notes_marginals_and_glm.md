# Why Marginal Tuning Curves and GLMs Can Mislead You

These notes explain two problems that motivate the cross-fold correlation method.

---

## Part 1: The Marginal Confound

### Setup

Suppose you record a neuron while an animal moves around a box. At every moment, two stimulus variables are changing simultaneously:

- **X** = the animal's position along one axis
- **Y** = the animal's position along the other axis (or head direction, or movement direction, etc.)

You want to know: does this neuron care about X, Y, or both?

### The naive approach

The standard first step is to compute a **marginal tuning curve**: average the neuron's firing rate at each value of X, collapsing across all values of Y. If the resulting curve has a bump, you conclude the neuron is tuned to X.

### Why it breaks

The problem is that **X and Y are not sampled independently by the animal**. Real animals don't visit every combination of (X, Y) equally. For example:

- A rat might tend to face a particular direction when it's in a particular corner.
- In an open field, head direction and position are correlated because the animal runs in curves, not random teleportation.

When the sampling of X and Y is correlated, the marginal tuning curve for Y inherits structure from the neuron's tuning to X, even if the neuron doesn't care about Y at all.

### Concrete example

Imagine a neuron that ONLY cares about X. Its true firing rate is:

    f(x, y) = exp(-x^2 / 2)

It's a Gaussian bump centered at x=0. It literally does not depend on y.

Now imagine the animal tends to be near y=0 when it's near x=0 (positive X-Y correlation, rho=0.6). When you compute the marginal tuning curve for Y:

1. You bin the data by Y values
2. At y=0, the animal was often near x=0 (where the neuron fires a lot), so the average rate is high
3. At y=3, the animal was rarely near x=0, so the average rate is lower

Result: a bump in the Y marginal curve. You'd conclude the neuron is tuned to Y. But it isn't -- you're seeing a shadow of the X tuning, projected through the correlated sampling.

### How bad is it?

The stronger the behavioral correlation between X and Y, the worse the confound. At rho=0.9, the marginal Y curve can look almost as peaked as the true X curve. At rho=0, there's no confound (independent sampling preserves the flat Y marginal).

This is not a rare edge case -- in real neural recordings, behavioral correlations between position, head direction, and movement direction are substantial and unavoidable.

### Why the cross-fold correlation method avoids this

The cross-fold method builds a **joint** tuning map f(X, Y) instead of marginals. It then asks: "do the tuning curves over Y change when I move to a different X position?" by correlating slices across folds. Because it operates on the joint map and compares structure across positions, it doesn't collapse the dimensions and can't be fooled by sampling correlations.

---

## Part 2: Why a GLM Doesn't Solve the Problem Either

- A **Poisson GLM** models spike counts as `log(lambda) = b0 + b1*x + b2*y + b3*x^2 + b4*y^2`. You read off the coefficients to ask "how much does X vs Y drive this neuron?"
- Poisson GLM: a regression framework that extends ordinary linear regression to non-normal response distributions
- The problem: every term in that model depends on **only one variable**. The x^2 term is a parabola along x, the y^2 term is a parabola along y. It can only capture **axis-aligned** tuning.
- Real tuning can be **rotated**. A Gaussian firing field at 45 degrees expands to `x^2*cos^2 - 2*x*y*cos*sin + y^2*sin^2` -- notice the **x\*y cross-term** that the standard model doesn't have.
- Without that term, the GLM approximates a diagonal ridge using equal parts x^2 and y^2, and you falsely conclude the neuron is equally tuned to both axes. Any rotation other than 0 or 90 degrees produces some misattribution.
- Adding an x\*y interaction term fixes the math but breaks the interpretability -- you can no longer cleanly say "this fraction of variance is from X." With more variables (position, HD, speed, MD...), the number of interactions explodes.
- More fundamentally, a GLM requires you to **commit to a parametric form** (polynomial, etc.) in advance. Real neurons can have multi-modal fields, sharp edges, or other shapes that low-order polynomials approximate poorly.
- The cross-fold correlation method is **nonparametric** -- it uses the empirical rate map directly and asks "do tuning curves differ across positions?" No functional form assumed, so it handles rotated, multi-modal, or arbitrarily shaped tuning without modification.
