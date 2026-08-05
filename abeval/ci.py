"""Confidence intervals for a single eval run."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ._dist import t_quantile, z_quantile


@dataclass
class Interval:
    estimate: float
    lo: float
    hi: float
    n: int
    method: str
    level: float
    se: float | None = None

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "method": self.method,
            "level": self.level,
            "se": self.se,
        }


def proportion_ci(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = z_quantile(1.0 - (1.0 - level) / 2.0)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    return Interval(
        estimate=p_hat,
        lo=max(0.0, center - half),
        hi=min(1.0, center + half),
        n=n,
        method="wilson",
        level=level,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_sd(values: list[float]) -> float:
    n = len(values)
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def mean_ci(
    values: list[float],
    level: float = 0.95,
    method: str = "t",
    reps: int = 10_000,
    seed: int = 0,
) -> Interval:
    """CI for the mean of per-item values.

    method="t" uses the t distribution; method="bootstrap" uses the percentile
    bootstrap with a seeded RNG so results are reproducible.
    """
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values")
    m = _mean(values)
    if method == "t":
        se = _sample_sd(values) / math.sqrt(n)
        t = t_quantile(1.0 - (1.0 - level) / 2.0, n - 1)
        return Interval(m, m - t * se, m + t * se, n, "t", level, se=se)
    if method == "bootstrap":
        rng = random.Random(seed)
        stats = sorted(
            _mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps)
        )
        alpha = (1.0 - level) / 2.0
        lo = stats[int(alpha * reps)]
        hi = stats[min(reps - 1, int((1.0 - alpha) * reps))]
        return Interval(m, lo, hi, n, "bootstrap", level)
    raise ValueError(f"unknown method {method!r}")


def clustered_mean_ci(
    values: list[float],
    clusters: list,
    level: float = 0.95,
) -> Interval:
    """CI for a mean when items are not independent within clusters.

    Uses the cluster-robust variance of the mean (sum of squared within-cluster
    residual sums over n^2) with a t reference on G-1 degrees of freedom, where
    G is the number of clusters. With few clusters (< ~20) treat the interval
    as approximate.
    """
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have the same length")
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values")
    m = _mean(values)
    sums: dict = {}
    for v, c in zip(values, clusters):
        sums[c] = sums.get(c, 0.0) + (v - m)
    n_clusters = len(sums)
    if n_clusters < 2:
        raise ValueError("need at least 2 clusters")
    var = sum(s * s for s in sums.values()) / (n * n)
    se = math.sqrt(var)
    t = t_quantile(1.0 - (1.0 - level) / 2.0, n_clusters - 1)
    return Interval(m, m - t * se, m + t * se, n, f"clustered-t ({n_clusters} clusters)", level, se=se)
