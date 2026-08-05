"""Paired comparison of two eval runs on the same items."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ._dist import t_cdf


@dataclass
class Comparison:
    n: int
    mean_a: float
    mean_b: float
    diff: float  # mean(b - a): positive means B is better
    ci_lo: float
    ci_hi: float
    level: float
    p_permutation: float
    p_t: float
    reps: int
    seed: int
    binary: bool
    mcnemar: dict | None = field(default=None)

    def as_dict(self) -> dict:
        out = {
            "n": self.n,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "diff": self.diff,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "level": self.level,
            "p_permutation": self.p_permutation,
            "p_t": self.p_t,
            "reps": self.reps,
            "seed": self.seed,
            "binary": self.binary,
        }
        if self.mcnemar is not None:
            out["mcnemar"] = self.mcnemar
        return out


def mcnemar_exact(a: list[float], b: list[float]) -> dict:
    """Exact McNemar test for paired binary outcomes.

    b_only = items B got right and A got wrong; a_only = the reverse.
    The p-value is the exact two-sided binomial sign test on the discordant
    pairs (p = 0.5 under the null), clamped to 1.
    """
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n_disc = a_only + b_only
    if n_disc == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(a_only, b_only)
    tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / 2.0**n_disc
    return {"a_only": a_only, "b_only": b_only, "p": min(1.0, 2.0 * tail)}


def paired_compare(
    a: list[float],
    b: list[float],
    level: float = 0.95,
    reps: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Compare two runs scored on the same items.

    Reports mean(b - a) with a paired percentile-bootstrap CI, a sign-flip
    permutation p-value (exact under item exchangeability), a paired-t
    p-value, and — when both runs are 0/1 — an exact McNemar test.
    """
    if len(a) != len(b):
        raise ValueError("runs must have the same length")
    n = len(a)
    if n < 2:
        raise ValueError("need at least 2 paired items")
    diffs = [y - x for x, y in zip(a, b)]
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)

    # Percentile bootstrap on the per-item differences.
    boot = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)
    )
    alpha = (1.0 - level) / 2.0
    ci_lo = boot[int(alpha * reps)]
    ci_hi = boot[min(reps - 1, int((1.0 - alpha) * reps))]

    # Sign-flip permutation test: under H0 each difference is symmetric
    # around 0, so flipping signs at random regenerates the null.
    obs = abs(mean_diff)
    hits = 0
    for _ in range(reps):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= obs - 1e-15:
            hits += 1
    p_perm = (1 + hits) / (reps + 1)

    # Paired t-test.
    sd = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (n - 1))
    if sd == 0.0:
        p_t = 1.0 if mean_diff == 0.0 else 0.0
    else:
        t_stat = mean_diff / (sd / math.sqrt(n))
        p_t = 2.0 * (1.0 - t_cdf(abs(t_stat), n - 1))

    binary = all(v in (0.0, 1.0) for v in a) and all(v in (0.0, 1.0) for v in b)
    return Comparison(
        n=n,
        mean_a=sum(a) / n,
        mean_b=sum(b) / n,
        diff=mean_diff,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        level=level,
        p_permutation=p_perm,
        p_t=p_t,
        reps=reps,
        seed=seed,
        binary=binary,
        mcnemar=mcnemar_exact(a, b) if binary else None,
    )
