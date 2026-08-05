"""Sample-size planning and minimum detectable effects for paired evals."""

from __future__ import annotations

import math
import random

from ._dist import z_quantile


def sd_diff_from_rates(p_a: float, p_b: float, corr: float) -> float:
    """SD of the per-item difference for paired binary outcomes.

    corr is the item-level correlation between the two runs' outcomes
    (0 = independent, higher = the models succeed and fail on the same items,
    which is the typical case and what makes paired designs cheap).
    """
    for p in (p_a, p_b):
        if not 0.0 < p < 1.0:
            raise ValueError("rates must be in (0, 1)")
    if not -1.0 < corr < 1.0:
        raise ValueError("corr must be in (-1, 1)")
    var = (
        p_a * (1.0 - p_a)
        + p_b * (1.0 - p_b)
        - 2.0 * corr * math.sqrt(p_a * (1.0 - p_a) * p_b * (1.0 - p_b))
    )
    if var <= 0.0:
        raise ValueError("implied variance is not positive; check the inputs")
    return math.sqrt(var)


def sample_size(
    delta: float,
    sd_diff: float,
    level: float = 0.95,
    power: float = 0.8,
) -> int:
    """Paired items needed to detect a true difference `delta`.

    Normal-approximation two-sided test on the mean of per-item differences.
    """
    if delta == 0.0:
        raise ValueError("delta must be nonzero")
    if sd_diff <= 0.0:
        raise ValueError("sd_diff must be positive")
    z_alpha = z_quantile(1.0 - (1.0 - level) / 2.0)
    z_power = z_quantile(power)
    n = ((z_alpha + z_power) * sd_diff / abs(delta)) ** 2
    return max(2, math.ceil(n))


def mde(
    n: int,
    sd_diff: float,
    level: float = 0.95,
    power: float = 0.8,
) -> float:
    """Minimum detectable difference at a given number of paired items."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if sd_diff <= 0.0:
        raise ValueError("sd_diff must be positive")
    z_alpha = z_quantile(1.0 - (1.0 - level) / 2.0)
    z_power = z_quantile(power)
    return (z_alpha + z_power) * sd_diff / math.sqrt(n)


def paired_binary_cells(p_a: float, p_b: float, corr: float) -> dict:
    """Joint distribution of paired 0/1 outcomes with the given margins/corr."""
    sd_a = math.sqrt(p_a * (1.0 - p_a))
    sd_b = math.sqrt(p_b * (1.0 - p_b))
    p_both = p_a * p_b + corr * sd_a * sd_b
    cells = {
        "both": p_both,
        "a_only": p_a - p_both,
        "b_only": p_b - p_both,
        "neither": 1.0 - p_a - p_b + p_both,
    }
    if any(v < -1e-12 for v in cells.values()):
        raise ValueError("that correlation is not achievable with these rates")
    return {k: max(0.0, v) for k, v in cells.items()}


def power_simulated(
    n: int,
    p_a: float,
    p_b: float,
    corr: float,
    level: float = 0.95,
    reps: int = 2_000,
    seed: int = 0,
) -> float:
    """Monte Carlo power of the paired z-test for binary outcomes.

    A check on the normal-approximation formulas: simulates `reps` evals of n
    paired items with the given accuracy margins and item-level correlation,
    and reports the fraction where the test rejects at the given level.
    """
    cells = paired_binary_cells(p_a, p_b, corr)
    thresholds = []
    acc = 0.0
    for key in ("both", "a_only", "b_only", "neither"):
        acc += cells[key]
        thresholds.append((acc, key))
    diff_of = {"both": 0.0, "neither": 0.0, "a_only": -1.0, "b_only": 1.0}
    z_alpha = z_quantile(1.0 - (1.0 - level) / 2.0)
    rng = random.Random(seed)
    rejections = 0
    for _ in range(reps):
        diffs = []
        for _ in range(n):
            u = rng.random()
            for cutoff, key in thresholds:
                if u <= cutoff:
                    diffs.append(diff_of[key])
                    break
            else:
                diffs.append(0.0)
        m = sum(diffs) / n
        var = sum((d - m) ** 2 for d in diffs) / (n - 1)
        if var == 0.0:
            continue
        z_stat = m / math.sqrt(var / n)
        if abs(z_stat) > z_alpha:
            rejections += 1
    return rejections / reps
