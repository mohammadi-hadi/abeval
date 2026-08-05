"""How noisy is the judge? Variance decomposition over repeated judgments."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Reliability:
    n_items: int
    n_judgments: int
    mean_reps: float
    icc: float
    sd_between: float  # spread of true item scores
    sd_within: float  # judge noise on a single judgment
    exact_agreement: float  # fraction of items where every repeat matched

    def repeats_for(self, noise_share: float = 0.1) -> int | None:
        """Repeats per item so judge noise is at most `noise_share` of the
        variance of an item's averaged score. None when the between-item
        variance is zero (no signal to protect)."""
        if not 0.0 < noise_share < 1.0:
            raise ValueError("noise_share must be in (0, 1)")
        var_b = self.sd_between**2
        var_w = self.sd_within**2
        if var_b <= 0.0:
            return None
        if var_w == 0.0:
            return 1
        return max(1, math.ceil(var_w * (1.0 - noise_share) / (noise_share * var_b)))

    def as_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_judgments": self.n_judgments,
            "mean_reps": self.mean_reps,
            "icc": self.icc,
            "sd_between": self.sd_between,
            "sd_within": self.sd_within,
            "exact_agreement": self.exact_agreement,
        }


def judge_reliability(groups: dict) -> Reliability:
    """One-way random-effects decomposition of repeated judge scores.

    `groups` maps item id -> list of scores from repeated judgments of the
    same item. Unbalanced designs are fine (k0 correction). Returns the
    intraclass correlation (share of variance that is real item signal rather
    than judge noise) plus both variance components.
    """
    groups = {k: [float(v) for v in vals] for k, vals in groups.items() if len(vals) >= 1}
    if len(groups) < 2:
        raise ValueError("need at least 2 items")
    if all(len(v) < 2 for v in groups.values()):
        raise ValueError("need repeated judgments (at least one item with 2+ scores)")

    n_total = sum(len(v) for v in groups.values())
    n_groups = len(groups)
    grand = sum(sum(v) for v in groups.values()) / n_total

    ss_between = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in groups.values())
    ss_within = sum(
        sum((x - sum(v) / len(v)) ** 2 for x in v) for v in groups.values()
    )
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups) if n_total > n_groups else 0.0

    k0 = (n_total - sum(len(v) ** 2 for v in groups.values()) / n_total) / (n_groups - 1)
    var_within = ms_within
    var_between = max(0.0, (ms_between - ms_within) / k0) if k0 > 0 else 0.0
    total = var_between + var_within
    icc = var_between / total if total > 0 else 0.0

    exact = sum(1 for v in groups.values() if len(set(v)) == 1) / n_groups
    return Reliability(
        n_items=n_groups,
        n_judgments=n_total,
        mean_reps=n_total / n_groups,
        icc=icc,
        sd_between=math.sqrt(var_between),
        sd_within=math.sqrt(var_within),
        exact_agreement=exact,
    )
