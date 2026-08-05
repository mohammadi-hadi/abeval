import math

import pytest

from abeval import judge_reliability


def test_perfect_judge():
    groups = {f"q{i}": [float(i % 5)] * 3 for i in range(10)}
    rel = judge_reliability(groups)
    assert rel.icc == 1.0
    assert rel.sd_within == 0.0
    assert rel.exact_agreement == 1.0
    assert rel.repeats_for(0.1) == 1


def test_pure_noise_judge():
    # Same true score everywhere; all variation is judge noise.
    groups = {
        "q1": [3.0, 5.0, 4.0],
        "q2": [4.0, 3.0, 5.0],
        "q3": [5.0, 4.0, 3.0],
    }
    rel = judge_reliability(groups)
    assert rel.icc == 0.0
    assert rel.sd_between == 0.0
    assert rel.repeats_for(0.1) is None


def test_anova_hand_computed():
    # Balanced one-way ANOVA, k=2 reps: groups (1,2), (4,5), (7,9).
    # Group means 1.5, 4.5, 8.0; grand mean 4.6667.
    # SS_within = 3.0 -> MS_within = 3/3 = 1.0
    # SS_between = 42.3333 -> MS_between = 21.1667; k0 = 2
    # var_between = (21.1667 - 1.0) / 2 = 10.0833
    groups = {"a": [1.0, 2.0], "b": [4.0, 5.0], "c": [7.0, 9.0]}
    rel = judge_reliability(groups)
    assert math.isclose(rel.sd_within**2, 1.0, abs_tol=1e-9)
    assert math.isclose(rel.sd_between**2, 10.08333, abs_tol=1e-4)
    assert math.isclose(rel.icc, 10.08333 / 11.08333, abs_tol=1e-4)


def test_unbalanced_designs_accepted():
    groups = {"a": [1.0, 1.0, 1.0], "b": [2.0], "c": [3.0, 3.0]}
    rel = judge_reliability(groups)
    assert rel.n_items == 3
    assert rel.n_judgments == 6
    assert rel.icc > 0.9


def test_needs_repeats():
    with pytest.raises(ValueError):
        judge_reliability({"a": [1.0], "b": [2.0]})
