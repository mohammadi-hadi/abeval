import math

import pytest

from abeval import mcnemar_exact, paired_compare


def test_mcnemar_exact_known_value():
    # 8 discordant one way, 2 the other: p = 2 * sum_{i<=2} C(10,i) / 2^10
    a = [1.0] * 2 + [0.0] * 8 + [1.0] * 10
    b = [0.0] * 2 + [1.0] * 8 + [1.0] * 10
    result = mcnemar_exact(a, b)
    assert result["a_only"] == 2
    assert result["b_only"] == 8
    assert math.isclose(result["p"], 112 / 1024, abs_tol=1e-12)


def test_mcnemar_no_discordant():
    a = [1.0, 0.0, 1.0]
    assert mcnemar_exact(a, a)["p"] == 1.0


def test_paired_compare_deterministic():
    a = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0] * 4
    b = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0] * 4
    r1 = paired_compare(a, b, seed=3)
    r2 = paired_compare(a, b, seed=3)
    assert r1.as_dict() == r2.as_dict()


def test_paired_compare_obvious_difference():
    a = [0.0] * 30
    b = [1.0] * 30
    result = paired_compare(a, b)
    assert result.diff == 1.0
    assert result.p_permutation < 0.01
    assert result.p_t < 1e-6
    assert result.mcnemar["p"] < 1e-6
    assert result.ci_lo == result.ci_hi == 1.0  # all diffs identical


def test_paired_compare_null_is_not_significant():
    a = [1.0, 0.0] * 20
    result = paired_compare(a, a)
    assert result.diff == 0.0
    assert result.p_permutation > 0.9
    assert result.p_t == 1.0


def test_ci_brackets_diff_and_directions_agree():
    a = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.55, 0.45, 0.5, 0.65]
    b = [0.7, 0.65, 0.6, 0.75, 0.6, 0.7, 0.6, 0.6, 0.7, 0.7]
    result = paired_compare(a, b)
    assert result.ci_lo <= result.diff <= result.ci_hi
    assert result.diff > 0
    assert not result.binary
    assert result.mcnemar is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_compare([1.0, 0.0], [1.0])
