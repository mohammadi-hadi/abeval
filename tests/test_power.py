import math

import pytest

from abeval import mde, power_simulated, sample_size, sd_diff_from_rates


def test_sample_size_hand_formula():
    # (z_.975 + z_.8)^2 * sd^2 / delta^2, z values 1.95996 and 0.84162
    sd = 0.5
    delta = 0.05
    expected = math.ceil(((1.959964 + 0.841621) * sd / delta) ** 2)
    assert sample_size(delta, sd) == expected  # 785


def test_mde_inverts_sample_size():
    sd = 0.62
    delta = 0.04
    n = sample_size(delta, sd)
    detectable = mde(n, sd)
    assert detectable <= delta
    assert mde(n - 5, sd) > detectable


def test_sd_diff_shrinks_with_correlation():
    uncorrelated = sd_diff_from_rates(0.7, 0.75, 0.0)
    correlated = sd_diff_from_rates(0.7, 0.75, 0.6)
    assert correlated < uncorrelated
    # Independent case is the sqrt of summed Bernoulli variances.
    expected = math.sqrt(0.7 * 0.3 + 0.75 * 0.25)
    assert math.isclose(uncorrelated, expected, rel_tol=1e-12)


def test_simulated_power_matches_formula_direction():
    # At the formula's n for 80% power the simulation should land near 0.8.
    sd = sd_diff_from_rates(0.70, 0.75, 0.5)
    n = sample_size(0.05, sd)
    p = power_simulated(n, 0.70, 0.75, 0.5, reps=400, seed=11)
    assert 0.65 < p < 0.95
    # Far below that n, power collapses.
    assert power_simulated(max(2, n // 10), 0.70, 0.75, 0.5, reps=400, seed=11) < 0.5


def test_validation():
    with pytest.raises(ValueError):
        sample_size(0.0, 0.5)
    with pytest.raises(ValueError):
        sd_diff_from_rates(0.0, 0.5, 0.2)
    with pytest.raises(ValueError):
        mde(1, 0.5)
