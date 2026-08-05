import math

import pytest

from abeval import clustered_mean_ci, mean_ci, proportion_ci


def test_wilson_known_value():
    # 8/10 at 95%: canonical Wilson interval.
    interval = proportion_ci(8, 10)
    assert math.isclose(interval.estimate, 0.8)
    assert math.isclose(interval.lo, 0.4902, abs_tol=1e-3)
    assert math.isclose(interval.hi, 0.9433, abs_tol=1e-3)


def test_wilson_extremes_stay_in_bounds():
    assert proportion_ci(0, 20).lo == 0.0
    assert proportion_ci(20, 20).hi == 1.0
    assert proportion_ci(0, 20).hi > 0.0  # never a zero-width interval at the edge


def test_mean_ci_t_known_value():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    interval = mean_ci(values)
    # mean 3, sd sqrt(2.5), se 0.7071, t(0.975, 4) = 2.7764 -> half-width 1.9633
    assert math.isclose(interval.estimate, 3.0)
    assert math.isclose(interval.hi - interval.estimate, 1.9633, abs_tol=1e-3)
    assert math.isclose(interval.estimate - interval.lo, 1.9633, abs_tol=1e-3)


def test_bootstrap_deterministic_and_sane():
    values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3, 0.7] * 5
    a = mean_ci(values, method="bootstrap", seed=7)
    b = mean_ci(values, method="bootstrap", seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert a.lo < a.estimate < a.hi


def test_clustered_wider_than_iid_when_clusters_dominate():
    # Two clusters with strongly separated values: within-cluster dependence
    # should widen the interval relative to the iid t interval.
    values = [0.9, 0.92, 0.88, 0.91, 0.3, 0.28, 0.33, 0.29]
    clusters = ["doc1"] * 4 + ["doc2"] * 4
    clustered = clustered_mean_ci(values, clusters)
    iid = mean_ci(values)
    assert (clustered.hi - clustered.lo) > (iid.hi - iid.lo)


def test_input_validation():
    with pytest.raises(ValueError):
        proportion_ci(5, 0)
    with pytest.raises(ValueError):
        mean_ci([1.0])
    with pytest.raises(ValueError):
        clustered_mean_ci([1.0, 2.0], ["a"])
    with pytest.raises(ValueError):
        mean_ci([1.0, 2.0], method="magic")
