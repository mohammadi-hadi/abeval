import math

from abeval._dist import t_cdf, t_quantile, z_quantile


def test_z_quantile_known_values():
    assert math.isclose(z_quantile(0.975), 1.959964, abs_tol=1e-4)
    assert math.isclose(z_quantile(0.8), 0.841621, abs_tol=1e-4)


def test_t_quantile_matches_tables():
    # Values from standard t tables.
    assert math.isclose(t_quantile(0.975, 1), 12.7062, rel_tol=1e-3)
    assert math.isclose(t_quantile(0.975, 10), 2.22814, rel_tol=1e-4)
    assert math.isclose(t_quantile(0.975, 30), 2.04227, rel_tol=1e-4)
    assert math.isclose(t_quantile(0.95, 5), 2.01505, rel_tol=1e-4)


def test_t_converges_to_normal():
    assert math.isclose(t_quantile(0.975, 100000), z_quantile(0.975), abs_tol=1e-3)


def test_t_cdf_symmetry_and_median():
    assert math.isclose(t_cdf(0.0, 7), 0.5, abs_tol=1e-12)
    assert math.isclose(t_cdf(1.5, 7) + t_cdf(-1.5, 7), 1.0, abs_tol=1e-9)


def test_cdf_quantile_roundtrip():
    for df in (1, 4, 27):
        for p in (0.6, 0.9, 0.99):
            assert math.isclose(t_cdf(t_quantile(p, df), df), p, abs_tol=1e-6)
