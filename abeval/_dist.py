"""Distribution helpers used by the estimators.

Everything here is standard-library only. The Student-t CDF goes through the
regularized incomplete beta function (continued-fraction expansion), and the
quantile inverts it by bisection — slower than a closed form but robust for
every df, which is what a stats tool should optimize for.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def z_quantile(p: float) -> float:
    """Standard normal quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return _NORMAL.inv_cdf(p)


def normal_cdf(x: float) -> float:
    return _NORMAL.cdf(x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    max_iter = 300
    eps = 3e-12
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def reg_inc_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with `df` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    x = df / (df + t * t)
    p_tail = 0.5 * reg_inc_beta(df / 2.0, 0.5, x)
    return 1.0 - p_tail if t > 0 else p_tail


def t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t, by bisection on the CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    lo, hi = -1e8, 1e8
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
