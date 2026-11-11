"""Across-seed statistics: means and Student-t confidence intervals."""
from typing import Tuple
import warnings

import numpy as np

try:  # scipy is optional; fall back to a small t-table.
    from scipy.stats import t as _student_t
except ImportError:  # pragma: no cover - exercised only without scipy
    _student_t = None

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045, 30: 2.042,
         40: 2.021, 60: 2.000, 120: 1.980}


def t_quantile(df: int, confidence: float = 0.95) -> float:
    if _student_t is not None:
        return float(_student_t.ppf(0.5 + confidence / 2.0, df))
    if abs(confidence - 0.95) > 1e-12:
        raise RuntimeError("Install scipy for confidence levels other than 0.95")
    keys = [k for k in _T975 if k <= df]
    return _T975[max(keys)] if keys else _T975[1]


def mean_ci(samples, axis: int = 0, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, half_width) along ``axis``, ignoring NaNs.

    Half-width is t_{df} * s / sqrt(k) with k the number of non-NaN samples;
    it is 0 where k < 2.
    """
    x = np.asarray(samples, dtype=float)
    k = np.sum(~np.isnan(x), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(x, axis=axis) if x.size else np.nan
        sd = np.nanstd(x, axis=axis, ddof=1)
        tq = np.vectorize(lambda d: t_quantile(int(d), confidence) if d >= 1 else 0.0)(np.maximum(k - 1, 0))
        half = np.where(k >= 2, tq * sd / np.sqrt(np.maximum(k, 1)), 0.0)
    return mean, half


def loglog_slope(t: np.ndarray, y: np.ndarray, start_frac: float = 0.1) -> float:
    """Least-squares slope of log y vs log t over t >= start_frac * T.

    For cumulative regret, a slope < 1 indicates sublinear growth (H1);
    sqrt(T)-type regret gives ~0.5, log(T)-type gives ~0.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = (t >= max(2.0, start_frac * t[-1])) & (y > 1e-12)
    if sel.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(t[sel]), np.log(y[sel]), 1)[0])
