"""
Profile-level evaluation metrics (proposal Section 3.7).

All quantities are computed from the *true* click probabilities f, which the
evaluator knows but learners never see. Vectorised with numpy so they can be
evaluated every round of a T = 10^4 simulation.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.tier1.knapsack import compute_best_response_knapsack


def winning_probabilities(bids: np.ndarray) -> np.ndarray:
    """q_ij(b) under uniform random tie-breaking. ``bids`` has shape (m, n)."""
    top = bids.max(axis=0)
    is_top = (bids == top) & (top > 0)
    return is_top / np.maximum(is_top.sum(axis=0), 1)


def competing_bids(bids: np.ndarray) -> np.ndarray:
    """c_{-i,j} = max_{k != i} b_kj, shape (m, n)."""
    m = bids.shape[0]
    if m == 1:
        return np.zeros_like(bids)
    out = np.empty_like(bids)
    for i in range(m):
        out[i] = np.delete(bids, i, axis=0).max(axis=0)
    return out


def expected_profits(bids: np.ndarray, f: np.ndarray, v: float, q: Optional[np.ndarray] = None) -> np.ndarray:
    """p_i(b) = sum_j q_ij (f_ij v - b_ij)."""
    q = winning_probabilities(bids) if q is None else q
    return (q * (f * v - bids)).sum(axis=1)


def platform_revenue(bids: np.ndarray, q: np.ndarray) -> float:
    return float((q * bids).sum())


def expected_welfare(f: np.ndarray, v: float, q: np.ndarray) -> float:
    return float((q * f * v).sum())


def best_advertiser_mask(f: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """mask[i, j] = True iff i in argmax_k f_kj."""
    return f >= f.max(axis=0) - tol


def allocation_efficiency(q_or_wins: np.ndarray, best_mask: np.ndarray) -> float:
    """Eff = (1/n) sum_j sum_{i in argmax_k f_kj} q_ij (pass realized wins for Eff_t)."""
    return float((q_or_wins * best_mask).sum() / q_or_wins.shape[1])


def fluid_knapsack_value(f_i: np.ndarray, c_i: np.ndarray, capacity: float, v: float) -> float:
    """Fractional (LP) knapsack over strict-win items: the per-round fluid
    benchmark for the cumulative-budget variant (bandits with knapsacks)."""
    w = c_i + 1.0
    r = f_i * v - w
    keep = r > 1e-9
    if capacity <= 0 or not keep.any():
        return 0.0
    w, r = w[keep], r[keep]
    order = np.argsort(-(r / w), kind="stable")
    val, cap = 0.0, float(capacity)
    for k in order:
        take = min(1.0, cap / w[k])
        val += take * r[k]
        cap -= take * w[k]
        if cap <= 1e-12:
            break
    return val


class BestResponseOracle:
    """Caches p_i^*(c_{-i}) - the knapsack best-response value - per (i, c_{-i}, cap).

    In Tier 2 the opponents are fixed, so after round 1 every lookup is a cache
    hit; in Tier 3 the profile usually settles, so the cache stays effective.
    """

    def __init__(self, f: np.ndarray, v: float, fluid_capacity: Optional[np.ndarray] = None):
        self.f = f
        self.v = v
        self.fluid_capacity = fluid_capacity  # per-agent, cumulative variant only
        self._cache: Dict[Tuple, float] = {}

    def value(self, i: int, c_i: np.ndarray, cap: int) -> float:
        if self.fluid_capacity is not None:
            # Fluid benchmark spends total_i / T every round for the whole horizon,
            # independent of how much the evaluated policy has already spent.
            cap = -1
        key = (i, cap, c_i.tobytes())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if self.fluid_capacity is not None:
            val = fluid_knapsack_value(self.f[i], c_i, self.fluid_capacity[i], self.v)
        elif cap <= 0:
            val = 0.0
        else:
            res = compute_best_response_knapsack(list(self.f[i]), [int(x) for x in c_i], int(cap), self.v)
            val = max(0.0, res.expected_profit)  # bidding 0 everywhere is always feasible
        self._cache[key] = val
        return val


@dataclass
class ProfileMetrics:
    profits: np.ndarray        # p_i(b^t), shape (m,)
    br_values: np.ndarray      # p_i^*(c_{-i}^t), shape (m,)
    regret: np.ndarray         # br_values - profits
    eq_gap: float              # max_i max(0, regret_i)
    revenue: float
    welfare: float
    efficiency: float          # expected (via q)
    eq_distance: float         # ||b^t - b*||_1


def evaluate_profile(
    bids: np.ndarray,
    f: np.ndarray,
    v: float,
    oracle: BestResponseOracle,
    caps: np.ndarray,
    best_mask: np.ndarray,
    b_star: Optional[np.ndarray] = None,
) -> ProfileMetrics:
    q = winning_probabilities(bids)
    profits = expected_profits(bids, f, v, q)
    comp = competing_bids(bids)
    br = np.array([oracle.value(i, comp[i], int(caps[i])) for i in range(bids.shape[0])])
    regret = br - profits
    return ProfileMetrics(
        profits=profits,
        br_values=br,
        regret=regret,
        eq_gap=float(max(0.0, regret.max())),
        revenue=platform_revenue(bids, q),
        welfare=expected_welfare(f, v, q),
        efficiency=allocation_efficiency(q, best_mask),
        eq_distance=float(np.abs(bids - b_star).sum()) if b_star is not None else float("nan"),
    )
