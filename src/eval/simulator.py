"""
Repeated batched first-price auction simulator.

Runs any list of ``BaseAdvertiserAgent`` for T rounds on an ``AuctionInstance``
under one of the three round structures of proposal Remark 1:

* ``canonical``  - per-round budget B_i resets every round.
* ``discounted`` - canonical constraint; round-t quantities are weighted by
                   gamma^(t-1) in every cumulative statistic.
* ``cumulative`` - one total budget per advertiser shared across all T rounds
                   (default ``rho * T * B_i``) with per-round cap B_i; an
                   advertiser is excluded once its spend reaches the total.
                   Regret is measured against the per-round fluid (LP)
                   relaxation with capacity min(B_i, total_i / T).

Realized outcomes (wins, clicks, payments) come from the shared
``BatchedFirstPriceAuction.simulate_round``; expected metrics use the true f.
"""
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.agents.base import BaseAdvertiserAgent
from src.env.auction import BatchedFirstPriceAuction
from src.env.instance import AuctionInstance
from src.eval.metrics import BestResponseOracle, best_advertiser_mask, allocation_efficiency, evaluate_profile

ROUND_STRUCTURES = ("canonical", "discounted", "cumulative")


@dataclass
class SimulationConfig:
    horizon: int = 1000
    round_structure: str = "canonical"
    gamma: float = 0.999            # discounted variant
    cumulative_rho: float = 0.5     # cumulative variant: total budget = rho * T * B_i
    on_infeasible: str = "zero"     # "zero" (replace with 0-bid and count) or "raise"
    seed: int = 0

    def __post_init__(self):
        if self.round_structure not in ROUND_STRUCTURES:
            raise ValueError(f"round_structure must be one of {ROUND_STRUCTURES}")
        if self.on_infeasible not in ("zero", "raise"):
            raise ValueError("on_infeasible must be 'zero' or 'raise'")

    def total_budgets(self, budgets: Sequence[int]) -> Optional[List[int]]:
        if self.round_structure != "cumulative":
            return None
        return [max(1, int(round(self.cumulative_rho * self.horizon * b))) for b in budgets]


@dataclass
class SimulationTrace:
    """Per-round time series. Arrays are indexed [t] or [t, i] with t = 0..T-1."""

    profits: np.ndarray            # (T, m) expected p_i(b^t)
    realized_profits: np.ndarray   # (T, m)
    br_values: np.ndarray          # (T, m) benchmark p_i^*(c_{-i}^t)
    regret: np.ndarray             # (T, m) per-round regret
    eq_gap: np.ndarray             # (T,)
    revenue: np.ndarray            # (T,)
    welfare: np.ndarray            # (T,)
    efficiency: np.ndarray         # (T,) expected Eff via q
    realized_efficiency: np.ndarray  # (T,) Eff_t via realized wins
    eq_distance: np.ndarray        # (T,)
    spend: np.ndarray              # (T, m) realized payments
    weights: np.ndarray            # (T,) discount weights (all 1 unless discounted)
    violations: np.ndarray         # (m,) count of infeasible bid vectors replaced by 0
    exhausted_round: List[Optional[int]] = field(default_factory=list)  # cumulative only

    def cumulative_regret(self) -> np.ndarray:
        """(T, m) weighted cumulative regret."""
        return np.cumsum(self.regret * self.weights[:, None], axis=0)

    def to_dict(self) -> Dict[str, list]:
        out = {k: getattr(self, k).tolist() for k in (
            "profits", "realized_profits", "br_values", "regret", "eq_gap", "revenue", "welfare",
            "efficiency", "realized_efficiency", "eq_distance", "spend", "weights", "violations")}
        out["exhausted_round"] = list(self.exhausted_round)
        return out


def _sanitize(bid, n: int, cap: int, agent_id: int, mode: str) -> Optional[List[int]]:
    arr = np.asarray(bid)
    ok = (
        arr.shape == (n,)
        and np.all(np.isfinite(arr))
        and np.all(arr == np.round(arr))
        and np.all(arr >= 0)
        and arr.sum() <= cap
    )
    if ok:
        return [int(x) for x in arr]
    if mode == "raise":
        raise ValueError(f"Agent {agent_id} submitted infeasible bid {arr.tolist()} (cap {cap})")
    return None


def run_simulation(
    instance: AuctionInstance,
    agents: Sequence[BaseAdvertiserAgent],
    config: SimulationConfig,
    reference_profile: Optional[List[List[int]]] = None,
) -> SimulationTrace:
    m, n, T = instance.num_advertisers, instance.num_users, config.horizon
    if len(agents) != m:
        raise ValueError(f"Need {m} agents, got {len(agents)}")

    auction = BatchedFirstPriceAuction(instance)
    f = np.asarray(instance.click_probabilities, dtype=float)
    v = float(instance.click_value)
    budgets = np.asarray(instance.budgets, dtype=int)
    totals = config.total_budgets(instance.budgets)
    remaining = None if totals is None else np.asarray(totals, dtype=int)
    fluid_cap = None if totals is None else np.minimum(budgets, np.asarray(totals, dtype=float) / T)
    oracle = BestResponseOracle(f, v, fluid_capacity=fluid_cap)
    best_mask = best_advertiser_mask(f)
    b_star = None if reference_profile is None else np.asarray(reference_profile, dtype=int)
    rng = random.Random(config.seed)

    for a in agents:
        a.reset()

    shape_tm = (T, m)
    tr = SimulationTrace(
        profits=np.zeros(shape_tm), realized_profits=np.zeros(shape_tm), br_values=np.zeros(shape_tm),
        regret=np.zeros(shape_tm), eq_gap=np.zeros(T), revenue=np.zeros(T), welfare=np.zeros(T),
        efficiency=np.zeros(T), realized_efficiency=np.zeros(T), eq_distance=np.zeros(T),
        spend=np.zeros(shape_tm),
        weights=(config.gamma ** np.arange(T)) if config.round_structure == "discounted" else np.ones(T),
        violations=np.zeros(m, dtype=int),
        exhausted_round=[None] * m if totals is not None else [],
    )

    last_comp: List[Optional[np.ndarray]] = [None] * m
    for t in range(1, T + 1):
        caps = budgets.copy() if remaining is None else np.minimum(budgets, remaining)
        profile: List[List[int]] = []
        for i, agent in enumerate(agents):
            if caps[i] <= 0:
                profile.append([0] * n)  # excluded (cumulative budget exhausted)
                continue
            bid = _sanitize(agent.get_bid(t, last_comp[i]), n, int(caps[i]), i, config.on_infeasible)
            if bid is None:
                tr.violations[i] += 1
                bid = [0] * n
            profile.append(bid)

        bids = np.asarray(profile, dtype=int)
        pm = evaluate_profile(bids, f, v, oracle, caps, best_mask, b_star)
        out = auction.simulate_round(profile, rng)
        wins = np.asarray(out.wins)
        pay = np.asarray(out.payments)

        k = t - 1
        tr.profits[k] = pm.profits
        tr.br_values[k] = pm.br_values
        tr.regret[k] = pm.regret
        tr.eq_gap[k] = pm.eq_gap
        tr.revenue[k] = pm.revenue
        tr.welfare[k] = pm.welfare
        tr.efficiency[k] = pm.efficiency
        tr.eq_distance[k] = pm.eq_distance
        tr.realized_profits[k] = out.realized_profits
        tr.realized_efficiency[k] = allocation_efficiency(wins, best_mask)
        tr.spend[k] = pay.sum(axis=1)

        comp = np.asarray(out.competing_bids, dtype=int)
        clicks = np.asarray(out.clicks)
        for i, agent in enumerate(agents):
            agent.observe(wins[i].astype(bool), clicks[i], pay[i], comp[i].copy())
            last_comp[i] = comp[i].copy()

        if remaining is not None:
            remaining = remaining - pay.sum(axis=1)
            for i in range(m):
                if remaining[i] <= 0 and tr.exhausted_round[i] is None:
                    tr.exhausted_round[i] = t

    return tr
