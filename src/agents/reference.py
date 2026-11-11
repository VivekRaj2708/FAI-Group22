"""
Reference agents used by the evaluation harness.

* ``fixed_bstar``  - plays its Tier-1 equilibrium bids b*_i every round
                     (the fixed opponents of Tier 2, and the Tier-1 scenario).
* ``oracle_br``    - knows f(., a_i) and best-responds (knapsack DP) to the last
                     observed competing bids: full-information repeated best
                     response, i.e. the Tier-3 dynamics with learning removed.
* ``random``       - uniformly random feasible strict-win subset; a floor
                     against which learners should clearly improve.

The bandit learners themselves (UCB, Thompson, greedy, paced) live in Tier 2.
"""
from typing import List, Optional

import numpy as np

from src.agents.base import BaseAdvertiserAgent
from src.agents.registry import AgentContext, register_agent
from src.tier1.knapsack import compute_best_response_knapsack


class FixedProfileAgent(BaseAdvertiserAgent):
    def __init__(self, agent_id: int, budget: int, num_users: int, bids: List[int],
                 click_value: float = 1.0):
        super().__init__(agent_id, budget, num_users, click_value)
        if len(bids) != num_users or sum(bids) > budget or min(bids, default=0) < 0:
            raise ValueError(f"Infeasible fixed bid vector {bids} for budget {budget}")
        self.bids = np.asarray(bids, dtype=int)

    def get_bid(self, t, last_competing_bids=None):
        return self.bids.copy()

    def observe(self, won, clicks, payments, competing_bids):
        pass

    def reset(self):
        pass


class OracleBestResponseAgent(BaseAdvertiserAgent):
    """Known-f knapsack best response to c_{-i}^{t-1} (zeros in round 1)."""

    def __init__(self, agent_id: int, budget: int, num_users: int, true_f: List[float],
                 click_value: float = 1.0, total_budget: Optional[int] = None):
        super().__init__(agent_id, budget, num_users, click_value)
        self.true_f = list(true_f)
        self.total_budget = total_budget
        self.reset()

    def reset(self):
        self.spent = 0

    def get_bid(self, t, last_competing_bids=None):
        c = [0] * self.num_users if last_competing_bids is None else [int(x) for x in last_competing_bids]
        cap = self.budget
        if self.total_budget is not None:
            cap = min(cap, self.total_budget - self.spent)
        if cap <= 0:
            return np.zeros(self.num_users, dtype=int)
        res = compute_best_response_knapsack(self.true_f, c, cap, self.click_value)
        return np.asarray(res.bids, dtype=int)

    def observe(self, won, clicks, payments, competing_bids):
        self.spent += int(np.sum(payments))


class RandomFeasibleAgent(BaseAdvertiserAgent):
    """Strict-win bids on a random subset of users that fits the budget."""

    def __init__(self, agent_id: int, budget: int, num_users: int, click_value: float = 1.0,
                 seed: int = 0):
        super().__init__(agent_id, budget, num_users, click_value)
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def get_bid(self, t, last_competing_bids=None):
        c = np.zeros(self.num_users, dtype=int) if last_competing_bids is None else np.asarray(last_competing_bids, dtype=int)
        bids = np.zeros(self.num_users, dtype=int)
        remaining = self.budget
        for j in self.rng.permutation(self.num_users):
            if self.rng.random() < 0.5 and c[j] + 1 <= remaining:
                bids[j] = c[j] + 1
                remaining -= bids[j]
        return bids

    def observe(self, won, clicks, payments, competing_bids):
        pass


@register_agent("fixed_bstar")
def _make_fixed(ctx: AgentContext, **_):
    if ctx.reference_bids is None:
        raise ValueError("fixed_bstar requires the Tier-1 reference profile")
    return FixedProfileAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.reference_bids, ctx.click_value)


@register_agent("oracle_br")
def _make_oracle(ctx: AgentContext, **_):
    if ctx.true_click_probs is None:
        raise ValueError("oracle_br requires the true click probabilities")
    return OracleBestResponseAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.true_click_probs,
                                   ctx.click_value, total_budget=ctx.total_budget)


@register_agent("random")
def _make_random(ctx: AgentContext, **_):
    return RandomFeasibleAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value, seed=ctx.seed)
