from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

from src.env.instance import AuctionInstance


@dataclass
class AuctionOutcome:
    """Realized results of a single simulated auction round."""

    wins: List[List[int]]  # Shape (m, n): 1 if won, 0 otherwise
    clicks: List[List[int]]  # Shape (m, n): 1 if clicked, 0 otherwise
    payments: List[List[int]]  # Shape (m, n): payment made per user
    competing_bids: List[List[int]]  # Shape (m, n): c_{-i, j} for each advertiser i
    realized_profits: List[float]  # Shape (m,): profit realized in this round
    platform_revenue: float  # Total revenue collected by platform


class BatchedFirstPriceAuction:
    """
    Simulates and evaluates a Batched First-Price Ad Auction with integer bids.
    """

    def __init__(self, instance: AuctionInstance):
        self.instance = instance
        self.m = instance.num_advertisers
        self.n = instance.num_users
        self.budgets = instance.budgets
        self.f = instance.click_probabilities
        self.v = instance.click_value

    def validate_bids(self, bid_profile: List[List[int]]) -> None:
        """Validates that bids are non-negative integers satisfying budget constraints."""
        if len(bid_profile) != self.m:
            raise ValueError(f"Expected {self.m} bid vectors, got {len(bid_profile)}")
        for i, b_i in enumerate(bid_profile):
            if len(b_i) != self.n:
                raise ValueError(
                    f"Advertiser {i} bid vector length {len(b_i)} != {self.n}"
                )
            for j, bid in enumerate(b_i):
                if not isinstance(bid, int) or bid < 0:
                    raise ValueError(f"Bid for advertiser {i}, user {j} must be a non-negative integer, got {bid}")
            total_spend = sum(b_i)
            if total_spend > self.budgets[i]:
                raise ValueError(
                    f"Advertiser {i} exceeded budget: sum(b_{i})={total_spend} > {self.budgets[i]}"
                )

    def get_competing_bids(self, bid_profile: List[List[int]]) -> List[List[int]]:
        """
        Computes the highest competing bid c_{-i, j} for every advertiser i and user j.
        c_{-i, j} = max_{k != i} b_{kj}
        """
        competing: List[List[int]] = [[0] * self.n for _ in range(self.m)]
        for j in range(self.n):
            for i in range(self.m):
                other_bids = [bid_profile[k][j] for k in range(self.m) if k != i]
                competing[i][j] = max(other_bids) if other_bids else 0
        return competing

    def compute_winning_probabilities(
        self, bid_profile: List[List[int]]
    ) -> List[List[float]]:
        """
        Computes analytical winning probabilities q_{ij}(b) under uniform random tie-breaking.
        """
        q: List[List[float]] = [[0.0] * self.n for _ in range(self.m)]
        for j in range(self.n):
            bids_on_j = [bid_profile[i][j] for i in range(self.m)]
            max_bid = max(bids_on_j)
            if max_bid <= 0:
                continue

            winners = [i for i, b in enumerate(bids_on_j) if b == max_bid]
            prob = 1.0 / len(winners)
            for w in winners:
                q[w][j] = prob
        return q

    def expected_utilities(self, bid_profile: List[List[int]]) -> List[float]:
        """
        Computes expected profit for each advertiser:
        p_i(b) = sum_{j=1}^n q_{ij}(b) * (f_{ij} * v - b_{ij})
        """
        self.validate_bids(bid_profile)
        q = self.compute_winning_probabilities(bid_profile)
        utilities: List[float] = [0.0] * self.m
        for i in range(self.m):
            u_i = 0.0
            for j in range(self.n):
                if q[i][j] > 0.0:
                    u_i += q[i][j] * (self.f[i][j] * self.v - bid_profile[i][j])
            utilities[i] = u_i
        return utilities

    def platform_revenue(self, bid_profile: List[List[int]]) -> float:
        """
        Computes expected platform revenue:
        Rev(b) = sum_{i, j} q_{ij}(b) * b_{ij}
        """
        q = self.compute_winning_probabilities(bid_profile)
        revenue = 0.0
        for i in range(self.m):
            for j in range(self.n):
                revenue += q[i][j] * bid_profile[i][j]
        return revenue

    def expected_welfare(self, bid_profile: List[List[int]]) -> float:
        """
        Computes expected social welfare:
        Wel(b) = sum_{i, j} q_{ij}(b) * f_{ij} * v
        """
        q = self.compute_winning_probabilities(bid_profile)
        welfare = 0.0
        for i in range(self.m):
            for j in range(self.n):
                welfare += q[i][j] * self.f[i][j] * self.v
        return welfare

    def allocation_efficiency(self, bid_profile: List[List[int]]) -> float:
        """
        Fraction of users allocated to the true highest-value advertiser:
        Eff(b) = (1 / n) * sum_{j=1}^n sum_{i in argmax_k f_{kj}} q_{ij}(b)
        """
        q = self.compute_winning_probabilities(bid_profile)
        efficiency = 0.0
        for j in range(self.n):
            user_values = [self.f[k][j] for k in range(self.m)]
            max_val = max(user_values)
            top_advertisers = [
                k for k, val in enumerate(user_values) if abs(val - max_val) < 1e-9
            ]
            win_prob_top = sum(q[i][j] for i in top_advertisers)
            efficiency += win_prob_top
        return efficiency / self.n

    def simulate_round(
        self,
        bid_profile: List[List[int]],
        rng: Optional[random.Random] = None,
    ) -> AuctionOutcome:
        """
        Simulates an actual auction round with Bernoulli click outcomes.
        """
        self.validate_bids(bid_profile)
        if rng is None:
            rng = random.Random()

        competing = self.get_competing_bids(bid_profile)
        wins = [[0] * self.n for _ in range(self.m)]
        clicks = [[0] * self.n for _ in range(self.m)]
        payments = [[0] * self.n for _ in range(self.m)]
        profits = [0.0] * self.m
        tot_rev = 0.0

        for j in range(self.n):
            bids_on_j = [bid_profile[i][j] for i in range(self.m)]
            max_bid = max(bids_on_j)
            if max_bid <= 0:
                continue

            max_bidders = [i for i, b in enumerate(bids_on_j) if b == max_bid]
            winner = rng.choice(max_bidders)

            wins[winner][j] = 1
            payments[winner][j] = max_bid
            tot_rev += max_bid

            # Click realization
            click = 1 if rng.random() < self.f[winner][j] else 0
            clicks[winner][j] = click
            profits[winner] += click * self.v - max_bid

        return AuctionOutcome(
            wins=wins,
            clicks=clicks,
            payments=payments,
            competing_bids=competing,
            realized_profits=profits,
            platform_revenue=tot_rev,
        )
