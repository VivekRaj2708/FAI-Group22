from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import json
import numpy as np

from src.agents.base import BaseAdvertiserAgent
from src.env.instance import AuctionInstance
from src.env.auction import BatchedFirstPriceAuction
from src.tier1.knapsack import compute_best_response_knapsack
from src.tier1.social_planner import compute_social_planner_benchmark, SocialPlannerOutcome
from src.tier1.equilibrium import (
    EquilibriumResult,
    solve_best_response_dynamics,
    solve_support_enumeration_small,
    solve_fictitious_play,
    generate_pure_strategies,
)


@dataclass
class Tier1Benchmark:
    """Complete Tier-1 benchmark reference data consumed by Tier 2 and Tier 3."""

    instance_name: str
    num_advertisers: int
    num_users: int
    budgets: List[int]
    reference_bid_profile: List[List[int]]  # b^*
    advertiser_profits: List[float]  # p_i^*
    platform_revenue: float  # Rev(b^*)
    expected_welfare: float  # Wel(b^*)
    allocation_efficiency: float  # Eff(b^*)
    equilibrium_gap: float  # epsilon(b^*)
    is_pure_equilibrium: bool
    solver_method: str

    # Social Planner comparison
    social_planner_welfare: float
    social_planner_assignment: List[int]
    welfare_loss: float
    efficiency_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "Tier1Benchmark":
        with open(file_path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class Tier1Solver:
    """
    Tier-1 Baseline Solver:
    Computes exact best responses, equilibria, and theoretical benchmarks
    for batched first-price ad auctions with known click probabilities.
    """

    def __init__(self, instance: AuctionInstance):
        self.instance = instance
        self.auction = BatchedFirstPriceAuction(instance)

    def solve(
        self,
        prefer_exact: bool = True,
        max_brd_iter: int = 100,
        seed: int = 42,
    ) -> Tier1Benchmark:
        """
        Solves the Tier-1 baseline:
        - Runs social planner benchmark.
        - Computes reference equilibrium profile b* via Support Enumeration (for small instances)
          or Best-Response Dynamics / Fictitious Play.
        """
        # 1. Social planner benchmark
        planner = compute_social_planner_benchmark(self.instance)

        # 2. Decide solver method
        m = self.instance.num_advertisers
        n = self.instance.num_users

        # Check total joint strategy size for exact support enumeration
        can_exact = False
        if prefer_exact and m <= 3 and n <= 5:
            joint_size = 1
            for b in self.instance.budgets:
                s_size = len(generate_pure_strategies(n, b))
                joint_size *= s_size
                if joint_size > 50000:
                    break
            can_exact = (joint_size <= 50000)

        if can_exact:
            eq_res = solve_support_enumeration_small(self.auction)
        else:
            eq_res = solve_best_response_dynamics(
                self.auction, max_iter=max_brd_iter
            )
            # If BRD did not find a pure equilibrium and size is small, try Fictitious Play
            if not eq_res.is_pure_equilibrium and m <= 3 and n <= 6:
                fp_res = solve_fictitious_play(self.auction, num_rounds=150, seed=seed)
                if fp_res.equilibrium_gap < eq_res.equilibrium_gap:
                    eq_res = fp_res

        b_star = eq_res.bid_profile
        profits = eq_res.advertiser_profits
        rev = eq_res.platform_revenue
        eff = eq_res.allocation_efficiency
        welfare = self.auction.expected_welfare(b_star)

        welfare_loss = max(0.0, planner.max_social_welfare - welfare)
        eff_loss = max(0.0, 1.0 - eff)

        return Tier1Benchmark(
            instance_name=self.instance.name or f"inst_{m}x{n}",
            num_advertisers=m,
            num_users=n,
            budgets=self.instance.budgets,
            reference_bid_profile=b_star,
            advertiser_profits=profits,
            platform_revenue=rev,
            expected_welfare=welfare,
            allocation_efficiency=eff,
            equilibrium_gap=eq_res.equilibrium_gap,
            is_pure_equilibrium=eq_res.is_pure_equilibrium,
            solver_method=eq_res.method,
            social_planner_welfare=planner.max_social_welfare,
            social_planner_assignment=planner.optimal_assignment,
            welfare_loss=welfare_loss,
            efficiency_loss=eff_loss,
        )


class Tier1EquilibriumAgent(BaseAdvertiserAgent):
    """
    Tier-1 Baseline Agent that persistently plays a fixed equilibrium strategy b_i*.
    """

    def __init__(
        self,
        agent_id: int,
        budget: int,
        num_users: int,
        equilibrium_bid: List[int],
        click_value: float = 1.0,
        true_f: Optional[List[float]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            name=name or f"Tier1Equilibrium_Agent_{agent_id}",
        )
        self.equilibrium_bid = [int(b) for b in equilibrium_bid]
        self.true_f = list(true_f) if true_f is not None else [0.0] * num_users
        self.pulls = [0] * num_users
        self.clicks = [0] * num_users
        self.last_competing_bids: Optional[List[int]] = None

    def reset(self) -> None:
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids = None

    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        if last_competing_bids is not None:
            self.last_competing_bids = [int(b) for b in last_competing_bids]
        return list(self.equilibrium_bid)

    def observe(
        self,
        won: Union[List[int], np.ndarray],
        clicks: Union[List[int], np.ndarray],
        payments: Union[List[int], np.ndarray],
        competing_bids: Union[List[int], np.ndarray],
    ) -> None:
        self.last_competing_bids = [int(b) for b in competing_bids]
        for j in range(self.num_users):
            if won[j]:
                self.pulls[j] += 1
                self.clicks[j] += int(clicks[j])

    def get_estimated_click_probabilities(self) -> List[float]:
        return list(self.true_f)


class Tier1KnapsackAgent(BaseAdvertiserAgent):
    """
    Tier-1 Best-Response Agent with perfect knowledge of true click probabilities f_ij.
    Dynamically computes best response via 0-1 knapsack DP against observed competing bids.
    """

    def __init__(
        self,
        agent_id: int,
        budget: int,
        num_users: int,
        true_f: List[float],
        click_value: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            name=name or f"Tier1Knapsack_Agent_{agent_id}",
        )
        self.true_f = list(true_f)
        self.pulls = [0] * num_users
        self.clicks = [0] * num_users
        self.last_competing_bids: Optional[List[int]] = None

    def reset(self) -> None:
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids = None

    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        if last_competing_bids is not None:
            self.last_competing_bids = [int(b) for b in last_competing_bids]

        if t <= 1 or self.last_competing_bids is None:
            return [0] * self.num_users

        knap_res = compute_best_response_knapsack(
            f_i=self.true_f,
            competing_bids=self.last_competing_bids,
            budget=self.budget,
            click_value=self.click_value,
        )
        return knap_res.bids

    def observe(
        self,
        won: Union[List[int], np.ndarray],
        clicks: Union[List[int], np.ndarray],
        payments: Union[List[int], np.ndarray],
        competing_bids: Union[List[int], np.ndarray],
    ) -> None:
        self.last_competing_bids = [int(b) for b in competing_bids]
        for j in range(self.num_users):
            if won[j]:
                self.pulls[j] += 1
                self.clicks[j] += int(clicks[j])

    def get_estimated_click_probabilities(self) -> List[float]:
        return list(self.true_f)
