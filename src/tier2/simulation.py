from dataclasses import dataclass, asdict
import json
import math
import random
from typing import List, Dict, Any, Optional, Union
import numpy as np

from src.env.instance import AuctionInstance
from src.env.auction import BatchedFirstPriceAuction, AuctionOutcome
from src.tier1.baseline import Tier1Solver
from src.tier1.knapsack import compute_best_response_knapsack
from src.agents.base import BaseAdvertiserAgent


@dataclass
class Tier2RoundMetrics:
    """Metrics recorded for a single round of Tier-2 simulation."""

    round_idx: int
    learner_bids: List[int]
    competing_bids: List[int]
    wins: List[int]
    clicks: List[int]
    payments: List[int]
    spend: int
    optimal_profit: float
    expected_profit: float
    realized_profit: float
    instantaneous_regret: float
    cumulative_regret: float
    platform_revenue: float
    allocation_efficiency: float
    expected_welfare: float


@dataclass
class Tier2TrialResult:
    """Complete results for one independent simulation run of Tier 2."""

    agent_name: str
    seed: int
    total_rounds: int
    round_metrics: List[Tier2RoundMetrics]
    cumulative_regrets: List[float]
    instantaneous_regrets: List[float]
    expected_profits: List[float]
    realized_profits: List[float]
    platform_revenues: List[float]
    allocation_efficiencies: List[float]
    total_cumulative_regret: float
    mean_expected_profit: float
    mean_efficiency: float
    mean_platform_revenue: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Tier2ExperimentResult:
    """Aggregated results across multiple independent trials (seeds) with 95% CIs."""

    agent_name: str
    instance_name: str
    num_trials: int
    total_rounds: int
    round_structure: str
    mean_cumulative_regret: List[float]
    std_cumulative_regret: List[float]
    ci_95_cumulative_regret: List[float]
    mean_instantaneous_regret: List[float]
    mean_expected_profit: List[float]
    mean_efficiency: List[float]
    mean_platform_revenue: List[float]
    final_regret_mean: float
    final_regret_ci95: float
    average_round_profit: float
    average_efficiency: float
    average_platform_revenue: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "Tier2ExperimentResult":
        with open(file_path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class Tier2Simulator:
    """
    Tier-2 Simulation Engine:
    Single-agent combinatorial semi-bandit learner competing against fixed
    opponents playing pure Tier-1 equilibrium bids.
    """

    def __init__(
        self,
        instance: AuctionInstance,
        learner_agent: BaseAdvertiserAgent,
        learner_id: int = 0,
        fixed_opponent_bids: Optional[List[List[int]]] = None,
        round_structure: str = "canonical",  # 'canonical', 'discounted', 'cumulative'
        discount_factor: float = 0.99,
        total_cumulative_budget: Optional[int] = None,
    ):
        self.instance = instance
        self.learner = learner_agent
        self.learner_id = learner_id
        self.m = instance.num_advertisers
        self.n = instance.num_users
        self.auction = BatchedFirstPriceAuction(instance)
        self.round_structure = round_structure
        self.discount_factor = discount_factor
        self.total_cumulative_budget = total_cumulative_budget

        # 1. Determine fixed opponent bids
        if fixed_opponent_bids is not None:
            self.reference_bid_profile = [list(b) for b in fixed_opponent_bids]
        else:
            solver = Tier1Solver(instance)
            benchmark = solver.solve()
            self.reference_bid_profile = [list(b) for b in benchmark.reference_bid_profile]

        # Extract fixed bids for opponents
        self.opponent_bids = [
            list(self.reference_bid_profile[i]) for i in range(self.m)
        ]

        # Learner faces constant competing bids c_{-i, j}^*
        self.competing_bids_learner = [0] * self.n
        for j in range(self.n):
            other_bids = [
                self.opponent_bids[k][j]
                for k in range(self.m)
                if k != self.learner_id
            ]
            self.competing_bids_learner[j] = max(other_bids) if other_bids else 0

        # Compute full-information optimal best-response profit p_i*(c_{-i})
        learner_budget = instance.budgets[self.learner_id]
        learner_f = instance.click_probabilities[self.learner_id]
        knap_opt = compute_best_response_knapsack(
            learner_f,
            self.competing_bids_learner,
            learner_budget,
            click_value=instance.click_value,
        )
        self.p_i_star = knap_opt.expected_profit
        self.opt_bids = knap_opt.bids

    def run_trial(self, total_rounds: int = 1000, seed: int = 42) -> Tier2TrialResult:
        """Runs a single simulation trial over T rounds."""
        rng = random.Random(seed)
        self.learner.reset()

        round_metrics: List[Tier2RoundMetrics] = []
        cumulative_regrets: List[float] = []
        instantaneous_regrets: List[float] = []
        expected_profits: List[float] = []
        realized_profits: List[float] = []
        platform_revenues: List[float] = []
        allocation_efficiencies: List[float] = []

        cum_regret = 0.0
        last_competing: Optional[List[int]] = None

        for t in range(1, total_rounds + 1):
            # 1. Learner chooses bids
            learner_bids = self.learner.get_bid(t, last_competing)

            # Build full bid profile
            full_bid_profile = [list(self.opponent_bids[i]) for i in range(self.m)]
            full_bid_profile[self.learner_id] = list(learner_bids)

            # 2. Simulate round
            outcome: AuctionOutcome = self.auction.simulate_round(full_bid_profile, rng=rng)

            # 3. Calculate analytical expected profit and efficiency under true parameters
            expected_utils = self.auction.expected_utilities(full_bid_profile)
            learner_exp_profit = expected_utils[self.learner_id]
            round_revenue = self.auction.platform_revenue(full_bid_profile)
            round_efficiency = self.auction.allocation_efficiency(full_bid_profile)
            round_welfare = self.auction.expected_welfare(full_bid_profile)

            # 4. Regret calculation
            inst_regret = max(0.0, self.p_i_star - learner_exp_profit)

            if self.round_structure == "discounted":
                discount = self.discount_factor ** (t - 1)
                effective_regret = discount * inst_regret
            else:
                effective_regret = inst_regret

            cum_regret += effective_regret

            # 5. Record round metrics
            learner_spend = sum(learner_bids)
            last_competing = outcome.competing_bids[self.learner_id]

            metric = Tier2RoundMetrics(
                round_idx=t,
                learner_bids=learner_bids,
                competing_bids=last_competing,
                wins=outcome.wins[self.learner_id],
                clicks=outcome.clicks[self.learner_id],
                payments=outcome.payments[self.learner_id],
                spend=learner_spend,
                optimal_profit=self.p_i_star,
                expected_profit=learner_exp_profit,
                realized_profit=outcome.realized_profits[self.learner_id],
                instantaneous_regret=effective_regret,
                cumulative_regret=cum_regret,
                platform_revenue=round_revenue,
                allocation_efficiency=round_efficiency,
                expected_welfare=round_welfare,
            )
            round_metrics.append(metric)

            cumulative_regrets.append(cum_regret)
            instantaneous_regrets.append(effective_regret)
            expected_profits.append(learner_exp_profit)
            realized_profits.append(outcome.realized_profits[self.learner_id])
            platform_revenues.append(round_revenue)
            allocation_efficiencies.append(round_efficiency)

            # 6. Learner observes feedback
            self.learner.observe(
                won=outcome.wins[self.learner_id],
                clicks=outcome.clicks[self.learner_id],
                payments=outcome.payments[self.learner_id],
                competing_bids=last_competing,
            )

        mean_profit = float(np.mean(expected_profits))
        mean_eff = float(np.mean(allocation_efficiencies))
        mean_rev = float(np.mean(platform_revenues))

        return Tier2TrialResult(
            agent_name=self.learner.name,
            seed=seed,
            total_rounds=total_rounds,
            round_metrics=round_metrics,
            cumulative_regrets=cumulative_regrets,
            instantaneous_regrets=instantaneous_regrets,
            expected_profits=expected_profits,
            realized_profits=realized_profits,
            platform_revenues=platform_revenues,
            allocation_efficiencies=allocation_efficiencies,
            total_cumulative_regret=cum_regret,
            mean_expected_profit=mean_profit,
            mean_efficiency=mean_eff,
            mean_platform_revenue=mean_rev,
        )

    def run_experiment(
        self,
        total_rounds: int = 1000,
        seeds: Optional[List[int]] = None,
        num_seeds: int = 10,
    ) -> Tier2ExperimentResult:
        """
        Runs multiple independent simulation trials and computes means & 95% confidence intervals.
        """
        if seeds is None:
            seeds = [100 + i for i in range(num_seeds)]

        all_cum_regrets = []
        all_inst_regrets = []
        all_profits = []
        all_efficiencies = []
        all_revenues = []

        for s in seeds:
            res = self.run_trial(total_rounds=total_rounds, seed=s)
            all_cum_regrets.append(res.cumulative_regrets)
            all_inst_regrets.append(res.instantaneous_regrets)
            all_profits.append(res.expected_profits)
            all_efficiencies.append(res.allocation_efficiencies)
            all_revenues.append(res.platform_revenues)

        regrets_mat = np.array(all_cum_regrets)  # (S, T)
        inst_mat = np.array(all_inst_regrets)
        profits_mat = np.array(all_profits)
        eff_mat = np.array(all_efficiencies)
        rev_mat = np.array(all_revenues)

        S = len(seeds)
        mean_cum_regret = np.mean(regrets_mat, axis=0).tolist()
        std_cum_regret = np.std(regrets_mat, axis=0, ddof=1 if S > 1 else 0).tolist()
        ci_95 = (1.96 * np.array(std_cum_regret) / math.sqrt(S)).tolist()

        final_regrets = regrets_mat[:, -1]
        final_mean = float(np.mean(final_regrets))
        final_ci = float(1.96 * np.std(final_regrets, ddof=1 if S > 1 else 0) / math.sqrt(S))

        return Tier2ExperimentResult(
            agent_name=self.learner.name,
            instance_name=self.instance.name or f"inst_{self.m}x{self.n}",
            num_trials=S,
            total_rounds=total_rounds,
            round_structure=self.round_structure,
            mean_cumulative_regret=mean_cum_regret,
            std_cumulative_regret=std_cum_regret,
            ci_95_cumulative_regret=ci_95,
            mean_instantaneous_regret=np.mean(inst_mat, axis=0).tolist(),
            mean_expected_profit=np.mean(profits_mat, axis=0).tolist(),
            mean_efficiency=np.mean(eff_mat, axis=0).tolist(),
            mean_platform_revenue=np.mean(rev_mat, axis=0).tolist(),
            final_regret_mean=final_mean,
            final_regret_ci95=final_ci,
            average_round_profit=float(np.mean(profits_mat)),
            average_efficiency=float(np.mean(eff_mat)),
            average_platform_revenue=float(np.mean(rev_mat)),
        )
