from dataclasses import dataclass, asdict
import json
import math
import random
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from src.env.instance import AuctionInstance
from src.env.auction import BatchedFirstPriceAuction, AuctionOutcome
from src.agents.base import BaseAdvertiserAgent
from src.tier1.baseline import Tier1Solver, Tier1Benchmark, Tier1EquilibriumAgent, Tier1KnapsackAgent
from src.tier1.equilibrium import compute_equilibrium_gap
from src.tier2.agents import (
    UCBAgent,
    ThompsonSamplingAgent,
    GreedyAgent,
    EpsGreedyAgent,
    CumulativeBudgetPacedAgent,
)
from src.tier3.convergence import (
    compute_equilibrium_distance,
    compute_normalized_distance,
    compute_convergence_rate,
    detect_cycling_or_stability,
    compute_time_to_convergence,
    compute_price_of_learning,
)


@dataclass
class Tier3RoundMetrics:
    """Metrics recorded for a single round of multi-agent concurrent learning."""

    round_idx: int
    joint_bids: List[List[int]]  # Shape (m, n)
    competing_bids: List[List[int]]  # Shape (m, n)
    wins: List[List[int]]  # Shape (m, n)
    clicks: List[List[int]]  # Shape (m, n)
    payments: List[List[int]]  # Shape (m, n)
    spends: List[int]  # Length m
    expected_profits: List[float]  # Length m
    realized_profits: List[float]  # Length m
    equilibrium_gap: float  # epsilon_t = max_i (BR_i - u_i)
    per_agent_gaps: List[float]  # Length m
    equilibrium_distance: float  # ||b^t - b*||_1
    per_agent_instantaneous_regrets: List[float]  # Length m
    per_agent_cumulative_regrets: List[float]  # Length m
    platform_revenue: float
    allocation_efficiency: float
    expected_welfare: float
    price_of_learning_profit: float
    price_of_learning_efficiency: float
    price_of_learning_revenue: float


@dataclass
class Tier3TrialResult:
    """Complete results for one independent multi-agent simulation run across T rounds."""

    trial_seed: int
    population_type: str
    agent_names: List[str]
    agent_types: List[str]
    total_rounds: int
    round_metrics: List[Tier3RoundMetrics]
    equilibrium_gaps: List[float]
    equilibrium_distances: List[float]
    per_agent_cumulative_regrets: List[List[float]]  # Shape (m, T)
    per_agent_instantaneous_regrets: List[List[float]]  # Shape (m, T)
    per_agent_expected_profits: List[List[float]]  # Shape (m, T)
    per_agent_realized_profits: List[List[float]]  # Shape (m, T)
    platform_revenues: List[float]  # Length T
    allocation_efficiencies: List[float]  # Length T
    expected_welfares: List[float]  # Length T

    # Summary scalars
    final_equilibrium_gap: float
    mean_equilibrium_gap: float
    final_equilibrium_distance: float
    mean_equilibrium_distance: float
    final_cumulative_regrets: List[float]  # Length m
    mean_expected_profits: List[float]  # Length m
    mean_platform_revenue: float
    mean_efficiency: float
    mean_welfare: float
    is_converged_to_pure_equilibrium: bool
    convergence_round: Optional[int]
    stability_analysis: Dict[str, Any]
    convergence_rate_fit: Dict[str, float]
    price_of_learning: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Tier3ExperimentResult:
    """Aggregated results across multiple independent trials (seeds) with 95% CIs."""

    instance_name: str
    population_type: str
    num_trials: int
    total_rounds: int
    round_structure: str
    agent_names: List[str]
    agent_types: List[str]

    # Time series (mean, std, ci95)
    mean_equilibrium_gap: List[float]
    ci95_equilibrium_gap: List[float]
    mean_equilibrium_distance: List[float]
    ci95_equilibrium_distance: List[float]
    mean_platform_revenue: List[float]
    ci95_platform_revenue: List[float]
    mean_allocation_efficiency: List[float]
    ci95_allocation_efficiency: List[float]
    mean_expected_welfare: List[float]
    ci95_expected_welfare: List[float]

    # Per-agent time series: Shape (m, T)
    mean_per_agent_cumulative_regret: List[List[float]]
    ci95_per_agent_cumulative_regret: List[List[float]]
    mean_per_agent_expected_profit: List[List[float]]
    ci95_per_agent_expected_profit: List[List[float]]

    # Final scalar benchmarks
    final_gap_mean: float
    final_gap_ci95: float
    final_distance_mean: float
    final_distance_ci95: float
    final_regrets_mean: List[float]
    final_regrets_ci95: List[float]
    average_efficiency_mean: float
    average_efficiency_ci95: float
    average_revenue_mean: float
    average_revenue_ci95: float
    average_profit_mean: List[float]
    average_profit_ci95: List[float]
    price_of_learning: Dict[str, Any]
    convergence_rate_summary: Dict[str, float]
    stability_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "Tier3ExperimentResult":
        with open(file_path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


def create_multi_agent_population(
    population_type: str,
    instance: AuctionInstance,
    reference_benchmark: Optional[Tier1Benchmark] = None,
    alpha: float = 2.0,
    epsilon: float = 0.1,
    total_rounds: int = 1000,
    seed: int = 42,
) -> List[BaseAdvertiserAgent]:
    """
    Factory function creating homogeneous or heterogeneous populations of bidding agents.
    """
    m = instance.num_advertisers
    n = instance.num_users
    budgets = instance.budgets
    v = instance.click_value
    agents: List[BaseAdvertiserAgent] = []

    pop_type = population_type.lower()

    if pop_type in ("all_ucb", "ucb"):
        for i in range(m):
            agent = UCBAgent(
                agent_id=i,
                budget=budgets[i],
                num_users=n,
                click_value=v,
                alpha=alpha,
                name=f"UCB_{i} (alpha={alpha})",
            )
            agents.append(agent)

    elif pop_type in ("all_ts", "ts", "thompson"):
        for i in range(m):
            agent_seed = seed * 1000 + i
            agent = ThompsonSamplingAgent(
                agent_id=i,
                budget=budgets[i],
                num_users=n,
                click_value=v,
                seed=agent_seed,
                name=f"ThompsonSampling_{i}",
            )
            agents.append(agent)

    elif pop_type in ("mixed_ucb_ts", "ucb_ts"):
        for i in range(m):
            if i % 2 == 0:
                agent = UCBAgent(
                    agent_id=i,
                    budget=budgets[i],
                    num_users=n,
                    click_value=v,
                    alpha=alpha,
                    name=f"UCB_{i}",
                )
            else:
                agent_seed = seed * 1000 + i
                agent = ThompsonSamplingAgent(
                    agent_id=i,
                    budget=budgets[i],
                    num_users=n,
                    click_value=v,
                    seed=agent_seed,
                    name=f"ThompsonSampling_{i}",
                )
            agents.append(agent)

    elif pop_type in ("mixed_all", "heterogeneous"):
        # Cycle through UCB, TS, Greedy, EpsGreedy
        for i in range(m):
            agent_seed = seed * 1000 + i
            archetype = i % 4
            if archetype == 0:
                agent = UCBAgent(
                    agent_id=i,
                    budget=budgets[i],
                    num_users=n,
                    click_value=v,
                    alpha=alpha,
                    name=f"UCB_{i}",
                )
            elif archetype == 1:
                agent = ThompsonSamplingAgent(
                    agent_id=i,
                    budget=budgets[i],
                    num_users=n,
                    click_value=v,
                    seed=agent_seed,
                    name=f"ThompsonSampling_{i}",
                )
            elif archetype == 2:
                agent = GreedyAgent(
                    agent_id=i,
                    budget=budgets[i],
                    num_users=n,
                    click_value=v,
                    name=f"Greedy_{i}",
                )
            else:
                agent = EpsGreedyAgent(
                    agent_id=i,
                    budget=budgets[i],
                    num_users=n,
                    click_value=v,
                    epsilon=epsilon,
                    seed=agent_seed,
                    name=f"EpsGreedy_{i}",
                )
            agents.append(agent)

    elif pop_type in ("learner_vs_equilibrium", "vs_eq"):
        if reference_benchmark is None:
            solver = Tier1Solver(instance)
            reference_benchmark = solver.solve()

        # Agent 0 learns via UCB; opponents play fixed equilibrium bids b*
        agents.append(
            UCBAgent(
                agent_id=0,
                budget=budgets[0],
                num_users=n,
                click_value=v,
                alpha=alpha,
                name=f"UCB_Learner_0",
            )
        )
        for i in range(1, m):
            b_star_i = reference_benchmark.reference_bid_profile[i]
            agents.append(
                Tier1EquilibriumAgent(
                    agent_id=i,
                    budget=budgets[i],
                    num_users=n,
                    equilibrium_bid=b_star_i,
                    click_value=v,
                    true_f=instance.click_probabilities[i],
                    name=f"EquilibriumOpponent_{i}",
                )
            )

    else:
        raise ValueError(f"Unknown population_type: {population_type}")

    return agents


class MultiAgentSimulator:
    """
    Tier-3 Multi-Agent Simulation Engine:
    Simulates repeated batched first-price ad auctions with all m advertisers
    learning and bidding concurrently in a non-stationary strategic environment.
    """

    def __init__(
        self,
        instance: AuctionInstance,
        agents: Optional[List[BaseAdvertiserAgent]] = None,
        population_type: str = "all_ucb",
        reference_benchmark: Optional[Tier1Benchmark] = None,
        round_structure: str = "canonical",  # 'canonical', 'discounted', 'cumulative'
        discount_factor: float = 0.99,
        alpha: float = 2.0,
        epsilon: float = 0.1,
        convergence_tol: float = 1e-4,
    ):
        self.instance = instance
        self.m = instance.num_advertisers
        self.n = instance.num_users
        self.auction = BatchedFirstPriceAuction(instance)
        self.round_structure = round_structure
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.convergence_tol = convergence_tol

        # Compute or store reference Tier-1 benchmark
        if reference_benchmark is None:
            solver = Tier1Solver(instance)
            self.benchmark = solver.solve()
        else:
            self.benchmark = reference_benchmark

        self.b_star = self.benchmark.reference_bid_profile
        self.p_star = self.benchmark.advertiser_profits

        self.population_type = population_type
        if agents is not None:
            if len(agents) != self.m:
                raise ValueError(f"Expected {self.m} agents, got {len(agents)}")
            self.agents = agents
        else:
            self.agents = create_multi_agent_population(
                population_type=self.population_type,
                instance=self.instance,
                reference_benchmark=self.benchmark,
                alpha=self.alpha,
                epsilon=self.epsilon,
            )

    def _reset_agents(self, trial_seed: int, total_rounds: int) -> None:
        """Resets all agent states and updates per-trial seeds."""
        for i, agent in enumerate(self.agents):
            agent.reset()
            # If agent has a seed, ensure distinct per-agent RNG
            if hasattr(agent, "seed") and agent.seed is not None:
                agent.seed = trial_seed * 1000 + i
                agent.rng = random.Random(agent.seed)

    def run_round(
        self,
        t: int,
        rng: random.Random,
        cumulative_regrets: List[float],
        remaining_cumulative_budgets: Optional[List[int]] = None,
    ) -> Tier3RoundMetrics:
        """Executes a single multi-agent auction round and computes metrics."""
        # 1. Query bid vectors from all m agents
        joint_bids: List[List[int]] = []
        for i, agent in enumerate(self.agents):
            b_i = agent.get_bid(t=t)
            # Enforce non-negativity and integer constraint
            b_i = [max(0, int(round(b))) for b in b_i]

            # In cumulative budget mode, cap bids by remaining cumulative budget
            if self.round_structure == "cumulative" and remaining_cumulative_budgets is not None:
                rem_b = remaining_cumulative_budgets[i]
                if sum(b_i) > rem_b:
                    scaled_b = [0] * self.n
                    current_spend = 0
                    for j in range(self.n):
                        if current_spend + b_i[j] <= rem_b:
                            scaled_b[j] = b_i[j]
                            current_spend += b_i[j]
                    b_i = scaled_b

            # Enforce per-round budget constraint sum(b_i) <= B_i
            tot_spend = sum(b_i)
            if tot_spend > self.instance.budgets[i]:
                # Proportionally scale down if agent violated budget
                excess = tot_spend - self.instance.budgets[i]
                for j in reversed(range(self.n)):
                    cut = min(b_i[j], excess)
                    b_i[j] -= cut
                    excess -= cut
                    if excess <= 0:
                        break

            joint_bids.append(b_i)

        # 2. Simulate auction resolution and click realization
        outcome = self.auction.simulate_round(joint_bids, rng=rng)

        # Deduct spends from cumulative budget if applicable
        if remaining_cumulative_budgets is not None:
            for i in range(self.m):
                spend_i = sum(outcome.payments[i])
                remaining_cumulative_budgets[i] = max(0, remaining_cumulative_budgets[i] - spend_i)

        # 3. Deliver semi-bandit and competing bid feedback to all agents
        for i, agent in enumerate(self.agents):
            agent.observe(
                won=outcome.wins[i],
                clicks=outcome.clicks[i],
                payments=outcome.payments[i],
                competing_bids=outcome.competing_bids[i],
            )

        # 4. Compute analytical game-theoretic metrics
        expected_utilities = self.auction.expected_utilities(joint_bids)
        eq_gap, per_agent_gaps, br_profiles = compute_equilibrium_gap(self.auction, joint_bids)
        eq_dist = compute_equilibrium_distance(joint_bids, self.b_star)

        rev = self.auction.platform_revenue(joint_bids)
        eff = self.auction.allocation_efficiency(joint_bids)
        welfare = self.auction.expected_welfare(joint_bids)

        # Instantaneous and cumulative regrets
        # Against opponents' bids b_{-i}^t, best response achieves max_{b_i'} p_i(b_i', b_{-i}^t)
        # Instantaneous regret is exactly the agent's gap: BR_i(b_{-i}^t) - u_i(b^t)
        instantaneous_regrets: List[float] = []
        discount = (self.discount_factor ** (t - 1)) if self.round_structure == "discounted" else 1.0

        for i in range(self.m):
            inst_reg = per_agent_gaps[i] * discount
            instantaneous_regrets.append(inst_reg)
            cumulative_regrets[i] += inst_reg

        # Price of learning relative to Tier 1 reference profile b*
        pol_profit = float(sum(self.p_star) - sum(expected_utilities))
        pol_eff = float(self.benchmark.allocation_efficiency - eff)
        pol_rev = float(self.benchmark.platform_revenue - rev)

        spends = [sum(b_i) for b_i in joint_bids]

        return Tier3RoundMetrics(
            round_idx=t,
            joint_bids=joint_bids,
            competing_bids=outcome.competing_bids,
            wins=outcome.wins,
            clicks=outcome.clicks,
            payments=outcome.payments,
            spends=spends,
            expected_profits=expected_utilities,
            realized_profits=outcome.realized_profits,
            equilibrium_gap=eq_gap,
            per_agent_gaps=per_agent_gaps,
            equilibrium_distance=eq_dist,
            per_agent_instantaneous_regrets=instantaneous_regrets,
            per_agent_cumulative_regrets=list(cumulative_regrets),
            platform_revenue=rev,
            allocation_efficiency=eff,
            expected_welfare=welfare,
            price_of_learning_profit=pol_profit,
            price_of_learning_efficiency=pol_eff,
            price_of_learning_revenue=pol_rev,
        )

    def run_trial(self, seed: int = 42, total_rounds: int = 500) -> Tier3TrialResult:
        """Runs a complete simulation trial across total_rounds with a fixed seed."""
        rng = random.Random(seed)
        self._reset_agents(trial_seed=seed, total_rounds=total_rounds)

        agent_names = [a.name for a in self.agents]
        agent_types = [a.__class__.__name__ for a in self.agents]

        cumulative_regrets = [0.0] * self.m
        remaining_cum_budgets = None
        if self.round_structure == "cumulative":
            remaining_cum_budgets = [
                self.instance.budgets[i] * total_rounds // 2 for i in range(self.m)
            ]

        round_metrics: List[Tier3RoundMetrics] = []
        eq_gaps: List[float] = []
        eq_distances: List[float] = []
        platform_revs: List[float] = []
        efficiencies: List[float] = []
        welfares: List[float] = []

        per_agent_cum_reg: List[List[float]] = [[] for _ in range(self.m)]
        per_agent_inst_reg: List[List[float]] = [[] for _ in range(self.m)]
        per_agent_exp_prof: List[List[float]] = [[] for _ in range(self.m)]
        per_agent_real_prof: List[List[float]] = [[] for _ in range(self.m)]
        joint_bid_history: List[List[List[int]]] = []

        for t in range(1, total_rounds + 1):
            m_t = self.run_round(
                t=t,
                rng=rng,
                cumulative_regrets=cumulative_regrets,
                remaining_cumulative_budgets=remaining_cum_budgets,
            )
            round_metrics.append(m_t)
            eq_gaps.append(m_t.equilibrium_gap)
            eq_distances.append(m_t.equilibrium_distance)
            platform_revs.append(m_t.platform_revenue)
            efficiencies.append(m_t.allocation_efficiency)
            welfares.append(m_t.expected_welfare)
            joint_bid_history.append(m_t.joint_bids)

            for i in range(self.m):
                per_agent_cum_reg[i].append(m_t.per_agent_cumulative_regrets[i])
                per_agent_inst_reg[i].append(m_t.per_agent_instantaneous_regrets[i])
                per_agent_exp_prof[i].append(m_t.expected_profits[i])
                per_agent_real_prof[i].append(m_t.realized_profits[i])

        # Summary calculations
        final_gap = eq_gaps[-1]
        mean_gap = float(np.mean(eq_gaps))
        final_dist = eq_distances[-1]
        mean_dist = float(np.mean(eq_distances))
        final_cum_regs = [per_agent_cum_reg[i][-1] for i in range(self.m)]
        mean_exp_profs = [float(np.mean(per_agent_exp_prof[i])) for i in range(self.m)]
        mean_rev = float(np.mean(platform_revs))
        mean_eff = float(np.mean(efficiencies))
        mean_wel = float(np.mean(welfares))

        is_converged = final_gap <= self.convergence_tol
        conv_round = compute_time_to_convergence(eq_gaps, threshold=0.05, consecutive_rounds=10)
        stability_res = detect_cycling_or_stability(joint_bid_history, window_size=min(50, total_rounds))
        rate_fit = compute_convergence_rate(eq_gaps, burn_in_fraction=0.1)

        pol = compute_price_of_learning(
            tier1_advertiser_profits=self.p_star,
            tier1_platform_revenue=self.benchmark.platform_revenue,
            tier1_allocation_efficiency=self.benchmark.allocation_efficiency,
            tier1_expected_welfare=self.benchmark.expected_welfare,
            tier3_mean_profits=mean_exp_profs,
            tier3_mean_revenue=mean_rev,
            tier3_mean_efficiency=mean_eff,
            tier3_mean_welfare=mean_wel,
        )

        return Tier3TrialResult(
            trial_seed=seed,
            population_type=self.population_type,
            agent_names=agent_names,
            agent_types=agent_types,
            total_rounds=total_rounds,
            round_metrics=round_metrics,
            equilibrium_gaps=eq_gaps,
            equilibrium_distances=eq_distances,
            per_agent_cumulative_regrets=per_agent_cum_reg,
            per_agent_instantaneous_regrets=per_agent_inst_reg,
            per_agent_expected_profits=per_agent_exp_prof,
            per_agent_realized_profits=per_agent_real_prof,
            platform_revenues=platform_revs,
            allocation_efficiencies=efficiencies,
            expected_welfares=welfares,
            final_equilibrium_gap=final_gap,
            mean_equilibrium_gap=mean_gap,
            final_equilibrium_distance=final_dist,
            mean_equilibrium_distance=mean_dist,
            final_cumulative_regrets=final_cum_regs,
            mean_expected_profits=mean_exp_profs,
            mean_platform_revenue=mean_rev,
            mean_efficiency=mean_eff,
            mean_welfare=mean_wel,
            is_converged_to_pure_equilibrium=is_converged,
            convergence_round=conv_round,
            stability_analysis=stability_res,
            convergence_rate_fit=rate_fit,
            price_of_learning=pol,
        )

    def run_experiment(
        self,
        total_rounds: int = 500,
        seeds: Optional[List[int]] = None,
        num_trials: int = 10,
        base_seed: int = 42,
    ) -> Tier3ExperimentResult:
        """
        Runs multi-trial experiments across independent seeds and aggregates
        trajectories with mean and 95% Confidence Intervals.
        """
        if seeds is None:
            seeds = [base_seed + 17 * k for k in range(num_trials)]
        else:
            num_trials = len(seeds)

        trials: List[Tier3TrialResult] = []
        for s in seeds:
            trial = self.run_trial(seed=s, total_rounds=total_rounds)
            trials.append(trial)

        # Statistical aggregation helper
        def calc_stats_1d(matrix_seeds: np.ndarray) -> Tuple[List[float], List[float]]:
            # matrix_seeds shape: (num_trials, total_rounds)
            means = np.mean(matrix_seeds, axis=0)
            stds = np.std(matrix_seeds, axis=0, ddof=1) if num_trials > 1 else np.zeros_like(means)
            ci95 = 1.96 * stds / math.sqrt(num_trials) if num_trials > 1 else np.zeros_like(means)
            return means.tolist(), ci95.tolist()

        # Matrices across trials
        gap_mat = np.array([tr.equilibrium_gaps for tr in trials])
        dist_mat = np.array([tr.equilibrium_distances for tr in trials])
        rev_mat = np.array([tr.platform_revenues for tr in trials])
        eff_mat = np.array([tr.allocation_efficiencies for tr in trials])
        wel_mat = np.array([tr.expected_welfares for tr in trials])

        mean_gap, ci_gap = calc_stats_1d(gap_mat)
        mean_dist, ci_dist = calc_stats_1d(dist_mat)
        mean_rev, ci_rev = calc_stats_1d(rev_mat)
        mean_eff, ci_eff = calc_stats_1d(eff_mat)
        mean_wel, ci_wel = calc_stats_1d(wel_mat)

        # Per-agent matrices
        mean_per_agent_cum_reg: List[List[float]] = []
        ci95_per_agent_cum_reg: List[List[float]] = []
        mean_per_agent_exp_prof: List[List[float]] = []
        ci95_per_agent_exp_prof: List[List[float]] = []

        final_regrets_trials: List[List[float]] = [[] for _ in range(self.m)]
        average_profits_trials: List[List[float]] = [[] for _ in range(self.m)]

        for i in range(self.m):
            agent_cum_reg_mat = np.array([tr.per_agent_cumulative_regrets[i] for tr in trials])
            m_reg, c_reg = calc_stats_1d(agent_cum_reg_mat)
            mean_per_agent_cum_reg.append(m_reg)
            ci95_per_agent_cum_reg.append(c_reg)

            agent_exp_prof_mat = np.array([tr.per_agent_expected_profits[i] for tr in trials])
            m_prof, c_prof = calc_stats_1d(agent_exp_prof_mat)
            mean_per_agent_exp_prof.append(m_prof)
            ci95_per_agent_exp_prof.append(c_prof)

            for tr in trials:
                final_regrets_trials[i].append(tr.final_cumulative_regrets[i])
                average_profits_trials[i].append(tr.mean_expected_profits[i])

        final_gap_mean = float(np.mean([tr.final_equilibrium_gap for tr in trials]))
        final_gap_std = float(np.std([tr.final_equilibrium_gap for tr in trials], ddof=1)) if num_trials > 1 else 0.0
        final_gap_ci95 = 1.96 * final_gap_std / math.sqrt(num_trials) if num_trials > 1 else 0.0

        final_dist_mean = float(np.mean([tr.final_equilibrium_distance for tr in trials]))
        final_dist_std = float(np.std([tr.final_equilibrium_distance for tr in trials], ddof=1)) if num_trials > 1 else 0.0
        final_dist_ci95 = 1.96 * final_dist_std / math.sqrt(num_trials) if num_trials > 1 else 0.0

        final_regrets_mean: List[float] = []
        final_regrets_ci95: List[float] = []
        avg_profit_mean: List[float] = []
        avg_profit_ci95: List[float] = []

        for i in range(self.m):
            r_mean = float(np.mean(final_regrets_trials[i]))
            r_std = float(np.std(final_regrets_trials[i], ddof=1)) if num_trials > 1 else 0.0
            final_regrets_mean.append(r_mean)
            final_regrets_ci95.append(1.96 * r_std / math.sqrt(num_trials) if num_trials > 1 else 0.0)

            p_mean = float(np.mean(average_profits_trials[i]))
            p_std = float(np.std(average_profits_trials[i], ddof=1)) if num_trials > 1 else 0.0
            avg_profit_mean.append(p_mean)
            avg_profit_ci95.append(1.96 * p_std / math.sqrt(num_trials) if num_trials > 1 else 0.0)

        all_effs = [tr.mean_efficiency for tr in trials]
        eff_mean = float(np.mean(all_effs))
        eff_std = float(np.std(all_effs, ddof=1)) if num_trials > 1 else 0.0
        eff_ci95 = 1.96 * eff_std / math.sqrt(num_trials) if num_trials > 1 else 0.0

        all_revs = [tr.mean_platform_revenue for tr in trials]
        rev_mean = float(np.mean(all_revs))
        rev_std = float(np.std(all_revs, ddof=1)) if num_trials > 1 else 0.0
        rev_ci95 = 1.96 * rev_std / math.sqrt(num_trials) if num_trials > 1 else 0.0

        all_wels = [tr.mean_welfare for tr in trials]
        wel_mean = float(np.mean(all_wels))

        pol_exp = compute_price_of_learning(
            tier1_advertiser_profits=self.p_star,
            tier1_platform_revenue=self.benchmark.platform_revenue,
            tier1_allocation_efficiency=self.benchmark.allocation_efficiency,
            tier1_expected_welfare=self.benchmark.expected_welfare,
            tier3_mean_profits=avg_profit_mean,
            tier3_mean_revenue=rev_mean,
            tier3_mean_efficiency=eff_mean,
            tier3_mean_welfare=wel_mean,
        )

        overall_rate_fit = compute_convergence_rate(mean_gap, burn_in_fraction=0.1)

        # Aggregate stability
        statuses = [tr.stability_analysis.get("status", "unknown") for tr in trials]
        stability_summary = {
            "status_distribution": {s: statuses.count(s) for s in set(statuses)},
            "dominant_status": max(set(statuses), key=statuses.count),
            "mean_stability_fraction": float(np.mean([tr.stability_analysis.get("profile_stability_fraction", 0.0) for tr in trials])),
        }

        return Tier3ExperimentResult(
            instance_name=self.instance.name or f"inst_{self.m}x{self.n}",
            population_type=self.population_type,
            num_trials=num_trials,
            total_rounds=total_rounds,
            round_structure=self.round_structure,
            agent_names=[a.name for a in self.agents],
            agent_types=[a.__class__.__name__ for a in self.agents],
            mean_equilibrium_gap=mean_gap,
            ci95_equilibrium_gap=ci_gap,
            mean_equilibrium_distance=mean_dist,
            ci95_equilibrium_distance=ci_dist,
            mean_platform_revenue=mean_rev,
            ci95_platform_revenue=ci_rev,
            mean_allocation_efficiency=mean_eff,
            ci95_allocation_efficiency=ci_eff,
            mean_expected_welfare=mean_wel,
            ci95_expected_welfare=ci_wel,
            mean_per_agent_cumulative_regret=mean_per_agent_cum_reg,
            ci95_per_agent_cumulative_regret=ci95_per_agent_cum_reg,
            mean_per_agent_expected_profit=mean_per_agent_exp_prof,
            ci95_per_agent_expected_profit=ci95_per_agent_exp_prof,
            final_gap_mean=final_gap_mean,
            final_gap_ci95=final_gap_ci95,
            final_distance_mean=final_dist_mean,
            final_distance_ci95=final_dist_ci95,
            final_regrets_mean=final_regrets_mean,
            final_regrets_ci95=final_regrets_ci95,
            average_efficiency_mean=eff_mean,
            average_efficiency_ci95=eff_ci95,
            average_revenue_mean=rev_mean,
            average_revenue_ci95=rev_ci95,
            average_profit_mean=avg_profit_mean,
            average_profit_ci95=avg_profit_ci95,
            price_of_learning=pol_exp,
            convergence_rate_summary=overall_rate_fit,
            stability_summary=stability_summary,
        )
