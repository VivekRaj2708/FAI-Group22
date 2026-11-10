import math
import random
from typing import List, Optional, Union
import numpy as np

from src.agents.base import BaseAdvertiserAgent
from src.tier1.knapsack import solve_knapsack_dp


class UCBAgent(BaseAdvertiserAgent):
    """
    Budget-aware Optimistic (UCB) Bidding Agent (Algorithm 1 in main.tex).
    Operates as a combinatorial semi-bandit learner under a knapsack constraint.
    """

    def __init__(
        self,
        agent_id: int,
        budget: int,
        num_users: int,
        click_value: float = 1.0,
        alpha: float = 2.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            name=name or f"UCB_Agent_{agent_id}",
        )
        self.alpha = alpha
        self.pulls = [0] * self.num_users  # N_j: impressions won
        self.clicks = [0] * self.num_users  # s_j: clicks observed
        self.last_competing_bids: Optional[List[int]] = None

    def reset(self) -> None:
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids = None

    def compute_ucb_index(self, user_idx: int, t: int) -> float:
        """
        Computes the Upper Confidence Bound for user click probability:
        f_tilde_j = 1 if N_j == 0 else min(1.0, s_j/N_j + sqrt(alpha * ln(t) / N_j))
        """
        n_j = self.pulls[user_idx]
        if n_j == 0:
            return 1.0
        empirical_mean = self.clicks[user_idx] / n_j
        # t should be >= 2 in round 2 onwards
        t_effective = max(2, t)
        bonus = math.sqrt((self.alpha * math.log(t_effective)) / n_j)
        return min(1.0, empirical_mean + bonus)

    def get_estimated_click_probabilities(self, t: int = 2) -> List[float]:
        return [self.compute_ucb_index(j, t) for j in range(self.num_users)]

    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        if last_competing_bids is not None:
            self.last_competing_bids = [int(b) for b in last_competing_bids]

        # Round 1: probe round, bid 0 everywhere (Assumption 2 & Algorithm 1)
        if t <= 1 or self.last_competing_bids is None:
            return [0] * self.num_users

        # Form knapsack weights and optimistic values
        weights: List[int] = []
        values: List[float] = []
        eligible_users: List[int] = []

        for j in range(self.num_users):
            c_j = self.last_competing_bids[j]
            w_j = c_j + 1  # Cost to strictly beat competing bid
            f_tilde = self.compute_ucb_index(j, t)
            r_j = f_tilde * self.click_value - w_j

            if r_j > 1e-9 and w_j <= self.budget:
                weights.append(w_j)
                values.append(r_j)
                eligible_users.append(j)

        # Solve knapsack DP exactly
        _, selected_sub = solve_knapsack_dp(weights, values, self.budget)

        bids = [0] * self.num_users
        for idx in selected_sub:
            user_idx = eligible_users[idx]
            bids[user_idx] = self.last_competing_bids[user_idx] + 1

        return bids

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


class ThompsonSamplingAgent(BaseAdvertiserAgent):
    """
    Budget-aware Thompson Sampling Bidding Agent.
    Maintains independent Beta(1 + s_j, 1 + N_j - s_j) posteriors for each user.
    """

    def __init__(
        self,
        agent_id: int,
        budget: int,
        num_users: int,
        click_value: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            name=name or f"ThompsonSampling_Agent_{agent_id}",
        )
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.seed = seed
        self.rng = random.Random(seed)
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids: Optional[List[int]] = None

    def reset(self) -> None:
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids = None
        if self.seed is not None:
            self.rng = random.Random(self.seed)

    def sample_posterior(self) -> List[float]:
        """Draws a posterior sample f_tilde_j ~ Beta(alpha + s_j, beta + N_j - s_j)."""
        samples: List[float] = []
        for j in range(self.num_users):
            a = self.prior_alpha + self.clicks[j]
            b = self.prior_beta + (self.pulls[j] - self.clicks[j])
            samples.append(self.rng.betavariate(a, b))
        return samples

    def get_estimated_click_probabilities(self) -> List[float]:
        """Returns posterior mean estimates."""
        means: List[float] = []
        for j in range(self.num_users):
            a = self.prior_alpha + self.clicks[j]
            b = self.prior_beta + (self.pulls[j] - self.clicks[j])
            means.append(a / (a + b))
        return means

    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        if last_competing_bids is not None:
            self.last_competing_bids = [int(b) for b in last_competing_bids]

        if t <= 1 or self.last_competing_bids is None:
            return [0] * self.num_users

        sampled_f = self.sample_posterior()

        weights: List[int] = []
        values: List[float] = []
        eligible_users: List[int] = []

        for j in range(self.num_users):
            c_j = self.last_competing_bids[j]
            w_j = c_j + 1
            f_tilde = sampled_f[j]
            r_j = f_tilde * self.click_value - w_j

            if r_j > 1e-9 and w_j <= self.budget:
                weights.append(w_j)
                values.append(r_j)
                eligible_users.append(j)

        _, selected_sub = solve_knapsack_dp(weights, values, self.budget)

        bids = [0] * self.num_users
        for idx in selected_sub:
            user_idx = eligible_users[idx]
            bids[user_idx] = self.last_competing_bids[user_idx] + 1

        return bids

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


class GreedyAgent(BaseAdvertiserAgent):
    """
    Greedy Bidding Agent without exploration bonus (alpha = 0).
    Uses empirical mean s_j / N_j (with optimistic 1.0 on unobserved users).
    """

    def __init__(
        self,
        agent_id: int,
        budget: int,
        num_users: int,
        click_value: float = 1.0,
        optimistic_unvisited: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            name=name or f"Greedy_Agent_{agent_id}",
        )
        self.optimistic_unvisited = optimistic_unvisited
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids: Optional[List[int]] = None

    def reset(self) -> None:
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids = None

    def get_estimated_click_probabilities(self) -> List[float]:
        estimates: List[float] = []
        for j in range(self.num_users):
            if self.pulls[j] == 0:
                estimates.append(self.optimistic_unvisited)
            else:
                estimates.append(self.clicks[j] / self.pulls[j])
        return estimates

    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        if last_competing_bids is not None:
            self.last_competing_bids = [int(b) for b in last_competing_bids]

        if t <= 1 or self.last_competing_bids is None:
            return [0] * self.num_users

        estimates = self.get_estimated_click_probabilities()

        weights: List[int] = []
        values: List[float] = []
        eligible_users: List[int] = []

        for j in range(self.num_users):
            c_j = self.last_competing_bids[j]
            w_j = c_j + 1
            f_hat = estimates[j]
            r_j = f_hat * self.click_value - w_j

            if r_j > 1e-9 and w_j <= self.budget:
                weights.append(w_j)
                values.append(r_j)
                eligible_users.append(j)

        _, selected_sub = solve_knapsack_dp(weights, values, self.budget)

        bids = [0] * self.num_users
        for idx in selected_sub:
            user_idx = eligible_users[idx]
            bids[user_idx] = self.last_competing_bids[user_idx] + 1

        return bids

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


class EpsGreedyAgent(BaseAdvertiserAgent):
    """
    Epsilon-Greedy Bidding Agent.
    With probability 1 - epsilon, plays greedily.
    With probability epsilon, explores by picking a random feasible subset of eligible users.
    """

    def __init__(
        self,
        agent_id: int,
        budget: int,
        num_users: int,
        click_value: float = 1.0,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            name=name or f"EpsGreedy_Agent_{agent_id}",
        )
        self.epsilon = epsilon
        self.seed = seed
        self.rng = random.Random(seed)
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids: Optional[List[int]] = None

    def reset(self) -> None:
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids = None
        if self.seed is not None:
            self.rng = random.Random(self.seed)

    def get_estimated_click_probabilities(self) -> List[float]:
        estimates: List[float] = []
        for j in range(self.num_users):
            if self.pulls[j] == 0:
                estimates.append(1.0)
            else:
                estimates.append(self.clicks[j] / self.pulls[j])
        return estimates

    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        if last_competing_bids is not None:
            self.last_competing_bids = [int(b) for b in last_competing_bids]

        if t <= 1 or self.last_competing_bids is None:
            return [0] * self.num_users

        estimates = self.get_estimated_click_probabilities()

        # Exploration branch
        if self.rng.random() < self.epsilon:
            # Random exploration: choose random feasible knapsack subset among users with non-negative margin
            candidates = []
            for j in range(self.num_users):
                w_j = self.last_competing_bids[j] + 1
                if w_j <= self.budget:
                    candidates.append((j, w_j))

            self.rng.shuffle(candidates)
            bids = [0] * self.num_users
            spend = 0
            for j, w_j in candidates:
                if spend + w_j <= self.budget:
                    bids[j] = w_j
                    spend += w_j
            return bids

        # Exploitation branch (Greedy)
        weights: List[int] = []
        values: List[float] = []
        eligible_users: List[int] = []

        for j in range(self.num_users):
            c_j = self.last_competing_bids[j]
            w_j = c_j + 1
            f_hat = estimates[j]
            r_j = f_hat * self.click_value - w_j

            if r_j > 1e-9 and w_j <= self.budget:
                weights.append(w_j)
                values.append(r_j)
                eligible_users.append(j)

        _, selected_sub = solve_knapsack_dp(weights, values, self.budget)

        bids = [0] * self.num_users
        for idx in selected_sub:
            user_idx = eligible_users[idx]
            bids[user_idx] = self.last_competing_bids[user_idx] + 1

        return bids

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


class CumulativeBudgetPacedAgent(BaseAdvertiserAgent):
    """
    Cumulative Budget Paced Bidding Agent (Remark 1, item 2 in main.tex).
    Designed for the Bandits with Knapsacks (BwK) setting where a single total budget
    is depleted across T rounds without replenishment.
    Employs adaptive dual Lagrangian pacing to pace spending across remaining rounds.
    """

    def __init__(
        self,
        agent_id: int,
        total_budget: int,
        num_users: int,
        total_rounds: int,
        click_value: float = 1.0,
        per_round_max_budget: Optional[int] = None,
        alpha: float = 2.0,
        eta: Optional[float] = None,
        name: Optional[str] = None,
    ):
        # Base class budget is per-round cap or total_budget
        per_round_cap = per_round_max_budget or total_budget
        super().__init__(
            agent_id=agent_id,
            budget=per_round_cap,
            num_users=num_users,
            click_value=click_value,
            name=name or f"CumulativePaced_Agent_{agent_id}",
        )
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.total_rounds = total_rounds
        self.per_round_max_budget = per_round_cap
        self.alpha = alpha

        # Dual multiplier lambda (shadow price of budget)
        self.dual_lambda = 0.0
        self.target_pace = total_budget / max(1, total_rounds)
        self.eta = eta or (1.0 / math.sqrt(max(1, total_rounds)))

        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids: Optional[List[int]] = None

    def reset(self) -> None:
        self.remaining_budget = self.total_budget
        self.dual_lambda = 0.0
        self.pulls = [0] * self.num_users
        self.clicks = [0] * self.num_users
        self.last_competing_bids = None

    def compute_ucb_index(self, user_idx: int, t: int) -> float:
        n_j = self.pulls[user_idx]
        if n_j == 0:
            return 1.0
        empirical_mean = self.clicks[user_idx] / n_j
        t_effective = max(2, t)
        bonus = math.sqrt((self.alpha * math.log(t_effective)) / n_j)
        return min(1.0, empirical_mean + bonus)

    def get_estimated_click_probabilities(self, t: int = 2) -> List[float]:
        return [self.compute_ucb_index(j, t) for j in range(self.num_users)]

    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        if last_competing_bids is not None:
            self.last_competing_bids = [int(b) for b in last_competing_bids]

        # If budget exhausted, cannot bid
        if self.remaining_budget <= 0:
            return [0] * self.num_users

        # Round 1: probe round
        if t <= 1 or self.last_competing_bids is None:
            return [0] * self.num_users

        # Remaining rounds & target allocation
        remaining_rounds = max(1, self.total_rounds - t + 1)
        target_round_budget = math.ceil(self.remaining_budget / remaining_rounds)
        # Capacity is bounded by available budget, per-round cap, and target pace with buffer
        effective_capacity = min(
            self.remaining_budget,
            self.per_round_max_budget,
            max(1, int(target_round_budget * 2)),
        )

        weights: List[int] = []
        values: List[float] = []
        eligible_users: List[int] = []

        for j in range(self.num_users):
            c_j = self.last_competing_bids[j]
            w_j = c_j + 1
            f_tilde = self.compute_ucb_index(j, t)

            # Lagrangian penalized profit: r_j(lambda) = f_tilde * v - (1 + lambda) * w_j
            r_j = f_tilde * self.click_value - (1.0 + self.dual_lambda) * w_j

            if r_j > 1e-9 and w_j <= effective_capacity:
                weights.append(w_j)
                values.append(r_j)
                eligible_users.append(j)

        _, selected_sub = solve_knapsack_dp(weights, values, effective_capacity)

        bids = [0] * self.num_users
        for idx in selected_sub:
            user_idx = eligible_users[idx]
            bids[user_idx] = self.last_competing_bids[user_idx] + 1

        return bids

    def observe(
        self,
        won: Union[List[int], np.ndarray],
        clicks: Union[List[int], np.ndarray],
        payments: Union[List[int], np.ndarray],
        competing_bids: Union[List[int], np.ndarray],
    ) -> None:
        self.last_competing_bids = [int(b) for b in competing_bids]
        round_spend = 0
        for j in range(self.num_users):
            if won[j]:
                self.pulls[j] += 1
                self.clicks[j] += int(clicks[j])
                round_spend += int(payments[j])

        # Deduct spend from remaining cumulative budget
        self.remaining_budget = max(0, self.remaining_budget - round_spend)

        # Update dual multiplier: lambda <- max(0, lambda + eta * (spend - target_pace))
        self.dual_lambda = max(
            0.0,
            self.dual_lambda + self.eta * (round_spend - self.target_pace),
        )
