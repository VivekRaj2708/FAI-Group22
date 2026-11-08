from dataclasses import dataclass
from typing import List
from src.env.instance import AuctionInstance


@dataclass
class SocialPlannerOutcome:
    """Welfare-maximizing benchmarks without budget or bidding frictions."""

    optimal_assignment: List[int]  # For each user j, index of best advertiser
    max_click_probabilities: List[float]  # Max click probability per user
    max_social_welfare: float  # sum_j max_i (f_ij * v)
    efficiency: float = 1.0  # By definition 1.0 under first-best


def compute_social_planner_benchmark(instance: AuctionInstance) -> SocialPlannerOutcome:
    """
    Computes the omniscient first-best assignment where each user
    is allocated to the advertiser who values them the most (argmax_i f_ij).
    """
    m = instance.num_advertisers
    n = instance.num_users
    v = instance.click_value

    assignment: List[int] = []
    max_probs: List[float] = []
    tot_welfare = 0.0

    for j in range(n):
        user_vals = [instance.click_probabilities[i][j] for i in range(m)]
        best_val = max(user_vals)
        best_i = user_vals.index(best_val)

        assignment.append(best_i)
        max_probs.append(best_val)
        tot_welfare += best_val * v

    return SocialPlannerOutcome(
        optimal_assignment=assignment,
        max_click_probabilities=max_probs,
        max_social_welfare=tot_welfare,
        efficiency=1.0,
    )
