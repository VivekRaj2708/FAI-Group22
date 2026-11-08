from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import itertools
import random

from src.env.instance import AuctionInstance
from src.env.auction import BatchedFirstPriceAuction
from src.tier1.knapsack import compute_best_response_knapsack


@dataclass
class EquilibriumResult:
    """Outcome of equilibrium computation."""

    method: str
    bid_profile: List[List[int]]  # Best profile found (shape m x n)
    is_pure_equilibrium: bool  # True if exploitability gap <= tolerance
    equilibrium_gap: float  # max_i (BR_i - u_i)
    advertiser_profits: List[float]
    platform_revenue: float
    allocation_efficiency: float
    iterations: int
    history_gap: Optional[List[float]] = None


def generate_pure_strategies(num_users: int, budget: int) -> List[List[int]]:
    """
    Generates all valid pure integer bid vectors b_i in {0, ..., budget}^n
    satisfying sum(b_ij) <= budget.
    """
    strategies: List[List[int]] = []

    def backtrack(user_idx: int, remaining_budget: int, current_bid: List[int]):
        if user_idx == num_users:
            strategies.append(list(current_bid))
            return
        # Bid on user_idx can be from 0 up to remaining_budget
        for b in range(remaining_budget + 1):
            current_bid.append(b)
            backtrack(user_idx + 1, remaining_budget - b, current_bid)
            current_bid.pop()

    backtrack(0, budget, [])
    return strategies


def compute_best_response_for_advertiser(
    auction: BatchedFirstPriceAuction,
    advertiser_idx: int,
    opponent_bids: List[List[int]],
) -> Tuple[List[int], float]:
    """
    Computes best response and maximum utility for advertiser_idx given fixed opponent_bids.
    Uses DP knapsack over strict-win bids and compares with the 0-bid.
    """
    m = auction.m
    n = auction.n
    i = advertiser_idx

    # Construct full profile with dummy bid for i
    dummy_bids = [list(opponent_bids[k]) if k < i else (list(opponent_bids[k - 1]) if k > i else [0] * n) for k in range(m)]
    competing_bids = auction.get_competing_bids(dummy_bids)[i]

    # Strict-win DP Knapsack
    knap_res = compute_best_response_knapsack(
        f_i=auction.f[i],
        competing_bids=competing_bids,
        budget=auction.budgets[i],
        click_value=auction.v,
    )

    # Verify against full auction expected utility
    test_profile = [list(row) for row in dummy_bids]
    test_profile[i] = knap_res.bids
    u_knap = auction.expected_utilities(test_profile)[i]

    # Zero bid utility
    test_profile[i] = [0] * n
    u_zero = auction.expected_utilities(test_profile)[i]

    if u_knap >= u_zero:
        return knap_res.bids, u_knap
    else:
        return [0] * n, u_zero


def compute_equilibrium_gap(
    auction: BatchedFirstPriceAuction,
    bid_profile: List[List[int]],
) -> Tuple[float, List[float], List[List[int]]]:
    """
    Computes epsilon(b) = max_i (BR_i(b_{-i}) - u_i(b)).

    Returns:
        (equilibrium_gap, per_agent_gaps, list_of_best_responses)
    """
    current_utilities = auction.expected_utilities(bid_profile)
    gaps: List[float] = []
    br_profiles: List[List[int]] = []

    for i in range(auction.m):
        opponents = [bid_profile[k] for k in range(auction.m) if k != i]
        br_bid, br_val = compute_best_response_for_advertiser(auction, i, opponents)
        br_profiles.append(br_bid)
        gap_i = max(0.0, br_val - current_utilities[i])
        gaps.append(gap_i)

    return max(gaps), gaps, br_profiles


def solve_best_response_dynamics(
    auction: BatchedFirstPriceAuction,
    max_iter: int = 100,
    tol: float = 1e-6,
    initial_profile: Optional[List[List[int]]] = None,
) -> EquilibriumResult:
    """
    Iterates Best-Response Dynamics until convergence or max_iter.
    Returns the profile with the smallest exploitability gap encountered.
    """
    if initial_profile is None:
        current_b = [[0] * auction.n for _ in range(auction.m)]
    else:
        current_b = [list(row) for row in initial_profile]

    best_profile = [list(row) for row in current_b]
    best_gap, _, _ = compute_equilibrium_gap(auction, current_b)
    history_gap: List[float] = [best_gap]

    if best_gap <= tol:
        return EquilibriumResult(
            method="BestResponseDynamics",
            bid_profile=best_profile,
            is_pure_equilibrium=True,
            equilibrium_gap=best_gap,
            advertiser_profits=auction.expected_utilities(best_profile),
            platform_revenue=auction.platform_revenue(best_profile),
            allocation_efficiency=auction.allocation_efficiency(best_profile),
            iterations=0,
            history_gap=history_gap,
        )

    for it in range(1, max_iter + 1):
        # Round-robin best response updates
        for i in range(auction.m):
            opponents = [current_b[k] for k in range(auction.m) if k != i]
            br_bid, _ = compute_best_response_for_advertiser(auction, i, opponents)
            current_b[i] = br_bid

        gap, _, _ = compute_equilibrium_gap(auction, current_b)
        history_gap.append(gap)

        if gap < best_gap:
            best_gap = gap
            best_profile = [list(row) for row in current_b]

        if gap <= tol:
            break

    return EquilibriumResult(
        method="BestResponseDynamics",
        bid_profile=best_profile,
        is_pure_equilibrium=(best_gap <= tol),
        equilibrium_gap=best_gap,
        advertiser_profits=auction.expected_utilities(best_profile),
        platform_revenue=auction.platform_revenue(best_profile),
        allocation_efficiency=auction.allocation_efficiency(best_profile),
        iterations=len(history_gap) - 1,
        history_gap=history_gap,
    )


def solve_fictitious_play(
    auction: BatchedFirstPriceAuction,
    num_rounds: int = 200,
    tol: float = 1e-4,
    seed: int = 42,
) -> EquilibriumResult:
    """
    Fictitious Play: agents choose best response against empirical historical frequency of opponent play.
    """
    rng = random.Random(seed)
    m = auction.m
    n = auction.n

    # Generate feasible action spaces
    action_spaces = [generate_pure_strategies(n, auction.budgets[i]) for i in range(m)]

    # Action counts for each player: list of counts for each strategy
    action_counts = [[0] * len(action_spaces[i]) for i in range(m)]

    # Initial random actions
    current_actions = [rng.randrange(len(action_spaces[i])) for i in range(m)]
    for i in range(m):
        action_counts[i][current_actions[i]] += 1

    history_gap: List[float] = []
    best_profile: List[List[int]] = [[0] * n for _ in range(m)]
    min_gap = float("inf")

    for t in range(1, num_rounds + 1):
        # Current pure bid vectors
        current_bids = [action_spaces[i][current_actions[i]] for i in range(m)]
        gap, _, _ = compute_equilibrium_gap(auction, current_bids)
        history_gap.append(gap)

        if gap < min_gap:
            min_gap = gap
            best_profile = [list(row) for row in current_bids]
            if min_gap <= tol:
                break

        # Compute next action for each player by responding to opponents' empirical play
        next_actions = list(current_actions)
        for i in range(m):
            best_action_idx = 0
            best_exp_payoff = -float("inf")

            # Evaluate expected utility of each action a_i against empirical distribution
            # For computational tractability, sample or average over opponent history
            for a_idx, candidate_bid in enumerate(action_spaces[i]):
                # Expected payoff against opponent historical distribution
                # We can sample opponent actions from history to approximate expected utility
                exp_u = 0.0
                samples = min(20, t)
                for _ in range(samples):
                    sampled_opponents: List[List[int]] = []
                    for k in range(m):
                        if k == i:
                            continue
                        # Sample according to action_counts[k]
                        k_weights = action_counts[k]
                        total_k = sum(k_weights)
                        r = rng.randint(1, total_k)
                        acc = 0
                        chosen_k_act = 0
                        for idx_act, cnt in enumerate(k_weights):
                            acc += cnt
                            if acc >= r:
                                chosen_k_act = idx_act
                                break
                        sampled_opponents.append(action_spaces[k][chosen_k_act])

                    # Build profile
                    temp_profile: List[List[int]] = []
                    opp_ptr = 0
                    for k in range(m):
                        if k == i:
                            temp_profile.append(candidate_bid)
                        else:
                            temp_profile.append(sampled_opponents[opp_ptr])
                            opp_ptr += 1

                    u = auction.expected_utilities(temp_profile)[i]
                    exp_u += u / samples

                if exp_u > best_exp_payoff:
                    best_exp_payoff = exp_u
                    best_action_idx = a_idx

            next_actions[i] = best_action_idx

        current_actions = next_actions
        for i in range(m):
            action_counts[i][current_actions[i]] += 1

    return EquilibriumResult(
        method="FictitiousPlay",
        bid_profile=best_profile,
        is_pure_equilibrium=(min_gap <= tol),
        equilibrium_gap=min_gap,
        advertiser_profits=auction.expected_utilities(best_profile),
        platform_revenue=auction.platform_revenue(best_profile),
        allocation_efficiency=auction.allocation_efficiency(best_profile),
        iterations=len(history_gap),
        history_gap=history_gap,
    )


def solve_support_enumeration_small(
    auction: BatchedFirstPriceAuction,
    tol: float = 1e-6,
) -> EquilibriumResult:
    """
    Exhaustively searches all pure strategy profiles for small instances (m <= 3, n <= 5).
    Finds exact Pure Nash Equilibria (gap <= tol). If multiple exist, chooses the one
    maximizing social welfare. If none exist, returns the profile with minimal gap.
    """
    m = auction.m
    n = auction.n

    # Generate action spaces for all players
    action_spaces = [generate_pure_strategies(n, auction.budgets[i]) for i in range(m)]

    pure_equilibria: List[Tuple[List[List[int]], float, float]] = []  # (profile, welfare, gap)
    best_profile: Optional[List[List[int]]] = None
    min_gap = float("inf")

    # Cartesian product over all strategy spaces
    for profile_tuple in itertools.product(*action_spaces):
        profile = [list(b) for b in profile_tuple]
        gap, _, _ = compute_equilibrium_gap(auction, profile)

        if gap < min_gap:
            min_gap = gap
            best_profile = profile

        if gap <= tol:
            welfare = auction.expected_welfare(profile)
            pure_equilibria.append((profile, welfare, gap))

    if pure_equilibria:
        # Pick the equilibrium with maximum welfare
        pure_equilibria.sort(key=lambda x: x[1], reverse=True)
        chosen_profile = pure_equilibria[0][0]
        chosen_gap = pure_equilibria[0][2]
        is_pure = True
    else:
        chosen_profile = best_profile or [[0] * n for _ in range(m)]
        chosen_gap = min_gap
        is_pure = False

    return EquilibriumResult(
        method="SupportEnumeration",
        bid_profile=chosen_profile,
        is_pure_equilibrium=is_pure,
        equilibrium_gap=chosen_gap,
        advertiser_profits=auction.expected_utilities(chosen_profile),
        platform_revenue=auction.platform_revenue(chosen_profile),
        allocation_efficiency=auction.allocation_efficiency(chosen_profile),
        iterations=1,
    )
