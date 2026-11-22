from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class KnapsackResult:
    """Output of the 0-1 Knapsack Best Response solver."""

    bids: List[int]  # Integer bid vector of length n
    selected_users: List[int]  # Indices of users targeted (x_j = 1)
    expected_profit: float  # Maximum expected profit sum_j r_{ij} x_j
    total_spend: int  # Total budget utilized sum_j b_{ij}


def solve_knapsack_dp(
    weights: List[int],
    values: List[float],
    capacity: int,
) -> Tuple[float, List[int]]:
    """
    Solves the standard 0-1 Knapsack problem exactly via dynamic programming in O(n * B).

    Args:
        weights: Integer costs/weights w_j for each item.
        values: Floating point profits/values r_j for each item.
        capacity: Integer maximum capacity B.

    Returns:
        (max_value, selected_indices)
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []

    # dp[k][b] stores the maximum value using a subset of the first k items with budget b
    dp: List[List[float]] = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    # keep track of decision: 1 if item k-1 was included, 0 otherwise
    take: List[List[bool]] = [[False] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for b in range(capacity + 1):
            # Option 1: Do not take item i-1
            dp[i][b] = dp[i - 1][b]

            # Option 2: Take item i-1 (if weight fits and gives positive incremental value)
            if w <= b and v > 0:
                candidate = dp[i - 1][b - w] + v
                if candidate > dp[i][b] + 1e-12:
                    dp[i][b] = candidate
                    take[i][b] = True

    # Backtrack to reconstruct the optimal item set
    selected: List[int] = []
    rem_b = capacity
    for i in range(n, 0, -1):
        if take[i][rem_b]:
            selected.append(i - 1)
            rem_b -= weights[i - 1]

    selected.reverse()
    return dp[n][capacity], selected


def compute_best_response_knapsack(
    f_i: List[float],
    competing_bids: List[int],
    budget: int,
    click_value: float = 1.0,
) -> KnapsackResult:
    """
    Computes advertiser a_i's best response against fixed competing bids c_{-i}
    under strict-win bidding (Eq. (1) in the proposal).

    To win user u_j with probability 1, the advertiser bids b_{ij} = c_{-i, j} + 1.
    Cost: w_j = c_{-i, j} + 1
    Value: r_{ij} = f_{ij} * v - (c_{-i, j} + 1)

    Items with r_{ij} <= 0 are ignored since winning them yields non-positive expected profit.
    """
    n = len(f_i)
    weights: List[int] = []
    values: List[float] = []
    eligible_mapping: List[int] = []

    for j in range(n):
        w_j = competing_bids[j] + 1
        r_j = f_i[j] * click_value - w_j

        if r_j > 1e-9 and w_j <= budget:
            weights.append(w_j)
            values.append(r_j)
            eligible_mapping.append(j)

    max_profit, selected_sub = solve_knapsack_dp(weights, values, budget)

    bids = [0] * n
    selected_users = [eligible_mapping[idx] for idx in selected_sub]
    for j in selected_users:
        bids[j] = competing_bids[j] + 1

    return KnapsackResult(
        bids=bids,
        selected_users=selected_users,
        expected_profit=max_profit,
        total_spend=sum(bids),
    )
