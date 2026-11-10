import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None

from src.env.instance import AuctionInstance, generate_synthetic_instance
from src.env.auction import BatchedFirstPriceAuction
from src.tier1.knapsack import solve_knapsack_dp, compute_best_response_knapsack
from src.tier1.social_planner import compute_social_planner_benchmark
from src.tier1.equilibrium import (
    generate_pure_strategies,
    compute_equilibrium_gap,
    solve_best_response_dynamics,
    solve_support_enumeration_small,
)
from src.tier1.baseline import Tier1Solver, Tier1Benchmark


def test_knapsack_dp_against_brute_force():
    weights = [2, 3, 1, 4]
    values = [3.5, 4.2, 1.8, 5.0]
    capacity = 5

    dp_val, dp_selected = solve_knapsack_dp(weights, values, capacity)

    # Brute force enumeration over all 2^4 = 16 subsets
    best_bf_val = 0.0
    best_bf_set = []
    n = len(weights)
    for mask in range(1 << n):
        subset = [i for i in range(n) if (mask & (1 << i))]
        w_sum = sum(weights[i] for i in subset)
        if w_sum <= capacity:
            v_sum = sum(values[i] for i in subset)
            if v_sum > best_bf_val:
                best_bf_val = v_sum
                best_bf_set = subset

    assert abs(dp_val - best_bf_val) < 1e-9
    assert sum(weights[i] for i in dp_selected) <= capacity


def test_auction_allocation_and_utilities():
    instance = AuctionInstance(
        num_advertisers=2,
        num_users=3,
        budgets=[3, 3],
        click_probabilities=[
            [0.8, 0.4, 0.1],
            [0.2, 0.9, 0.5],
        ],
        click_value=5.0,
    )
    auction = BatchedFirstPriceAuction(instance)

    # Valid profile: a0 bids [2, 1, 0], a1 bids [1, 2, 0]
    bids = [[2, 1, 0], [1, 2, 0]]
    q = auction.compute_winning_probabilities(bids)

    # User 0: a0 bids 2 > a1 bids 1 -> a0 wins with prob 1
    assert q[0][0] == 1.0 and q[1][0] == 0.0
    # User 1: a1 bids 2 > a0 bids 1 -> a1 wins with prob 1
    assert q[0][1] == 0.0 and q[1][1] == 1.0
    # User 2: both bid 0 -> no one wins
    assert q[0][2] == 0.0 and q[1][2] == 0.0

    # Utilities
    utils = auction.expected_utilities(bids)
    # a0 utility: 1.0 * (0.8 * 5.0 - 2) = 2.0
    assert abs(utils[0] - 2.0) < 1e-9
    # a1 utility: 1.0 * (0.9 * 5.0 - 2) = 2.5
    assert abs(utils[1] - 2.5) < 1e-9

    # Revenue: user 0 payment is 2, user 1 payment is 2 -> total 4.0
    rev = auction.platform_revenue(bids)
    assert abs(rev - 4.0) < 1e-9

    # Social welfare: 0.8 * 5.0 + 0.9 * 5.0 = 8.5
    wel = auction.expected_welfare(bids)
    assert abs(wel - 8.5) < 1e-9


def test_strict_win_knapsack_best_response():
    # User click probabilities for advertiser 0
    f_0 = [0.8, 0.5, 0.2]
    competing_bids = [1, 1, 0]
    budget = 4
    v = 5.0

    # User 0: competing bid 1 -> strict win cost = 2. Value = 0.8 * 5 - 2 = 2.0 > 0
    # User 1: competing bid 1 -> strict win cost = 2. Value = 0.5 * 5 - 2 = 0.5 > 0
    # User 2: competing bid 0 -> strict win cost = 1. Value = 0.2 * 5 - 1 = 0.0 <= 0 (excluded)
    # With budget 4, can afford User 0 and User 1 (total cost 2 + 2 = 4)
    res = compute_best_response_knapsack(f_0, competing_bids, budget, click_value=v)

    assert res.bids == [2, 2, 0]
    assert set(res.selected_users) == {0, 1}
    assert abs(res.expected_profit - 2.5) < 1e-9
    assert res.total_spend == 4


def test_social_planner_benchmark():
    instance = AuctionInstance(
        num_advertisers=2,
        num_users=3,
        budgets=[2, 2],
        click_probabilities=[
            [0.7, 0.3, 0.1],
            [0.2, 0.8, 0.5],
        ],
        click_value=10.0,
    )
    outcome = compute_social_planner_benchmark(instance)

    # Best advertiser per user: user 0 -> a0 (0.7), user 1 -> a1 (0.8), user 2 -> a1 (0.5)
    assert outcome.optimal_assignment == [0, 1, 1]
    assert abs(outcome.max_social_welfare - (0.7 + 0.8 + 0.5) * 10.0) < 1e-9
    assert outcome.efficiency == 1.0


def test_support_enumeration_small_equilibrium():
    instance = AuctionInstance(
        num_advertisers=2,
        num_users=2,
        budgets=[1, 1],
        click_probabilities=[
            [0.6, 0.1],
            [0.1, 0.7],
        ],
        click_value=4.0,
    )
    auction = BatchedFirstPriceAuction(instance)
    res = solve_support_enumeration_small(auction)

    assert res.is_pure_equilibrium
    assert res.equilibrium_gap < 1e-6
    # a0 values user 0 much higher, a1 values user 1 much higher
    assert res.bid_profile[0][0] == 1
    assert res.bid_profile[1][1] == 1


def test_tier1_solver_and_json_serialization():
    instance = generate_synthetic_instance(
        num_advertisers=2,
        num_users=3,
        budget_tightness=0.5,
        click_value=5.0,
        seed=100,
    )
    solver = Tier1Solver(instance)
    benchmark = solver.solve()

    assert benchmark.num_advertisers == 2
    assert benchmark.num_users == 3
    assert len(benchmark.reference_bid_profile) == 2
    assert len(benchmark.advertiser_profits) == 2

    # Test JSON dump and load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        benchmark.save_json(tmp_path)
        loaded = Tier1Benchmark.load_json(tmp_path)
        assert loaded.instance_name == benchmark.instance_name
        assert loaded.reference_bid_profile == benchmark.reference_bid_profile
        assert abs(loaded.platform_revenue - benchmark.platform_revenue) < 1e-9
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("All Tier 1 tests passed successfully!")
