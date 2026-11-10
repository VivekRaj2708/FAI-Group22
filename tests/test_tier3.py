import math
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    class _PytestStub:
        @staticmethod
        def fixture(func):
            return func
    pytest = _PytestStub()

import numpy as np

from src.env.instance import AuctionInstance, generate_synthetic_instance
from src.env.auction import BatchedFirstPriceAuction
from src.tier1.baseline import Tier1Solver
from src.tier2.agents import UCBAgent, ThompsonSamplingAgent
from src.tier3.convergence import (
    compute_equilibrium_distance,
    compute_normalized_distance,
    compute_convergence_rate,
    detect_cycling_or_stability,
    compute_time_to_convergence,
    compute_price_of_learning,
)
from src.tier3.simulation import (
    Tier3RoundMetrics,
    Tier3TrialResult,
    Tier3ExperimentResult,
    create_multi_agent_population,
    MultiAgentSimulator,
)


@pytest.fixture
def test_instance():
    return AuctionInstance(
        name="test_inst_2x3",
        num_advertisers=2,
        num_users=3,
        budgets=[3, 3],
        click_probabilities=[
            [0.8, 0.4, 0.2],
            [0.3, 0.7, 0.5],
        ],
        click_value=5.0,
    )


def test_multi_agent_population_factory(test_instance):
    # 1. all_ucb
    ucb_pop = create_multi_agent_population("all_ucb", test_instance, alpha=2.5)
    assert len(ucb_pop) == 2
    assert all(isinstance(a, UCBAgent) for a in ucb_pop)
    assert ucb_pop[0].alpha == 2.5

    # 2. all_ts
    ts_pop = create_multi_agent_population("all_ts", test_instance, seed=99)
    assert len(ts_pop) == 2
    assert all(isinstance(a, ThompsonSamplingAgent) for a in ts_pop)

    # 3. mixed_all with m=4
    inst4 = generate_synthetic_instance(num_advertisers=4, num_users=3, budget_tightness=0.5, seed=1)
    mixed_pop = create_multi_agent_population("mixed_all", inst4)
    assert len(mixed_pop) == 4
    type_names = [a.__class__.__name__ for a in mixed_pop]
    assert type_names == ["UCBAgent", "ThompsonSamplingAgent", "GreedyAgent", "EpsGreedyAgent"]


def test_round_1_probe_and_budget_constraints(test_instance):
    sim = MultiAgentSimulator(instance=test_instance, population_type="all_ucb")

    # Round 1: probe round
    m1 = sim.run_round(t=1, rng=np.random.default_rng(1), cumulative_regrets=[0.0, 0.0])
    for i in range(sim.m):
        assert m1.joint_bids[i] == [0] * sim.n
        assert m1.spends[i] == 0
        assert m1.competing_bids[i] == [0] * sim.n

    # Simulate subsequent 10 rounds: verify budget and integer constraints
    cum_regs = [0.0, 0.0]
    import random
    rng = random.Random(42)
    for t in range(2, 12):
        m_t = sim.run_round(t=t, rng=rng, cumulative_regrets=cum_regs)
        for i in range(sim.m):
            b_i = m_t.joint_bids[i]
            assert len(b_i) == sim.n
            assert all(isinstance(b, int) for b in b_i)
            assert all(b >= 0 for b in b_i)
            assert sum(b_i) <= sim.instance.budgets[i]


def test_equilibrium_gap_and_distance_metrics(test_instance):
    sim = MultiAgentSimulator(instance=test_instance, population_type="all_ucb")

    # Distance to b* when playing b* should be 0.0
    dist_zero = compute_equilibrium_distance(sim.b_star, sim.b_star)
    assert dist_zero == 0.0
    assert compute_normalized_distance(sim.b_star, sim.b_star) == 0.0

    # Perturbed bids should have positive distance
    perturbed = [list(row) for row in sim.b_star]
    perturbed[0][0] += 1
    dist_pert = compute_equilibrium_distance(perturbed, sim.b_star)
    assert dist_pert == 1.0


def test_cumulative_regret_growth(test_instance):
    sim = MultiAgentSimulator(instance=test_instance, population_type="all_ucb")
    trial = sim.run_trial(seed=42, total_rounds=25)

    assert trial.total_rounds == 25
    assert len(trial.round_metrics) == 25
    assert len(trial.equilibrium_gaps) == 25

    # Regrets should be non-negative and cumulative regret monotonically non-decreasing
    for i in range(sim.m):
        inst_regs = trial.per_agent_instantaneous_regrets[i]
        cum_regs = trial.per_agent_cumulative_regrets[i]
        assert len(inst_regs) == 25
        assert len(cum_regs) == 25
        for t in range(25):
            assert inst_regs[t] >= -1e-9
            if t > 0:
                assert cum_regs[t] >= cum_regs[t - 1] - 1e-9


def test_convergence_rate_synthetic():
    # Construct synthetic power law: y_t = 10 * t^(-0.5)
    t_vals = np.arange(1, 101)
    gaps = (10.0 * (t_vals ** -0.5)).tolist()

    res = compute_convergence_rate(gaps, burn_in_fraction=0.1)
    assert abs(res["beta"] - 0.5) < 0.05
    assert res["r_squared"] > 0.95
    assert res["is_converging"] == 1.0


def test_cycling_and_stability_detection():
    # 1. Constant sequence (fixed point)
    const_history = [[[1, 0], [0, 1]] for _ in range(30)]
    res_const = detect_cycling_or_stability(const_history, window_size=20)
    assert res_const["status"] == "converged_fixed_point"
    assert res_const["cycle_period"] == 1
    assert res_const["profile_stability_fraction"] == 1.0

    # 2. Limit cycle with period 2
    prof_a = [[1, 0], [0, 1]]
    prof_b = [[0, 1], [1, 0]]
    cycle_history = [prof_a if k % 2 == 0 else prof_b for k in range(40)]
    res_cycle = detect_cycling_or_stability(cycle_history, window_size=20)
    assert res_cycle["status"] == "limit_cycle"
    assert res_cycle["cycle_period"] == 2

    # 3. Time to convergence
    gaps = [0.8, 0.6, 0.4, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    conv_t = compute_time_to_convergence(gaps, threshold=0.05, consecutive_rounds=10)
    assert conv_t == 5  # index 4 is round 5


def test_price_of_learning_math():
    t1_profs = [1.5, 2.0]
    t1_rev = 3.0
    t1_eff = 0.8
    t1_wel = 4.0

    t3_profs = [1.0, 1.8]
    t3_rev = 2.5
    t3_eff = 0.6
    t3_wel = 3.2

    pol = compute_price_of_learning(
        t1_profs, t1_rev, t1_eff, t1_wel,
        t3_profs, t3_rev, t3_eff, t3_wel,
    )
    assert abs(pol["delta_total_profit"] - (3.5 - 2.8)) < 1e-9
    assert abs(pol["delta_platform_revenue"] - 0.5) < 1e-9
    assert abs(pol["delta_allocation_efficiency"] - 0.2) < 1e-9
    assert abs(pol["delta_social_welfare"] - 0.8) < 1e-9


def test_multi_seed_experiment_aggregation(test_instance):
    sim = MultiAgentSimulator(instance=test_instance, population_type="all_ucb")
    exp = sim.run_experiment(total_rounds=15, seeds=[10, 20, 30])

    assert exp.num_trials == 3
    assert exp.total_rounds == 15
    assert len(exp.mean_equilibrium_gap) == 15
    assert len(exp.ci95_equilibrium_gap) == 15
    assert len(exp.mean_per_agent_cumulative_regret) == 2
    assert len(exp.final_regrets_mean) == 2
    assert len(exp.final_regrets_ci95) == 2
    assert 0.0 <= exp.average_efficiency_mean <= 1.0


def test_experiment_serialization(test_instance):
    sim = MultiAgentSimulator(instance=test_instance, population_type="all_ts")
    exp = sim.run_experiment(total_rounds=10, seeds=[1, 2])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        exp.save_json(tmp_path)
        loaded = Tier3ExperimentResult.load_json(tmp_path)
        assert loaded.instance_name == exp.instance_name
        assert loaded.num_trials == exp.num_trials
        assert loaded.total_rounds == exp.total_rounds
        assert abs(loaded.final_gap_mean - exp.final_gap_mean) < 1e-9
        assert abs(loaded.average_revenue_mean - exp.average_revenue_mean) < 1e-9
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_round_structures_variants(test_instance):
    # Discounted variant
    sim_disc = MultiAgentSimulator(
        instance=test_instance,
        population_type="all_ucb",
        round_structure="discounted",
        discount_factor=0.95,
    )
    trial_disc = sim_disc.run_trial(seed=42, total_rounds=10)
    assert trial_disc.total_rounds == 10

    # Cumulative budget variant
    sim_cum = MultiAgentSimulator(
        instance=test_instance,
        population_type="all_ucb",
        round_structure="cumulative",
    )
    trial_cum = sim_cum.run_trial(seed=42, total_rounds=10)
    assert trial_cum.total_rounds == 10


if __name__ == "__main__":
    inst = test_instance()
    test_multi_agent_population_factory(inst)
    test_round_1_probe_and_budget_constraints(inst)
    test_equilibrium_gap_and_distance_metrics(inst)
    test_cumulative_regret_growth(inst)
    test_convergence_rate_synthetic()
    test_cycling_and_stability_detection()
    test_price_of_learning_math()
    test_multi_seed_experiment_aggregation(inst)
    test_experiment_serialization(inst)
    test_round_structures_variants(inst)
    print("All Tier 3 tests passed successfully!")

