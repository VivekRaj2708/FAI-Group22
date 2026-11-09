import math
import os
import tempfile
import pytest
import numpy as np

from src.env.instance import AuctionInstance, generate_synthetic_instance
from src.env.auction import BatchedFirstPriceAuction
from src.tier2.agents import (
    UCBAgent,
    ThompsonSamplingAgent,
    GreedyAgent,
    EpsGreedyAgent,
    CumulativeBudgetPacedAgent,
)
from src.tier2.simulation import (
    Tier2Simulator,
    Tier2TrialResult,
    Tier2ExperimentResult,
)


@pytest.fixture
def standard_instance():
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


def test_ucb_agent_probing_and_budget_constraint(standard_instance):
    agent = UCBAgent(
        agent_id=0,
        budget=3,
        num_users=3,
        click_value=5.0,
        alpha=2.0,
    )

    # Round 1: probe round with no previous competing bids -> must bid [0, 0, 0]
    b1 = agent.get_bid(t=1)
    assert b1 == [0, 0, 0]

    # Observe round 1 feedback
    competing = [1, 1, 0]
    agent.observe(
        won=[0, 0, 0],
        clicks=[0, 0, 0],
        payments=[0, 0, 0],
        competing_bids=competing,
    )

    # Round 2: should bid strict-win bids (c_j + 1) within budget 3
    b2 = agent.get_bid(t=2)
    assert sum(b2) <= 3
    for j, bid in enumerate(b2):
        if bid > 0:
            assert bid == competing[j] + 1


def test_ucb_index_math():
    agent = UCBAgent(agent_id=0, budget=4, num_users=2, click_value=1.0, alpha=2.0)
    # Untried user -> UCB index is 1.0
    assert agent.compute_ucb_index(user_idx=0, t=2) == 1.0

    # User 0 won 4 times with 2 clicks
    agent.pulls[0] = 4
    agent.clicks[0] = 2

    # Expected: 2/4 + sqrt(2.0 * ln(10) / 4)
    expected = 0.5 + math.sqrt((2.0 * math.log(10)) / 4)
    expected = min(1.0, expected)
    computed = agent.compute_ucb_index(user_idx=0, t=10)
    assert abs(computed - expected) < 1e-9


def test_thompson_sampling_agent(standard_instance):
    ts_agent = ThompsonSamplingAgent(
        agent_id=0,
        budget=3,
        num_users=3,
        click_value=5.0,
        seed=123,
    )

    # Round 1: probe
    assert ts_agent.get_bid(t=1) == [0, 0, 0]

    competing = [1, 1, 0]
    ts_agent.observe([0, 0, 0], [0, 0, 0], [0, 0, 0], competing)

    bids = ts_agent.get_bid(t=2)
    assert sum(bids) <= 3
    for j, bid in enumerate(bids):
        if bid > 0:
            assert bid == competing[j] + 1


def test_greedy_and_eps_greedy_agents(standard_instance):
    greedy = GreedyAgent(agent_id=0, budget=3, num_users=3, click_value=5.0)
    assert greedy.get_bid(t=1) == [0, 0, 0]

    competing = [1, 1, 0]
    greedy.observe([0, 0, 0], [0, 0, 0], [0, 0, 0], competing)
    bids_greedy = greedy.get_bid(t=2)
    assert sum(bids_greedy) <= 3

    eps_agent = EpsGreedyAgent(
        agent_id=0, budget=3, num_users=3, click_value=5.0, epsilon=0.5, seed=42
    )
    assert eps_agent.get_bid(t=1) == [0, 0, 0]
    eps_agent.observe([0, 0, 0], [0, 0, 0], [0, 0, 0], competing)
    bids_eps = eps_agent.get_bid(t=2)
    assert sum(bids_eps) <= 3


def test_cumulative_budget_paced_agent():
    total_budget = 10
    agent = CumulativeBudgetPacedAgent(
        agent_id=0,
        total_budget=total_budget,
        num_users=2,
        total_rounds=5,
        click_value=5.0,
        per_round_max_budget=4,
    )

    assert agent.remaining_budget == 10
    # Round 1 probe
    assert agent.get_bid(t=1) == [0, 0]

    competing = [0, 0]
    # Simulate spending 3
    agent.observe([1, 1], [1, 0], [2, 1], competing)
    assert agent.remaining_budget == 7

    # Simulate depleting remaining budget
    agent.observe([1, 1], [0, 0], [4, 3], competing)
    assert agent.remaining_budget == 0

    # Exhausted budget -> bids 0
    assert agent.get_bid(t=4, last_competing_bids=competing) == [0, 0]


def test_tier2_simulation_single_trial(standard_instance):
    agent = UCBAgent(agent_id=0, budget=3, num_users=3, click_value=5.0)
    sim = Tier2Simulator(
        instance=standard_instance,
        learner_agent=agent,
        learner_id=0,
    )

    T = 50
    trial = sim.run_trial(total_rounds=T, seed=99)

    assert trial.total_rounds == T
    assert len(trial.round_metrics) == T
    assert len(trial.cumulative_regrets) == T

    # In round 1, learner bids 0 everywhere, so instantaneous regret is p_i*
    assert abs(trial.instantaneous_regrets[0] - sim.p_i_star) < 1e-6
    # Cumulative regret is non-decreasing
    for t in range(1, T):
        assert trial.cumulative_regrets[t] >= trial.cumulative_regrets[t - 1] - 1e-9


def test_tier2_multi_seed_experiment(standard_instance):
    agent = ThompsonSamplingAgent(agent_id=0, budget=3, num_users=3, click_value=5.0)
    sim = Tier2Simulator(
        instance=standard_instance,
        learner_agent=agent,
        learner_id=0,
    )

    T = 40
    exp = sim.run_experiment(total_rounds=T, seeds=[10, 20, 30])

    assert exp.num_trials == 3
    assert len(exp.mean_cumulative_regret) == T
    assert len(exp.ci_95_cumulative_regret) == T
    assert exp.final_regret_mean >= 0.0
    assert exp.average_round_profit >= -1e-6


def test_sublinear_regret_growth():
    # Large synthetic instance where optimal strategy gives positive profit
    inst = AuctionInstance(
        num_advertisers=2,
        num_users=3,
        budgets=[2, 2],
        click_probabilities=[
            [0.9, 0.8, 0.1],
            [0.2, 0.1, 0.1],
        ],
        click_value=5.0,
    )
    agent = UCBAgent(agent_id=0, budget=2, num_users=3, click_value=5.0, alpha=1.0)
    sim = Tier2Simulator(instance=inst, learner_agent=agent, learner_id=0)

    T = 300
    res = sim.run_trial(total_rounds=T, seed=42)

    # Reg_t / t should be strictly lower at t=300 than at t=20
    avg_reg_early = res.cumulative_regrets[19] / 20.0
    avg_reg_late = res.cumulative_regrets[-1] / float(T)
    assert avg_reg_late < avg_reg_early


def test_discounted_round_structure(standard_instance):
    agent = UCBAgent(agent_id=0, budget=3, num_users=3, click_value=5.0)
    gamma = 0.9
    sim = Tier2Simulator(
        instance=standard_instance,
        learner_agent=agent,
        learner_id=0,
        round_structure="discounted",
        discount_factor=gamma,
    )

    trial = sim.run_trial(total_rounds=20, seed=123)
    # Total discounted regret must be bounded by p_i^* / (1 - gamma)
    upper_bound = sim.p_i_star / (1.0 - gamma) + 1.0
    assert trial.total_cumulative_regret <= upper_bound


def test_tier2_experiment_serialization(standard_instance):
    agent = UCBAgent(agent_id=0, budget=3, num_users=3, click_value=5.0)
    sim = Tier2Simulator(instance=standard_instance, learner_agent=agent, learner_id=0)
    exp = sim.run_experiment(total_rounds=15, seeds=[1, 2])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        exp.save_json(tmp_path)
        loaded = Tier2ExperimentResult.load_json(tmp_path)
        assert loaded.num_trials == exp.num_trials
        assert loaded.total_rounds == exp.total_rounds
        assert abs(loaded.final_regret_mean - exp.final_regret_mean) < 1e-9
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
