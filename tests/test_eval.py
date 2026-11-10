import json
import os
import random

import numpy as np
import pytest

from src.agents.base import BaseAdvertiserAgent
from src.agents.reference import FixedProfileAgent, OracleBestResponseAgent
from src.agents.registry import AgentContext, available_agents, make_agent, register_agent
from src.env.auction import BatchedFirstPriceAuction
from src.env.instance import generate_synthetic_instance
from src.eval import metrics as M
from src.eval.experiment import ExperimentConfig, run_experiment, run_seed, tier1_reference
from src.eval.simulator import SimulationConfig, run_simulation
from src.eval.stats import loglog_slope, mean_ci
from src.eval.sweep import SweepSpec, run_sweep
from src.tier1.equilibrium import compute_best_response_for_advertiser


def _instance(m=3, n=6, kappa=0.5, seed=1):
    return generate_synthetic_instance(num_advertisers=m, num_users=n, budget_tightness=kappa,
                                       click_value=5.0, seed=seed)


def _random_profile(inst, rng):
    prof = []
    for b in inst.budgets:
        row = [0] * inst.num_users
        rem = b
        for j in rng.sample(range(inst.num_users), inst.num_users):
            x = rng.randint(0, rem)
            row[j] = x
            rem -= x
        prof.append(row)
    return prof


def test_vectorised_metrics_match_environment():
    inst = _instance()
    auc = BatchedFirstPriceAuction(inst)
    f = np.asarray(inst.click_probabilities)
    rng = random.Random(0)
    mask = M.best_advertiser_mask(f)
    for _ in range(50):
        prof = _random_profile(inst, rng)
        b = np.asarray(prof)
        q = M.winning_probabilities(b)
        assert np.allclose(q, auc.compute_winning_probabilities(prof))
        assert np.allclose(M.expected_profits(b, f, inst.click_value), auc.expected_utilities(prof))
        assert M.platform_revenue(b, q) == pytest.approx(auc.platform_revenue(prof))
        assert M.expected_welfare(f, inst.click_value, q) == pytest.approx(auc.expected_welfare(prof))
        assert M.allocation_efficiency(q, mask) == pytest.approx(auc.allocation_efficiency(prof))
        assert np.array_equal(M.competing_bids(b), np.asarray(auc.get_competing_bids(prof)))


def test_best_response_oracle_matches_tier1():
    inst = _instance()
    auc = BatchedFirstPriceAuction(inst)
    f = np.asarray(inst.click_probabilities)
    oracle = M.BestResponseOracle(f, inst.click_value)
    rng = random.Random(3)
    for _ in range(30):
        prof = _random_profile(inst, rng)
        comp = M.competing_bids(np.asarray(prof))
        for i in range(inst.num_advertisers):
            opp = [prof[k] for k in range(inst.num_advertisers) if k != i]
            _, ref_val = compute_best_response_for_advertiser(auc, i, opp)
            assert oracle.value(i, comp[i], inst.budgets[i]) == pytest.approx(ref_val)


def test_fluid_value_upper_bounds_integer_knapsack():
    rng = np.random.default_rng(0)
    for _ in range(50):
        f_i = rng.uniform(0, 1, 8)
        c = rng.integers(0, 3, 8)
        oracle = M.BestResponseOracle(f_i[None, :], 5.0)
        cap = int(rng.integers(1, 10))
        assert M.fluid_knapsack_value(f_i, c, cap, 5.0) >= oracle.value(0, c, cap) - 1e-9


def test_tier1_scenario_reproduces_reference_metrics():
    cfg = ExperimentConfig(m=2, n=5, kappa=0.5, scenario="tier1", horizon=50, seeds=[0, 1, 2])
    for seed in cfg.seeds:
        _, bench = tier1_reference(cfg, seed)
        res = run_seed(cfg, seed, keep_trace=True)
        tr = res.trace
        assert np.allclose(tr.eq_distance, 0.0)
        assert tr.eq_gap[0] == pytest.approx(bench.equilibrium_gap, abs=1e-9)
        assert tr.revenue[0] == pytest.approx(bench.platform_revenue)
        assert tr.efficiency[0] == pytest.approx(bench.allocation_efficiency)
        assert np.allclose(tr.profits[0], bench.advertiser_profits)
        if bench.is_pure_equilibrium:
            assert res.summary["final_regret"] == pytest.approx(0.0, abs=1e-9)


def test_tier2_oracle_learner_has_zero_regret_after_probe():
    cfg = ExperimentConfig(m=2, n=6, kappa=0.5, scenario="tier2", learner="oracle_br",
                           horizon=60, seeds=list(range(4)))
    agg, results = run_experiment(cfg)
    for r in results:
        if not r.summary["excluded"]:
            # best-responding to the (stationary) b*_{-i} is optimal from round 2 on
            assert r.summary["eq_gap_last"] == pytest.approx(0.0, abs=1e-9)
    assert agg["n_used"] + agg["n_excluded"] == 4


class _Greedy(BaseAdvertiserAgent):
    """Bids its whole budget on user 0 every round."""

    def get_bid(self, t, last_competing_bids=None):
        b = np.zeros(self.num_users, dtype=int)
        b[0] = self.budget
        return b

    def observe(self, won, clicks, payments, competing_bids):
        self.last_payment = int(np.sum(payments))

    def reset(self):
        self.last_payment = 0


class _Overbidder(_Greedy):
    def get_bid(self, t, last_competing_bids=None):
        return np.full(self.num_users, self.budget, dtype=int)


def test_infeasible_bids_are_zeroed_or_raise():
    inst = _instance(m=2, n=4)
    agents = [_Overbidder(0, inst.budgets[0], 4), _Greedy(1, inst.budgets[1], 4)]
    tr = run_simulation(inst, agents, SimulationConfig(horizon=10))
    assert tr.violations.tolist() == [10, 0]
    assert np.all(tr.spend[:, 0] == 0)
    with pytest.raises(ValueError):
        run_simulation(inst, agents, SimulationConfig(horizon=1, on_infeasible="raise"))


def test_cumulative_budget_excludes_after_exhaustion():
    inst = _instance(m=2, n=4)
    cfg = SimulationConfig(horizon=40, round_structure="cumulative", cumulative_rho=0.25)
    totals = cfg.total_budgets(inst.budgets)
    agents = [_Greedy(0, inst.budgets[0], 4), _Greedy(1, inst.budgets[1], 4)]
    tr = run_simulation(inst, agents, cfg)
    assert np.all(tr.spend.sum(axis=0) <= np.asarray(totals))
    assert any(r is not None for r in tr.exhausted_round)
    first = min(r for r in tr.exhausted_round if r is not None)
    assert tr.spend[first:].sum() >= 0 and np.all(tr.spend[-1] == 0)
    # cumulative regret is measured against the per-round fluid benchmark
    assert np.all(tr.br_values >= -1e-12)


def test_oracle_respects_cumulative_budget():
    inst = _instance(m=2, n=8, kappa=1.0)
    cfg = SimulationConfig(horizon=30, round_structure="cumulative", cumulative_rho=0.3, on_infeasible="raise")
    totals = cfg.total_budgets(inst.budgets)
    agents = [OracleBestResponseAgent(i, inst.budgets[i], 8, inst.click_probabilities[i], 5.0, totals[i])
              for i in range(2)]
    tr = run_simulation(inst, agents, cfg)  # would raise on any infeasible bid
    assert np.all(tr.spend.sum(axis=0) <= np.asarray(totals))


def test_discount_weights():
    inst = _instance(m=2, n=4)
    agents = [FixedProfileAgent(i, inst.budgets[i], 4, [0, 0, 0, 0]) for i in range(2)]
    tr = run_simulation(inst, agents, SimulationConfig(horizon=5, round_structure="discounted", gamma=0.5))
    assert np.allclose(tr.weights, [1, 0.5, 0.25, 0.125, 0.0625])


def test_simulation_is_deterministic():
    cfg = ExperimentConfig(m=3, n=5, kappa=0.5, scenario="tier3", learner="random", horizon=40, seeds=[7])
    a = run_seed(cfg, 7).summary
    b = run_seed(cfg, 7).summary
    assert a == b


def test_mean_ci_and_slope():
    mu, hw = mean_ci([1.0, 2.0, 3.0])
    assert mu == pytest.approx(2.0)
    assert hw == pytest.approx(4.302652729911275 / np.sqrt(3), rel=1e-3)
    mu, hw = mean_ci([[1.0, np.nan], [3.0, 5.0]], axis=0)
    assert np.allclose(mu, [2.0, 5.0]) and hw[1] == 0.0
    t = np.arange(1, 5001)
    assert loglog_slope(t, np.sqrt(t)) == pytest.approx(0.5, abs=1e-6)
    assert loglog_slope(t, 3.0 * t) == pytest.approx(1.0, abs=1e-6)


def test_registry_plugin_and_custom_scenario():
    assert {"fixed_bstar", "oracle_br", "random"} <= set(available_agents())
    if "_test_greedy" not in available_agents():
        register_agent("_test_greedy")(lambda ctx, **kw: _Greedy(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value))
    with pytest.raises(KeyError):
        make_agent("does_not_exist", AgentContext(0, 1, 1, 1.0, 1, 0))
    cfg = ExperimentConfig(m=2, n=4, scenario="custom", agents=["_test_greedy", "fixed_bstar"],
                           horizon=20, seeds=[0, 1])
    agg, _ = run_experiment(cfg)
    assert agg["n_used"] == 2 and "final_regret" in agg["summary"]


def test_sweep_and_report_outputs(tmp_path):
    spec = SweepSpec(ms=[2], ns=[4], kappas=[0.5], structures=["canonical", "cumulative"],
                     runs=[("tier1", "-"), ("tier3", "oracle_br")], horizon=30, num_seeds=2)
    out = tmp_path / "sweep"
    aggs = run_sweep(spec, str(out), workers=1, verbose=False)
    assert len(aggs) == 3  # tier1 canonical only + tier3 x 2 structures
    for name in ("summary.csv", "per_seed.csv", "aggregates.json", "spec.json"):
        assert (out / name).exists()
    assert len(json.loads((out / "aggregates.json").read_text())) == 3

    from src.eval.report import build_report
    manifest = build_report(str(out), str(tmp_path / "report"), fig_ext="png")
    text = open(manifest, encoding="utf-8").read()
    assert r"\input{generated/tier1_m2}" in text
    assert os.path.exists(tmp_path / "report" / "figures" / "tier3_curves_eq_gap.png")


TIER2_NAMES = ("ucb", "thompson", "greedy", "eps_greedy", "paced")


@pytest.mark.parametrize("name", TIER2_NAMES)
@pytest.mark.parametrize("structure", ["canonical", "cumulative"])
def test_tier2_learners_run_through_harness(name, structure):
    assert set(TIER2_NAMES) <= set(available_agents())
    cfg = ExperimentConfig(m=2, n=5, scenario="tier2", learner=name, horizon=60,
                           round_structure=structure, seeds=[0, 1])
    used = [r for r in (run_seed(cfg, s, keep_trace=True) for s in cfg.seeds) if not r.summary["excluded"]]
    assert used
    for r in used:
        # Per-round budgets: every learner stays feasible. Cumulative budgets: only
        # the paced agent tracks remaining spend; the others (and fixed-b*
        # opponents) can overbid near exhaustion and the harness zeroes those bids.
        if structure == "canonical" or name == "paced":
            assert r.trace.violations[cfg.focal_agent] == 0
        assert r.summary["final_regret"] >= -1e-9  # BR is an upper bound per round


def test_tier2_learners_are_deterministic_and_beat_random():
    def regret(name):
        cfg = ExperimentConfig(m=2, n=8, scenario="tier2", learner=name, horizon=600, seeds=[0, 1, 2])
        return run_experiment(cfg)[0]["summary"]["final_regret"]["mean"]

    assert regret("thompson") == regret("thompson")
    assert regret("ucb") < regret("random")
    assert regret("thompson") < regret("random")
