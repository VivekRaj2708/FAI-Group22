"""Smoke tests for the live demo (run_demo.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_demo  # noqa: E402


def _market(**overrides):
    args = run_demo.argparse.Namespace(scenario="tier2", learner="thompson", agents=None, m=2, n=6,
                                       kappa=0.5, v=5.0, T=40, seed=1)
    for k, v in overrides.items():
        setattr(args, k, v)
    return run_demo.build_market(args)


def test_step_market_produces_feasible_rounds():
    cfg, inst, bench, agents, names = _market()
    snaps = list(run_demo.step_market(inst, bench, agents, 40, seed=1))
    assert len(snaps) == 40
    for s in snaps:
        assert s.bids.shape == (2, 6)
        assert (s.bids.sum(axis=1) <= np.asarray(inst.budgets)).all()
        assert (s.wins.sum(axis=0) <= 1).all()          # at most one winner per user
        assert (s.clicks <= s.wins).all()               # clicks only on won users
        assert s.eq_gap >= 0 and 0 <= s.efficiency <= 1
    assert (snaps[0].bids[0] == 0).all()               # round-1 probe for the learner
    assert names == ["thompson", "fixed b*"]


def test_custom_market_and_headless_cli(capsys):
    _, _, _, agents, names = _market(agents=["ucb", "greedy", "thompson"])
    assert len(agents) == 3 and names == ["ucb", "greedy", "thompson"]
    assert run_demo.main(["--headless", "--scenario", "tier3", "--learner", "ucb", "--T", "20", "--every", "10"]) == 0
    out = capsys.readouterr().out
    assert "cumulative regret" in out
