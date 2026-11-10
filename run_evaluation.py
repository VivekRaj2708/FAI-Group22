#!/usr/bin/env python3
"""
Evaluation harness CLI (Phases 4 & 5).

Examples
--------
  # list agents the harness can run (Tier-2/3 agents appear once registered)
  python run_evaluation.py agents

  # one configuration, 30 seeds, curves + summary
  python run_evaluation.py run --scenario tier2 --learner ucb --m 2 --n 10 --kappa 0.5 --T 2000

  # sensitivity sweep over (m, n, kappa) x round structure
  python run_evaluation.py sweep --preset tier1 --out results/sweep_tier1
  python run_evaluation.py sweep --preset paper --learners ucb thompson greedy --out results/sweep_paper

  # build report tables/figures from a sweep
  python run_evaluation.py report --sweep-dir results/sweep_tier1 --report-dir Doc/report
"""
import argparse
import json
import os
import sys

from src.agents.registry import available_agents
from src.eval.experiment import ExperimentConfig, run_experiment
from src.eval.plotting import plot_curves
from src.eval.simulator import ROUND_STRUCTURES
from src.eval.sweep import PRESETS, SweepSpec, run_sweep, write_outputs


def _kv(s: str):
    return json.loads(s) if s else {}


def cmd_agents(_):
    for name in available_agents():
        print(name)


def cmd_run(a):
    cfg = ExperimentConfig(
        m=a.m, n=a.n, kappa=a.kappa, click_value=a.v, scenario=a.scenario, learner=a.learner,
        learner_kwargs=_kv(a.learner_kwargs), agents=a.agents, horizon=a.T, round_structure=a.structure,
        gamma=a.gamma, cumulative_rho=a.rho, seeds=list(range(a.seed_offset, a.seed_offset + a.seeds)),
        prefer_exact=not a.no_exact,
    )
    print(f"Running {cfg.label} over {len(cfg.seeds)} seeds ...")
    agg, _ = run_experiment(cfg)
    os.makedirs(a.out, exist_ok=True)
    write_outputs([agg], a.out)
    print(f"seeds used: {agg['n_used']} (excluded non-pure Tier-1: {agg['n_excluded']})")
    for k, s in agg["summary"].items():
        print(f"  {k:28s} {s['mean']:12.4f}  +/- {s['ci']:.4f}")
    if agg["n_used"]:
        for metric in ("cum_regret", "eq_gap", "eq_distance", "efficiency", "total_profit"):
            plot_curves([agg], metric, os.path.join(a.out, f"{metric}.png"), title=cfg.label)
    print(f"Wrote {a.out}/summary.csv, aggregates.json and curve PNGs")


def cmd_sweep(a):
    spec = SweepSpec.from_dict(PRESETS[a.preset].to_dict())
    if a.learners:
        spec.runs = [r for r in spec.runs if r[0] == "tier1"] + \
                    [(sc, l) for sc in a.scenarios for l in a.learners]
    for field, val in (("ms", a.m), ("ns", a.n), ("kappas", a.kappa), ("structures", a.structures),
                       ("horizon", a.T), ("num_seeds", a.seeds), ("click_value", a.v)):
        if val is not None:
            setattr(spec, field, val)
    if a.no_exact:
        spec.prefer_exact = False
    missing = sorted({l for sc, l in spec.runs if sc != "tier1"} - set(available_agents()))
    if missing:
        sys.exit(f"Agents not registered yet: {missing}. Available: {available_agents()}")
    print(f"Sweep '{a.preset}': {len(spec.all_configs())} configurations x {spec.num_seeds} seeds, T={spec.horizon}")
    run_sweep(spec, a.out, workers=a.workers)


def cmd_report(a):
    from src.eval.report import build_report
    path = build_report(a.sweep_dir, a.report_dir, fig_ext=a.fig_ext)
    print(f"Wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("agents", help="list registered agents").set_defaults(fn=cmd_agents)

    r = sub.add_parser("run", help="one configuration over many seeds")
    r.add_argument("--scenario", choices=["tier1", "tier2", "tier3", "custom"], default="tier2")
    r.add_argument("--learner", default="oracle_br")
    r.add_argument("--learner-kwargs", default="", help='JSON, e.g. \'{"alpha": 2}\'')
    r.add_argument("--agents", nargs="+", help="custom scenario: one agent name per advertiser")
    r.add_argument("--m", type=int, default=2)
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--kappa", type=float, default=0.5)
    r.add_argument("--v", type=float, default=5.0)
    r.add_argument("--T", type=int, default=1000)
    r.add_argument("--structure", choices=ROUND_STRUCTURES, default="canonical")
    r.add_argument("--gamma", type=float, default=0.999)
    r.add_argument("--rho", type=float, default=0.5, help="cumulative budget = rho * T * B_i")
    r.add_argument("--seeds", type=int, default=30)
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--no-exact", action="store_true", help="skip support enumeration in Tier 1")
    r.add_argument("--out", default="results/run")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep", help="sensitivity sweep over (m, n, kappa) x structure")
    s.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    s.add_argument("--learners", nargs="+", help="override learners (registry names)")
    s.add_argument("--scenarios", nargs="+", default=["tier2", "tier3"], help="used with --learners")
    s.add_argument("--m", type=int, nargs="+")
    s.add_argument("--n", type=int, nargs="+")
    s.add_argument("--kappa", type=float, nargs="+")
    s.add_argument("--structures", nargs="+", choices=ROUND_STRUCTURES)
    s.add_argument("--T", type=int)
    s.add_argument("--seeds", type=int)
    s.add_argument("--v", type=float)
    s.add_argument("--no-exact", action="store_true")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--out", default="results/sweep")
    s.set_defaults(fn=cmd_sweep)

    rp = sub.add_parser("report", help="build LaTeX tables + figures from a sweep")
    rp.add_argument("--sweep-dir", required=True, nargs="+", help="one or more sweep dirs (first wins on duplicate configs)")
    rp.add_argument("--report-dir", default="Doc/report")
    rp.add_argument("--fig-ext", default="pdf", choices=["pdf", "png"])
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
