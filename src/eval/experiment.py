"""
Experiment runner: one configuration x many seeds, with Tier-1 reference
values, per-seed summaries and across-seed curves with 95% CIs.

Scenarios (proposal Section 4):
* ``tier1``  - every advertiser plays its Tier-1 reference bids b* (sanity
               check: regret = equilibrium gap of b*, distance = 0).
* ``tier2``  - advertiser ``focal_agent`` runs ``learner``; all others play
               their fixed Tier-1 bids. Instances whose Tier-1 profile is not
               a pure equilibrium are excluded (proposal Section 4.2).
* ``tier3``  - every advertiser runs ``learner`` concurrently.
* ``custom`` - ``agents`` lists one registry name per advertiser.
"""
from dataclasses import dataclass, field, asdict
from functools import lru_cache
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.registry import AgentContext, make_agent
from src.env.instance import AuctionInstance, generate_synthetic_instance
from src.eval.simulator import SimulationConfig, SimulationTrace, run_simulation
from src.eval.stats import loglog_slope, mean_ci
from src.tier1.baseline import Tier1Benchmark, Tier1Solver

SCENARIOS = ("tier1", "tier2", "tier3", "custom")

# Market-level curves stored for plotting (focal-agent curves are added separately).
CURVE_METRICS = ("cum_regret", "regret", "eq_gap", "eq_distance", "efficiency",
                 "revenue", "welfare", "total_profit", "focal_profit")


@dataclass
class ExperimentConfig:
    m: int = 2
    n: int = 5
    kappa: float = 0.5
    click_value: float = 5.0
    beta_params: Tuple[float, float] = (2.0, 5.0)
    scenario: str = "tier2"
    learner: str = "oracle_br"
    learner_kwargs: Dict[str, Any] = field(default_factory=dict)
    agents: Optional[List[str]] = None       # custom scenario only
    focal_agent: int = 0
    horizon: int = 1000
    round_structure: str = "canonical"
    gamma: float = 0.999
    cumulative_rho: float = 0.5
    seeds: List[int] = field(default_factory=lambda: list(range(30)))
    window_frac: float = 0.1                 # last-window fraction for "final" metrics
    prefer_exact: bool = True
    exclude_mixed: Optional[bool] = None     # default: True for tier2 only
    on_infeasible: str = "zero"

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if self.scenario == "custom" and (self.agents is None or len(self.agents) != self.m):
            raise ValueError("custom scenario needs one agent name per advertiser")
        self.beta_params = tuple(self.beta_params)

    @property
    def label(self) -> str:
        who = "+".join(self.agents) if self.scenario == "custom" else (
            "bstar" if self.scenario == "tier1" else self.learner)
        return (f"{self.scenario}|{who}|{self.round_structure}|m{self.m}|n{self.n}"
                f"|k{self.kappa:g}|T{self.horizon}")

    def should_exclude_mixed(self) -> bool:
        return self.scenario == "tier2" if self.exclude_mixed is None else self.exclude_mixed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["beta_params"] = list(self.beta_params)
        d["label"] = self.label
        return d


# --------------------------------------------------------------------------- Tier 1

@lru_cache(maxsize=256)
def _tier1_cached(m, n, kappa, v, beta, seed, prefer_exact) -> Tuple[AuctionInstance, Tier1Benchmark]:
    inst = generate_synthetic_instance(num_advertisers=m, num_users=n, budget_tightness=kappa,
                                       click_value=v, seed=seed, beta_params=beta)
    return inst, Tier1Solver(inst).solve(prefer_exact=prefer_exact, seed=seed)


def tier1_reference(cfg: ExperimentConfig, seed: int) -> Tuple[AuctionInstance, Tier1Benchmark]:
    return _tier1_cached(cfg.m, cfg.n, float(cfg.kappa), float(cfg.click_value),
                         tuple(cfg.beta_params), int(seed), bool(cfg.prefer_exact))


def realized_kappa(inst: AuctionInstance, b_star: List[List[int]]) -> float:
    """kappa_i = B_i / sum_j (c*_{-i,j} + 1), averaged over advertisers."""
    bs = np.asarray(b_star)
    vals = []
    for i in range(inst.num_advertisers):
        others = np.delete(bs, i, axis=0)
        c = others.max(axis=0) if others.size else np.zeros(inst.num_users)
        vals.append(inst.budgets[i] / float((c + 1).sum()))
    return float(np.mean(vals))


# --------------------------------------------------------------------------- agents

def build_agents(cfg: ExperimentConfig, inst: AuctionInstance, bench: Tier1Benchmark, seed: int):
    sim = SimulationConfig(horizon=cfg.horizon, round_structure=cfg.round_structure,
                           cumulative_rho=cfg.cumulative_rho)
    totals = sim.total_budgets(inst.budgets)
    if cfg.scenario == "tier1":
        names = ["fixed_bstar"] * cfg.m
    elif cfg.scenario == "tier2":
        names = ["fixed_bstar"] * cfg.m
        names[cfg.focal_agent] = cfg.learner
    elif cfg.scenario == "tier3":
        names = [cfg.learner] * cfg.m
    else:
        names = list(cfg.agents)

    agents = []
    for i, name in enumerate(names):
        ctx = AgentContext(
            agent_id=i, budget=inst.budgets[i], num_users=inst.num_users,
            click_value=inst.click_value, horizon=cfg.horizon, seed=seed * 1000 + i,
            round_structure=cfg.round_structure,
            total_budget=None if totals is None else totals[i],
            reference_bids=list(bench.reference_bid_profile[i]),
            true_click_probs=list(inst.click_probabilities[i]),
        )
        kwargs = cfg.learner_kwargs if name == cfg.learner else {}
        agents.append(make_agent(name, ctx, **kwargs))
    return agents


# --------------------------------------------------------------------------- one seed

def curve_grid(T: int, points: int = 200) -> np.ndarray:
    """1-indexed rounds at which curves are stored (all rounds if T is small)."""
    if T <= points:
        return np.arange(1, T + 1)
    g = np.concatenate([np.geomspace(1, T, points // 2), np.linspace(1, T, points // 2)])
    return np.unique(np.clip(np.round(g).astype(int), 1, T))


@dataclass
class SeedResult:
    seed: int
    summary: Dict[str, Any]
    curves: Dict[str, List[float]]   # metric -> values on curve_grid(T)
    trace: Optional[SimulationTrace] = None


def summarize_trace(cfg: ExperimentConfig, tr: SimulationTrace, inst: AuctionInstance,
                    bench: Tier1Benchmark) -> Dict[str, Any]:
    T = cfg.horizon
    W = max(1, int(math.ceil(cfg.window_frac * T)))
    i0 = cfg.focal_agent
    cum = tr.cumulative_regret()
    total_profit = tr.profits.sum(axis=1)
    last = slice(T - W, T)
    t1_profit = float(sum(bench.advertiser_profits))

    s: Dict[str, Any] = {
        "final_regret": float(cum[-1, i0]),
        "final_regret_mean_agents": float(cum[-1].mean()),
        "avg_regret_per_round": float(cum[-1, i0] / tr.weights.sum()),
        "regret_slope": loglog_slope(np.arange(1, T + 1), cum[:, i0]),
        "eq_gap_last": float(tr.eq_gap[last].mean()),
        "eq_distance_last": float(tr.eq_distance[last].mean()),
        "efficiency_last": float(tr.efficiency[last].mean()),
        "realized_efficiency_last": float(tr.realized_efficiency[last].mean()),
        "revenue_last": float(tr.revenue[last].mean()),
        "welfare_last": float(tr.welfare[last].mean()),
        "total_profit_last": float(total_profit[last].mean()),
        "focal_profit_last": float(tr.profits[last, i0].mean()),
        "focal_discounted_profit": float((tr.profits[:, i0] * tr.weights).sum()),
        "t1_total_profit": t1_profit,
        "t1_revenue": float(bench.platform_revenue),
        "t1_efficiency": float(bench.allocation_efficiency),
        "t1_welfare": float(bench.expected_welfare),
        "t1_eq_gap": float(bench.equilibrium_gap),
        "planner_welfare": float(bench.social_planner_welfare),
        "realized_kappa": realized_kappa(inst, bench.reference_bid_profile),
        "budget": int(inst.budgets[0]),
        "violations": int(tr.violations.sum()),
        "is_pure": bool(bench.is_pure_equilibrium),
        "solver_method": bench.solver_method,
    }
    # Price of learning: Tier-1 quantity minus learned quantity over the last window.
    s["pol_total_profit"] = t1_profit - s["total_profit_last"]
    s["pol_revenue"] = s["t1_revenue"] - s["revenue_last"]
    s["pol_efficiency"] = s["t1_efficiency"] - s["efficiency_last"]
    s["pol_welfare"] = s["t1_welfare"] - s["welfare_last"]
    pw = s["planner_welfare"]
    s["t1_welfare_ratio"] = s["t1_welfare"] / pw if pw > 0 else float("nan")
    s["welfare_ratio_last"] = s["welfare_last"] / pw if pw > 0 else float("nan")
    if tr.exhausted_round:
        ex = [r for r in tr.exhausted_round if r is not None]
        s["frac_budget_exhausted"] = len(ex) / len(tr.exhausted_round)
    return s


def run_seed(cfg: ExperimentConfig, seed: int, keep_trace: bool = False) -> SeedResult:
    inst, bench = tier1_reference(cfg, seed)
    if cfg.should_exclude_mixed() and not bench.is_pure_equilibrium:
        return SeedResult(seed=seed, summary={"excluded": True, "is_pure": False}, curves={})

    agents = build_agents(cfg, inst, bench, seed)
    sim = SimulationConfig(horizon=cfg.horizon, round_structure=cfg.round_structure, gamma=cfg.gamma,
                           cumulative_rho=cfg.cumulative_rho, on_infeasible=cfg.on_infeasible,
                           seed=seed + 7919)
    tr = run_simulation(inst, agents, sim, reference_profile=bench.reference_bid_profile)
    summary = summarize_trace(cfg, tr, inst, bench)
    summary["excluded"] = False

    idx = curve_grid(cfg.horizon) - 1
    cum = tr.cumulative_regret()[:, cfg.focal_agent]
    full = {
        "cum_regret": cum,
        "regret": tr.regret[:, cfg.focal_agent],
        "eq_gap": tr.eq_gap,
        "eq_distance": tr.eq_distance,
        "efficiency": tr.efficiency,
        "revenue": tr.revenue,
        "welfare": tr.welfare,
        "total_profit": tr.profits.sum(axis=1),
        "focal_profit": tr.profits[:, cfg.focal_agent],
    }
    curves = {k: v[idx].tolist() for k, v in full.items()}
    return SeedResult(seed=seed, summary=summary, curves=curves, trace=tr if keep_trace else None)


# --------------------------------------------------------------------------- aggregation

NON_NUMERIC = {"solver_method", "excluded", "is_pure"}


def aggregate(cfg: ExperimentConfig, results: List[SeedResult], confidence: float = 0.95) -> Dict[str, Any]:
    used = [r for r in results if not r.summary.get("excluded")]
    out: Dict[str, Any] = {
        "label": cfg.label,
        "config": cfg.to_dict(),
        "n_seeds": len(results),
        "n_used": len(used),
        "n_excluded": len(results) - len(used),
        "summary": {},
        "curves": {"t": curve_grid(cfg.horizon).tolist()},
    }
    if not used:
        return out
    keys = [k for k in used[0].summary if k not in NON_NUMERIC]
    for k in keys:
        vals = [r.summary.get(k, float("nan")) for r in used]
        mu, hw = mean_ci(np.asarray(vals, dtype=float), confidence=confidence)
        out["summary"][k] = {"mean": float(mu), "ci": float(hw)}
    out["summary"]["frac_pure"] = {"mean": float(np.mean([r.summary["is_pure"] for r in used])), "ci": 0.0}
    for k in CURVE_METRICS:
        arr = np.asarray([r.curves[k] for r in used], dtype=float)
        mu, hw = mean_ci(arr, axis=0, confidence=confidence)
        out["curves"][k] = {"mean": np.atleast_1d(mu).tolist(), "ci": np.atleast_1d(hw).tolist()}
    return out


def run_experiment(cfg: ExperimentConfig, confidence: float = 0.95) -> Tuple[Dict[str, Any], List[SeedResult]]:
    results = [run_seed(cfg, s) for s in cfg.seeds]
    return aggregate(cfg, results, confidence), results
