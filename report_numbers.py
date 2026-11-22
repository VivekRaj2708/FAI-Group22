"""Recompute every number quoted in the final report from the committed sweep outputs.

    python report_numbers.py            # prints all tables / in-text numbers
    python report_numbers.py --section t2

Each block below is labelled with the table or paragraph of report/final_report.tex it feeds.
Inputs are the per-cell summaries written by `run_evaluation.py sweep` (results/sweep_*/summary.csv,
one row per (scenario, learner, round structure, m, n, kappa), already averaged over 30 seeds).
Everything except the last block only aggregates. The "lim" block (Limitations paragraph) re-runs one
Tier-2 cell (30 seeds, about 5 s), because the per-seed slopes and per-round curves it needs are not saved.
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEPS = ["tier1", "learners", "discounted", "ucb_alpha", "horizon"]


def load() -> pd.DataFrame:
    frames = []
    for s in SWEEPS:
        d = pd.read_csv(os.path.join(ROOT, "results", f"sweep_{s}", "summary.csv"))
        d["sweep"] = s
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # T=10^3 sweeps share the tier1 rows; keep the first copy of each config (as report.py does).
    key = ["scenario", "learner", "round_structure", "m", "n", "kappa", "horizon"]
    return d.drop_duplicates(subset=key, keep="first")


def sel(d, scenario, learner=None, structure="canonical", horizon=1000):
    q = (d.scenario == scenario) & (d.round_structure == structure) & (d.horizon == horizon)
    if learner is not None:
        q &= d.learner == learner
    return d[q]


def hdr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


T2_LEARNERS = ["thompson", "ucb_a05", "ucb_a1", "ucb", "eps_greedy", "greedy", "paced"]
T3_LEARNERS = ["oracle_br", "thompson", "ucb_a05", "ucb", "eps_greedy", "greedy", "paced"]
HEAD2HEAD = ["thompson", "ucb", "greedy", "eps_greedy"]


def tier1(d):
    hdr("Table 3 (Tier-1 reference profile b*, by nominal kappa; mean over m, n, 30 seeds)")
    t1 = sel(d, "tier1", "fixed_bstar")
    g = t1.groupby("kappa")[["realized_kappa", "t1_efficiency", "t1_welfare_ratio",
                             "t1_revenue", "t1_total_profit", "frac_pure"]].mean()
    print(g.round(3).to_string())
    hdr("Tier 1: fraction of seeds with a certified pure equilibrium")
    print("by m:", t1.groupby("m").frac_pure.mean().round(3).to_dict())
    print(t1.pivot_table(index=["m", "n"], columns="kappa", values="frac_pure").round(2).to_string())
    hard = t1[(t1.m == 5) & (t1.n >= 20) & (t1.kappa >= 0.25)]
    print("m=5, n>=20, kappa>=0.25 frac_pure range: "
          f"{hard.frac_pure.min():.2f}-{hard.frac_pure.max():.2f}")
    ts = sel(d, "tier2", "thompson")
    print(f"Tier-2 seeds excluded (no pure eq): {int(ts.n_excluded.sum())} of "
          f"{int((ts.n_excluded + ts.n_used).sum())}; cells with >=1 usable seed: {(ts.n_used > 0).sum()}")
    hdr("Tier 1 by m (mean over n, kappa)")
    print(t1.groupby("m")[["t1_efficiency", "t1_welfare_ratio", "t1_revenue", "t1_total_profit"]]
          .mean().round(3).to_string())


def tier2(d):
    hdr("Table 4 (Tier 2, canonical, T=1000): mean over cells with a pure equilibrium")
    rows = {}
    for L in T2_LEARNERS:
        x = sel(d, "tier2", L)
        x = x[x.n_used > 0]
        rows[L] = dict(cells=len(x), regret_per_round=x.avg_regret_per_round.mean(),
                       slope=x.regret_slope.mean(), focal_profit=x.focal_profit_last.mean())
    print(pd.DataFrame(rows).T.round(3).to_string())

    piv = sel(d, "tier2").query("n_used > 0").pivot_table(
        index=["m", "n", "kappa"], columns="learner", values="final_regret")
    best = piv[HEAD2HEAD].idxmin(axis=1)
    print("wins among", HEAD2HEAD, best.value_counts().to_dict(), "of", len(best))
    a05 = (piv["ucb_a05"] < piv["thompson"]).sum()
    print(f"UCB alpha=0.5 beats TS in {a05} of {len(piv)} cells")
    cut = 1 - piv["ucb_a05"].mean() / piv["ucb"].mean()
    print(f"alpha 2 -> 0.5 reduces mean final regret by {100 * cut:.0f}%")

    hdr("Tier 2 by n (regret exponent), canonical")
    x = sel(d, "tier2").query("n_used > 0")
    print(x.pivot_table(index="n", columns="learner", values="regret_slope").round(2).to_string())
    hdr("Tier 2 regret per round by kappa (H3), canonical")
    print(x.pivot_table(index="kappa", columns="learner", values="avg_regret_per_round").round(2).to_string())
    hdr("Tier 2 regret per round by m, canonical")
    print(x.pivot_table(index="m", columns="learner", values="avg_regret_per_round").round(2).to_string())

    hdr("Tier 2 long horizon T=10^4 (m in {2,3}, n in {10,20}, kappa=0.5)")
    lo = sel(d, "tier2", horizon=10000)
    print(lo.pivot_table(index=["m", "n"], columns="learner", values="regret_slope").round(2).to_string())
    print(lo.pivot_table(index=["m", "n"], columns="learner", values="avg_regret_per_round").round(2).to_string())
    same = sel(d, "tier2", "ucb").query("kappa == 0.5 and m in [2, 3] and n in [10, 20]")
    print("UCB alpha=2 regret/round on the same 4 cells at T=1000:",
          same.avg_regret_per_round.round(2).tolist())


def tier3(d):
    hdr("Table 5 (Tier 3, canonical, T=1000): final eq gap by m, price of learning (grid mean)")
    rows = {}
    for L in T3_LEARNERS:
        x = sel(d, "tier3", L)
        r = {f"gap_m{m}": x[x.m == m].eq_gap_last.mean() for m in (2, 3, 5)}
        r.update(pol_eff=x.pol_efficiency.mean(), pol_rev=x.pol_revenue.mean(),
                 pol_profit=x.pol_total_profit.mean(), eff_last=x.efficiency_last.mean(),
                 dist=x.eq_distance_last.mean())
        rows[L] = r
    print(pd.DataFrame(rows).T.round(2).to_string())
    t1 = sel(d, "tier1", "fixed_bstar")
    print(f"Tier-1 reference: efficiency {t1.t1_efficiency.mean():.2f}, revenue {t1.t1_revenue.mean():.1f}")

    hdr("Tier 3 final efficiency by kappa, canonical")
    x = sel(d, "tier3")
    print(x.pivot_table(index="kappa", columns="learner", values="efficiency_last").round(2).to_string())
    print("Tier-1:", t1.groupby("kappa").t1_efficiency.mean().round(2).to_dict())

    hdr("Tier 3 long horizon: final eq gap, T=1000 vs T=10^4 (kappa=0.5)")
    for L in ["thompson", "ucb", "ucb_a05", "eps_greedy", "greedy"]:
        a = sel(d, "tier3", L).query("kappa == 0.5 and m in [2, 3] and n in [10, 20]").set_index(["m", "n"]).eq_gap_last
        b = sel(d, "tier3", L, horizon=10000).set_index(["m", "n"]).eq_gap_last
        print(L, pd.DataFrame({"T1e3": a, "T1e4": b}).round(2).to_dict("index"))


def robustness(d):
    hdr("Discounted (gamma=0.999), Tier 2")
    x = sel(d, "tier2", structure="discounted").query("n_used > 0")
    print("mean exponent:", x.groupby("learner").regret_slope.mean().round(2).to_dict())
    piv = x.pivot_table(index=["m", "n", "kappa"], columns="learner", values="final_regret")
    print("wins:", piv[HEAD2HEAD].idxmin(axis=1).value_counts().to_dict(), "of", len(piv))

    hdr("Table 6 (Cumulative budget, rho=0.5): Tier 2 vs fluid LP benchmark")
    x = sel(d, "tier2", structure="cumulative").query("n_used > 0")
    print(x.groupby("learner")[["regret_slope", "final_regret", "focal_profit_last",
                                "frac_budget_exhausted"]].mean().round(2).to_string())
    piv = x.pivot_table(index=["m", "n", "kappa"], columns="learner", values="final_regret")
    print("wins (all 5):", piv.idxmin(axis=1).value_counts().to_dict(), "of", len(piv))
    hdr("Cumulative, Tier 3")
    x = sel(d, "tier3", structure="cumulative")
    out = x.pivot_table(index="learner", columns="m", values="eq_gap_last").round(2)
    out["pol_rev"] = x.groupby("learner").pol_revenue.mean().round(1)
    out["pol_eff"] = x.groupby("learner").pol_efficiency.mean().round(2)
    print(out.to_string())


def limitations(d):
    hdr("Limitations: where the excluded Tier-2 seeds are (no pure equilibrium)")
    x = sel(d, "tier2", "thompson")
    dropped = x.n_excluded.sum()
    big = x.query("m == 5 and n >= 20").n_excluded.sum()
    print(f"excluded {dropped} seeds; {big} ({big / dropped:.0%}) of them in m=5, n>=20")

    hdr("Limitations: Tier 2 on all 1440 seeds, including those without a pure equilibrium")
    # results/sweep_all_seeds was run with --include-mixed: on non-pure seeds the opponents play the
    # final best-response-dynamics profile. Regret is against the best response to those fixed bids.
    ps = pd.read_csv(os.path.join(ROOT, "results", "sweep_all_seeds", "per_seed.csv"))
    ps = ps[(ps.scenario == "tier2") & ps.learner.isin(HEAD2HEAD)]
    ps["is_pure"] = ps.is_pure.astype(str).str.lower() == "true"
    for name, part in (("all seeds", ps), ("non-pure seeds only", ps[~ps.is_pure])):
        n_seeds = part[part.learner == "thompson"].shape[0]
        cells = part.pivot_table(index=["m", "n", "kappa"], columns="learner", values="final_regret")
        wins = cells[HEAD2HEAD].idxmin(axis=1).value_counts().to_dict()
        rpr = part.groupby("learner").avg_regret_per_round.mean().round(2).to_dict()
        slope = part.groupby("learner").regret_slope.mean().round(2).to_dict()
        print(f"{name}: {n_seeds} seeds in {len(cells)} cells; cell wins {wins}")
        print(f"  regret per round {rpr}")
        print(f"  regret exponent  {slope}")

    hdr("Limitations: UCB (alpha=2) regret exponent above 1 (re-runs one cell, ~5 s)")
    x = sel(d, "tier2", "ucb")
    print("cells with exponent > 1:", x.query("regret_slope > 1")[["m", "n", "kappa", "regret_slope"]]
          .round(2).to_dict("records"))
    import numpy as np
    from src.eval.experiment import ExperimentConfig, run_seed
    cfg = ExperimentConfig(m=2, n=10, kappa=0.5, scenario="tier2", learner="ucb", horizon=1000)
    runs = [r for r in (run_seed(cfg, s, keep_trace=True) for s in cfg.seeds) if not r.summary["excluded"]]
    slopes = np.array([r.summary["regret_slope"] for r in runs])
    windows = np.mean([r.trace.regret[:, 0].reshape(10, 100).mean(axis=1) for r in runs], axis=0)
    print(f"m=2, n=10, kappa=0.5: {len(runs)} seeds, mean exponent {slopes.mean():.2f}, "
          f"median {np.median(slopes):.2f}, seeds > 1: {(slopes > 1).sum()}")
    print("mean regret per round in each 100-round window:", windows.round(2).tolist())


SECTIONS = {"t1": tier1, "t2": tier2, "t3": tier3, "rob": robustness, "lim": limitations}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=sorted(SECTIONS), nargs="*")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    data = load()
    for name in (args.section or ["t1", "t2", "t3", "rob", "lim"]):
        SECTIONS[name](data)
