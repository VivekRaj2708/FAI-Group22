#!/usr/bin/env python3
"""
Automated Convergence Data Collection Pipeline for Tier 3 (Phase 3 Deliverable).
Systematically collects raw multi-agent convergence logs as the numbers of
advertisers (m), users (n), budget tightness (kappa), and populations vary.
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env.instance import generate_synthetic_instance
from src.tier1.baseline import Tier1Solver
from src.tier3.simulation import MultiAgentSimulator, Tier3ExperimentResult
from run_tier3 import plot_convergence_curves


def run_single_benchmark(
    name: str,
    output_dir: str,
    m: int = 2,
    n: int = 4,
    kappa: float = 0.5,
    v: float = 5.0,
    population: str = "all_ucb",
    round_structure: str = "canonical",
    discount_factor: float = 0.99,
    rounds: int = 200,
    trials: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"\n>>> Running Experiment: {name}")
    print(f"    Params: m={m}, n={n}, kappa={kappa}, pop={population}, struct={round_structure}, T={rounds}, trials={trials}")

    t0 = time.time()
    instance = generate_synthetic_instance(
        num_advertisers=m,
        num_users=n,
        budget_tightness=kappa,
        click_value=v,
        seed=seed,
    )

    solver = Tier1Solver(instance)
    tier1_bm = solver.solve()

    sim = MultiAgentSimulator(
        instance=instance,
        population_type=population,
        reference_benchmark=tier1_bm,
        round_structure=round_structure,
        discount_factor=discount_factor,
    )

    seeds = [seed + 101 * k for k in range(trials)]
    exp_res = sim.run_experiment(total_rounds=rounds, seeds=seeds)

    # Save outputs
    json_path = os.path.join(output_dir, f"{name}.json")
    plot_path = os.path.join(output_dir, f"{name}.png")

    exp_res.save_json(json_path)
    plot_convergence_curves(exp_res, plot_path, tier1_bm)

    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.2f}s | Gap={exp_res.final_gap_mean:.4f} | Dist={exp_res.final_distance_mean:.2f} | Eff={exp_res.average_efficiency_mean * 100:.1f}%")

    return {
        "experiment_name": name,
        "m": m,
        "n": n,
        "kappa": kappa,
        "population": population,
        "round_structure": round_structure,
        "total_rounds": rounds,
        "trials": trials,
        "final_equilibrium_gap": exp_res.final_gap_mean,
        "final_gap_ci95": exp_res.final_gap_ci95,
        "final_equilibrium_distance": exp_res.final_distance_mean,
        "final_distance_ci95": exp_res.final_distance_ci95,
        "average_efficiency": exp_res.average_efficiency_mean,
        "average_platform_revenue": exp_res.average_revenue_mean,
        "stability_status": exp_res.stability_summary.get("dominant_status", "unknown"),
        "decay_exponent_beta": exp_res.convergence_rate_summary.get("beta", 0.0),
        "is_converging": exp_res.convergence_rate_summary.get("is_converging", 0.0),
        "price_of_learning_welfare": exp_res.price_of_learning.get("delta_social_welfare", 0.0),
        "json_path": json_path,
        "plot_path": plot_path,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Automated Multi-Agent Convergence Data Collection Pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="convergence_logs",
        help="Directory to store raw convergence JSON logs and plots (default: convergence_logs)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of rounds per trial (default: 200)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of independent random seeds per setting (default: 5)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: run a concise subset of experiments",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print(" PHASE 3: CONVERGENCE DATA COLLECTION SUITE")
    print(f" Target Directory: {os.path.abspath(args.output_dir)}")
    print(f" Rounds per Trial: {args.rounds} | Trials: {args.trials}")
    print("=" * 75)

    experiments = []

    # 1. Population Comparison
    experiments.append(("pop_all_ucb", {"m": 2, "n": 4, "kappa": 0.5, "population": "all_ucb"}))
    experiments.append(("pop_all_ts", {"m": 2, "n": 4, "kappa": 0.5, "population": "all_ts"}))
    experiments.append(("pop_mixed_ucb_ts", {"m": 2, "n": 4, "kappa": 0.5, "population": "mixed_ucb_ts"}))
    experiments.append(("pop_mixed_all", {"m": 4, "n": 4, "kappa": 0.5, "population": "mixed_all"}))

    # 2. Competitor Scaling (m varying)
    experiments.append(("scaling_m2_n4", {"m": 2, "n": 4, "kappa": 0.5, "population": "all_ucb"}))
    experiments.append(("scaling_m3_n4", {"m": 3, "n": 4, "kappa": 0.5, "population": "all_ucb"}))
    if not args.fast:
        experiments.append(("scaling_m5_n4", {"m": 5, "n": 4, "kappa": 0.5, "population": "all_ucb"}))

    # 3. User Scaling (n varying)
    experiments.append(("scaling_m2_n8", {"m": 2, "n": 8, "kappa": 0.5, "population": "all_ucb"}))
    if not args.fast:
        experiments.append(("scaling_m2_n12", {"m": 2, "n": 12, "kappa": 0.5, "population": "all_ucb"}))

    # 4. Budget Tightness (kappa varying)
    experiments.append(("tightness_kappa_0.25", {"m": 2, "n": 4, "kappa": 0.25, "population": "all_ucb"}))
    experiments.append(("tightness_kappa_0.50", {"m": 2, "n": 4, "kappa": 0.50, "population": "all_ucb"}))
    experiments.append(("tightness_kappa_1.00", {"m": 2, "n": 4, "kappa": 1.00, "population": "all_ucb"}))

    # 5. Round Structures
    experiments.append(("struct_canonical", {"m": 2, "n": 4, "kappa": 0.5, "population": "all_ucb", "round_structure": "canonical"}))
    experiments.append(("struct_discounted", {"m": 2, "n": 4, "kappa": 0.5, "population": "all_ucb", "round_structure": "discounted", "discount_factor": 0.95}))
    experiments.append(("struct_cumulative", {"m": 2, "n": 4, "kappa": 0.5, "population": "all_ucb", "round_structure": "cumulative"}))

    results_summary = []
    total_t0 = time.time()

    for exp_name, params in experiments:
        res = run_single_benchmark(
            name=exp_name,
            output_dir=args.output_dir,
            rounds=args.rounds,
            trials=args.trials,
            **params,
        )
        results_summary.append(res)

    # Save summary manifest
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Generate Markdown Report
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Tier-3 Multi-Agent Convergence Data Logs\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"This directory contains raw convergence logs and publication figures for the Phase 3 deliverable.\n\n")
        f.write("## Summary Table across Experimental Configurations\n\n")
        f.write("| Experiment | m | n | kappa | Population | Round Struct | Gap (eps_T ± CI) | Dist (||b-b*||) | Eff (%) | Revenue | Terminal State | Beta (Decay) |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |\n")
        for r in results_summary:
            f.write(
                f"| `{r['experiment_name']}` | {r['m']} | {r['n']} | {r['kappa']} | {r['population']} | {r['round_structure']} | "
                f"{r['final_equilibrium_gap']:.4f} ± {r['final_gap_ci95']:.4f} | {r['final_equilibrium_distance']:.2f} | "
                f"{r['average_efficiency'] * 100:.1f}% | {r['average_platform_revenue']:.2f} | "
                f"`{r['stability_status']}` | {r['decay_exponent_beta']:.4f} |\n"
            )
        f.write("\n## Artifact Files\n")
        for r in results_summary:
            f.write(f"- **{r['experiment_name']}**: [`{r['experiment_name']}.json`](./{r['experiment_name']}.json) | [Plot](./{r['experiment_name']}.png)\n")

    total_time = time.time() - total_t0
    print("\n" + "=" * 75)
    print(f" CONVERGENCE DATA COLLECTION COMPLETE: {len(results_summary)} experiments executed in {total_time:.1f}s")
    print(f" Summary JSON : {summary_path}")
    print(f" Summary Table: {readme_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
