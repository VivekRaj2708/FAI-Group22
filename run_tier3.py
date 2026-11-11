#!/usr/bin/env python3
"""
CLI runner for Tier-3: Multi-Agent Concurrent Learning & Non-Stationary Convergence
in Batched First-Price Ad Auctions under Knapsack Constraints.
"""

import argparse
import os
import sys
import tempfile
from typing import List, Dict, Optional

# Ensure writable matplotlib cache dir
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np

from src.env.instance import generate_synthetic_instance, AuctionInstance
from src.tier1.baseline import Tier1Solver
from src.tier3.simulation import MultiAgentSimulator, Tier3ExperimentResult


def plot_convergence_curves(
    exp_result: Tier3ExperimentResult,
    output_image_path: str,
    tier1_benchmark,
):
    """
    Generates publication-quality 5-panel multi-agent convergence figures:
    1. Equilibrium Gap epsilon_t vs Rounds (with 95% CI band)
    2. L1 Distance to Tier-1 Equilibrium ||b^t - b*||_1 vs Rounds (with 95% CI band)
    3. Per-Agent Cumulative Regret Reg_t^(i) vs Rounds
    4. Allocation Efficiency Eff_t & Platform Revenue vs Tier-1 Benchmark
    5. Mean Expected Profit per Agent vs Tier-1 Equilibrium Benchmark
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed; skipping plot generation.")
        return

    rounds = np.arange(1, exp_result.total_rounds + 1)
    m = len(exp_result.agent_names)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Tier-3 Multi-Agent Learning Dynamics & Convergence\n"
        f"Population: {exp_result.population_type} | m={m}, n={len(tier1_benchmark.reference_bid_profile[0])}, "
        f"Horizon T={exp_result.total_rounds} ({exp_result.num_trials} seeds)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Equilibrium Gap
    ax1 = axes[0, 0]
    mean_gap = np.array(exp_result.mean_equilibrium_gap)
    ci_gap = np.array(exp_result.ci95_equilibrium_gap)
    ax1.plot(rounds, mean_gap, color="crimson", lw=2, label="Mean $\\varepsilon_t$")
    ax1.fill_between(
        rounds,
        np.maximum(0, mean_gap - ci_gap),
        mean_gap + ci_gap,
        color="crimson",
        alpha=0.2,
        label="95% CI",
    )
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Round $t$")
    ax1.set_ylabel("Equilibrium Gap $\\varepsilon_t$")
    ax1.set_title("Equilibrium Exploitability Gap $\\varepsilon_t$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # 2. L1 Distance to Tier-1 Equilibrium b*
    ax2 = axes[0, 1]
    mean_dist = np.array(exp_result.mean_equilibrium_distance)
    ci_dist = np.array(exp_result.ci95_equilibrium_distance)
    ax2.plot(rounds, mean_dist, color="navy", lw=2, label="$\\|b^t - b^*\\|_1$")
    ax2.fill_between(
        rounds,
        np.maximum(0, mean_dist - ci_dist),
        mean_dist + ci_dist,
        color="navy",
        alpha=0.2,
        label="95% CI",
    )
    ax2.set_xlabel("Round $t$")
    ax2.set_ylabel("L1 Distance $\\|b^t - b^*\\|_1$")
    ax2.set_title("Distance to Tier-1 Equilibrium Profile $b^*$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # 3. Per-Agent Cumulative Regret
    ax3 = axes[0, 2]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i in range(m):
        c = colors[i % len(colors)]
        name = exp_result.agent_names[i]
        reg_mean = np.array(exp_result.mean_per_agent_cumulative_regret[i])
        reg_ci = np.array(exp_result.ci95_per_agent_cumulative_regret[i])
        ax3.plot(rounds, reg_mean, color=c, lw=1.8, label=f"Agent {i}: {name}")
        ax3.fill_between(
            rounds,
            np.maximum(0, reg_mean - reg_ci),
            reg_mean + reg_ci,
            color=c,
            alpha=0.15,
        )
    ax3.set_xlabel("Round $t$")
    ax3.set_ylabel("Cumulative Regret $\\mathrm{Reg}_t^{(i)}$")
    ax3.set_title("Per-Agent Cumulative Regret Growth")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8)

    # 4. Allocation Efficiency & Platform Revenue
    ax4 = axes[1, 0]
    mean_eff = np.array(exp_result.mean_allocation_efficiency)
    ci_eff = np.array(exp_result.ci95_allocation_efficiency)
    ax4.plot(rounds, mean_eff * 100, color="forestgreen", lw=2, label="Multi-Agent Efficiency")
    ax4.fill_between(
        rounds,
        np.maximum(0, (mean_eff - ci_eff) * 100),
        np.minimum(100, (mean_eff + ci_eff) * 100),
        color="forestgreen",
        alpha=0.2,
    )
    ax4.axhline(
        tier1_benchmark.allocation_efficiency * 100,
        color="black",
        linestyle="--",
        label=f"Tier-1 Benchmark ({tier1_benchmark.allocation_efficiency * 100:.1f}%)",
    )
    ax4.set_xlabel("Round $t$")
    ax4.set_ylabel("Allocation Efficiency (%)")
    ax4.set_title("Allocation Efficiency $\\mathrm{Eff}_t$ vs Tier 1")
    ax4.set_ylim(-5, 105)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="lower right")

    # Platform Revenue
    ax5 = axes[1, 1]
    mean_rev = np.array(exp_result.mean_platform_revenue)
    ci_rev = np.array(exp_result.ci95_platform_revenue)
    ax5.plot(rounds, mean_rev, color="purple", lw=2, label="Multi-Agent Revenue")
    ax5.fill_between(
        rounds,
        np.maximum(0, mean_rev - ci_rev),
        mean_rev + ci_rev,
        color="purple",
        alpha=0.2,
    )
    ax5.axhline(
        tier1_benchmark.platform_revenue,
        color="black",
        linestyle="--",
        label=f"Tier-1 Revenue ({tier1_benchmark.platform_revenue:.2f})",
    )
    ax5.set_xlabel("Round $t$")
    ax5.set_ylabel("Platform Revenue")
    ax5.set_title("Platform Revenue $\\mathrm{Rev}(b^t)$ vs Tier 1")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper right")

    # 5. Profit Comparison Bar Chart
    ax6 = axes[1, 2]
    x_pos = np.arange(m)
    width = 0.35
    t1_profs = tier1_benchmark.advertiser_profits
    t3_profs = exp_result.average_profit_mean
    t3_errs = exp_result.average_profit_ci95

    ax6.bar(x_pos - width / 2, t1_profs, width, label="Tier 1 Full Info $p_i^*$", color="gray", alpha=0.8)
    ax6.bar(
        x_pos + width / 2,
        t3_profs,
        width,
        yerr=t3_errs,
        capsize=4,
        label="Tier 3 Multi-Agent $\\bar{p}_i$",
        color="royalblue",
        alpha=0.9,
    )
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f"Agent {i}" for i in range(m)])
    ax6.set_ylabel("Expected Profit per Round")
    ax6.set_title("Advertiser Profit: Tier 1 vs Tier 3")
    ax6.grid(True, alpha=0.3, axis="y")
    ax6.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"\nSaved convergence plots to: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tier-3: Multi-Agent Concurrent Learning Simulator & Convergence Evaluator"
    )
    parser.add_argument(
        "--m", type=int, default=2, help="Number of advertisers (default: 2)"
    )
    parser.add_argument(
        "--n", type=int, default=4, help="Number of users (default: 4)"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.5,
        help="Budget tightness kappa in [0.1, 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--v",
        type=float,
        default=5.0,
        help="Click value v (default: 5.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for instance (default: 42)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=300,
        help="Horizon T per trial (default: 300)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of independent simulation seeds (default: 5)",
    )
    parser.add_argument(
        "--population",
        type=str,
        default="all_ucb",
        choices=["all_ucb", "all_ts", "mixed_ucb_ts", "mixed_all", "learner_vs_equilibrium"],
        help="Multi-agent population configuration (default: all_ucb)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="UCB exploration bonus parameter alpha (default: 2.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="EpsGreedy exploration parameter (default: 0.1)",
    )
    parser.add_argument(
        "--round-structure",
        type=str,
        default="canonical",
        choices=["canonical", "discounted", "cumulative"],
        help="Round structure: canonical, discounted, or cumulative (default: canonical)",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=0.99,
        help="Discount factor gamma for discounted variant (default: 0.99)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tier3_benchmark.json",
        help="Output JSON file for benchmark data",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="tier3_convergence_plots.png",
        help="Output PNG path for convergence curves",
    )
    parser.add_argument(
        "--load-instance",
        type=str,
        default=None,
        help="Path to load pre-existing instance JSON",
    )

    args = parser.parse_args()

    print("=" * 75)
    print(" TIER 3: MULTI-AGENT CONCURRENT LEARNING & CONVERGENCE")
    print("=" * 75)

    if args.load_instance:
        print(f"Loading instance from {args.load_instance}...")
        instance = AuctionInstance.load_json(args.load_instance)
    else:
        print(
            f"Generating synthetic instance: m={args.m}, n={args.n}, kappa={args.kappa}, "
            f"v={args.v}, seed={args.seed}"
        )
        instance = generate_synthetic_instance(
            num_advertisers=args.m,
            num_users=args.n,
            budget_tightness=args.kappa,
            click_value=args.v,
            seed=args.seed,
        )

    print(f"\nAdvertisers (m): {instance.num_advertisers}")
    print(f"Users (n): {instance.num_users}")
    print(f"Budgets: {instance.budgets}")
    print(f"Click Value (v): {instance.click_value}")
    print(f"Population Archetype: {args.population}")
    print(f"Round Structure: {args.round_structure}")
    print(f"Horizons (T): {args.rounds} rounds across {args.trials} seeds")

    # Step 1: Precompute Tier-1 Reference Equilibrium
    print("\nComputing Tier-1 Reference Benchmark...")
    solver = Tier1Solver(instance)
    tier1_bm = solver.solve()
    print(f"  Tier-1 Solver Method    : {tier1_bm.solver_method}")
    print(f"  Pure Equilibrium Found? : {tier1_bm.is_pure_equilibrium}")
    print(f"  Tier-1 Reference Profile: {tier1_bm.reference_bid_profile}")
    print(f"  Tier-1 Profits          : {[round(p, 4) for p in tier1_bm.advertiser_profits]}")
    print(f"  Tier-1 Platform Revenue : {tier1_bm.platform_revenue:.4f}")
    print(f"  Tier-1 Allocation Eff   : {tier1_bm.allocation_efficiency * 100:.2f}%")

    # Step 2: Initialize Multi-Agent Simulator
    print("\nInitializing Multi-Agent Simulator...")
    sim = MultiAgentSimulator(
        instance=instance,
        population_type=args.population,
        reference_benchmark=tier1_bm,
        round_structure=args.round_structure,
        discount_factor=args.discount_factor,
        alpha=args.alpha,
        epsilon=args.epsilon,
    )
    print("  Agent Roster:")
    for a in sim.agents:
        print(f"    - {a.name} (Budget: {a.budget})")

    # Step 3: Run Multi-Agent Experiments across Seeds
    print(f"\nSimulating {args.trials} independent runs of {args.rounds} rounds...")
    seeds = [args.seed + 101 * k for k in range(args.trials)]
    exp_result = sim.run_experiment(
        total_rounds=args.rounds,
        seeds=seeds,
    )

    # Step 4: Display Summary Results
    print("\n" + "=" * 75)
    print(" TIER 3 RESULTS SUMMARY")
    print("=" * 75)
    print(f"Final Equilibrium Gap (eps_T) : {exp_result.final_gap_mean:.4f} ± {exp_result.final_gap_ci95:.4f}")
    print(f"Final Distance to b* (L1)     : {exp_result.final_distance_mean:.2f} ± {exp_result.final_distance_ci95:.2f}")
    print(f"Mean Allocation Efficiency    : {exp_result.average_efficiency_mean * 100:.2f}% ± {exp_result.average_efficiency_ci95 * 100:.2f}%")
    print(f"Mean Platform Revenue         : {exp_result.average_revenue_mean:.4f} ± {exp_result.average_revenue_ci95:.4f}")

    print("\nPer-Agent Learning Performance:")
    print(f"{'Agent':<24} | {'Final Regret (Reg_T ± 95%CI)':<28} | {'Avg Profit':<12} | {'T1 Profit':<10}")
    print("-" * 80)
    for i in range(instance.num_advertisers):
        name = exp_result.agent_names[i]
        reg_str = f"{exp_result.final_regrets_mean[i]:.2f} ± {exp_result.final_regrets_ci95[i]:.2f}"
        prof_str = f"{exp_result.average_profit_mean[i]:.4f}"
        t1_str = f"{tier1_bm.advertiser_profits[i]:.4f}"
        print(f"{name:<24} | {reg_str:<28} | {prof_str:<12} | {t1_str:<10}")
    print("-" * 80)

    # Price of Learning Report
    pol = exp_result.price_of_learning
    print("\nPrice of Learning (Delta = Metric(Tier 1) - Metric(Tier 3)):")
    print(f"  Delta Profit (Total Welfare Loss): {pol['delta_total_profit']:.4f}")
    print(f"  Delta Platform Revenue          : {pol['delta_platform_revenue']:.4f}")
    print(f"  Delta Allocation Efficiency     : {pol['delta_allocation_efficiency'] * 100:.2f}%")
    print(f"  Delta Expected Social Welfare   : {pol['delta_social_welfare']:.4f}")

    # Stability & Convergence Rate
    stab = exp_result.stability_summary
    rate = exp_result.convergence_rate_summary
    print("\nConvergence Dynamics:")
    print(f"  Dominant Terminal State : {stab['dominant_status']}")
    print(f"  Profile Stability Ratio : {stab['mean_stability_fraction'] * 100:.1f}%")
    print(f"  Gap Decay Exponent beta : {rate['beta']:.4f} (R^2 = {rate['r_squared']:.4f})")
    print(f"  Empirically Converging  : {'Yes' if rate['is_converging'] > 0.5 else 'Inconclusive / Cyclic'}")

    # Step 5: Save JSON Artifact
    exp_result.save_json(args.output)
    print(f"\nSaved Tier-3 experiment data to: {args.output}")

    # Step 6: Generate Convergence Curves Plot
    if args.plot:
        plot_convergence_curves(exp_result, args.plot, tier1_bm)

    print("=" * 75)


if __name__ == "__main__":
    main()

