#!/usr/bin/env python3
"""
CLI runner for Tier-2: Single-Agent Combinatorial Semi-Bandit Learning
in Batched First-Price Ad Auctions under Knapsack Constraints.
"""

import argparse
import os
import sys
from typing import List, Dict

import numpy as np

from src.env.instance import generate_synthetic_instance, AuctionInstance
from src.tier1.baseline import Tier1Solver
from src.tier2.agents import (
    UCBAgent,
    ThompsonSamplingAgent,
    GreedyAgent,
    EpsGreedyAgent,
    CumulativeBudgetPacedAgent,
)
from src.tier2.simulation import Tier2Simulator, Tier2ExperimentResult


def create_agent(
    learner_type: str,
    agent_id: int,
    budget: int,
    num_users: int,
    click_value: float,
    total_rounds: int,
    alpha: float = 2.0,
    epsilon: float = 0.1,
    seed: int = 42,
):
    learner_type = learner_type.lower()
    if learner_type == "ucb":
        return UCBAgent(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            alpha=alpha,
            name=f"UCB (alpha={alpha})",
        )
    elif learner_type in ("thompson", "ts"):
        return ThompsonSamplingAgent(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            seed=seed,
            name="Thompson Sampling",
        )
    elif learner_type == "greedy":
        return GreedyAgent(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            name="Greedy (alpha=0)",
        )
    elif learner_type in ("eps_greedy", "eps-greedy"):
        return EpsGreedyAgent(
            agent_id=agent_id,
            budget=budget,
            num_users=num_users,
            click_value=click_value,
            epsilon=epsilon,
            seed=seed,
            name=f"Eps-Greedy (eps={epsilon})",
        )
    elif learner_type in ("paced", "cumulative_paced"):
        total_budget = budget * total_rounds // 2  # Tight cumulative budget
        return CumulativeBudgetPacedAgent(
            agent_id=agent_id,
            total_budget=total_budget,
            num_users=num_users,
            total_rounds=total_rounds,
            click_value=click_value,
            per_round_max_budget=budget,
            alpha=alpha,
            name=f"Cumulative Paced UCB (B_tot={total_budget})",
        )
    else:
        raise ValueError(f"Unknown learner type: {learner_type}")


def plot_regret_curves(
    results: Dict[str, Tier2ExperimentResult],
    output_image_path: str,
    total_rounds: int,
):
    try:
        import matplotlib
        # Avoid DISPLAY issues in headless/terminal environments
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5.5), dpi=150)
        rounds_x = np.arange(1, total_rounds + 1)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for idx, (name, exp_res) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            mean = np.array(exp_res.mean_cumulative_regret)
            ci = np.array(exp_res.ci_95_cumulative_regret)

            plt.plot(rounds_x, mean, label=name, color=color, linewidth=2)
            plt.fill_between(
                rounds_x,
                np.maximum(0, mean - ci),
                mean + ci,
                color=color,
                alpha=0.15,
            )

        plt.title(
            f"Tier-2 Cumulative Regret Comparison (T={total_rounds}, 95% CI)",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Round $t$", fontsize=11)
        plt.ylabel("Cumulative Regret $\\mathrm{Reg}_t$", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        plt.savefig(output_image_path)
        plt.close()
        print(f"\nSaved regret curves plot to: {output_image_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate plot ({e})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Tier-2 Single-Agent Combinatorial Semi-Bandit Learning Runner"
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
        help="Click value v (default: 5.0 to allow profitable integer bids)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for instance (default: 42)"
    )
    parser.add_argument(
        "--load-instance",
        type=str,
        default=None,
        help="Path to load pre-existing instance JSON",
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="all",
        choices=["all", "ucb", "thompson", "greedy", "eps_greedy", "paced"],
        help="Bandit algorithm to simulate (default: all)",
    )
    parser.add_argument(
        "-T",
        "--rounds",
        type=int,
        default=1000,
        help="Number of rounds per trial (default: 1000)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of independent seeds/trials for confidence intervals (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="UCB exploration bonus hyperparameter alpha (default: 2.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon parameter for eps-greedy (default: 0.1)",
    )
    parser.add_argument(
        "--round-structure",
        type=str,
        default="canonical",
        choices=["canonical", "discounted", "cumulative"],
        help="Round structure (default: canonical)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for discounted round structure (default: 0.99)",
    )
    parser.add_argument(
        "--learner-id",
        type=int,
        default=0,
        help="Index of the learning advertiser (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tier2_benchmark.json",
        help="Output JSON file path for experiment results",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="tier2_regret_curves.png",
        help="Output image path for regret curves plot (set empty to disable)",
    )

    args = parser.parse_args()

    print("=" * 75)
    print(" TIER 2: SINGLE-AGENT COMBINATORIAL SEMI-BANDIT LEARNING")
    print("=" * 75)

    # 1. Load or generate instance
    if args.load_instance:
        print(f"Loading instance from {args.load_instance}...")
        instance = AuctionInstance.load_json(args.load_instance)
    else:
        print(
            f"Generating synthetic instance: m={args.m}, n={args.n}, kappa={args.kappa}, v={args.v}, seed={args.seed}"
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
    print(f"Learner ID: Advertiser {args.learner_id}")
    print(f"Round Structure: {args.round_structure}")
    print(f"Horizons (T): {args.rounds} rounds across {args.trials} seeds")

    # 2. Compute Tier-1 Equilibrium to get fixed opponent bids
    print("\nComputing Tier-1 Equilibrium for opponents' reference strategy...")
    tier1_solver = Tier1Solver(instance)
    t1_benchmark = tier1_solver.solve()

    print(f"  Tier-1 Solver Method    : {t1_benchmark.solver_method}")
    print(f"  Pure Equilibrium Found? : {t1_benchmark.is_pure_equilibrium}")
    print(f"  Tier-1 Reference Profile: {t1_benchmark.reference_bid_profile}")

    opponents_bids = [
        t1_benchmark.reference_bid_profile[i]
        for i in range(instance.num_advertisers)
    ]
    learner_c_star = [0] * instance.num_users
    for j in range(instance.num_users):
        others = [
            opponents_bids[k][j]
            for k in range(instance.num_advertisers)
            if k != args.learner_id
        ]
        learner_c_star[j] = max(others) if others else 0

    print(f"  Learner Faces Constant Competing Bids c_{{-i}}^*: {learner_c_star}")

    # Determine learners to run
    if args.learner == "all":
        learner_keys = ["ucb", "thompson", "greedy", "eps_greedy"]
        if args.round_structure == "cumulative":
            learner_keys.append("paced")
    else:
        learner_keys = [args.learner]

    results_dict: Dict[str, Tier2ExperimentResult] = {}
    seeds = [args.seed * 100 + i for i in range(args.trials)]

    print("\n" + "-" * 75)
    print(" RUNNING TIER-2 BANDIT SIMULATIONS")
    print("-" * 75)

    for l_key in learner_keys:
        agent = create_agent(
            learner_type=l_key,
            agent_id=args.learner_id,
            budget=instance.budgets[args.learner_id],
            num_users=instance.num_users,
            click_value=instance.click_value,
            total_rounds=args.rounds,
            alpha=args.alpha,
            epsilon=args.epsilon,
            seed=args.seed,
        )

        sim = Tier2Simulator(
            instance=instance,
            learner_agent=agent,
            learner_id=args.learner_id,
            fixed_opponent_bids=t1_benchmark.reference_bid_profile,
            round_structure=args.round_structure,
            discount_factor=args.gamma,
        )

        print(f"Simulating {agent.name} across {args.trials} seeds...")
        exp_res = sim.run_experiment(
            total_rounds=args.rounds,
            seeds=seeds,
        )
        results_dict[agent.name] = exp_res

    # 3. Print Results Summary Table
    print("\n" + "=" * 75)
    print(" TIER 2 RESULTS SUMMARY")
    print("=" * 75)
    print(
        f"{'Algorithm':<28} | {'Final Regret (Reg_T ± 95%CI)':<26} | {'Avg Profit':<10} | {'Avg Eff':<8}"
    )
    print("-" * 75)

    for name, res in results_dict.items():
        reg_str = f"{res.final_regret_mean:.2f} ± {res.final_regret_ci95:.2f}"
        print(
            f"{name:<28} | {reg_str:<26} | {res.average_round_profit:<10.3f} | {res.average_efficiency * 100:<7.1f}%"
        )

    print("-" * 75)
    print(f"Optimal Full-Info Profit p_i*(c_{{-i}}) = {sim.p_i_star:.4f}")

    # 4. Save results to JSON
    if args.output:
        serializable = {
            "instance": instance.to_dict(),
            "tier1_reference_profile": t1_benchmark.reference_bid_profile,
            "learner_competing_bids": learner_c_star,
            "optimal_expected_profit": sim.p_i_star,
            "round_structure": args.round_structure,
            "total_rounds": args.rounds,
            "trials": args.trials,
            "algorithms": {name: res.to_dict() for name, res in results_dict.items()},
        }
        import json

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved Tier-2 experiment data to: {args.output}")

    # 5. Plot regret curves if requested
    if args.plot:
        plot_regret_curves(results_dict, args.plot, args.rounds)

    print("=" * 75)


if __name__ == "__main__":
    main()
