#!/usr/bin/env python3
"""
CLI runner for Tier-1: Batched First-Price Ad Auction Baseline & Equilibrium Solver.
"""

import argparse
import os
import sys
from pprint import pprint

from src.env.instance import generate_synthetic_instance, AuctionInstance
from src.tier1.baseline import Tier1Solver


def main():
    parser = argparse.ArgumentParser(
        description="Tier-1 Ground Truth Baseline & Equilibrium Solver"
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
        "--output",
        type=str,
        default="tier1_benchmark.json",
        help="Output JSON file for benchmark data",
    )
    parser.add_argument(
        "--load-instance",
        type=str,
        default=None,
        help="Path to load pre-existing instance JSON",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" TIER 1: KNOWN-PARAMETER BASELINE & EQUILIBRIUM SOLVER")
    print("=" * 70)

    if args.load_instance:
        print(f"Loading instance from {args.load_instance}...")
        instance = AuctionInstance.load_json(args.load_instance)
    else:
        print(f"Generating synthetic instance: m={args.m}, n={args.n}, kappa={args.kappa}, v={args.v}, seed={args.seed}")
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
    print("\nClick Probabilities matrix f(u_j, a_i):")
    for i in range(instance.num_advertisers):
        row_str = ", ".join(f"{val:.4f}" for val in instance.click_probabilities[i])
        print(f"  Advertiser {i}: [{row_str}]")

    print("\nSolving Tier-1 baseline...")
    solver = Tier1Solver(instance)
    benchmark = solver.solve()

    print("\n" + "-" * 70)
    print(" TIER 1 RESULTS SUMMARY")
    print("-" * 70)
    print(f"Solver Method Used      : {benchmark.solver_method}")
    print(f"Pure Equilibrium Found? : {benchmark.is_pure_equilibrium}")
    print(f"Equilibrium Gap (eps)   : {benchmark.equilibrium_gap:.6f}")
    print("\nReference Bid Profile b*:")
    for i, bids in enumerate(benchmark.reference_bid_profile):
        spend = sum(bids)
        budget = instance.budgets[i]
        print(f"  Advertiser {i}: {bids} (Spend: {spend}/{budget})")

    print("\nEconomic Metrics:")
    for i, p_i in enumerate(benchmark.advertiser_profits):
        print(f"  Advertiser {i} Profit (p_{i}*): {p_i:.4f}")
    print(f"  Platform Revenue Rev(b*)    : {benchmark.platform_revenue:.4f}")
    print(f"  Expected Welfare Wel(b*)    : {benchmark.expected_welfare:.4f}")
    print(f"  Allocation Efficiency Eff   : {benchmark.allocation_efficiency * 100:.2f}%")

    print("\nSocial Planner Benchmark (First-Best Comparison):")
    print(f"  First-Best Max Welfare      : {benchmark.social_planner_welfare:.4f}")
    print(f"  Welfare Loss (DWL)          : {benchmark.welfare_loss:.4f}")
    print(f"  Efficiency Loss             : {benchmark.efficiency_loss * 100:.2f}%")
    print(f"  Optimal User Assignments    : {benchmark.social_planner_assignment}")
    print("-" * 70)

    if args.output:
        benchmark.save_json(args.output)
        print(f"Saved Tier-1 benchmark reference artifact to: {args.output}")

    print("=" * 70)


if __name__ == "__main__":
    main()
