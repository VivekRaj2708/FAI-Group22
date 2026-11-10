#!/usr/bin/env python3
"""
Consolidated Test Suite Runner for FAI-Group22 Project:
Runs Tier-1, Tier-2, and Tier-3 unit and integration test suites.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print("=" * 70)
    print(" FAI-Group22: CONSOLIDATED TEST SUITE RUNNER")
    print("=" * 70)

    total_start = time.time()
    passed_suites = 0
    total_suites = 3

    # --- Test Suite 1: Tier 1 ---
    print("\n[1/3] Running Tier-1 Baseline & Equilibrium Tests...")
    t0 = time.time()
    try:
        import tests.test_tier1 as t1
        t1_tests = [getattr(t1, f) for f in dir(t1) if f.startswith("test_") and callable(getattr(t1, f))]
        for fn in t1_tests:
            fn()
        elapsed = time.time() - t0
        print(f"  -> PASSED: {len(t1_tests)} tests executed ({elapsed:.3f}s)")
        passed_suites += 1
    except Exception as e:
        print(f"  -> FAILED Tier-1 tests: {e}")

    # --- Test Suite 2: Tier 2 ---
    print("\n[2/3] Running Tier-2 Single-Agent Bandit Tests...")
    t0 = time.time()
    try:
        import tests.test_tier2 as t2
        inst = t2.standard_instance()
        t2.test_ucb_agent_probing_and_budget_constraint(inst)
        t2.test_ucb_index_math()
        t2.test_thompson_sampling_agent(inst)
        t2.test_greedy_and_eps_greedy_agents(inst)
        t2.test_cumulative_budget_paced_agent()
        t2.test_tier2_simulation_single_trial(inst)
        t2.test_tier2_multi_seed_experiment(inst)
        t2.test_sublinear_regret_growth()
        t2.test_discounted_round_structure(inst)
        t2.test_tier2_experiment_serialization(inst)
        elapsed = time.time() - t0
        print(f"  -> PASSED: 10 tests executed ({elapsed:.3f}s)")
        passed_suites += 1
    except Exception as e:
        print(f"  -> FAILED Tier-2 tests: {e}")

    # --- Test Suite 3: Tier 3 ---
    print("\n[3/3] Running Tier-3 Multi-Agent Simulator & Convergence Tests...")
    t0 = time.time()
    try:
        import tests.test_tier3 as t3
        inst = t3.test_instance()
        t3.test_multi_agent_population_factory(inst)
        t3.test_round_1_probe_and_budget_constraints(inst)
        t3.test_equilibrium_gap_and_distance_metrics(inst)
        t3.test_cumulative_regret_growth(inst)
        t3.test_convergence_rate_synthetic()
        t3.test_cycling_and_stability_detection()
        t3.test_price_of_learning_math()
        t3.test_multi_seed_experiment_aggregation(inst)
        t3.test_experiment_serialization(inst)
        t3.test_round_structures_variants(inst)
        elapsed = time.time() - t0
        print(f"  -> PASSED: 10 tests executed ({elapsed:.3f}s)")
        passed_suites += 1
    except Exception as e:
        print(f"  -> FAILED Tier-3 tests: {e}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f" TEST RUNNER SUMMARY: {passed_suites}/{total_suites} test suites passed in {total_elapsed:.3f}s")
    print("=" * 70)

    if passed_suites < total_suites:
        sys.exit(1)


if __name__ == "__main__":
    main()

