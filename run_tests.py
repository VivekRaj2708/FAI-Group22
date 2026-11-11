#!/usr/bin/env python3
"""
Consolidated Test Suite Runner for FAI-Group22 Project:
Runs the Tier-1, Tier-2, Tier-3 and evaluation-harness test suites through pytest
(the suites use pytest fixtures, so the test functions cannot be called directly).
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SUITES = [
    ("Tier-1 Baseline & Equilibrium", "tests/test_tier1.py"),
    ("Tier-2 Single-Agent Bandit", "tests/test_tier2.py"),
    ("Tier-3 Multi-Agent Simulator & Convergence", "tests/test_tier3.py"),
    ("Evaluation Harness (Phase 4/5)", "tests/test_eval.py"),
]


def main():
    print("=" * 70)
    print(" FAI-Group22: CONSOLIDATED TEST SUITE RUNNER")
    print("=" * 70)

    total_start = time.time()
    passed_suites = 0

    for k, (label, path) in enumerate(SUITES, start=1):
        print(f"\n[{k}/{len(SUITES)}] Running {label} tests...")
        t0 = time.time()
        code = pytest.main(["-q", str(ROOT / path)])
        elapsed = time.time() - t0
        if code == 0:
            print(f"  -> PASSED ({elapsed:.3f}s)")
            passed_suites += 1
        else:
            print(f"  -> FAILED {label} tests (pytest exit code {int(code)})")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f" TEST RUNNER SUMMARY: {passed_suites}/{len(SUITES)} test suites passed in {total_elapsed:.3f}s")
    print("=" * 70)

    if passed_suites < len(SUITES):
        sys.exit(1)


if __name__ == "__main__":
    main()
