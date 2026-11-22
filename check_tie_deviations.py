"""Check the Tier-1 pure equilibria against deviations that include tie bids.

The Tier-1 solver certifies b* only against strict-win and all-zero deviations. Here we compute the
exact best response over the whole strategy set S_i: on each user j, with c = highest competing bid
and M = number of opponents bidding exactly c, the only undominated bids are
    0        (cost 0,   expected profit 0)
    c, c>0   (cost c,   expected profit (f_ij v - c) / (1 + M))   -- a tie
    c + 1    (cost c+1, expected profit  f_ij v - c - 1)          -- a strict win
(bids in (0, c) win nothing, bids above c+1 cost more for the same win). The best response is a
multiple-choice knapsack over these options, solved exactly by DP. The utility of the best-response
bid vector is then re-evaluated with the auction engine, so the check does not trust the DP's arithmetic.

    python check_tie_deviations.py                 # the full grid used in the report (1440 instances)
    python check_tie_deviations.py --workers 8
"""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

from src.env.auction import BatchedFirstPriceAuction
from src.eval.experiment import ExperimentConfig, tier1_reference

TOL = 1e-6


def exact_best_response(engine, inst, profile, i):
    n, B, v = inst.num_users, inst.budgets[i], inst.click_value
    others = [profile[k] for k in range(inst.num_advertisers) if k != i]
    # options[j] = list of (cost, value, bid)
    options = []
    for j in range(n):
        col = [b[j] for b in others]
        c = max(col)
        f = inst.click_probabilities[i][j]
        opts = []
        if c > 0:
            opts.append((c, (f * v - c) / (1 + col.count(c)), c))
        opts.append((c + 1, f * v - c - 1, c + 1))
        options.append([o for o in opts if o[0] <= B and o[1] > 0])
    best = [0.0] * (B + 1)        # best[beta] = best value with total cost <= beta
    choice = []
    for j in range(n):
        new, pick = best[:], [None] * (B + 1)
        for cost, val, bid in options[j]:
            for beta in range(cost, B + 1):
                if best[beta - cost] + val > new[beta] + 1e-12:
                    new[beta], pick[beta] = best[beta - cost] + val, (cost, bid)
        choice.append(pick)
        best = new
    bids, beta = [0] * n, B
    for j in reversed(range(n)):
        if choice[j][beta] is not None:
            cost, bids[j] = choice[j][beta]
            beta -= cost
    dev = [row[:] for row in profile]
    dev[i] = bids
    return engine.expected_utilities(dev)[i]


def check(args):
    m, n, kappa, seed = args
    inst, bench = tier1_reference(ExperimentConfig(m=m, n=n, kappa=kappa, scenario="tier1"), seed)
    if not bench.is_pure_equilibrium:
        return None
    engine = BatchedFirstPriceAuction(inst)
    b = bench.reference_bid_profile
    current = engine.expected_utilities(b)
    gain = max(exact_best_response(engine, inst, b, i) - current[i] for i in range(m))
    return (m, n, kappa, seed, gain)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    grid = list(itertools.product([2, 3, 5], [5, 10, 20, 50], [0.1, 0.25, 0.5, 1.0], range(30)))
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for r in ex.map(check, grid, chunksize=8) if r is not None]
    bad = [r for r in res if r[4] > TOL]
    print(f"pure Tier-1 profiles checked: {len(res)}")
    print(f"profiles where some advertiser gains by a deviation that includes ties: {len(bad)}")
    if bad:
        print(f"largest gain: {max(r[4] for r in bad):.4f}")
        for r in sorted(bad, key=lambda r: -r[4])[:20]:
            print("  m=%d n=%d kappa=%g seed=%d gain=%.4f" % r)
