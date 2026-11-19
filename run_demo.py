#!/usr/bin/env python3
"""
Live demo: watch advertisers learn to bid, round by round.

Uses the project's own code end to end (instance generator, Tier-1 solver,
registered agents, BatchedFirstPriceAuction, harness metrics), but steps the
auction one round at a time so it can be animated.

Examples
--------
  # Tier 2: one Thompson learner (advertiser 1) vs. opponents fixed at b*
  python run_demo.py --scenario tier2 --learner thompson

  # Tier 3: every advertiser runs UCB at the same time
  python run_demo.py --scenario tier3 --learner ucb --m 3 --n 8

  # Head-to-head market (any registered agents, one per advertiser)
  python run_demo.py --agents thompson ucb greedy --n 8

  # No window (e.g. over SSH): print a live text log instead
  python run_demo.py --headless --T 200

  # Save the animation instead of showing it
  python run_demo.py --save demo.gif --T 150

Panels: true click probabilities with b*; the bids submitted this round
(winners outlined); the focal learner's empirical click rate vs. the truth;
cumulative regret per advertiser; equilibrium gap; allocation efficiency.
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from src.agents.registry import available_agents
from src.env.auction import BatchedFirstPriceAuction
from src.eval.experiment import ExperimentConfig, build_agents, tier1_reference
from src.eval.metrics import BestResponseOracle, best_advertiser_mask, evaluate_profile
from src.eval.simulator import _sanitize


@dataclass
class RoundSnapshot:
    t: int
    bids: np.ndarray            # (m, n)
    wins: np.ndarray            # (m, n)
    clicks: np.ndarray          # (m, n)
    profits: np.ndarray         # (m,) expected p_i(b^t)
    regret: np.ndarray          # (m,) per-round regret
    eq_gap: float
    efficiency: float
    revenue: float
    eq_distance: float
    pulls: Optional[np.ndarray]     # focal learner's N_j
    est_f: Optional[np.ndarray]     # focal learner's s_j / N_j (nan if N_j = 0)


def build_market(args):
    if args.agents:
        scenario, learner, m = "custom", args.agents[0], len(args.agents)
    else:
        scenario, learner, m = args.scenario, args.learner, args.m
    cfg = ExperimentConfig(m=m, n=args.n, kappa=args.kappa, click_value=args.v,
                           scenario=scenario, learner=learner, agents=args.agents,
                           horizon=args.T, seeds=[args.seed])
    inst, bench = tier1_reference(cfg, args.seed)
    agents = build_agents(cfg, inst, bench, args.seed)
    names = (list(args.agents) if args.agents else
             ([learner] * m if scenario == "tier3" else
              [learner] + ["fixed b*"] * (m - 1)))
    return cfg, inst, bench, agents, names


def step_market(inst, bench, agents, T: int, seed: int, focal: int = 0) -> Iterator[RoundSnapshot]:
    """Canonical round structure, one round per iteration."""
    import random

    m, n = inst.num_advertisers, inst.num_users
    f = np.asarray(inst.click_probabilities, dtype=float)
    v = float(inst.click_value)
    caps = np.asarray(inst.budgets, dtype=int)
    oracle = BestResponseOracle(f, v)
    best_mask = best_advertiser_mask(f)
    b_star = np.asarray(bench.reference_bid_profile, dtype=int)
    auction = BatchedFirstPriceAuction(inst)
    rng = random.Random(seed)
    for a in agents:
        a.reset()

    last_comp: List[Optional[np.ndarray]] = [None] * m
    for t in range(1, T + 1):
        profile = []
        for i, agent in enumerate(agents):
            bid = _sanitize(agent.get_bid(t, last_comp[i]), n, int(caps[i]), i, "zero")
            profile.append(bid if bid is not None else [0] * n)
        bids = np.asarray(profile, dtype=int)
        pm = evaluate_profile(bids, f, v, oracle, caps, best_mask, b_star)
        out = auction.simulate_round(profile, rng)
        wins, clicks, pay = (np.asarray(out.wins), np.asarray(out.clicks), np.asarray(out.payments))
        comp = np.asarray(out.competing_bids, dtype=int)
        for i, agent in enumerate(agents):
            agent.observe(wins[i].astype(bool), clicks[i], pay[i], comp[i].copy())
            last_comp[i] = comp[i].copy()

        pulls = est = None
        fa = agents[focal]
        if hasattr(fa, "pulls") and hasattr(fa, "clicks"):
            pulls = np.asarray(fa.pulls, dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(pulls > 0, np.asarray(fa.clicks, dtype=float) / pulls, np.nan)
        yield RoundSnapshot(t, bids, wins, clicks, pm.profits, pm.regret, pm.eq_gap,
                            pm.efficiency, pm.revenue, pm.eq_distance, pulls, est)


def describe(inst, bench, names, args):
    f = np.asarray(inst.click_probabilities)
    print("=" * 72)
    print(f" LIVE DEMO  m={inst.num_advertisers}  n={inst.num_users}  B={inst.budgets}  "
          f"v={inst.click_value:g}  T={args.T}  seed={args.seed}")
    print("=" * 72)
    for i, name in enumerate(names):
        print(f"  a{i+1}: {name:<10s} f = {np.round(f[i], 2).tolist()}")
    print(f"  Tier-1 reference b* ({bench.solver_method}, pure={bench.is_pure_equilibrium}):")
    for i, row in enumerate(bench.reference_bid_profile):
        print(f"    b*_{i+1} = {row}   p*_{i+1} = {bench.advertiser_profits[i]:.3f}")
    print(f"  Eff(b*) = {bench.allocation_efficiency:.2f}   Rev(b*) = {bench.platform_revenue:.2f}")
    print("-" * 72)


def run_headless(gen, names, every: int):
    cum = None
    for s in gen:
        cum = s.regret.copy() if cum is None else cum + s.regret
        if s.t == 1 or s.t % every == 0:
            regs = "  ".join(f"R{i+1}={r:7.1f}" for i, r in enumerate(cum))
            print(f"t={s.t:5d}  gap={s.eq_gap:5.2f}  eff={s.efficiency:4.2f}  "
                  f"rev={s.revenue:5.1f}  |b-b*|={s.eq_distance:4.0f}  {regs}")
    print("-" * 72)
    for i, name in enumerate(names):
        print(f"  a{i+1} ({name}): cumulative regret {cum[i]:.1f}, per round {cum[i] / s.t:.3f}")


def run_animated(gen, inst, bench, names, args):
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    m, n, T = inst.num_advertisers, inst.num_users, args.T
    f = np.asarray(inst.click_probabilities)
    b_star = np.asarray(bench.reference_bid_profile)
    labels = [f"a{i+1} {nm}" for i, nm in enumerate(names)]
    colors = plt.cm.tab10(np.arange(m))
    bmax = max(inst.budgets)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.3, top=0.88)
    title = fig.suptitle("", fontsize=13)
    ax_f, ax_b, ax_est, ax_reg, ax_gap, ax_eff = axes.ravel()

    # (a) true f with b* annotated
    ax_f.imshow(f, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m):
        for j in range(n):
            ax_f.text(j, i, f"{f[i, j]:.2f}\nb*={b_star[i, j]}", ha="center", va="center",
                      fontsize=7, color="white" if f[i, j] > 0.55 else "black")
    ax_f.set_title("True click prob. $f_{ij}$ (hidden) and $b^*$")
    ax_f.set_yticks(range(m), labels, fontsize=8)
    ax_f.set_xticks(range(n), [f"u{j+1}" for j in range(n)], fontsize=8)

    # (b) bids this round, winners outlined
    im_b = ax_b.imshow(np.zeros((m, n)), cmap="Oranges", vmin=0, vmax=bmax, aspect="auto")
    bid_txt = [[ax_b.text(j, i, "", ha="center", va="center", fontsize=9) for j in range(n)]
               for i in range(m)]
    win_boxes = [[ax_b.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=0, ec="green"))
                  for j in range(n)] for i in range(m)]
    ax_b.set_title("Bids this round (green = won, * = click)")
    ax_b.set_yticks(range(m), labels, fontsize=8)
    ax_b.set_xticks(range(n), [f"u{j+1}" for j in range(n)], fontsize=8)

    # (c) focal learner's estimates
    x = np.arange(n)
    ax_est.bar(x - 0.2, f[0], 0.4, color="lightgray", label="true $f_{1j}$")
    est_bars = ax_est.bar(x + 0.2, np.zeros(n), 0.4, color=colors[0], label="learned $s_j/N_j$")
    n_txt = [ax_est.text(j + 0.2, 0.02, "", ha="center", fontsize=7, rotation=90) for j in range(n)]
    ax_est.set_ylim(0, 1)
    ax_est.set_xticks(x, [f"u{j+1}" for j in range(n)], fontsize=8)
    ax_est.set_title(f"What a1 ({names[0]}) has learned")
    ax_est.legend(fontsize=8, loc="upper right")

    # (d)-(f) time series
    reg_lines = [ax_reg.plot([], [], color=colors[i], label=labels[i])[0] for i in range(m)]
    ax_reg.set_title("Cumulative regret vs. best response")
    ax_reg.legend(fontsize=8, loc="upper left")
    gap_line, = ax_gap.plot([], [], color="tab:red", lw=0.8, alpha=0.4, label="$\\varepsilon_t$")
    gap_avg, = ax_gap.plot([], [], color="tab:red", lw=2, label="moving avg (20)")
    ax_gap.axhline(bench.equilibrium_gap, ls="--", color="k", lw=1, label="$\\varepsilon(b^*)$")
    ax_gap.set_title("Equilibrium gap $\\varepsilon_t$")
    ax_gap.legend(fontsize=8)
    eff_line, = ax_eff.plot([], [], color="tab:green", lw=2, label="moving avg (20)")
    ax_eff.axhline(bench.allocation_efficiency, ls="--", color="k", lw=1, label="Eff$(b^*)$")
    ax_eff.set_ylim(0, 1.05)
    ax_eff.set_title("Allocation efficiency")
    ax_eff.legend(fontsize=8, loc="lower right")
    for ax in (ax_reg, ax_gap, ax_eff):
        ax.set_xlim(1, T)
        ax.set_xlabel("round $t$")

    hist = {"t": [], "cum": [], "gap": [], "eff": []}

    def ma(a, w=20):
        a = np.asarray(a, dtype=float)
        c = np.cumsum(np.insert(a, 0, 0.0))
        k = np.minimum(np.arange(1, len(a) + 1), w)
        return (c[1:] - c[np.arange(1, len(a) + 1) - k]) / k

    def update(s: RoundSnapshot):
        prev = hist["cum"][-1] if hist["cum"] else np.zeros(m)
        hist["t"].append(s.t)
        hist["cum"].append(prev + s.regret)
        hist["gap"].append(s.eq_gap)
        hist["eff"].append(s.efficiency)
        ts = np.asarray(hist["t"])
        cum = np.asarray(hist["cum"])

        im_b.set_data(s.bids)
        for i in range(m):
            for j in range(n):
                b = s.bids[i, j]
                bid_txt[i][j].set_text(f"{b}{'*' if s.clicks[i, j] else ''}" if b else "")
                win_boxes[i][j].set_linewidth(3 if s.wins[i, j] else 0)
        if s.est_f is not None:
            for j, bar in enumerate(est_bars):
                bar.set_height(0 if np.isnan(s.est_f[j]) else s.est_f[j])
                n_txt[j].set_text(f"N={int(s.pulls[j])}")
        for i, ln in enumerate(reg_lines):
            ln.set_data(ts, cum[:, i])
        ax_reg.set_ylim(min(0, cum.min()) - 1, max(1, cum.max()) * 1.1)
        gap_line.set_data(ts, hist["gap"])
        gap_avg.set_data(ts, ma(hist["gap"]))
        ax_gap.set_ylim(0, max(1.0, max(hist["gap"])) * 1.1)
        eff_line.set_data(ts, ma(hist["eff"]))
        prof = "  ".join(f"p{i+1}={p:.2f}" for i, p in enumerate(s.profits))
        title.set_text(f"Round {s.t}/{T}   |   {prof}   |   revenue {s.revenue:.1f}   |   "
                       f"$\\|b^t-b^*\\|_1$ = {s.eq_distance:.0f}")
        return []

    anim = FuncAnimation(fig, update, frames=gen, interval=args.interval, blit=False,
                         repeat=False, save_count=T, cache_frame_data=False)
    if args.save:
        if args.save.lower().endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(args.save, writer=PillowWriter(fps=max(1, 1000 // args.interval)))
        else:
            anim.save(args.save, fps=max(1, 1000 // args.interval))
        print(f"Saved animation to {args.save}")
    else:
        plt.show()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=["tier2", "tier3"], default="tier2",
                   help="tier2: advertiser 1 learns, others fixed at b*; tier3: all learn")
    p.add_argument("--learner", default="thompson", help="registered agent name (see --list)")
    p.add_argument("--agents", nargs="+", help="custom market: one registered agent per advertiser")
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--v", type=float, default=5.0, help="click value (5 in all experiments)")
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--interval", type=int, default=60, help="ms between rounds")
    p.add_argument("--save", help="write animation to .gif / .mp4 instead of opening a window")
    p.add_argument("--headless", action="store_true", help="text log only, no window")
    p.add_argument("--every", type=int, default=25, help="headless: print every k rounds")
    p.add_argument("--list", action="store_true", help="list registered agents and exit")
    args = p.parse_args(argv)

    if args.list:
        print("\n".join(available_agents()))
        return 0
    known = set(available_agents())
    for name in (args.agents or [args.learner]):
        if name not in known:
            p.error(f"unknown agent '{name}'; choose from {sorted(known)}")

    cfg, inst, bench, agents, names = build_market(args)
    describe(inst, bench, names, args)
    gen = step_market(inst, bench, agents, args.T, args.seed)
    if args.headless:
        run_headless(gen, names, args.every)
    else:
        run_animated(gen, inst, bench, names, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
