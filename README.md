# Learning to Bid Under a Budget

**A batched first-price ad auction with unknown click probabilities.**
CS329 Foundations of AI, Group 22, IIT Gandhinagar. Vivek Raj, Anvit Aggarwal, Eshaan Ashish Jaiswal, Gella Naga Sai Krishna.

Repository: <https://github.com/VivekRaj2708/FAI-Group22>

| Deliverable | Where |
|---|---|
| **Final report (8 pages, PDF)** | [`Doc/report/final_report.pdf`](Doc/report/final_report.pdf) (source: [`final_report.tex`](Doc/report/final_report.tex)) |
| Supplementary material (every per-cell table and figure, 95% CIs) | [`Doc/report/supplementary.tex`](Doc/report/supplementary.tex) |
| Slides | [`Doc/presentation/slides.tex`](Doc/presentation/slides.tex) |
| Live demo | `python run_demo.py`, or open [`demo/index.html`](demo/index.html) in a browser |
| Submitted proposal and TA review | [`proposal/`](proposal/) (LaTeX source: [`Doc/main.tex`](Doc/main.tex)) |

---

## 1. The problem

`m` advertisers compete for a fixed set of `n` users over `T` rounds. In each round every advertiser submits
**one integer bid per user**, with the bids summing to at most its budget `B_i`. On each user the highest bid
wins, ties are broken at random, and the winner **pays its own bid** whether or not the user clicks. User `j`
clicks advertiser `i`'s ad with probability `f_ij`. The advertisers **do not know** `f_ij` and only see clicks
on users they won.

When `f` is known, the best response to competing bids `c` is a **0-1 knapsack**: winning user `j` costs
`c_j + 1` and is worth `f_ij * v - (c_j + 1)`. It is solved exactly by dynamic programming in `O(n B_i)`.
Every learner in this project plugs an estimate of `f` into the same knapsack.

We relax one assumption at a time:

| Tier | Setting | What we compute |
|---|---|---|
| **1** | `f` known | Exact knapsack best responses, reference equilibrium `b*`, social-planner benchmark |
| **2** | One advertiser learns `f`; opponents fixed at `b*` | Regret of Thompson sampling, UCB, greedy, ε-greedy and a budget-paced learner |
| **3** | All advertisers learn at once | Equilibrium gap, distance to `b*`, efficiency, revenue, price of learning |

No model is trained offline. Each advertiser learns online, as a combinatorial semi-bandit: per user it keeps
win and click counts, which feed a UCB index or a Beta posterior.

---

## 2. How to run

Requires Python 3.10+.

```bash
pip install -r requirements.txt

python run_tests.py         # 54 tests: Tier 1, Tier 2, Tier 3, evaluation harness, demo
python report_numbers.py    # recomputes every number in the final report (see Section 4)
python run_demo.py          # live, animated market using the real agents
```

### Single experiments for each tier

```bash
# Tier 1: knapsack best responses and reference equilibrium b* for one instance
python run_tier1.py --m 2 --n 4 --kappa 0.5 --v 5.0 --output tier1_benchmark.json

# Tier 2: all learners against fixed b* opponents, regret curves
python run_tier2.py --rounds 200 --trials 5 --plot tier2_regret_curves.png --output tier2_benchmark.json
python run_tier2.py --learners paced --round-structure cumulative --rounds 200

# Tier 3: every advertiser learns (populations: all_ucb, all_ts, mixed_ucb_ts, mixed_all, learner_vs_equilibrium)
python run_tier3.py --m 2 --n 4 --population all_ts --rounds 300 --trials 5 --plot tier3_convergence_plots.png

# Evaluation harness: any registered agent, 30 seeds, curves and a summary with 95% CIs
python run_evaluation.py agents
python run_evaluation.py run --scenario tier3 --learner thompson --m 2 --n 10 --kappa 0.5 --T 1000
```

Run each script with `--help` to see all of its options (round structure, α, ε, discount factor, seeds).

### Live demo

```bash
python run_demo.py --scenario tier2 --learner thompson    # one learner vs. b* opponents
python run_demo.py --scenario tier3 --learner ucb --m 3   # everyone learns
python run_demo.py --agents thompson ucb greedy           # head-to-head market
python run_demo.py --headless --T 200                     # text log only, no window
```

`demo/index.html` is a JavaScript version of the same demo that runs in any browser without installing anything.

---

## 3. Results

Every result uses 30 seeds per cell over `m ∈ {2,3,5}`, `n ∈ {5,10,20,50}` and budget level `κ ∈ {0.1,0.25,0.5,1}`
(48 cells), with `f_ij ~ Beta(2,5)` and click value `v = 5`. Shaded bands in the figures are 95% confidence intervals.

**Tier 2: one learner, horizon T = 1000** (means over the 47 cells that have a pure equilibrium)

| Learner | Regret / round | Regret exponent (T = 10³ / 10⁴) | Cells with lowest regret |
|---|---|---|---|
| Thompson sampling | **0.49** | **0.33 / ~0.2** | **47 / 47** |
| UCB, α = 0.5 | 0.73 | 0.46 / 0.23–0.38 | – |
| UCB, α = 2 | 1.77 | 0.73 / 0.50–0.58 | 0 |
| ε-greedy (ε = 0.1) | 1.22 | 0.66 / 0.80–0.84 | 0 |
| Greedy | 2.95 | 0.79 / 0.96–1.00 | 0 |

A regret exponent of 1 means linear regret, 0.5 means √T-type, and a value near 0 means close to logarithmic.

**Tier 3: all advertisers learn** (canonical budgets, T = 1000)

| Learner | Final equilibrium gap, m = 2 / 3 / 5 | Final efficiency | Efficiency loss vs. Tier 1 |
|---|---|---|---|
| Known-f best response (control) | 0.76 / 1.42 / 2.37 | 0.53 | 0.04 |
| Thompson sampling | **0.67 / 1.25** / 2.48 | **0.56** | **0.02** |
| UCB, α = 2 | 2.32 / 3.67 / 4.93 | 0.35 | 0.23 |
| Greedy | 4.79 / 4.82 / 4.42 | 0.29 | 0.29 |

**Findings**
- **H1 holds.** Regret is sublinear for every learner whose exploration decays. Thompson sampling is best in all 47 cells; greedy's regret is close to linear.
- **H2 holds.** The gap to equilibrium grows with `m`. Most of the gap is already present with known `f`, because simultaneous best responses cycle.
- **H3 is not supported.** Scarce budgets lower efficiency through the equilibrium itself, not by slowing learning.
- **Cumulative (campaign) budgets:** the budget-paced learner is the only one still earning at the end of the campaign, and it has the smallest Tier-3 gap.

### Screenshots

**Tier 1: efficiency of the reference equilibrium.** Scarce budgets cost efficiency even when `f` is known.

![Tier-1 efficiency heatmap](docs/screenshots/tier1_efficiency_heatmap.png)

**Tier 2: cumulative regret** (m = 2, n = 10, κ = 0.5), T = 1000 (left) and T = 10⁴ (right).

<p>
<img src="docs/screenshots/tier2_cum_regret_T1000.png" width="49%" alt="Tier-2 cumulative regret, T=1000">
<img src="docs/screenshots/tier2_cum_regret_T10000.png" width="49%" alt="Tier-2 cumulative regret, T=10000">
</p>

**Tier 2: Thompson sampling final regret in every (m, n, κ) cell.**

![Thompson sampling regret heatmap](docs/screenshots/tier2_thompson_regret_heatmap.png)

**Tier 3: equilibrium gap and allocation efficiency** (m = 2, n = 10, κ = 0.5).

<p>
<img src="docs/screenshots/tier3_equilibrium_gap.png" width="49%" alt="Tier-3 equilibrium gap">
<img src="docs/screenshots/tier3_efficiency.png" width="49%" alt="Tier-3 allocation efficiency">
</p>

**Tier 3 over T = 10⁴:** the gap keeps shrinking for Thompson sampling and UCB, but not for greedy.

![Tier-3 equilibrium gap, T=10^4](docs/screenshots/tier3_equilibrium_gap_T10000.png)

**Live demo:** bids, wins, clicks, learned vs. true click rates, regret, gap and efficiency.

![Live demo](docs/screenshots/live_demo.png)

---

## 4. Checking the numbers in the report

The sweep outputs behind the report are committed in `results/sweep_*/` (`summary.csv`, plus the exact settings in `spec.json`).

```bash
python report_numbers.py                 # every table and in-text number of the final report
python report_numbers.py --section t2    # one section: t1 | t2 | t3 | rob
```

To rerun the sweeps from scratch, then rebuild the figures and per-cell tables used by the report:

```bash
python run_evaluation.py sweep --preset tier1 --T 1000 --out results/sweep_tier1
python run_evaluation.py sweep --preset paper --T 1000 --structures canonical cumulative \
    --learners ucb thompson greedy eps_greedy paced --scenarios tier2 tier3 --out results/sweep_learners
python run_evaluation.py sweep --preset paper --T 1000 --structures discounted \
    --learners ucb thompson greedy eps_greedy paced --scenarios tier2 tier3 --out results/sweep_discounted
python run_evaluation.py sweep --preset paper --T 1000 --structures canonical \
    --learners ucb_a05 ucb_a1 --scenarios tier2 tier3 --out results/sweep_ucb_alpha
python run_evaluation.py sweep --preset paper --T 10000 --m 2 3 --n 10 20 --kappa 0.5 --structures canonical \
    --learners ucb ucb_a05 thompson greedy eps_greedy --scenarios tier2 tier3 --out results/sweep_horizon
python run_evaluation.py report --sweep-dir results/sweep_tier1 results/sweep_learners \
    results/sweep_discounted results/sweep_ucb_alpha results/sweep_horizon --report-dir Doc/report
```

The full learner sweep takes about 20 minutes on 14 cores; add `--workers N` to set the number of parallel processes.
To compile the report, run `pdflatex final_report` twice from `Doc/report/`, or upload `Doc/` to Overleaf.

---

## 5. Repository layout

```
src/
  env/        auction engine (batched first-price, tie-breaking, feedback) and instance generator
  tier1/      knapsack DP, equilibrium solvers (exhaustive search, BRD, fictitious play), social planner
  tier2/      learners (UCB, Thompson sampling, greedy, ε-greedy, budget-paced) and single-learner simulation
  tier3/      multi-agent simulator, convergence-rate and cycling analysis, price of learning
  agents/     shared agent interface (get_bid / observe / reset) and the agent registry
  eval/       evaluation harness: metrics, 95% CIs, parameter sweeps, plots, LaTeX tables
tests/        test suites for every tier, the harness and the demo
results/      committed sweep summaries used by the report
Doc/report/   final_report.tex/.pdf, supplementary.tex, figures/, generated/ tables
Doc/presentation/  slides
demo/         browser demo
docs/screenshots/  figures shown in this README
run_tier1.py, run_tier2.py, run_tier3.py, run_evaluation.py, run_demo.py, run_tests.py, report_numbers.py
collect_convergence_logs.py   batch collection of Tier-3 convergence logs
Agents.md                     architecture and agent specifications
```

---

## 6. Team and contributions

| Member | Assigned task | Status |
|---|---|---|
| Vivek Raj | Tier 1: knapsack best response, equilibrium computation, reference `b*` | Fulfilled |
| Anvit Aggarwal | Tier 2: UCB, Thompson sampling, greedy baselines, budget-paced learner, regret evaluation | Fulfilled |
| Eshaan Ashish Jaiswal | Tier 3: multi-agent simulator and convergence data | Fulfilled |
| Gella Naga Sai Krishna | Evaluation harness, sensitivity study, report, slides, demo | Fulfilled |

Section 6 of the report gives the details.
