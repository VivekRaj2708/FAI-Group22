# FAI-Group22: Learning to Bid Under a Budget

**Project Title:** Learning to Bid Under a Budget: A Batched First-Price Ad Auction with Unknown Click Probabilities  
**Authors:** Vivek Raj, Anvit Aggarwal, Eshaan Ashish Jaiswal, Gella Naga Sai Krishna  
**Course:** Foundations of Artificial Intelligence (FAI)

---

## 1. System Overview & Architecture

This repository implements a repeated batched first-price ad auction environment where budget-constrained advertisers compete to display ads to $n$ users with unknown click probabilities $f_{ij} = f(u_j, a_i) \in [0, 1]$:

```
                      ┌────────────────────────────────────────┐
                      │          Auction Platform              │
                      │  - Resolves highest bids per user      │
                      │  - Charges winners their exact bid     │
                      │  - Simulates clicks: X ~ Ber(f_ij)     │
                      └────▲──────────────────────────────┬─────┘
                           │                              │
         Submits Bid Vector│                              │ Feedback:
        b_i = (b_i1,..,b_in│                              │ {w_ij, X_ij, b_ij, c_{-i,j}}
       s.t. sum(b_ij) <= B_│                              │
                           │                              ▼
                   ┌───────┴────────┐            ┌─────────────────┐
                   │ Advertiser a_1 │    ...     │ Advertiser a_m  │
                   │ (Learner/Equil)│            │ (Learner/Equil) │
                   └────────────────┘            └─────────────────┘
```

The experimental study is organized across three tiers:
1. **Tier 1 (Full Information Baseline & Equilibrium):** Exact 0-1 knapsack best-response dynamic programming, pure/approximate Nash equilibria ($b^*$), and social planner welfare ceiling.
2. **Tier 2 (Single-Agent Combinatorial Semi-Bandit):** Single budget-aware bandit learners (Optimistic UCB, Thompson Sampling, Greedy, $\epsilon$-Greedy, Cumulative Budget Paced) competing against stationary opponents playing Tier-1 equilibrium strategies.
3. **Tier 3 (Multi-Agent Concurrent Learning & Non-Stationary Convergence - Phase 3):** All $m$ advertisers learn and bid concurrently. Evaluates exploitability gap $\varepsilon_t$, distance to equilibrium $\|b^t - b^*\|_1$, per-agent regret, market efficiency, stability/cycling, and the Price of Learning.

---

## 2. Directory Structure

```
.
├── src/
│   ├── agents/
│   │   ├── base.py                   # BaseAdvertiserAgent abstract class
│   │   └── __init__.py
│   ├── env/
│   │   ├── auction.py                # BatchedFirstPriceAuction mechanics
│   │   ├── instance.py               # AuctionInstance & synthetic generators
│   │   └── __init__.py
│   ├── tier1/
│   │   ├── baseline.py               # Tier1Solver & baseline equilibrium agents
│   │   ├── equilibrium.py            # BRD, Support Enumeration, Fictitious Play
│   │   ├── knapsack.py               # 0-1 Knapsack DP solver
│   │   ├── social_planner.py         # First-best welfare planner
│   │   └── __init__.py
│   ├── tier2/
│   │   ├── agents.py                 # UCB, Thompson Sampling, Greedy, Paced
│   │   ├── simulation.py             # Single-agent simulation engine
│   │   └── __init__.py
│   └── tier3/
│       ├── convergence.py            # Convergence rate, cycling, Price of Learning
│       ├── simulation.py             # MultiAgentSimulator & trial/experiment aggregation
│       └── __init__.py
├── tests/
│   ├── test_tier1.py                 # Tier-1 test suite
│   ├── test_tier2.py                 # Tier-2 test suite
│   └── test_tier3.py                 # Tier-3 test suite
├── run_tier1.py                      # Tier-1 CLI runner
├── run_tier2.py                      # Tier-2 CLI runner
├── run_tier3.py                      # Tier-3 Multi-agent CLI runner
├── run_tests.py                      # Consolidated test runner
├── Agents.md                         # Architecture and specifications
└── pyproject.toml
```

---

## 3. Project Deliverables & Run Instructions

All deliverables defined in `Doc/main.tex` and `AGENTS.md` are fully implemented and verifiable via the commands below.

### 3.0 Consolidated Test Suite Runner
Executes all 26 unit and integration tests across Tier 1, Tier 2, and Tier 3:
```bash
python run_tests.py
```
Or run each tier's test suite individually:
```bash
python tests/test_tier1.py
python tests/test_tier2.py
python tests/test_tier3.py
```

---

### 3.1 Deliverable (i): Tier-1 Solver with Equilibrium Computation (Phase 1)
Solves exact knapsack best responses via DP and computes reference equilibrium profiles $b^*$ via Support Enumeration or Best-Response Dynamics / Fictitious Play:
```bash
# Standard 2-advertiser, 4-user instance with budget tightness kappa=0.5
python run_tier1.py --m 2 --n 4 --kappa 0.5 --v 5.0 --output tier1_benchmark.json

# 3-advertiser, 5-user instance with tighter budget kappa=0.25
python run_tier1.py --m 3 --n 5 --kappa 0.25 --v 5.0 --output tier1_benchmark_3x5.json
```
**Artifact Produced:**
- `tier1_benchmark.json`: Contains reference equilibrium profile $b^*$, expected profits $p^*$, platform revenue $\mathrm{Rev}(b^*)$, welfare $\mathrm{Wel}(b^*)$, and social planner benchmark.

**CLI Options for `run_tier1.py`:**
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--m` | `int` | `2` | Number of advertisers |
| `--n` | `int` | `4` | Number of users |
| `--kappa` | `float` | `0.5` | Budget tightness $\kappa \in [0.1, 1.0]$ |
| `--v` | `float` | `5.0` | Value per click $v$ |
| `--seed` | `int` | `42` | Random seed for instance generation |
| `--output` | `str` | `tier1_benchmark.json` | Output JSON path |
| `--load-instance` | `str` | `None` | Path to load pre-existing instance JSON |

---

### 3.2 Deliverable (ii): UCB and Thompson Learners with Regret Curves (Phase 2)
Simulates a single budget-aware learner against stationary opponents playing pure Tier-1 equilibrium bids, verifying sublinear regret:
```bash
# Compare all four learners (UCB, TS, Greedy, Eps-Greedy) across 5 seeds
python run_tier2.py --rounds 200 --trials 5 --plot tier2_regret_curves.png --output tier2_benchmark.json

# Run Thompson Sampling specifically with 500 rounds
python run_tier2.py --learners thompson --rounds 500 --trials 10

# Robustness check: Discounted variant (gamma = 0.95)
python run_tier2.py --round-structure discounted --discount-factor 0.95 --rounds 200

# Robustness check: Cumulative budget pacing
python run_tier2.py --learners paced --round-structure cumulative --rounds 200
```
**Artifacts Produced:**
- `tier2_benchmark.json`: Per-round instantaneous & cumulative regrets, profits, efficiencies, and 95% confidence intervals.
- `tier2_regret_curves.png`: Comparative regret curves confirming sublinear growth.

**CLI Options for `run_tier2.py`:**
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--m`, `--n`, `--kappa`, `--v` | - | `2, 4, 0.5, 5.0` | Instance parameters |
| `--rounds` | `int` | `200` | Horizon $T$ per simulation trial |
| `--trials` | `int` | `5` | Number of independent random seeds |
| `--learner-id` | `int` | `0` | Index of learning agent |
| `--learners` | `list` | `all` | Subset of: `ucb`, `thompson`, `greedy`, `eps_greedy`, `paced` |
| `--alpha` | `float` | `2.0` | UCB exploration parameter $\alpha$ |
| `--epsilon` | `float` | `0.1` | $\epsilon$-Greedy exploration parameter |
| `--round-structure` | `str` | `canonical` | `canonical`, `discounted`, or `cumulative` |
| `--discount-factor` | `float` | `0.99` | Discount factor $\gamma$ |
| `--output` | `str` | `tier2_benchmark.json` | Output JSON path |
| `--plot` | `str` | `tier2_regret_curves.png`| Output plot PNG path |

---

### 3.3 Deliverable (iii): Multi-Agent Simulator & Convergence Data (Phase 3 - Eshaan Jaiswal)
Simulates repeated auctions where all $m$ advertisers learn and bid concurrently in a non-stationary market. Computes equilibrium gap $\varepsilon_t$, distance to equilibrium $\|b^t - b^*\|_1$, per-agent regrets, stability/cycling, and the Price of Learning:

```bash
# 1. Homogeneous UCB Market (All advertisers use UCB, alpha=2.0)
python run_tier3.py --m 2 --n 4 --population all_ucb --rounds 300 --trials 5 --plot tier3_convergence_plots.png --output tier3_benchmark.json

# 2. Homogeneous Thompson Sampling Market (All advertisers use TS)
python run_tier3.py --m 2 --n 4 --population all_ts --rounds 300 --trials 5 --output tier3_benchmark_ts.json

# 3. Mixed UCB vs. Thompson Sampling Market
python run_tier3.py --m 2 --n 4 --population mixed_ucb_ts --rounds 300 --trials 5

# 4. Heterogeneous Market (m=4: UCB, Thompson Sampling, Greedy, EpsGreedy)
python run_tier3.py --m 4 --n 4 --population mixed_all --rounds 300 --trials 5

# 5. Online Learner vs. Fixed Tier-1 Equilibrium Opponents
python run_tier3.py --m 3 --n 5 --population learner_vs_equilibrium --rounds 300 --trials 5

# 6. Cumulative Budget Pacing Structure
python run_tier3.py --m 2 --n 4 --population all_ucb --round-structure cumulative --rounds 300 --trials 5
```
**Artifacts Produced:**
- `tier3_benchmark.json`: Raw convergence logs containing round-by-round equilibrium gaps $\varepsilon_t$, distances $\|b^t - b^*\|_1$, cumulative regrets $\mathrm{Reg}_T^{(i)}$, market efficiencies, stability analysis, power-law decay rate fits, and the complete Price of Learning vector with 95% confidence intervals.
- `tier3_convergence_plots.png`: Publication-grade 5-panel figure displaying:
  1. Equilibrium Exploitability Gap $\varepsilon_t$ vs. Rounds (mean with shaded 95% CI)
  2. L1 Distance to Tier-1 Equilibrium $\|b^t - b^*\|_1$ vs. Rounds
  3. Per-Agent Cumulative Regret Growth $\mathrm{Reg}_t^{(i)}$
  4. Allocation Efficiency $\mathrm{Eff}_t$ & Platform Revenue $\mathrm{Rev}_t$ vs. Tier-1 Baseline
  5. Advertiser Profit Comparison (Tier 1 Full-Info vs. Tier 3 Multi-Agent)

**CLI Options for `run_tier3.py`:**
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--m` | `int` | `2` | Number of advertisers |
| `--n` | `int` | `4` | Number of users |
| `--kappa` | `float` | `0.5` | Budget tightness $\kappa \in [0.1, 1.0]$ |
| `--v` | `float` | `5.0` | Value per click $v$ |
| `--rounds` | `int` | `300` | Horizon $T$ per simulation trial |
| `--trials` | `int` | `5` | Number of independent simulation seeds |
| `--population` | `str` | `all_ucb` | Choices: `all_ucb`, `all_ts`, `mixed_ucb_ts`, `mixed_all`, `learner_vs_equilibrium` |
| `--alpha` | `float` | `2.0` | UCB exploration parameter $\alpha$ |
| `--epsilon` | `float` | `0.1` | $\epsilon$-Greedy exploration parameter |
| `--round-structure` | `str` | `canonical` | `canonical`, `discounted`, or `cumulative` |
| `--discount-factor` | `float` | `0.99` | Discount factor $\gamma$ for discounted variant |
| `--output` | `str` | `tier3_benchmark.json` | Output JSON path |
| `--plot` | `str` | `tier3_convergence_plots.png`| Output plot PNG path |

---

### 3.4 Deliverables (iv) & (v): Sensitivity Analysis & Evaluation Across Factor Matrix (Phase 4 & 5)
Table 1 of `Doc/main.tex` and Section 6 of `AGENTS.md` specify evaluating market dynamics across advertisers $m \in \{2, 3, 5\}$, users $n \in \{5, 10, 20\}$, and budget tightness $\kappa \in \{0.1, 0.25, 0.5, 1.0\}$:

```bash
# 1. Budget Tightness Sweep (Scarce vs. Abundant)
python run_tier3.py --m 2 --n 5 --kappa 0.1 --rounds 200 --trials 5 --output tier3_kappa_0.10.json
python run_tier3.py --m 2 --n 5 --kappa 0.25 --rounds 200 --trials 5 --output tier3_kappa_0.25.json
python run_tier3.py --m 2 --n 5 --kappa 0.50 --rounds 200 --trials 5 --output tier3_kappa_0.50.json
python run_tier3.py --m 2 --n 5 --kappa 1.00 --rounds 200 --trials 5 --output tier3_kappa_1.00.json

# 2. Competitor Scaling Sweep (m = 2, 3, 5)
python run_tier3.py --m 2 --n 5 --kappa 0.5 --rounds 200 --trials 5 --output tier3_m2_n5.json
python run_tier3.py --m 3 --n 5 --kappa 0.5 --rounds 200 --trials 5 --output tier3_m3_n5.json
python run_tier3.py --m 5 --n 5 --kappa 0.5 --rounds 200 --trials 5 --output tier3_m5_n5.json

# 3. User Scaling Sweep (n = 5, 10, 20)
python run_tier3.py --m 2 --n 5 --kappa 0.5 --rounds 200 --trials 5 --output tier3_m2_n5.json
python run_tier3.py --m 2 --n 10 --kappa 0.5 --rounds 200 --trials 5 --output tier3_m2_n10.json
python run_tier3.py --m 2 --n 20 --kappa 0.5 --rounds 200 --trials 5 --output tier3_m2_n20.json
```

---

## 4. Summary of Output Metrics

| Metric | Formula / Definition | Evaluated in | Target Behavior |
| :--- | :--- | :--- | :--- |
| **Cumulative Regret** | $\mathrm{Reg}_T^{(i)} = \sum_{t=1}^T \left( p_i^*(c_{-i}^t) - p_i(b_i^t, b_{-i}^t) \right)$ | Tier 2 & Tier 3 | Sublinear $o(T)$ growth |
| **Equilibrium Gap** | $\varepsilon_t = \max_i \left( \max_{b_i'} p_i(b_i', b_{-i}^t) - p_i(b^t) \right)$ | Tier 1 & Tier 3 | $\varepsilon_t \to 0$ or small cycle bound |
| **Equilibrium Distance**| $\|b^t - b^*\|_1 = \sum_{i,j} |b_{ij}^t - b_{ij}^*|$ | Tier 3 | Empirical convergence to $b^*$ |
| **Allocation Efficiency**| $\mathrm{Eff}_t = \frac{1}{n} \sum_{j=1}^n \sum_{i \in \arg\max_k f_{kj}} q_{ij}(b^t)$ | All Tiers | Near-optimal user matching |
| **Platform Revenue** | $\mathrm{Rev}(b^t) = \sum_{i,j} q_{ij}(b^t) b_{ij}^t$ | All Tiers | Platform revenue generation |
| **Price of Learning** | $\Delta_{\mathrm{metric}} = \mathrm{Metric}(\text{Tier 1}) - \mathrm{Metric}(\text{Tier 3})$ | Tier 3 | Economic loss from online learning |
| **Power-Law Decay** | $\varepsilon_t \approx c \cdot t^{-\beta}$ | Tier 3 | $\beta > 0$ indicates decay towards equilibrium |
| **Stability State** | Fixed point, Limit cycle, or Wandering | Tier 3 | Detects asymptotic dynamical behavior |
