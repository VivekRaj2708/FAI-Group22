# FAI-Group22

Learning to Bid Under a Budget: a batched first-price ad auction with unknown click probabilities.
See [`Agents.md`](Agents.md) for the agent specification and [`Doc/main.tex`](Doc/main.tex) for the proposal.

```bash
pip install -r requirements.txt
python -m pytest
```

## Layout

| Path | Contents | Owner |
|---|---|---|
| `src/env/` | instance generator, auction mechanics | Phase 0 |
| `src/tier1/` | knapsack best response, equilibria, social planner | Phase 1 |
| `src/agents/` | shared `BaseAdvertiserAgent` interface, agent registry, reference agents | Phase 0 / 4 |
| `src/eval/` | evaluation harness: simulator, metrics, stats, sweeps, plots, report | Phase 4 / 5 |
| `Doc/report/` | final report (`report.tex`) + generated tables/figures | Phase 5 |

## Evaluation harness (Phases 4 & 5)

```bash
python run_evaluation.py agents                                   # list registered agents
python run_evaluation.py run --scenario tier2 --learner ucb --m 2 --n 10 --kappa 0.5 --T 2000
python run_evaluation.py sweep --preset tier1 --T 1000 --out results/sweep_tier1
python run_evaluation.py sweep --preset paper --out results/sweep_paper   # Tier-2 + Tier-3 learners
python run_evaluation.py report --sweep-dir results/sweep_tier1 --report-dir Doc/report
```

Scenarios: `tier1` (everyone plays the Tier-1 reference b*), `tier2` (agent 0 learns, others fixed at b*;
non-pure Tier-1 instances excluded), `tier3` (everyone runs the learner), `custom` (one agent name per advertiser).
Round structures: `canonical`, `discounted`, `cumulative` (fluid-LP regret benchmark).

Every run reports, per seed and as mean ± 95% CI over seeds: cumulative regret (and its log-log slope),
equilibrium gap ε_t, ‖b^t − b*‖₁, allocation efficiency (expected and realized), platform revenue, welfare,
advertiser profit, and price of learning (Tier-1 value − learned value).

### Registered learners

The Tier-2 agents in `src/tier2/agents.py` are registered in `src/agents/tier2_adapters.py`:
`ucb`, `thompson`, `greedy`, `eps_greedy` and `paced` (`CumulativeBudgetPacedAgent`; it gets the real total
budget under `cumulative`, and `B_i * T` otherwise). The same names work for `tier3` (all advertisers learn).

### Plugging in a new agent

Implement `src.agents.base.BaseAdvertiserAgent` and register a factory anywhere under `src/tier2/` or
`src/tier3/` (the harness imports those packages automatically):

```python
from src.agents.registry import register_agent, AgentContext

@register_agent("ucb")
def make_ucb(ctx: AgentContext, alpha: float = 2.0):
    return UCBAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value, alpha=alpha, seed=ctx.seed)
```

`ctx` also carries `horizon`, `round_structure`, `total_budget` (cumulative variant) and, for oracle agents
only, `true_click_probs` / `reference_bids`. Learners must not read `true_click_probs`.
Bids must be non-negative integers with `sum <= budget` (and `<=` remaining budget under `cumulative`);
infeasible vectors are replaced by zero bids and counted in the `violations` metric.
