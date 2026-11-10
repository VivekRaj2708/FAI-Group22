"""
Registers the Tier-2 bandit learners (``src.tier2.agents``) with the harness.

Kept here instead of in ``src/tier2`` so the Tier-2 package needs no changes.

Registry names (used by ``run_evaluation.py --learner`` and sweep presets):

* ``ucb``         - UCBAgent (Algorithm 1), ``alpha`` default 2
* ``thompson``    - ThompsonSamplingAgent, Beta(1, 1) prior
* ``greedy``      - GreedyAgent (alpha = 0)
* ``eps_greedy``  - EpsGreedyAgent, ``epsilon`` default 0.1
* ``paced``       - CumulativeBudgetPacedAgent (dual pacing). Under the
                    cumulative structure it gets the real total budget; under
                    canonical/discounted it gets ``B_i * T``, so the pacing
                    constraint is slack and it behaves like a per-round-capped
                    UCB learner with a dual penalty.
"""
from src.agents.registry import AgentContext, register_agent
from src.tier2.agents import (
    CumulativeBudgetPacedAgent,
    EpsGreedyAgent,
    GreedyAgent,
    ThompsonSamplingAgent,
    UCBAgent,
)


@register_agent("ucb")
def make_ucb(ctx: AgentContext, alpha: float = 2.0):
    return UCBAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value, alpha=alpha)


@register_agent("thompson")
def make_thompson(ctx: AgentContext, prior_alpha: float = 1.0, prior_beta: float = 1.0):
    return ThompsonSamplingAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value,
                                 prior_alpha=prior_alpha, prior_beta=prior_beta, seed=ctx.seed)


@register_agent("greedy")
def make_greedy(ctx: AgentContext, optimistic_unvisited: float = 1.0):
    return GreedyAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value,
                       optimistic_unvisited=optimistic_unvisited)


@register_agent("eps_greedy")
def make_eps_greedy(ctx: AgentContext, epsilon: float = 0.1):
    return EpsGreedyAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value,
                          epsilon=epsilon, seed=ctx.seed)


@register_agent("paced")
def make_paced(ctx: AgentContext, alpha: float = 2.0, eta: float = None):
    total = ctx.total_budget if ctx.total_budget is not None else ctx.budget * ctx.horizon
    return CumulativeBudgetPacedAgent(ctx.agent_id, total_budget=total, num_users=ctx.num_users,
                                      total_rounds=ctx.horizon, click_value=ctx.click_value,
                                      per_round_max_budget=ctx.budget, alpha=alpha, eta=eta)
