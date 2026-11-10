"""
Agent registry used by the evaluation harness.

Tier-2 / Tier-3 modules make their agents available to the harness by
registering a factory::

    from src.agents.registry import register_agent, AgentContext

    @register_agent("ucb")
    def make_ucb(ctx: AgentContext, alpha: float = 2.0):
        return UCBAgent(ctx.agent_id, ctx.budget, ctx.num_users, ctx.click_value,
                        alpha=alpha, seed=ctx.seed)

The harness imports ``src.tier2`` and ``src.tier3`` (if they exist) before
looking agents up, so registering at import time is enough.
"""
from dataclasses import dataclass, field
import importlib
from typing import Callable, Dict, List, Optional

from src.agents.base import BaseAdvertiserAgent


@dataclass
class AgentContext:
    """Everything the harness knows when constructing agent ``agent_id``.

    ``true_click_probs`` is the ground-truth row f(., a_i). Only oracle /
    Tier-1 agents may use it; learners must ignore it.
    """

    agent_id: int
    budget: int                       # per-round bid cap B_i
    num_users: int
    click_value: float
    horizon: int                      # T
    seed: int                         # agent-private RNG seed
    round_structure: str = "canonical"
    total_budget: Optional[int] = None  # cumulative variant only
    reference_bids: Optional[List[int]] = None  # Tier-1 b*_i
    true_click_probs: Optional[List[float]] = None  # oracle-only
    extra: Dict = field(default_factory=dict)


AgentFactory = Callable[..., BaseAdvertiserAgent]

_REGISTRY: Dict[str, AgentFactory] = {}

# Packages that may register agents on import (owned by other phases).
_PLUGIN_PACKAGES = ("src.tier2", "src.agents.tier2_adapters", "src.tier3", "src.agents.reference")
_plugins_loaded = False


def register_agent(name: str) -> Callable[[AgentFactory], AgentFactory]:
    """Decorator registering ``factory(ctx, **kwargs) -> agent`` under ``name``."""

    def deco(factory: AgentFactory) -> AgentFactory:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"Agent '{name}' is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    for pkg in _PLUGIN_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ModuleNotFoundError as exc:
            # Only swallow "package does not exist yet"; real import errors surface.
            if exc.name is None or not pkg.startswith(exc.name):
                raise
    _plugins_loaded = True


def available_agents() -> List[str]:
    _load_plugins()
    return sorted(_REGISTRY)


def make_agent(name: str, ctx: AgentContext, **kwargs) -> BaseAdvertiserAgent:
    _load_plugins()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown agent '{name}'. Registered agents: {sorted(_REGISTRY)}. "
            "Tier-2/3 agents must call register_agent() in src/tier2 or src/tier3."
        )
    return _REGISTRY[name](ctx, **kwargs)
