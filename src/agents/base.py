"""
Shared advertiser-agent interface (Agents.md, Section 4).

Every agent in every tier (Tier-1 oracles, Tier-2 bandit learners, Tier-3
concurrent learners) implements this interface so the evaluation harness in
``src.eval`` can run any of them unchanged.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseAdvertiserAgent(ABC):
    """Abstract base class for all bidding agents."""

    def __init__(self, agent_id: int, budget: int, num_users: int, click_value: float = 1.0):
        self.agent_id = agent_id
        self.budget = budget
        self.num_users = num_users
        self.click_value = click_value

    @abstractmethod
    def get_bid(self, t: int, last_competing_bids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes and returns integer bid vector b_i of shape (num_users,).

        ``t`` is the 1-indexed round number. ``last_competing_bids`` is
        c_{-i}^{t-1} (``None`` in round 1).

        Must satisfy:
            b_ij in {0, ..., budget}
            sum(b_ij) <= budget
        """

    @abstractmethod
    def observe(
        self,
        won: np.ndarray,            # bool array of shape (num_users,)
        clicks: np.ndarray,         # binary array (1 if click, 0 otherwise) for won users
        payments: np.ndarray,       # payments deducted for won users
        competing_bids: np.ndarray,  # highest competing bid per user c_{-i, j}
    ) -> None:
        """Updates internal beliefs and statistics from round feedback."""

    @abstractmethod
    def reset(self) -> None:
        """Resets beliefs and statistics for a new trial/seed."""
