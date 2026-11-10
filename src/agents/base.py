from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np


class BaseAdvertiserAgent(ABC):
    """
    Abstract base class for all bidding agents.
    Complies with the standard interface defined in Section 4 of Agents.md.
    """

    def __init__(
        self,
        agent_id: int,
        budget: int,
        num_users: int,
        click_value: float = 1.0,
        name: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.budget = budget
        self.num_users = num_users
        self.click_value = click_value
        self.name = name or f"{self.__class__.__name__}_{agent_id}"

    @abstractmethod
    def get_bid(
        self,
        t: int,
        last_competing_bids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[int]:
        """
        Computes and returns integer bid vector b_i of length num_users.
        Must satisfy:
            b_ij in {0, ..., budget}
            sum(b_ij) <= budget
        """
        pass

    @abstractmethod
    def observe(
        self,
        won: Union[List[int], np.ndarray],
        clicks: Union[List[int], np.ndarray],
        payments: Union[List[int], np.ndarray],
        competing_bids: Union[List[int], np.ndarray],
    ) -> None:
        """
        Updates internal beliefs, counts, and statistics from round feedback.
        Args:
            won: Indicator (0 or 1) per user if impression was won.
            clicks: Realized click indicator (0 or 1) for won users.
            payments: Payment deducted for won users.
            competing_bids: Highest competing bid per user c_{-i, j}.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets beliefs and statistics for a new trial/seed."""
        pass

    def get_estimated_click_probabilities(self) -> List[float]:
        """Returns the agent's current point or upper-bound click probability estimates."""
        return [0.0] * self.num_users
