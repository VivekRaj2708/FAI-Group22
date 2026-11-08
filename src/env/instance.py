from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import json
import random


@dataclass
class AuctionInstance:
    """Represents a batched first-price ad auction environment instance."""

    num_advertisers: int
    num_users: int
    budgets: List[int]
    click_probabilities: List[List[float]]  # Shape: (m, n) where [i][j] = f(u_j, a_i)
    click_value: float = 1.0
    name: Optional[str] = None

    def __post_init__(self):
        if len(self.budgets) != self.num_advertisers:
            raise ValueError(
                f"Budgets length {len(self.budgets)} != num_advertisers {self.num_advertisers}"
            )
        if len(self.click_probabilities) != self.num_advertisers:
            raise ValueError(
                f"Click probabilities outer length {len(self.click_probabilities)} != {self.num_advertisers}"
            )
        for i, row in enumerate(self.click_probabilities):
            if len(row) != self.num_users:
                raise ValueError(
                    f"Row {i} length {len(row)} != num_users {self.num_users}"
                )
            for j, val in enumerate(row):
                if not (0.0 <= val <= 1.0):
                    raise ValueError(
                        f"f[{i}][{j}] = {val} not in [0, 1]"
                    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AuctionInstance":
        return cls(**data)

    def save_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "AuctionInstance":
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def generate_synthetic_instance(
    num_advertisers: int = 3,
    num_users: int = 5,
    budget_tightness: float = 0.5,
    click_value: float = 1.0,
    seed: Optional[int] = 42,
    beta_params: Tuple[float, float] = (2.0, 5.0),
    name: Optional[str] = None,
) -> AuctionInstance:
    """
    Generates a reproducible synthetic auction instance.

    Args:
        num_advertisers (m): Number of competing advertisers.
        num_users (n): Number of impression opportunities per round.
        budget_tightness (kappa): Fraction of total full-market win cost.
            Nominal full cost to buy all n users at price 1 is n.
            Budget B_i = max(1, round(kappa * n)).
        click_value (v): Value per realized click. Default 1.0.
        seed: Random seed for reproducibility.
        beta_params: Shape parameters (alpha, beta) for Beta-distributed click rates.
        name: Optional descriptor.
    """
    rng = random.Random(seed)

    # Click probabilities drawn from Beta distribution
    click_probabilities: List[List[float]] = []
    alpha_param, beta_param = beta_params
    for _ in range(num_advertisers):
        row = [
            round(rng.betavariate(alpha_param, beta_param), 4)
            for _ in range(num_users)
        ]
        click_probabilities.append(row)

    # Budgets determined by budget tightness
    # If kappa=1.0, budget is enough to bid 1 on every user (or 2 on half).
    base_budget = max(1, int(round(budget_tightness * num_users)))
    budgets = [base_budget] * num_advertisers

    instance_name = (
        name
        or f"synth_m{num_advertisers}_n{num_users}_k{budget_tightness}_seed{seed}"
    )

    return AuctionInstance(
        num_advertisers=num_advertisers,
        num_users=num_users,
        budgets=budgets,
        click_probabilities=click_probabilities,
        click_value=click_value,
        name=instance_name,
    )
