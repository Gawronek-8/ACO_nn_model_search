"""Shared building blocks: the layer description, the ACS selection rule and pheromone tables."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


@dataclass(frozen=True)
class LayerConfig:
    """One hidden FC layer chosen by an ant."""

    neurons: int
    activation: str
    dropout: float


class End(Enum):
    """Marker for the ant's decision to stop adding layers."""

    END = "END"


END = End.END
Step = LayerConfig | End

# LayerConfig fields the ants choose; SearchSpace has an options list under the same names
PARAMS = ("neurons", "activation", "dropout")


def acs_select(
    scores: np.ndarray,
    q0: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> int:
    """ACS transition rule: with probability q0 take the best option, otherwise roulette wheel.

    Ties in the greedy branch are broken at random. Options where `mask` is False are never chosen.
    Returns the index of the chosen option.
    """
    candidates = np.arange(len(scores)) if mask is None else np.flatnonzero(mask)
    if candidates.size == 0:
        raise ValueError("The mask does not allow any option")

    values = np.asarray(scores, dtype=float)[candidates]
    if rng.random() < q0:
        best = np.flatnonzero(values == values.max())
        return int(candidates[rng.choice(best)])

    total = values.sum()
    probabilities = values / total if total > 0 else None
    return int(rng.choice(candidates, p=probabilities))


class PheromoneTable:
    """Pheromone values tau[depth, context, value] with the ACS local and global updates.

    `context` is what the choice depends on (nothing, the previous value, the previous config),
    `value` is the option being chosen.
    """

    def __init__(self, depth: int, n_contexts: int, n_values: int, tau0: float):
        self.tau0 = tau0
        self.tau = np.full((depth, n_contexts, n_values), tau0, dtype=float)

    def select(
        self,
        depth: int,
        context: int,
        q0: float,
        rng: np.random.Generator,
        mask: np.ndarray | None = None,
    ) -> int:
        return acs_select(self.tau[depth, context], q0, rng, mask)

    def local_update(self, depth: int, context: int, value: int, decay: float) -> None:
        """tau <- (1 - decay) * tau + decay * tau0, pulls a used cell back so other ants explore."""
        old = self.tau[depth, context, value]
        self.tau[depth, context, value] = (1 - decay) * old + decay * self.tau0

    def global_update(self, depth: int, context: int, value: int, evaporation: float, score: float) -> None:
        """tau <- (1 - evaporation) * tau + evaporation * score, reinforces the best path."""
        old = self.tau[depth, context, value]
        self.tau[depth, context, value] = (1 - evaporation) * old + evaporation * score


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch (all devices) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
