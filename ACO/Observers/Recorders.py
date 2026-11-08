from __future__ import annotations

from pathlib import Path

import numpy as np

from ..Strategies.base import IterationContext, SearchContext
from .base import BaseObserver, Pheromones


class PheromoneRecorder(BaseObserver):
    """Keeps a copy of every pheromone table at the start of the search and after each iteration.

    Snapshot 0 is the initial state (all tau0), snapshot i is the state after the global update of iteration i.
    Strategies without pheromones (random search) leave nothing to record.
    """

    def __init__(self) -> None:
        self.snapshots: list[Pheromones] = []

    def on_search_start(self, ctx: SearchContext, pheromones: Pheromones) -> None:
        self.snapshots = [pheromones]

    def on_global_step_end(self, ctx: IterationContext, pheromones: Pheromones) -> None:
        self.snapshots.append(pheromones)

    def tables(self) -> dict[str, np.ndarray]:
        """For every table an array of shape (snapshots, depth, contexts, values); empty without pheromones."""
        if not self.snapshots or not self.snapshots[0]:
            return {}
        return {name: np.stack([snapshot[name] for snapshot in self.snapshots]) for name in self.snapshots[0]}

    def save(self, path: str | Path) -> bool:
        """Writes the tables to a compressed .npz (one array per table). Returns False if there was nothing to save."""
        tables = self.tables()
        if not tables:
            return False
        np.savez_compressed(path, **tables)
        return True
