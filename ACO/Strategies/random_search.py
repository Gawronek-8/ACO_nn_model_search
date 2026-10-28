from __future__ import annotations

from collections.abc import Iterable

from ..utils import END, PARAMS, LayerConfig, Step
from .base import Cell, MoveContext, PheromoneStrategy


class RandomSearchStrategy(PheromoneStrategy):
    """Baseline: the same choices as the ACO strategies (FC/END, then each parameter), all uniform.

    Keeps no pheromones, so both updates do nothing.
    """

    def build_tables(self) -> None:
        pass

    def on_ant_movement(self, ctx: MoveContext) -> Step:
        if ctx.can_end and self.rng.random() < 0.5:
            return END
        return LayerConfig(**{
            param: self.space.options(param)[self.rng.integers(len(self.space.options(param)))]
            for param in PARAMS
        })

    def cells(self, depth: int, prev: LayerConfig | None, step: Step) -> Iterable[Cell]:
        return []
