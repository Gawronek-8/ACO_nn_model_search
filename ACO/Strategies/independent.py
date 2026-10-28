from __future__ import annotations

from collections.abc import Iterable

from ..utils import END, PARAMS, LayerConfig, PheromoneTable, Step
from .base import TYPE_END, TYPE_FC, Cell, MoveContext, PheromoneStrategy, type_mask


class IndependentStrategy(PheromoneStrategy):
    """Pheromone per (depth, value) for every parameter: tau_p[depth, value].

    Choices ignore the previous layer. Every table has a single context (index 0).
    """

    def build_tables(self) -> None:
        depth = self.space.max_depth
        tau0 = self.config.pheromone.start
        self.tables["type"] = PheromoneTable(depth, 1, 2, tau0)
        for param in PARAMS:
            self.tables[param] = PheromoneTable(depth, 1, len(self.space.options(param)), tau0)

    def on_ant_movement(self, ctx: MoveContext) -> Step:
        if self.select("type", ctx.depth, 0, type_mask(ctx)) == TYPE_END:
            return END
        return LayerConfig(**{
            param: self.space.options(param)[self.select(param, ctx.depth, 0)] # THE LAST VALUE IN SELECT IS 0 BECAUSSE IN INDEPENDENT WE DO NOT LOOK BACK
            for param in PARAMS
        })

    def cells(self, depth: int, prev: LayerConfig | None, step: Step) -> Iterable[Cell]:
        if step is END:
            return [(self.tables["type"], 0, TYPE_END)]
        return [(self.tables["type"], 0, TYPE_FC)] + [
            (self.tables[param], 0, self.space.value_index(param, getattr(step, param))) # THIS RETURNS THE CORRECT PHEROMONE TABLE, THE CONTEXT AND THE VALUE INDEXES
            for param in PARAMS
        ]
