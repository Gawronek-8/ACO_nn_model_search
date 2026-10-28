from __future__ import annotations

from collections.abc import Iterable

from ..utils import END, LayerConfig, PheromoneTable, Step
from .base import TYPE_END, TYPE_FC, Cell, MoveContext, PheromoneStrategy, type_mask


class ConfigDependentStrategy(PheromoneStrategy):
    """Whole layer configs as nodes: tau[depth, prev_config, config].

    Context 0 is the input, context i + 1 is config i (see SearchSpace.config_index).
    FC/END is a separate choice with the same context rather than an extra config column,
    so every variant starts with the same 50/50 chance to stop.
    """

    def build_tables(self) -> None:
        depth = self.space.max_depth
        tau0 = self.config.pheromone.start
        n_configs = self.space.n_configs
        self.tables["type"] = PheromoneTable(depth, n_configs + 1, 2, tau0)
        self.tables["config"] = PheromoneTable(depth, n_configs + 1, n_configs, tau0)

    def on_ant_movement(self, ctx: MoveContext) -> Step:
        context = self._context(ctx.prev)
        if self.select("type", ctx.depth, context, type_mask(ctx)) == TYPE_END:
            return END
        return self.space.config_at(self.select("config", ctx.depth, context))

    def cells(self, depth: int, prev: LayerConfig | None, step: Step) -> Iterable[Cell]:
        context = self._context(prev)
        if step is END:
            return [(self.tables["type"], context, TYPE_END)]
        return [
            (self.tables["type"], context, TYPE_FC),
            (self.tables["config"], context, self.space.config_index(step)),
        ]

    def _context(self, prev: LayerConfig | None) -> int:
        return 0 if prev is None else self.space.config_index(prev) + 1 # IT USES THE CONFIG_INDEX, CUZ EVERY CONFIG IS ONE ELEMENT HERE
