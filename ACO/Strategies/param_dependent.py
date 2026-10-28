from __future__ import annotations

from collections.abc import Iterable

from ..utils import END, PARAMS, LayerConfig, PheromoneTable, Step
from .base import TYPE_END, TYPE_FC, Cell, MoveContext, PheromoneStrategy, type_mask


class ParamDependentStrategy(PheromoneStrategy):
    """Each parameter depends on the same parameter of the previous layer: tau_p[depth, prev_value, value].

    Context 0 is the input (no previous layer), context i + 1 is the previous layer's i-th option.
    The FC/END choice is conditioned on the previous layer's `END_CONTEXT_PARAM`.
    """

    END_CONTEXT_PARAM = "neurons"

    def build_tables(self) -> None:
        depth = self.space.max_depth
        tau0 = self.config.pheromone.start
        n_end_contexts = len(self.space.options(self.END_CONTEXT_PARAM)) + 1
        self.tables["type"] = PheromoneTable(depth, n_end_contexts, 2, tau0)
        for param in PARAMS:
            n_values = len(self.space.options(param))
            self.tables[param] = PheromoneTable(depth, n_values + 1, n_values, tau0) # N_VALUES +1 CUZ FIRST ONE IS NO_CONTEXT

    def on_ant_movement(self, ctx: MoveContext) -> Step:
        end_context = self._context(self.END_CONTEXT_PARAM, ctx.prev)
        if self.select("type", ctx.depth, end_context, type_mask(ctx)) == TYPE_END:
            return END
        return LayerConfig(**{
            param: self.space.options(param)[self.select(param, ctx.depth, self._context(param, ctx.prev))]
            for param in PARAMS
        })

    def cells(self, depth: int, prev: LayerConfig | None, step: Step) -> Iterable[Cell]:
        end_context = self._context(self.END_CONTEXT_PARAM, prev)
        if step is END:
            return [(self.tables["type"], end_context, TYPE_END)]
        return [(self.tables["type"], end_context, TYPE_FC)] + [
            (self.tables[param], self._context(param, prev), self.space.value_index(param, getattr(step, param))) # THE SAME AS INDEPENDENT BUT WE USE THE PREVIOUS VALUE INDEX AS CONTEXT
            for param in PARAMS
        ]

    def _context(self, param: str, prev: LayerConfig | None) -> int:
        if prev is None: # THIS IS NO_CONTEXT INDEX
            return 0
        return self.space.value_index(param, getattr(prev, param)) + 1 # PLUS ONE CUZ FIRST ONE IS NO_CONTEXT
