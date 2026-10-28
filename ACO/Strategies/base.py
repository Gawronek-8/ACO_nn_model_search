"""Strategy interface: the colony calls the hooks, the strategy owns the pheromones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..utils import LayerConfig, PheromoneTable, Step

if TYPE_CHECKING:
    from ..Colony import Ant, ColonyConfig, SearchSpace

TYPE_FC, TYPE_END = 0, 1

# (table, context, value) touched by one step
Cell = tuple[PheromoneTable, int, int]


@dataclass(frozen=True)
class SearchContext:
    space: SearchSpace # POSSIBLE VALUES FOR THE LAYERS
    config: ColonyConfig
    rng: np.random.Generator


@dataclass(frozen=True)
class IterationContext:
    iteration: int
    depth_limit: int
    ants: tuple[Ant, ...] = () # elipsis means here a tuple full of Ants
    best_ant: Ant | None = None  # best ant found so far
    update_ant: Ant | None = None  # ant whose path gets the global update


@dataclass(frozen=True)
class MoveContext:
    depth: int  # index of the layer being chosen, 0 = first hidden layer
    prev: LayerConfig | None  # previous hidden layer, None when coming from the input
    steps: tuple[Step, ...] # elpsis here means that this is a tuple gfull of Steps
    depth_limit: int
    can_end: bool  # False until the path has min_depth layers


@dataclass(frozen=True)
class AntContext:
    iteration: int
    ant: Ant


def type_mask(ctx: MoveContext) -> np.ndarray | None: # THIS  COULD MAKE SENSE 
    """Mask for the type table that blocks END until the path is deep enough."""
    return None if ctx.can_end else np.array([True, False])


class PheromoneStrategy(ABC):
    """Base for all pheromone models."""

    def on_search_start(self, ctx: SearchContext) -> None:
        self.space = ctx.space
        self.config = ctx.config
        self.rng = ctx.rng
        self.tables: dict[str, PheromoneTable] = {}
        self.build_tables() # CREATES THE TABLES FOR EACH PARAM, in CONFIG DEPENDENT IT IS ONE ARRAY - THEY STORE DEPTH, CONTEXT, VALUE AND PHEROMONE FOR THAT VALUE

    def on_iteration_start(self, ctx: IterationContext) -> None:
        pass

    @abstractmethod
    def build_tables(self) -> None:
        """Fills self.tables, sized from self.space."""

    @abstractmethod
    def on_ant_movement(self, ctx: MoveContext) -> Step:
        """Chooses the ant's next step: a LayerConfig or END."""

    @abstractmethod
    def cells(self, depth: int, prev: LayerConfig | None, step: Step) -> Iterable[Cell]:
        """Pheromone cells behind one step, used by both updates."""

    def on_ant_step_end(self, ctx: MoveContext, step: Step) -> None: # DOES THE SAME THING AS BELOW (IF THE ANTS WERE PARALLEL IT WOULD DIFFER)
        if self.config.local_update == "per_move":
            self._local_update(ctx.depth, ctx.prev, step)

    def on_local_step_end(self, ctx: AntContext) -> None: # LITERALLY THE SAME BUT DONE IN DIFFERENT STEP
        if self.config.local_update == "per_ant":
            for depth, prev, step in ctx.ant.transitions():
                self._local_update(depth, prev, step)

    def on_global_step_end(self, ctx: IterationContext) -> None:
        if ctx.update_ant is None:
            return
        evaporation = self.config.pheromone.evaporation
        score = ctx.update_ant.score
        for depth, prev, step in ctx.update_ant.transitions(): # FOR EVERY STEP OF BEST ANT
            for table, context, value in self.cells(depth, prev, step): # FOR EVERY CONTEXT AND VALUE INDEX IN THE CORRECT TABLE
                table.global_update(depth, context, value, evaporation, score) # UPDATE THE PHEROMONE VALYUE FOR THAT MOVE

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copies of all pheromone tables, e.g. for heatmaps."""
        return {name: table.tau.copy() for name, table in self.tables.items()}

    def select(self, table: str, depth: int, context: int, mask: np.ndarray | None = None) -> int:
        return self.tables[table].select(depth, context, self.config.greediness, self.rng, mask)

    def _local_update(self, depth: int, prev: LayerConfig | None, step: Step) -> None:
        decay = self.config.pheromone.local_decay
        for table, context, value in self.cells(depth, prev, step):
            table.local_update(depth, context, value, decay)
